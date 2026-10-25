"""The NCA step of each backend, and blocks of steps for compilation.

`make_step(model, opts)` returns a function with the signature of `NCA.step` extended by the
noise mask: `step(x, given4, clamp, fire, noise, noise_mask, clock) -> x`. The reference backend
is `NCA.step` itself; the fused backend replaces the fixed-attention perception by
`project_values` + `fixed_attention_mix` and the residual / noise / normalisation / clamp chain by
`fused_cell_update` (see `ops.py`). Each replacement can be switched on its own so that the
benchmark can attribute the gains.

`run_block` runs `n` consecutive steps and accumulates the supervised loss; it is what the
engine hands to `torch.compile` (one compiled graph per block of `n` steps instead of one per
step, and no Python between the steps of a block).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model import NCA, FixedAttention, KernelPerception, SobelPerception, _pad
from ..train import step_loss
from .ops import (
    fixed_attention_mix,
    fused_cell_update,
    linear_cell_update,
    linear_projection,
    project_values,
    shared_kernel_mix,
)
from .rng import StepDraw

_compiled_cell_update = torch.compile(fused_cell_update, dynamic=False)


@dataclass(frozen=True)
class StepOptions:
    """Which pieces of the fused backend are active."""

    project_before_pad: bool = True
    fused_mix: bool = True
    fused_update: bool = True
    kernel_mix: bool = False
    kernel_backward: bool | str = False
    triton_mix: bool = False
    normalized_backward: bool = False
    compile_update: bool = False
    sparse_mlp: bool = False
    sparse_sobel: bool = False
    checkpoint_mlp: bool = False
    linear_mlp: bool = False
    linear_value: bool = False
    attention_split_k: int = 1
    triton_mlp: bool = False

    @classmethod
    def for_backend(cls, backend: str) -> StepOptions:
        if backend == "reference":
            return cls(False, False, False)
        if backend == "fused":
            return cls(True, True, True)
        raise ValueError(f"unknown backend {backend!r} (reference | fused)")


def reference_step(model: NCA):
    """`NCA.step`, which ignores the noise mask (it recomputes it from the noise)."""

    def step(x, given4, clamp, fire, noise, noise_mask, clock):
        return model.step(x, given4, clamp, fire, noise, clock)

    return step


def make_step(model: NCA, opts: StepOptions):
    """The step function of the fused backend with the options `opts` (all off = reference)."""
    if not (
        opts.project_before_pad
        or opts.fused_mix
        or opts.fused_update
        or opts.kernel_mix
        or opts.kernel_backward
        or opts.triton_mix
        or opts.compile_update
        or opts.sparse_mlp
        or opts.sparse_sobel
        or opts.checkpoint_mlp
        or opts.linear_mlp
        or opts.linear_value
        or opts.triton_mlp
    ):
        return reference_step(model)
    cfg = model.cfg
    if opts.checkpoint_mlp and not opts.sparse_mlp:
        raise ValueError("checkpoint_mlp requires sparse_mlp")
    perceive = model.perceive
    is_attn = isinstance(perceive, FixedAttention)
    is_kernel = isinstance(perceive, KernelPerception)
    if opts.sparse_sobel and not (
        opts.sparse_mlp
        and isinstance(perceive, SobelPerception)
        and cfg.padding in ("wrap", "zero")
    ):
        raise ValueError(
            "sparse_sobel requires sparse_mlp and Sobel perception with wrap/zero padding"
        )

    def update(z):
        if opts.triton_mlp:
            if cfg.update != "mlp":
                raise ValueError("triton_mlp supports the ReLU MLP only")
            from .triton_mlp import fused_mlp_update

            return fused_mlp_update(z, model.update)
        return (
            linear_cell_update(z, model.update, cfg.update) if opts.linear_mlp else model.update(z)
        )

    def sense(xin: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
        """Perception + update MLP: the proposed dx (same as `NCA.sense_update`)."""
        if opts.sparse_mlp:
            from .ops import sparse_mlp_update, sparse_sobel_mlp_update

            B, C, H, W = xin.shape
            if cfg.update != "mlp":
                raise ValueError("sparse_mlp currently supports the ReLU MLP update only")
            if opts.sparse_sobel:
                return sparse_sobel_mlp_update(
                    xin, model.update, fire, cfg.padding, opts.checkpoint_mlp
                )
            z = perceive(_pad(xin, cfg.padding, model.pad_vec), H, W)
            return sparse_mlp_update(z, model.update, fire, opts.checkpoint_mlp)
        if is_kernel and (opts.kernel_mix or opts.kernel_backward):
            return update(
                shared_kernel_mix(
                    _pad(xin, cfg.padding, model.pad_vec),
                    perceive.weight,
                    reference_forward=not opts.kernel_mix,
                    triton_backward=opts.kernel_backward == "triton",
                )
            )
        if not is_attn or not (opts.project_before_pad or opts.fused_mix or opts.triton_mix):
            if opts.linear_mlp or opts.triton_mlp:
                H, W = xin.shape[-2:]
                return update(perceive(_pad(xin, cfg.padding, model.pad_vec), H, W))
            return model.sense_update(xin)
        B, C, H, W = xin.shape
        if opts.triton_mix:
            if cfg.padding not in ("wrap", "zero") or not xin.is_cuda:
                raise ValueError(
                    "triton_mix requires fixed attention with wrap or zero padding on CUDA"
                )
            from .triton_ops import wrapped_attention_mix_cuda

            v = (
                linear_projection(xin, perceive.value) if opts.linear_value else perceive.value(xin)
            ).view(B, perceive.heads, C // perceive.heads, H, W)
            z = wrapped_attention_mix_cuda(v, perceive.mix, cfg.padding, opts.attention_split_k)
            return update(z.reshape(B, C, H, W))
        if opts.project_before_pad:
            if opts.linear_value and cfg.padding in ("wrap", "zero"):
                vp = _pad(linear_projection(xin, perceive.value), cfg.padding, None)
            else:
                vp = project_values(xin, perceive.value.weight, cfg.padding, model.pad_vec)
        else:
            padded = _pad(xin, cfg.padding, model.pad_vec)
            vp = (
                linear_projection(padded, perceive.value)
                if opts.linear_value
                else perceive.value(padded)
            )
        vp = vp.view(B, perceive.heads, C // perceive.heads, H + 2, W + 2)
        if opts.fused_mix:
            z = fixed_attention_mix(vp, perceive.mix)
        else:  # the reference loop over the 9 neighbours, on the pre-projected values
            from ..model import OFFSETS

            z = None
            for n, (dy, dx) in enumerate(OFFSETS):
                vn = vp[:, :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
                term = vn * perceive.mix[None, :, n, None]
                z = term if z is None else z + term
        return update(z.reshape(B, C, H, W))

    def step(x, given4, clamp, fire, noise, noise_mask, clock):
        te = model.step_embedding(x, clock)  # sync-free since 0c1e3d4 (embedding lookup)
        if te is None:
            dx = sense(x, fire)
        else:
            dx = sense(x + te, fire) + te
        if (not opts.fused_update and not opts.compile_update) or model.public_update:
            # the fused cell update implements the default update only (no projection, noise
            # before the normalisation, sqrt_eps); the public-code switches run unfused
            return model.residual_update(x, dx, fire, noise, given4, clamp, noise_mask)
        update = _compiled_cell_update if opts.compile_update else fused_cell_update
        return update(
            x,
            dx,
            fire,
            noise,
            noise_mask,
            given4,
            clamp,
            cfg.norm_groups,
            bool(cfg.norm_groups and cfg.norm_fired_only),
            cfg.in_slice,
            opts.normalized_backward,
        )

    return step


def run_block(
    model: NCA,
    step,
    x: torch.Tensor,
    given4: torch.Tensor,
    clamp: torch.Tensor,
    fires: list[torch.Tensor],
    noises: list[torch.Tensor] | None,
    noise_masks: list[torch.Tensor] | None,
    clocks: torch.Tensor | None,
    targets: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    loss_weights: torch.Tensor | None,
    loss_kind: str,
    amp: bool,
    active_losses: tuple[bool, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run `len(fires)` steps. With `loss_weights` (one per step, 0 = not supervised) the
    per-step losses are accumulated with those weights and returned; the loss is computed at
    every step so that the graph has the same shape whatever the supervision window (it is a
    small elementwise reduction next to the update MLP). `clocks` (n,) long is only used by
    models with a time embedding (ARC)."""
    loss_sum = None
    device_type = x.device.type
    for i in range(len(fires)):
        noise = None if noises is None else noises[i]
        nmask = None if noise_masks is None else noise_masks[i]
        clock = None if clocks is None else clocks[i]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            x = step(x, given4, clamp, fires[i], noise, nmask, clock)
        x = x.float()
        if loss_weights is not None and (active_losses is None or active_losses[i]):
            term = loss_weights[i] * step_loss(model, x, targets, loss_kind, loss_mask)
            loss_sum = term if loss_sum is None else loss_sum + term
    return x, loss_sum


def draws_to_lists(draws: list[StepDraw]):
    """Split a list of `StepDraw` into the tensor lists `run_block` takes."""
    fires = [d.fire for d in draws]
    if draws and draws[0].noise is not None:
        return fires, [d.noise for d in draws], [d.noise_mask for d in draws]
    return fires, None, None
