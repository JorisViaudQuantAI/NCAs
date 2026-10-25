"""Fused operators of the fast engine, with explicit backward passes.

Two pieces of the NCA step are rewritten as `torch.autograd.Function`s so that the forward
pass saves fewer tensors and the backward pass runs fewer kernels than autograd's chain of
elementwise operations. The mathematics is unchanged; `tests/test_engine_ops.py` checks both
operators against the reference modules of `nca.model` (outputs and gradients) and with
`torch.autograd.gradcheck` in float64.

1. **Fixed-attention mixing** (`fixed_attention_mix`). The paper's fixed-attention sensing is

       z_i^(k) = sum_{delta in N} A_{i,delta}^(k) v_{i+delta}^(k),    v_j^(k) = V^(k) x_j,

   with per-position, per-head mixing weights A over the 9 neighbours. Its backward is

       dL/dv_j^(k)      = sum_delta A_{j-delta,delta}^(k) dL/dz_{j-delta}^(k)
       dL/dA_{i,delta}  = sum_{b,c} dL/dz_{b,i,c}^(k) v_{b,i+delta,c}^(k).

   The forward accumulates the 9 shifted products in place into one output (no temporary per
   neighbour); the backward scatters into the padded gradient of v in place and reduces the
   mixing-weight gradient head by head, without materialising a 9-copy "unfolded" tensor.

2. **Projection before padding** (`project_values`). The value projection V is linear and has
   no bias, so projecting the H x W grid and then padding the projected values gives the same
   padded tensor as projecting the (H+2) x (W+2) padded grid, for circular padding (border
   cells are copies), zero padding (V 0 = 0) and wall padding (V pad_vec, a constant computed
   once). For a 9 x 9 Sudoku this projects 81 cells instead of 121.

3. **Fused cell update** (`fused_cell_update`): residual update with the firing mask, additive
   noise, per-cell group normalisation and clamping of the given cells,

       u = x + m * dx + eta,   y_g = u_g / sqrt(||u_g||^2 + eps) (per group g),   out = clamp(y),

   with the normalisation applied to every cell, or only to the cells that fired or received
   noise (`norm_fired_only`). The backward of the normalisation is

       g_u = g_y / r - u (u . g_y) / r^3,   r = sqrt(||u||^2 + eps),

   per group, followed by the identity on the cells that were not normalised and zero on the
   clamped channels of given cells. The forward saves `u` and the small masks instead of the
   half-dozen state-sized intermediates autograd keeps for the same chain.

Mixed precision: under bf16 autocast the value projection and the update MLP run in bf16
while the mixing weights and the state stay fp32, exactly like the reference modules, so both
operators compute in the promoted dtype of their inputs and cast their gradients back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model import OFFSETS, _pad

NORM_EPS = 1e-4  # the reference normalisation's epsilon (nca.model.NCA.normalize)


def linear_projection(x, layer):
    """A 1x1 convolution expressed as GEMM, retaining its autocast boundary."""
    B, C, H, W = x.shape
    rows = x.permute(0, 2, 3, 1).reshape(-1, C)
    values = F.linear(rows, layer.weight[:, :, 0, 0], layer.bias)
    return values.reshape(B, H, W, -1).permute(0, 3, 1, 2)


def linear_cell_update(z, update, kind):
    """The same 1x1 MLP/SwiGLU weights through GEMM, with channels last in the output."""
    B, C, H, W = z.shape
    rows = z.permute(0, 2, 3, 1).reshape(-1, C)
    hidden = F.linear(rows, update.fc1.weight[:, :, 0, 0], update.fc1.bias)
    if kind == "swiglu":
        gate, value = hidden.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
    else:
        hidden = F.relu(hidden)
    values = F.linear(hidden, update.fc2.weight[:, :, 0, 0], update.fc2.bias)
    return values.reshape(B, H, W, -1).permute(0, 3, 1, 2)


def _shared_kernel_backward(gout, xp, weight):
    B, C, Hp, Wp = xp.shape
    K, H, W = weight.shape[0], Hp - 2, Wp - 2
    dtype = torch.float32 if xp.dtype in (torch.float16, torch.bfloat16) else xp.dtype
    g = gout.reshape(B, K, C, H, W).to(dtype)
    gx = torch.zeros(xp.shape, device=xp.device, dtype=dtype)
    gw = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    for dy in range(3):
        for dx in range(3):
            gx[:, :, dy : dy + H, dx : dx + W] += (
                g * weight[None, :, :, dy, dx, None, None].to(dtype)
            ).sum(dim=1)
            gw[:, 0, dy, dx] = (g * xp[:, None, :, dy : dy + H, dx : dx + W].to(dtype)).sum(
                dim=(0, 2, 3, 4)
            )
    return gx.to(xp.dtype), gw.to(weight.dtype)


_compiled_kernel_backward = torch.compile(_shared_kernel_backward, dynamic=False)


class _SharedKernelMix(torch.autograd.Function):
    """Shared 3x3 kernels in head-major order, without B*C single-channel convs."""

    @staticmethod
    def forward(ctx, xp, weight, reference_forward, triton_backward):
        B, C, Hp, Wp = xp.shape
        K, H, W = weight.shape[0], Hp - 2, Wp - 2
        dtype = torch.float32 if xp.dtype in (torch.float16, torch.bfloat16) else xp.dtype
        ctx.save_for_backward(xp, weight)
        ctx.reference_forward = reference_forward
        ctx.triton_backward = triton_backward
        if reference_forward:
            z = F.conv2d(xp.reshape(B * C, 1, Hp, Wp), weight)
            return z.view(B, C, K, H, W).transpose(1, 2).reshape(B, K * C, H, W)
        z = torch.zeros((B, K, C, H, W), device=xp.device, dtype=dtype)
        for dy in range(3):
            for dx in range(3):
                z.addcmul_(
                    xp[:, None, :, dy : dy + H, dx : dx + W].to(dtype),
                    weight[None, :, :, dy, dx, None, None].to(dtype),
                )
        return z.reshape(B, K * C, H, W).to(xp.dtype)

    @staticmethod
    def backward(ctx, gout):
        xp, weight = ctx.saved_tensors
        if ctx.triton_backward:
            from .triton_ops import shared_kernel_backward_cuda

            gx, gw = shared_kernel_backward_cuda(gout, xp, weight)
        elif ctx.reference_forward and xp.is_cuda and not torch.compiler.is_compiling():
            gx, gw = _compiled_kernel_backward(gout, xp, weight)
        else:
            gx, gw = _shared_kernel_backward(gout, xp, weight)
        return gx, gw, None, None


def shared_kernel_mix(
    xp: torch.Tensor, weight: torch.Tensor, reference_forward=False, triton_backward=False
) -> torch.Tensor:
    """Same head-major output and autocast boundaries as KernelPerception's conv2d.

    The fp32 accumulation is cast back to the convolution dtype in both forward
    and backward. Compilation can fuse the nine spatial terms and reductions.
    """
    if torch.is_autocast_enabled(xp.device.type):
        dtype = torch.get_autocast_dtype(xp.device.type)
        xp, weight = xp.to(dtype), weight.to(dtype)
    if triton_backward and not xp.is_cuda:
        raise ValueError("Triton kernel backward requires CUDA")
    return _SharedKernelMix.apply(xp, weight, reference_forward, triton_backward)


def _slices(dy: int, dx: int, H: int, W: int) -> tuple[slice, ...]:
    return (
        slice(None),
        slice(None),
        slice(None),
        slice(1 + dy, 1 + dy + H),
        slice(1 + dx, 1 + dx + W),
    )


class _FixedAttentionMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vp: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        # vp: (B, K, D, H+2, W+2) padded projected values; mix: (K, 9, H, W)
        B, K, D, Hp, Wp = vp.shape
        H, W = Hp - 2, Wp - 2
        dtype = torch.promote_types(vp.dtype, mix.dtype)
        z = torch.zeros((B, K, D, H, W), dtype=dtype, device=vp.device)
        for n, (dy, dx) in enumerate(OFFSETS):
            z.addcmul_(vp[_slices(dy, dx, H, W)], mix[None, :, n, None])
        ctx.save_for_backward(vp, mix)
        return z

    @staticmethod
    def backward(ctx, gz: torch.Tensor):
        vp, mix = ctx.saved_tensors
        H, W = gz.shape[-2:]
        need_v, need_a = ctx.needs_input_grad[:2]
        dtype = torch.promote_types(vp.dtype, mix.dtype)
        gz = gz.to(dtype)
        gvp = torch.zeros(vp.shape, dtype=dtype, device=vp.device) if need_v else None
        gmix = torch.empty(mix.shape, dtype=dtype, device=mix.device) if need_a else None
        for n, (dy, dx) in enumerate(OFFSETS):
            sl = _slices(dy, dx, H, W)
            if gvp is not None:
                gvp[sl].addcmul_(gz, mix[None, :, n, None])
            if gmix is not None:
                gmix[:, n] = (gz * vp[sl]).sum(dim=(0, 2))
        return (
            None if gvp is None else gvp.to(vp.dtype),
            None if gmix is None else gmix.to(mix.dtype),
        )


def fixed_attention_mix(vp: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """`z[b, k, c, i, j] = sum_n mix[k, n, i, j] * vp[b, k, c, i + dy_n, j + dx_n]` for the padded
    projected values `vp` (B, K, D, H+2, W+2) and the mixing weights `mix` (K, 9, H, W)."""
    return _FixedAttentionMix.apply(vp, mix)


def projected_pad_vec(value_weight: torch.Tensor, pad_vec: torch.Tensor) -> torch.Tensor:
    """`V pad_vec`: the border value of the projected grid under wall padding."""
    w = value_weight.reshape(value_weight.shape[0], -1)
    return w @ pad_vec.to(w.dtype)


def project_values(
    x: torch.Tensor, value_weight: torch.Tensor, padding: str, pad_vec: torch.Tensor
) -> torch.Tensor:
    """Project the unpadded grid with the 1x1 value projection, then pad: (B, C, H+2, W+2)."""
    v = F.conv2d(x, value_weight)
    if padding == "wall":
        return _pad(v, "wall", projected_pad_vec(value_weight, pad_vec).to(v.dtype))
    return _pad(v, padding, None)


class _FusedCellUpdate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dx: torch.Tensor,
        fire: torch.Tensor,
        noise: torch.Tensor | None,
        noise_mask: torch.Tensor | None,
        given4: torch.Tensor,
        clamp: torch.Tensor,
        groups: int,
        fired_only: bool,
        in_lo: int,
        in_hi: int,
        normalized_backward: bool,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        u = torch.addcmul(x, fire.to(x.dtype), dx.to(x.dtype))
        if noise is not None:
            u = u + noise
        r = moved = None
        if groups:
            ug = u.view(B, groups, C // groups, H, W)
            r = (ug.pow(2).sum(dim=2, keepdim=True) + NORM_EPS).sqrt()
            y = (ug / r).view(B, C, H, W)
            if fired_only:
                moved = fire > 0
                if noise_mask is not None:
                    moved = moved | noise_mask
                y = torch.where(moved, y, u)
        else:
            # Without normalisation the Jacobian is the identity before clamping.
            # u is freshly allocated, so it can be clamped in place and discarded
            # after forward instead of retaining a full state for every BPTT step.
            y = u
        y[:, in_lo:in_hi] = torch.where(given4, clamp.to(y.dtype), y[:, in_lo:in_hi])
        # A fully clamped normalization group has zero upstream gradient.
        # For aligned groups, the returned normalized state therefore suffices
        # for the Jacobian and can share storage with the next step's input.
        group_width = C // groups if groups else C
        ctx.normalized_backward = bool(
            groups and normalized_backward and in_lo % group_width == 0 and in_hi % group_width == 0
        )
        saved = y if ctx.normalized_backward else u
        ctx.save_for_backward(saved if groups else None, fire, given4, r, moved)
        ctx.groups, ctx.fired_only, ctx.in_lo, ctx.in_hi = groups, fired_only, in_lo, in_hi
        ctx.dx_dtype = dx.dtype
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        u, fire, given4, r, moved = ctx.saved_tensors
        lo, hi = ctx.in_lo, ctx.in_hi
        g_clamp = None
        if ctx.needs_input_grad[6]:
            g_clamp = g[:, lo:hi] * given4.to(g.dtype)
        gy = g.clone()
        gy[:, lo:hi].masked_fill_(given4, 0.0)
        if ctx.groups:
            B, C, H, W = u.shape
            G = ctx.groups
            ug = u.view(B, G, C // G, H, W)
            gyg = gy.view(B, G, C // G, H, W)
            dot = (ug * gyg).sum(dim=2, keepdim=True)
            if ctx.normalized_backward:
                gu = ((gyg - ug * dot) / r).view(B, C, H, W)
            else:
                gu = (gyg / r - ug * (dot / r.pow(3))).view(B, C, H, W)
            if ctx.fired_only:
                gu = torch.where(moved, gu, gy)
        else:
            gu = gy
        g_dx = (gu * fire).to(ctx.dx_dtype) if ctx.needs_input_grad[1] else None
        g_noise = gu if ctx.needs_input_grad[3] else None
        return gu, g_dx, None, g_noise, None, None, g_clamp, None, None, None, None, None


def fused_cell_update(
    x: torch.Tensor,
    dx: torch.Tensor,
    fire: torch.Tensor,
    noise: torch.Tensor | None,
    noise_mask: torch.Tensor | None,
    given4: torch.Tensor,
    clamp: torch.Tensor,
    groups: int,
    fired_only: bool,
    in_slice: slice,
    normalized_backward: bool = False,
) -> torch.Tensor:
    """Residual update + noise + group normalisation + clamping in one autograd node (see the
    module docstring). `noise_mask` marks the cells that received noise (used by
    `fired_only`); pass None when there is no noise."""
    return _FusedCellUpdate.apply(
        x,
        dx,
        fire,
        noise,
        noise_mask,
        given4,
        clamp,
        groups,
        fired_only,
        in_slice.start,
        in_slice.stop,
        normalized_backward,
    )


@torch.compiler.disable
def sparse_mlp_update(z, update, fire, checkpoint_mlp=False):
    """Evaluate the cellwise ReLU MLP only on fired cells, with exact autograd.

    Dynamic compaction synchronizes CUDA to get the row count. Measure its cost;
    intended for large pixel grids at low firing rates, not a default optimization.
    """
    B, C, H, W = z.shape
    rows = torch.nonzero(fire.reshape(-1) > 0, as_tuple=False).flatten()
    selected = z.permute(0, 2, 3, 1).reshape(-1, C).index_select(0, rows)
    return _selected_mlp_update(selected, update, rows, B, H, W, checkpoint_mlp)


def _selected_mlp_update(selected, update, rows, B, H, W, checkpoint_mlp=False):
    def mlp(selected):
        hidden = F.relu(F.linear(selected, update.fc1.weight[:, :, 0, 0], update.fc1.bias))
        return F.linear(hidden, update.fc2.weight[:, :, 0, 0], update.fc2.bias)

    if checkpoint_mlp and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint

        values = checkpoint(mlp, selected, use_reentrant=False)
    else:
        values = mlp(selected)
    dense = values.new_zeros((B * H * W, values.shape[1])).index_copy(0, rows, values)
    return dense.reshape(B, H, W, -1).permute(0, 3, 1, 2)


def selected_sobel_torch(x, rows, padding):
    """Sobel features at selected cells, preserving channel/filter interleaving.

    CPU oracle for the CUDA implementation. Padding and autocast match SobelPerception;
    omitting inactive rows is valid because their proposed update is multiplied by zero.
    """
    if padding not in ("wrap", "zero"):
        raise ValueError("selected Sobel supports wrap/zero padding")
    if torch.is_autocast_enabled(x.device.type):
        x = x.to(torch.get_autocast_dtype(x.device.type))
    B, C, H, W = x.shape
    xp = _pad(x, padding, None).permute(0, 2, 3, 1)
    b, i, j = rows // (H * W), (rows // W) % H, rows % W
    # Match a convolution's FP32 accumulator with BF16 inputs/output.
    dtype = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
    sx = torch.zeros((len(rows), C), device=x.device, dtype=dtype)
    sy = torch.zeros_like(sx)
    for dy in range(3):
        for dx in range(3):
            v = xp[b, i + dy, j + dx].to(dtype)
            sx = sx + v * ((dx - 1) * (2 if dy == 1 else 1) / 8)
            sy = sy + v * ((dy - 1) * (2 if dx == 1 else 1) / 8)
    center = xp[b, i + 1, j + 1].to(dtype)
    return torch.stack((center, sx, sy), dim=2).reshape(len(rows), 3 * C).to(x.dtype)


@torch.compiler.disable
def sparse_sobel_mlp_update(x, update, fire, padding, checkpoint_mlp=False):
    """Avoid the dense 3*C perception tensor for cells which do not fire."""
    B, _, H, W = x.shape
    rows = torch.nonzero(fire.reshape(-1) > 0, as_tuple=False).flatten()
    if x.is_cuda:
        from .triton_sobel import selected_sobel_cuda

        selected = selected_sobel_cuda(x, rows, padding)
    else:
        selected = selected_sobel_torch(x, rows, padding)
    return _selected_mlp_update(selected, update, rows, B, H, W, checkpoint_mlp)
