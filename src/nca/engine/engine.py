"""The engine: runs a training rollout with a chosen backend, block compilation and
segmented gradient checkpointing, behind one interface the training loop and the benchmark
share.

    engine = Engine(model, EngineConfig(backend="fused", compile=True, block=4), train_cfg, dev)
    x, loss, grad_norm = engine.train_iteration(batch, seeds, c0, loss_weights, grad_clip, apply_update)

Options (config section `engine:`; see docs/FAST_ENGINE.md):

* `backend`: `reference` (the step of `nca.model.NCA`) or `fused` (`engine.ops`); the three
  pieces of the fused backend can be toggled one by one (`project_before_pad`, `fused_mix`,
  `fused_update`) for ablations;
* `compile` / `block` / `compile_mode`: `torch.compile` blocks of `block` consecutive steps
  (`compile_mode: reduce-overhead` adds CUDA graphs; the compiled function is the same for
  every block of the rollout, so a run compiles once per distinct block length and grad mode);
* `checkpoint_segment`: `none`, an integer q (recompute the activations of segments of q steps
  during the backward pass, keeping only the segment boundaries) or `auto` (try the candidates
  of `checkpoint_candidates` on one iteration and keep the fastest whose peak memory stays
  under `vram_budget` x the GPU's memory);
* `update_after_rollout`: apply the optimiser and EMA update after the last step of the
  rollout (the paper's protocol: the whole rollout uses one set of weights; what
  `nca.train.train` does too since 2026-10-13) instead of right after the backward pass of
  the chunk (the order of the runs published before that date, which lets the steps after
  the chunk run with the new weights; only relevant when `chunk < rollout`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..model import NCA
from .rng import StepDraw, draw_step
from .step import StepOptions, draws_to_lists, make_step, run_block

DEFAULT_CANDIDATES = (0, 16, 8, 4, 2)  # 0 = no checkpointing, tried first (fastest when it fits)


def clone_batch(batch: dict, model: NCA | None = None) -> dict:
    """A copy of a training batch that can be rolled out and backpropagated again.

    `ReplayPool.sample` writes the *current* task embedding into the fresh rows of an ARC
    batch with its autograd history, and a backward pass frees that history; a plain
    `clone()` would keep a dead graph and the second backward would fail. The copy therefore
    detaches the state and, for models with a task embedding, re-writes the embedding of the
    fresh rows (`age == 0`, the rows `sample` wrote it into) from the live parameter, so the
    tensor values are unchanged and the embedding's gradient is still measured."""
    out = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    if model is not None and model.cfg.n_tasks and out.get("task_ids") is not None:
        rows = out["age"] == 0 if out.get("age") is not None else None
        out["x"] = model.write_task_embedding(out["x"], out["task_ids"], rows)
    return out


@dataclass
class EngineConfig:
    backend: str = "fused"
    compile: bool = False
    block: int = 1
    compile_mode: str | None = None
    compile_backward_only: bool = False
    checkpoint_segment: int | str | None = "auto"
    checkpoint_candidates: tuple[int, ...] = DEFAULT_CANDIDATES
    vram_budget: float = 0.85
    update_after_rollout: bool = True
    project_before_pad: bool | None = None
    fused_mix: bool | None = None
    fused_update: bool | None = None
    kernel_mix: bool = False
    kernel_backward: bool | str = False
    triton_mix: bool = False
    microbatch: int | None = None
    checkpoint_recursive: bool = False
    channels_last: bool = False
    normalized_backward: bool = False
    compile_update: bool = False
    sparse_mlp: bool = False
    sparse_sobel: bool = False
    cache_fire: bool = False
    checkpoint_mlp: bool = False
    skip_zero_loss: bool = False
    linear_mlp: bool = False
    clone_microbatch_grads: bool = False
    linear_value: bool = False
    attention_split_k: int = 1
    iteration_graph: bool = False
    triton_mlp: bool = False
    dynamo_cache_size: int = 64

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_segment, str):
            s = self.checkpoint_segment.lower()
            if s in ("none", "off", "false", ""):
                self.checkpoint_segment = None
            elif s == "auto":
                self.checkpoint_segment = "auto"
            else:
                self.checkpoint_segment = int(s)
        if self.checkpoint_segment == 0:
            self.checkpoint_segment = None
        self.checkpoint_candidates = tuple(int(c) for c in self.checkpoint_candidates)
        assert self.block >= 1, "block must be >= 1"
        assert self.backend in ("reference", "fused"), f"unknown backend {self.backend!r}"
        if self.compile_backward_only and (not self.compile or self.compile_mode is not None):
            raise ValueError("compile_backward_only requires compile=true and compile_mode=null")
        if self.microbatch is not None and self.microbatch < 1:
            raise ValueError("microbatch must be positive")
        if (
            self.microbatch is not None
            and self.compile
            and self.compile_mode in ("reduce-overhead", "max-autotune")
            and not self.clone_microbatch_grads
        ):
            # A later microbatch can overwrite captured gradient buffers still used
            # by accumulation (observed on ARC/H100). Plain compilation is validated.
            raise ValueError(
                "CUDA Graphs with microbatch gradient accumulation are unsupported; "
                "use compile_mode=null with microbatching, or clone_microbatch_grads=true"
            )
        if self.kernel_backward not in (False, True, "triton"):
            raise ValueError("kernel_backward must be false, true, or 'triton'")
        if self.attention_split_k not in (1, 2, 4, 8, 16, 32):
            raise ValueError("attention_split_k must be a power of two in 1..32")

    @classmethod
    def from_dict(cls, d: dict | None) -> EngineConfig:
        return cls(**(d or {}))

    def step_options(self) -> StepOptions:
        base = StepOptions.for_backend(self.backend)
        return StepOptions(
            base.project_before_pad if self.project_before_pad is None else self.project_before_pad,
            base.fused_mix if self.fused_mix is None else self.fused_mix,
            base.fused_update if self.fused_update is None else self.fused_update,
            self.kernel_mix,
            self.kernel_backward,
            self.triton_mix,
            self.normalized_backward,
            self.compile_update,
            self.sparse_mlp,
            self.sparse_sobel,
            self.checkpoint_mlp,
            self.linear_mlp,
            self.linear_value,
            self.attention_split_k,
            self.triton_mlp,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EngineState:
    """What the engine decided or measured (saved in run.json and the checkpoint)."""

    segment: int | None = None  # the checkpoint segment in use (None = no checkpointing)
    compile_seconds: float = 0.0
    autotune: list[dict] = field(default_factory=list)
    compiled: bool = False


class Engine:
    def __init__(
        self, model: NCA, cfg: EngineConfig, train_cfg: dict, device: torch.device
    ) -> None:
        self.model = model
        if cfg.channels_last:
            model.to(memory_format=torch.channels_last)
        self.cfg = cfg
        self.tr = train_cfg
        self.device = device
        self.opts = cfg.step_options()
        self.step = make_step(model, self.opts)
        self.state = EngineState()
        self._fire_cache = None
        self._loss_mask_cache = None
        self._iteration_graph = None
        self._cuda_graphs = (
            cfg.compile
            and device.type == "cuda"
            and cfg.compile_mode in ("reduce-overhead", "max-autotune")
        )
        self.N, self.Nc = int(train_cfg["rollout"]), int(train_cfg["chunk"])
        if cfg.checkpoint_segment != "auto":
            self.state.segment = cfg.checkpoint_segment
        self._validate_checkpoint_mode()
        self._block_fn = run_block
        if cfg.compile:
            from torch import _dynamo

            _dynamo.config.cache_size_limit = max(
                _dynamo.config.cache_size_limit, cfg.dynamo_cache_size
            )
            if cfg.compile_backward_only:
                from .compile import backward_only_backend

                self._block_fn = torch.compile(
                    run_block, dynamic=False, backend=backward_only_backend()
                )
            else:
                self._block_fn = torch.compile(run_block, dynamic=False, mode=cfg.compile_mode)
            self.state.compiled = True

    def _validate_checkpoint_mode(self) -> None:
        if self._cuda_graphs and self.state.segment and self.state.segment < self.Nc:
            raise ValueError(
                "CUDA Graphs with checkpoint recomputation are unsupported in this engine; "
                "use compile_mode=null or checkpoint_segment=null"
            )

    # ------------------------------------------------------------------ random draws
    @property
    def noise_cfg(self) -> dict | None:
        return self.tr["noise"] if self.tr.get("perturb", True) else None

    def draws(self, seeds, shape) -> list[StepDraw]:
        p = self.model.cfg.fire_rate
        if self._fire_cache is None:
            return [draw_step(int(s), shape, p, self.noise_cfg, self.device) for s in seeds]
        draws = []
        for seed in seeds:
            key = (int(seed), tuple(shape))
            if key not in self._fire_cache:
                self._fire_cache[key] = draw_step(int(seed), shape, p, None, self.device)
            draws.append(self._fire_cache[key])
        return draws

    # ------------------------------------------------------------------ running steps
    def _blocks(
        self, seeds, x, given4, clamp, clocks, targets, loss_mask, weights, amp, active_losses=None
    ):
        """Run `len(seeds)` steps in blocks of `cfg.block` (graph-building or not depends on
        the caller's grad mode). `amp` enables bf16 autocast for these steps: the training
        loop applies it to the differentiable chunk only, like `nca.train.train`, so the
        steps before and after the chunk run in fp32 whatever `train.amp` says."""
        loss_sum = None
        shape = tuple(x.shape)
        b = self.cfg.block
        for i in range(0, len(seeds), b):
            draws = self.draws(seeds[i : i + b], shape)
            fires, noises, nmasks = draws_to_lists(draws)
            x, part = self._block_fn(
                self.model,
                self.step,
                x,
                given4,
                clamp,
                fires,
                noises,
                nmasks,
                None if clocks is None else clocks[i : i + b],
                targets,
                loss_mask,
                None if weights is None else weights[i : i + b],
                self.tr["loss"],
                bool(amp),
                None if active_losses is None else active_losses[i : i + b],
            )
            if self._cuda_graphs and not torch.is_grad_enabled():
                # Inference captures can release their outputs when the next graph
                # enters training mode. Own the state across that transition.
                x = x.clone()
            if part is not None:
                loss_sum = part if loss_sum is None else loss_sum + part
        return x, loss_sum

    def run_steps(
        self,
        x: torch.Tensor,
        given4: torch.Tensor,
        clamp: torch.Tensor,
        seeds,
        clocks: torch.Tensor | None,
        grad: bool,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        active_losses: tuple[bool, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the steps `seeds` from state `x`; with `grad`, build the graph (in checkpointed
        segments when configured, under bf16 autocast when `train.amp`) and return the
        weighted loss sum; without, run in fp32 under `no_grad` (the reference policy: the
        steps outside the BPTT chunk never use autocast)."""
        if not grad:
            with torch.no_grad():
                return self._blocks(
                    seeds, x, given4, clamp, clocks, targets, loss_mask, None, False
                )
        amp = bool(self.tr["amp"])
        q = self.state.segment
        if not q or q >= len(seeds):
            return self._blocks(
                seeds, x, given4, clamp, clocks, targets, loss_mask, weights, amp, active_losses
            )
        if self.cfg.checkpoint_recursive:

            def recurse(state, lo, hi):
                if hi - lo <= q:
                    return self._blocks(
                        seeds[lo:hi],
                        state,
                        given4,
                        clamp,
                        None if clocks is None else clocks[lo:hi],
                        targets,
                        loss_mask,
                        None if weights is None else weights[lo:hi],
                        amp,
                        None if active_losses is None else active_losses[lo:hi],
                    )
                mid = (lo + hi) // 2
                state, left = grad_checkpoint(
                    lambda a: recurse(a, lo, mid), state, use_reentrant=False
                )
                state, right = grad_checkpoint(
                    lambda a: recurse(a, mid, hi), state, use_reentrant=False
                )
                return state, right if left is None else left if right is None else left + right

            return recurse(x, 0, len(seeds))
        loss_sum = None
        for i in range(0, len(seeds), q):
            seg_seeds = seeds[i : i + q]
            seg_clocks = None if clocks is None else clocks[i : i + q]
            seg_w = None if weights is None else weights[i : i + q]
            seg_active = None if active_losses is None else active_losses[i : i + q]

            def segment(
                x,
                given4,
                clamp,
                targets,
                loss_mask,
                seg_w,
                seeds=seg_seeds,
                clk=seg_clocks,
                active=seg_active,
            ):
                return self._blocks(
                    seeds, x, given4, clamp, clk, targets, loss_mask, seg_w, amp, active
                )

            x, part = grad_checkpoint(
                segment, x, given4, clamp, targets, loss_mask, seg_w, use_reentrant=False
            )
            if part is not None:
                loss_sum = part if loss_sum is None else loss_sum + part
        return x, loss_sum

    def _active_losses(self, weights):
        if not self.cfg.skip_zero_loss or weights.requires_grad:
            return None
        cached = self._loss_mask_cache
        if cached is None or cached[0] is not weights or cached[1] != weights._version:
            mask = tuple(bool(w != 0) for w in weights.detach().cpu().tolist())
            # Retain a differentiable zero loss when every weight is zero.
            self._loss_mask_cache = (weights, weights._version, mask if any(mask) else None)
        return self._loss_mask_cache[2]

    # ------------------------------------------------------------------ one training iteration
    def train_iteration(
        self,
        batch: dict,
        seeds,
        c0: int,
        weights: torch.Tensor,
        grad_clip: float,
        apply_update: Callable[[torch.Tensor], None] | None,
        *,
        _loss_scale=1.0,
        _microbatching=False,
        _cache_active=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll the batch out for `rollout` steps with the BPTT chunk at `[c0, c0 + chunk)`,
        backpropagate the loss (weighted per chunk step by `weights`, normalised by their sum),
        clip the gradients and call `apply_update(grad_norm)` (optimiser + EMA) either after the rollout
        (`update_after_rollout`) or right after the backward pass. Returns the final state, the
        loss and the gradient norm (a tensor: no CPU synchronisation here)."""
        model = self.model
        self._validate_checkpoint_mode()
        if self.cfg.iteration_graph:
            if self._iteration_graph is None:
                from .iteration_graph import IterationGraph

                self._iteration_graph = IterationGraph(self, batch, seeds, weights, grad_clip)
            return self._iteration_graph(self, batch, seeds, c0, weights, grad_clip, apply_update)
        noise = self.noise_cfg
        noiseless = noise is None or any(noise[k] <= 0 for k in ("sigma", "p_t", "p_s"))
        B, _, H, W = batch["x"].shape
        # Cache masks only within this iteration; never reuse draws across iterations.
        # Bound extra storage at 64 MiB and keep dense perturbation RNG unchanged.
        if (
            self.cfg.cache_fire
            and not _microbatching
            and not _cache_active
            and noiseless
            and B * H * W * len(seeds) * 4 <= 64 * 2**20
        ):
            self._fire_cache = {}
            try:
                return self.train_iteration(
                    batch,
                    seeds,
                    c0,
                    weights,
                    grad_clip,
                    apply_update,
                    _loss_scale=_loss_scale,
                    _cache_active=True,
                )
            finally:
                self._fire_cache = None
        if self.cfg.microbatch and batch["x"].shape[0] > self.cfg.microbatch and not _microbatching:
            if not self.cfg.update_after_rollout:
                raise ValueError("microbatch requires update_after_rollout=true")
            size = batch["x"].shape[0]
            shape = tuple(batch["x"].shape)
            old_draws = self.draws
            states, losses = [], []
            mask = batch.get("loss_mask")
            total = None if mask is None else mask.sum().clamp_min(1)
            try:
                for lo in range(0, size, self.cfg.microbatch):
                    hi = min(size, lo + self.cfg.microbatch)
                    part = {
                        k: (v[lo:hi] if torch.is_tensor(v) and v.shape[:1] == (size,) else v)
                        for k, v in batch.items()
                    }
                    part = clone_batch(part, model)
                    scale = (hi - lo) / size if mask is None else mask[lo:hi].sum() / total

                    def sliced_draws(step_seeds, ignored_shape, lo=lo, hi=hi):
                        return [
                            StepDraw(
                                d.fire[lo:hi].clone(),
                                None if d.noise is None else d.noise[lo:hi].clone(),
                                None if d.noise_mask is None else d.noise_mask[lo:hi].clone(),
                            )
                            for d in old_draws(step_seeds, shape)
                        ]

                    self.draws = sliced_draws
                    state, loss, _ = self.train_iteration(
                        part, seeds, c0, weights, None, None, _loss_scale=scale, _microbatching=True
                    )
                    states.append(state)
                    losses.append(loss * scale)
                    if self._cuda_graphs and self.cfg.clone_microbatch_grads:
                        # AccumulateGrad can adopt a graph-owned output on the first
                        # backward. Own every gradient before the next microbatch
                        # marks a new CUDA Graph iteration and reuses its buffers.
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad = parameter.grad.detach().clone()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if apply_update is not None:
                    apply_update(gn)
                return torch.cat(states), torch.stack(losses).sum(), gn
            finally:
                self.draws = old_draws
        if self._cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        x, given4, clamp = batch["x"], batch["given4"], batch["clamp"]
        if self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets, loss_mask = batch["targets"], batch.get("loss_mask")
        N, Nc = self.N, self.Nc
        clocks = torch.arange(N, device=self.device) if model.cfg.time_steps else None
        if c0 > 0:
            x, _ = self.run_steps(
                x, given4, clamp, seeds[:c0], None if clocks is None else clocks[:c0], False
            )
        x, loss_sum = self.run_steps(
            x,
            given4,
            clamp,
            seeds[c0 : c0 + Nc],
            None if clocks is None else clocks[c0 : c0 + Nc],
            True,
            targets,
            loss_mask,
            weights,
            self._active_losses(weights),
        )
        loss = loss_sum / weights.sum().clamp_min(1)
        # With normalized_backward the fused update saves its output for the backward, so the
        # final state is a saved activation: a compiled backward may reuse that buffer (seen with
        # triton_mix and norm_fired_only=false, where the returned state came back overwritten
        # with gradient-sized values). Own the state before backpropagating.
        saved_x = x.detach().clone() if self._cuda_graphs or self.cfg.normalized_backward else None
        saved_loss = loss.detach().clone() if self._cuda_graphs else None
        (loss * _loss_scale).backward()
        gn = (
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if grad_clip is not None
            else loss.new_zeros(())
        )
        x = saved_x if saved_x is not None else x.detach()
        if apply_update is not None and not self.cfg.update_after_rollout:
            apply_update(gn)
        if c0 + Nc < N:
            x, _ = self.run_steps(
                x,
                given4,
                clamp,
                seeds[c0 + Nc :],
                None if clocks is None else clocks[c0 + Nc :],
                False,
            )
        if apply_update is not None and self.cfg.update_after_rollout:
            apply_update(gn)
        return x, saved_loss if saved_loss is not None else loss.detach(), gn

    # ------------------------------------------------------------------ autotuning
    def autotune(
        self,
        batch: dict,
        seeds,
        c0: int,
        weights: torch.Tensor,
        quiet: bool = True,
        budget_bytes: float | None = None,
    ) -> int | None:
        """Pick the checkpoint segment: the fastest candidate whose peak allocated memory fits
        the budget. On CPU (no memory accounting) the fastest candidate wins. Runs one warm-up
        and one timed iteration per candidate on copies of the batch (`clone_batch`), without
        touching the weights (gradients are zeroed afterwards). Raises `RuntimeError` when no
        candidate fits (out of memory or above the budget): the budget is a contract, not a
        preference, so a run never silently starts with a configuration that exceeds it."""
        cuda = self.device.type == "cuda"
        budget = None
        if cuda:
            total = torch.cuda.get_device_properties(self.device).total_memory
            budget = self.cfg.vram_budget * total
        if budget_bytes is not None:
            budget = float(budget_bytes)
        cands = [c for c in self.cfg.checkpoint_candidates if c < self.Nc or c == 0]
        if self._cuda_graphs:
            # Repeated checkpointed backward calls violate CUDA Graph input-lifetime
            # invariants in PyTorch 2.8, even when firing-mask caching is disabled.
            cands = [0]
        if self.cfg.compile:  # keep whole blocks inside a segment
            cands = [c for c in cands if c == 0 or c % self.cfg.block == 0]
        if not cands:
            cands = [0]
        results = []
        for q in cands:
            self.state.segment = q or None
            ok, peak, secs = True, 0.0, float("inf")
            try:
                for _rep in range(2):
                    b = clone_batch(batch, self.model)
                    if cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(self.device)
                    t0 = time.perf_counter()
                    self.train_iteration(b, seeds, c0, weights, 1.0, None)
                    if cuda:
                        torch.cuda.synchronize()
                    secs = time.perf_counter() - t0
                    if cuda:
                        peak = torch.cuda.max_memory_allocated(self.device)
                    self.model.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                ok = False
                self.model.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.empty_cache()
            fits = ok and (budget is None or peak <= budget)
            results.append(dict(segment=q, seconds=secs, peak_bytes=peak, fits=fits))
            if not quiet:
                print(
                    f"  autotune segment={q or 'none'}: {secs:.3f} s/iter, peak {peak / 2**30:.2f} GiB, "
                    f"{'ok' if fits else 'too big'}"
                )
        fitting = [r for r in results if r["fits"]]
        self.state.autotune = results
        if not fitting:
            self.state.segment = None
            budget_txt = "no budget" if budget is None else f"budget {budget / 2**30:.2f} GiB"
            trials = ", ".join(
                f"segment {r['segment'] or 'none'}: "
                + (
                    "out of memory"
                    if r["seconds"] == float("inf")
                    else f"peak {r['peak_bytes'] / 2**30:.2f} GiB"
                )
                for r in results
            )
            raise RuntimeError(
                f"no checkpoint segment fits the memory budget ({budget_txt}; {trials}); "
                "lower train.batch_size or train.chunk, add smaller candidates to "
                "engine.checkpoint_candidates, or raise engine.vram_budget"
            )
        best = min(fitting, key=lambda r: r["seconds"])
        self.state.segment = best["segment"] or None
        return self.state.segment

    def to_dict(self) -> dict:
        return dict(config=self.cfg.to_dict(), options=asdict(self.opts), **asdict(self.state))
