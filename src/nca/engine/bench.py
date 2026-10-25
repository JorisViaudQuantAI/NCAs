"""Measurements for the engine: time per iteration, peak memory, FLOPs, compile time, and
the numerical distance between two backends on the same rollout.

Everything here is used by `scripts/bench_engine.py` and `scripts/check_engine.py`; the JSON
they write carries the hardware, software and commit so that two measurements can be compared
later (docs/FAST_ENGINE.md, "How to read a benchmark file").
"""

from __future__ import annotations

import copy
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..model import NCA
from ..pool import ReplayPool
from ..tasks.base import Boards, Task
from ..train import lr_at
from .engine import Engine, EngineConfig, clone_batch
from .rng import step_seeds


def git_commit(root: str | Path | None = None) -> str | None:
    try:
        root = Path(root or Path(__file__).resolve().parents[3])
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - no git, no repo: the field is just missing
        return None


def git_dirty(root: str | Path | None = None) -> bool | None:
    """True when the working tree has uncommitted changes (a benchmark then does not measure
    the recorded commit); None when git is not available."""
    try:
        root = Path(root or Path(__file__).resolve().parents[3])
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return None


def hardware_info(device: torch.device) -> dict:
    info = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        device=str(device),
        cpu_threads=torch.get_num_threads(),
        commit=git_commit(),
        dirty=git_dirty(),
    )
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        info["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(
            gpu=p.name,
            gpu_memory_gb=p.total_memory / 2**30,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        info["triton"] = None
    return info


def probe_compile(device: torch.device) -> dict:
    """Can `torch.compile` run a tiny graph on this device (with its backward)? Reports the
    error text when it cannot (Windows without a Triton build, missing compiler, ...)."""
    t0 = time.perf_counter()
    try:
        import torch._dynamo

        torch._dynamo.reset()

        def f(a, w):
            return torch.relu(a @ w).sum()

        g = torch.compile(f)
        a = torch.randn(8, 8, device=device)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        g(a, w).backward()
        ok, err = bool(torch.isfinite(w.grad).all()), None
    except Exception as e:  # noqa: BLE001 - the point is to report it
        ok, err = False, f"{type(e).__name__}: {str(e)[:400]}"
    return dict(ok=ok, error=err, seconds=time.perf_counter() - t0)


def make_pool(
    model: NCA,
    task: Task,
    boards: Boards,
    batch_size: int,
    seed: int = 0,
    train_cfg: dict | None = None,
) -> ReplayPool:
    """Create a fixed micro-benchmark pool, or the actual training pool when configured."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    size = batch_size
    augment = False
    if train_cfg is not None:
        size *= train_cfg["pool_mult"] if train_cfg["replay"] else 1
        augment = train_cfg["augment"]
    return ReplayPool(
        model,
        task,
        boards,
        size,
        rng,
        gen,
        augment=augment,
        task_ids=boards.meta.get("task_id"),
    )


def make_batch(model: NCA, task: Task, boards: Boards, batch_size: int, seed: int = 0) -> dict:
    """A batch of fresh boards with freshly initialised states, like the first iteration. For
    a model with task embeddings (ARC) the rows are drawn as *fresh* rows so that, as in
    training, the task slice carries the differentiable embedding (`write_task_embedding`)."""
    ratio = 1.0 if model.cfg.n_tasks else 0.0
    _, batch = make_pool(model, task, boards, batch_size, seed).sample(batch_size, ratio)
    return batch


__all__ = ["clone_batch"]  # re-exported from .engine for the scripts and tests


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compiled_graphs() -> int:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def _custom_kernel_variants() -> int:
    """Count our direct Triton specializations, outside Inductor's graph counters.

    Triton 3.4 exposes each device cache as (kernel_cache, target, backend, binder).
    Do not import CUDA-only modules when measuring on CPU.
    """
    count = 0
    for name in ("nca.engine.triton_ops", "nca.engine.triton_sobel", "nca.engine.triton_mlp"):
        module = sys.modules.get(name)
        if module is not None:
            for value in vars(module).values():
                caches = getattr(value, "device_caches", None)
                if isinstance(caches, dict):
                    count += sum(len(cache[0]) for cache in caches.values())
    return count


def _cuda_graph_recordings(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    from torch._inductor.cudagraph_trees import get_container

    index = torch.cuda.current_device() if device.index is None else device.index
    manager = get_container(index).tree_manager
    return 0 if manager is None else int(manager.graph_counter.__reduce__()[1][0])


def time_iterations(
    engine: Engine,
    batch: dict,
    seeds: np.ndarray,
    c0: int,
    weights: torch.Tensor,
    iters: int,
    warmup: int,
    grad_clip: float = 1.0,
) -> dict:
    """Time `iters` training iterations (rollout + backward + clip, no optimiser step) after
    `warmup` untimed ones (which include compilation and autotuning caches). The weights are
    not changed; gradients are zeroed after every iteration. See `time_full_iterations` for
    the complete iteration (optimiser, EMA, replay)."""
    dev = engine.device
    first = None
    first_memory = {}
    warmup_memory = {}
    graphs_before_timing = _compiled_graphs()
    captures_before_timing = _cuda_graph_recordings(dev)
    custom_before_timing = _custom_kernel_variants()
    times = []
    for i in range(warmup + iters):
        b = clone_batch(batch, engine.model)
        if i == warmup:
            graphs_before_timing = _compiled_graphs()
            captures_before_timing = _cuda_graph_recordings(dev)
            custom_before_timing = _custom_kernel_variants()
        _sync(dev)
        if dev.type == "cuda" and i == warmup:
            warmup_memory = dict(
                warmup_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                warmup_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30,
            )
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        engine.train_iteration(b, seeds, c0, weights, grad_clip, None)
        _sync(dev)
        dt = time.perf_counter() - t0
        engine.model.zero_grad(set_to_none=True)
        if i == 0:
            first = dt
            if dev.type == "cuda":
                first_memory = dict(
                    first_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                    first_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30,
                )
        if i >= warmup:
            times.append(dt)
    out = dict(
        iters=iters,
        warmup=warmup,
        first_iteration_seconds=first,
        seconds_mean=float(np.mean(times)),
        seconds_std=float(np.std(times)),
        seconds_min=float(np.min(times)),
        seconds=times,
        **first_memory,
        **warmup_memory,
        timed_compiled_graphs=_compiled_graphs() - graphs_before_timing,
        timed_custom_kernel_variants=_custom_kernel_variants() - custom_before_timing,
        timed_cuda_graph_recordings=_cuda_graph_recordings(dev) - captures_before_timing,
        total_cuda_graph_recordings=_cuda_graph_recordings(dev),
    )
    if dev.type == "cuda":
        out["peak_allocated_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 2**30
    return out


def time_full_iterations(
    engine: Engine,
    pool: ReplayPool,
    train_cfg: dict,
    iters: int,
    warmup: int,
    seed: int = 0,
) -> dict:
    """Time `iters` *complete* training iterations after `warmup` untimed ones: replay sampling
    (with the configured perturbations), the rollout, backward, clipping, the AdamW step, the
    scheduler, the EMA update and the write-back to the pool, i.e. everything
    `nca.engine.train.train_fast` does per step except logging and evaluation. The model's
    weights (and the pool) are modified by the timed updates; the weights are restored
    afterwards so a benchmark can time several variants from one initialisation. Mirrors
    the loop body of `train_fast`."""
    from ..perturb import apply_damage, swap_indices

    tr = train_cfg
    model, dev = engine.model, engine.device
    # Keep the restoration copy off the GPU so it does not inflate the VRAM measurement.
    state0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    training0 = model.training
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, tr["steps"], tr))
    # train_fast shares this generator between replay sampling, perturbations and step seeds.
    rng = pool.rng
    N, Nc = engine.N, engine.Nc
    weights = _loss_weights(tr["loss_window"], Nc, dev)
    seed_ratio = tr["seed_ratio"] if tr["replay"] else 1.0
    perturb = bool(tr["perturb"])
    times = []
    model.train()
    first = None
    first_memory = {}
    warmup_memory = {}
    graphs_before_timing = _compiled_graphs()
    captures_before_timing = _cuda_graph_recordings(dev)
    custom_before_timing = _custom_kernel_variants()
    skipped = 0
    try:
        for i in range(warmup + iters):
            if i == warmup:
                graphs_before_timing = _compiled_graphs()
                captures_before_timing = _cuda_graph_recordings(dev)
                custom_before_timing = _custom_kernel_variants()
            _sync(dev)
            if dev.type == "cuda" and i == warmup:
                warmup_memory = dict(
                    warmup_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                    warmup_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30,
                )
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            idx, batch = pool.sample(tr["batch_size"], seed_ratio)
            if perturb:
                pool.replace_boards(batch, swap_indices(tr["batch_size"], tr["swap"]["p"], rng))
                d = tr["damage"]
                batch["x"], _ = apply_damage(
                    batch["x"], d["p"], d["n"], d["r_min"], d["r_max"], rng
                )
                batch["x"] = model.apply_clamp(batch["x"], batch["given4"], batch["clamp"])
            c0 = int(rng.integers(0, N - Nc + 1))
            seeds = step_seeds(rng, N)

            def apply_update(gn: torch.Tensor, step: int = i) -> None:
                nonlocal skipped
                if bool(torch.isfinite(gn)):
                    opt.step()
                else:
                    skipped += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                with torch.no_grad():
                    d = min(tr["ema"], (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)

            x, _, _ = engine.train_iteration(
                batch, seeds, c0, weights, tr["grad_clip"], apply_update
            )
            batch["x"] = x.detach()
            pool.write_back(idx, batch)
            _sync(dev)
            dt = time.perf_counter() - t0
            if first is None:
                first = dt
                if dev.type == "cuda":
                    first_memory = dict(
                        first_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                        first_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30,
                    )
            if i >= warmup:
                times.append(dt)
        memory = {}
        if dev.type == "cuda":
            memory = dict(
                peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30,
            )
    finally:
        model.load_state_dict(state0)
        model.zero_grad(set_to_none=True)
        model.train(training0)
    return dict(
        iters=iters,
        warmup=warmup,
        first_iteration_seconds=first,
        skipped_steps=skipped,
        pool_size=pool.size,
        augment=pool.augment,
        perturb=perturb,
        seconds_mean=float(np.mean(times)),
        seconds_std=float(np.std(times)),
        seconds_min=float(np.min(times)),
        seconds=times,
        **first_memory,
        **warmup_memory,
        timed_compiled_graphs=_compiled_graphs() - graphs_before_timing,
        timed_custom_kernel_variants=_custom_kernel_variants() - custom_before_timing,
        timed_cuda_graph_recordings=_cuda_graph_recordings(dev) - captures_before_timing,
        total_cuda_graph_recordings=_cuda_graph_recordings(dev),
        **memory,
    )


def _loss_weights(window, chunk: int, device: torch.device) -> torch.Tensor:
    from .train import loss_weights

    return loss_weights(window, chunk, device)


def time_forward(engine: Engine, batch: dict, seeds: np.ndarray, iters: int, warmup: int) -> dict:
    """Time the no-grad rollout of `len(seeds)` steps (inference-like)."""
    dev = engine.device
    times = []
    clocks = torch.arange(len(seeds), device=dev) if engine.model.cfg.time_steps else None
    for i in range(warmup + iters):
        b = clone_batch(batch, engine.model)
        _sync(dev)
        t0 = time.perf_counter()
        engine.run_steps(b["x"], b["given4"], b["clamp"], seeds, clocks, False)
        _sync(dev)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return dict(
        steps=len(seeds),
        seconds_mean=float(np.mean(times)),
        seconds_per_step=float(np.mean(times)) / len(seeds),
    )


def count_step_flops(engine: Engine, batch: dict, seed: int = 0) -> dict:
    """FLOPs of one step (forward only, and forward + backward) counted by
    `torch.utils.flop_counter.FlopCounterMode` (matmuls and convolutions only: elementwise
    work is not counted)."""
    from torch.utils.flop_counter import FlopCounterMode

    draw = engine.draws([seed], tuple(batch["x"].shape))[0]
    clock = torch.tensor(0, device=engine.device) if engine.model.cfg.time_steps else None
    x = batch["x"].detach().clone()
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            engine.step(
                x, batch["given4"], batch["clamp"], draw.fire, draw.noise, draw.noise_mask, clock
            )
    fwd = fc.get_total_flops()
    x = batch["x"].detach().clone().requires_grad_(True)
    with FlopCounterMode(display=False) as fc:
        y = engine.step(
            x, batch["given4"], batch["clamp"], draw.fire, draw.noise, draw.noise_mask, clock
        )
        y.float().pow(2).mean().backward()
    engine.model.zero_grad(set_to_none=True)
    return dict(forward=fwd, forward_backward=fc.get_total_flops())


def compare_backends(
    model: NCA,
    cfg_a: EngineConfig,
    cfg_b: EngineConfig,
    train_cfg: dict,
    batch: dict,
    seeds: np.ndarray,
    c0: int,
    weights: torch.Tensor,
) -> dict:
    """Run the same iteration with two engine configurations on copies of the model and report
    the largest absolute differences of the final state, the loss and the parameter
    gradients (and the gradient norms). A parameter with a gradient on one side only
    (`grad_only_a` / `grad_only_b`) or a non-finite value anywhere (`nonfinite`) is reported,
    and `grad_max_abs` is then infinite so that no tolerance passes: a missing gradient is a
    wrong backward, not a small difference."""
    outs = []
    for ecfg in (cfg_a, cfg_b):
        m = copy.deepcopy(model)
        eng = Engine(m, ecfg, train_cfg, model.device)
        if eng.cfg.checkpoint_segment == "auto":
            # the copy's task embedding (ARC) is written from *this* model copy, so that its
            # gradient lands on the parameters compared below
            eng.state.segment = None
        x, loss, gn = eng.train_iteration(clone_batch(batch, m), seeds, c0, weights, 1e9, None)
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        # CUDA graph outputs may alias reusable graph-owned storage.
        outs.append((x.detach().clone(), loss.detach().clone(), gn.detach().clone(), grads))
    (xa, la, ga, grad_a), (xb, lb, gb, grad_b) = outs
    only_a = sorted(set(grad_a) - set(grad_b))
    only_b = sorted(set(grad_b) - set(grad_a))
    per_param = {n: float((grad_a[n] - grad_b[n]).abs().max()) for n in grad_a if n in grad_b}
    nonfinite = sorted(
        {n for n, g in grad_a.items() if not torch.isfinite(g).all()}
        | {n for n, g in grad_b.items() if not torch.isfinite(g).all()}
    )
    if not (torch.isfinite(xa).all() and torch.isfinite(xb).all()):
        nonfinite.append("<state>")
    if not (torch.isfinite(la) and torch.isfinite(lb)):
        nonfinite.append("<loss>")
    grad_max = max(per_param.values()) if per_param else 0.0
    if only_a or only_b or nonfinite:
        grad_max = float("inf")
    return dict(
        state_max_abs=float((xa - xb).abs().max()),
        state_max_abs_ref=float(xa.abs().max()),
        loss_a=float(la),
        loss_b=float(lb),
        loss_abs=float((la - lb).abs()),
        grad_norm_a=float(ga),
        grad_norm_b=float(gb),
        grad_max_abs=grad_max,
        grad_max_abs_ref=max(float(g.abs().max()) for g in grad_a.values()) if grad_a else 0.0,
        grad_only_a=only_a,
        grad_only_b=only_b,
        nonfinite=nonfinite,
        grad_per_param=per_param,
    )


# Tolerances of the trajectory-level check (scripts/check_engine.py): fp32 differences come
# from reduction order only (the fused operators reduce the 9 neighbours and the group norm in
# another order); bf16 differences also carry the rounding of a different operator sequence.
# The state is compared absolutely (states are O(1) after normalisation), the gradients
# relative to the largest reference gradient.
DEFAULT_TOL = dict(
    fp32=dict(state=1e-4, grad_rel=1e-4),
    bf16=dict(state=2e-2, grad_rel=2e-2),
    segment=dict(state=1e-6, grad_rel=1e-6),
)


def within_tolerance(r: dict, tol: dict) -> tuple[bool, str]:
    """Judge state, gradients and loss; the second value says what failed.

    Loss uses `loss_rel` when given, otherwise `grad_rel`, times the larger of
    the absolute reference loss and 1e-3. Non-finite comparison metrics fail.
    """
    if r["nonfinite"]:
        return False, "non-finite values: " + ", ".join(r["nonfinite"])
    if r["grad_only_a"] or r["grad_only_b"]:
        return False, (
            "gradient missing on one side: "
            + ", ".join(f"-{n}" for n in r["grad_only_a"])
            + ", ".join(f"+{n}" for n in r["grad_only_b"])
        )
    metrics = ("state_max_abs", "grad_max_abs", "grad_max_abs_ref", "loss_a", "loss_abs")
    if any(not math.isfinite(r[key]) for key in metrics):
        return False, "non-finite comparison metric"
    reasons = []
    if r["state_max_abs"] > tol["state"]:
        reasons.append(f"state {r['state_max_abs']:.2e} > {tol['state']:.0e}")
    grad_tol = tol["grad_rel"] * max(r["grad_max_abs_ref"], 1e-30)
    if r["grad_max_abs"] > grad_tol:
        reasons.append(
            f"grad {r['grad_max_abs']:.2e} > {tol['grad_rel']:.0e} x {r['grad_max_abs_ref']:.2e}"
        )
    # The same relative tolerance judges the loss, with an absolute floor for
    # near-zero reference losses. A matching final state cannot certify the
    # accumulated losses of the intermediate supervised steps.
    loss_tol = tol.get("loss_rel", tol["grad_rel"]) * max(abs(r["loss_a"]), 1e-3)
    if r["loss_abs"] > loss_tol:
        reasons.append(f"loss {r['loss_abs']:.2e} > {loss_tol:.2e}")
    return not reasons, "; ".join(reasons)


def iteration_seeds(train_cfg: dict, seed: int = 0) -> tuple[np.ndarray, int]:
    """Deterministic step seeds and chunk start for a benchmark iteration."""
    rng = np.random.default_rng(seed)
    N, Nc = int(train_cfg["rollout"]), int(train_cfg["chunk"])
    return step_seeds(rng, N), int(rng.integers(0, N - Nc + 1))
