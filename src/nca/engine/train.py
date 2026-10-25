"""Training with the fast engine: the recipe of `nca.train.train` (replay pool, perturbations,
truncated BPTT, AdamW, EMA, resumable checkpoints, the same `final.pt` / `run.json` /
`log.jsonl` / `summary.json` files) with the rollout executed by `Engine`.

Differences from `nca.train.train`, all documented in docs/FAST_ENGINE.md and DEVIATIONS.md:

* the firing masks and the noise of every step come from per-step seeds (`engine.rng`), with
  no CPU synchronisation and no stored noise tensors;
* the optimiser and EMA update is applied after the last step of the rollout
  (`engine.update_after_rollout`, default true), so the whole rollout runs with one set of
  weights as in paper Appendix B.2 and as `nca.train.train` since 2026-10-13; `false` gives
  the earlier order (update right after the chunk) for A/B tests;
* `train.grad_checkpoint` is replaced by `engine.checkpoint_segment` (segments of q steps,
  `auto` picks the fastest that fits the VRAM budget);
* `log.jsonl` records carry `iter_seconds` (mean over the log window) and, on CUDA,
  `peak_mem_gb`;
* the public-code TBPTT options `train.chunk_start: per_sample` and `train.writeback:
  chunk_end` are not supported (ValueError at startup); the other public-code switches
  (`betas`, `init_lr_ratio`, `loss_reduce`, `ema_warmup`, `damage.fill` / `geometry`,
  `batch_noise`, `swap_source`) are honoured as in the reference.

`train.stop_at` pauses the run as in the reference (checkpoint only, no `final.pt` or
`summary.json`); the `stop_at_step` argument is the bounded-training API, which writes them
with `training_complete: false`.

Everything else (configs, data, pool, losses, schedules, evaluation, checkpoint keys) is
shared with the reference loop, so the eval scripts read the engine's runs unchanged.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..pool import ReplayPool
from ..rollout import decode_boards
from ..tasks.base import overlap_counts
from ..train import (
    DEFAULTS,
    _json_line,
    build_run_model,
    build_task_and_data,
    ema_decay,
    evaluate_quick,
    lr_at,
    perturb_batch,
    restore_model_config,
)
from ..train import resolve_train_cfg as _resolve_train_cfg
from ..utils import count_parameters, get_device, human_time, save_json, seed_everything
from .engine import Engine, EngineConfig
from .rng import step_seeds


def loss_weights(window, chunk: int, device: torch.device) -> torch.Tensor:
    """Per-chunk-step loss weights for `loss_window` ("all", "last" or an integer w)."""
    w = torch.zeros(chunk, device=device)
    if window == "all":
        w[:] = 1.0
    elif window == "last":
        w[-1] = 1.0
    else:
        w[chunk - int(window) :] = 1.0
    return w


def resolve_train_cfg(cfg: dict) -> dict:
    """`nca.train.resolve_train_cfg`, plus a ValueError for the options the engine does not
    implement (the public code's per-sample TBPTT chunk and chunk-end write-back, extra EMAs)."""
    tr = _resolve_train_cfg(cfg)
    if tr["ema_extra"]:
        raise ValueError("train.ema_extra is implemented by the reference trainer only")
    for k in ("chunk_start", "writeback"):
        if tr[k] != DEFAULTS[k]:
            raise ValueError(
                f"train.{k}: {tr[k]} is implemented by the reference trainer only "
                f"(nca.train.train); the fast engine supports {DEFAULTS[k]!r}"
            )
    return tr


def train_fast(
    cfg: dict,
    out_dir: str | Path,
    seed: int = 0,
    device: str | None = None,
    resume: bool = True,
    quiet: bool = False,
    stop_at_step: int | None = None,
) -> dict:
    """Train one model; optionally pause at a checkpoint without changing the LR horizon."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["task"]
    mcfg = cfg.get("model", {})
    tr = resolve_train_cfg(cfg)
    # `train.stop_at` (a probe of the reference trainer): pause at that step with a checkpoint
    # only, no final.pt; 0 / null, or a value past `steps`, runs to the end
    pause = stop_at_step is None and bool(tr["stop_at"]) and tr["stop_at"] < tr["steps"]
    if pause:
        end_step = int(tr["stop_at"])
    else:
        end_step = tr["steps"] if stop_at_step is None else stop_at_step
    if not isinstance(end_step, int) or not 0 < end_step <= tr["steps"]:
        raise ValueError("stop_at_step must be in 1..train.steps")
    ecfg = EngineConfig.from_dict(cfg.get("engine"))
    dev = get_device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    seed_everything(seed)
    gen_dev = torch.Generator(device=dev)
    gen_dev.manual_seed(seed)
    rng = np.random.default_rng(seed)

    task, train_boards, val_boards = build_task_and_data(tcfg, cfg.get("data_root", "data"))
    model = build_run_model(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"], betas=tuple(tr["betas"])
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, tr["steps"], tr))

    pool_mult = tr["pool_mult"] if tr["replay"] else 1
    seed_ratio = tr["seed_ratio"] if tr["replay"] else 1.0
    task_ids = train_boards.meta.get("task_id")
    pool = ReplayPool(
        model,
        task,
        train_boards,
        pool_mult * tr["batch_size"],
        rng,
        gen_dev,
        augment=tr["augment"],
        task_ids=task_ids,
    )
    perturb = tr["perturb"]
    if dev.type != "cuda" and ecfg.checkpoint_segment == "auto":
        ecfg.checkpoint_segment = None  # nothing to save on CPU
    engine = Engine(model, ecfg, tr, dev)
    N, Nc = engine.N, engine.Nc
    assert Nc <= N
    weights = loss_weights(tr["loss_window"], Nc, dev)

    step0 = 0
    log_path = out / "log.jsonl"
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        restore_model_config(model, ema, out / "run.json")
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        pool.load_state_dict(ck["pool"])
        rng.bit_generator.state = ck["rng"]
        gen_dev.set_state(ck["gen_dev"].cpu())
        step0 = ck["step"]
        if "engine_segment" in ck:  # keep the segment the run was tuned with
            engine.state.segment = ck["engine_segment"]
            ecfg.checkpoint_segment = ck["engine_segment"]
        if not quiet:
            print(f"resumed from step {step0}")
        if step0 >= tr["steps"] and (out / "summary.json").exists():
            with open(out / "summary.json") as f:
                return json.load(f).get("final_eval", {})
        if step0 >= end_step:
            if pause:
                raise ValueError(
                    f"train.stop_at ({end_step}) must be after the resumed checkpoint (step "
                    f"{step0}); clear it to continue the run"
                )
            raise ValueError("stop_at_step must be after the resumed checkpoint")
    elif not quiet:
        print(
            f"{task.name}: {len(train_boards)} train boards ({train_boards.meta.get('source')}), "
            f"model {mcfg.get('preset')} with {count_parameters(model):,} parameters on {dev}, "
            f"engine {ecfg.backend}{' + compile' if ecfg.compile else ''} block {ecfg.block}"
        )

    if ecfg.checkpoint_segment == "auto":
        # tune on a batch drawn with a *copy* of the RNG state, so the run itself is unchanged
        rng_state = rng.bit_generator.state
        gen_state = gen_dev.get_state()
        _, b = pool.sample(tr["batch_size"], seed_ratio)
        seeds = step_seeds(rng, N)
        t0 = time.time()
        if not quiet:
            print("autotuning the checkpoint segment (includes compilation):")
        engine.autotune(b, seeds, int(rng.integers(0, N - Nc + 1)), weights, quiet=quiet)
        engine.state.compile_seconds = time.time() - t0
        rng.bit_generator.state = rng_state
        gen_dev.set_state(gen_state)
        if not quiet:
            print(f"  -> segment {engine.state.segment or 'none'} ({human_time(time.time() - t0)})")
    elif ecfg.compile:
        t0 = time.time()
        rng_state = rng.bit_generator.state
        gen_state = gen_dev.get_state()
        _, b = pool.sample(tr["batch_size"], seed_ratio)
        seeds = step_seeds(rng, N)
        engine.train_iteration(b, seeds, int(rng.integers(0, N - Nc + 1)), weights, 1.0, None)
        model.zero_grad(set_to_none=True)
        engine.state.compile_seconds = time.time() - t0
        rng.bit_generator.state = rng_state
        gen_dev.set_state(gen_state)
        if not quiet:
            print(f"compiled in {human_time(engine.state.compile_seconds)}")

    data_info = {
        "train_source": train_boards.meta.get("source"),
        "train_n": len(train_boards),
        "train_shape": list(train_boards.shape),
        "val_source": None if val_boards is None else val_boards.meta.get("source"),
        "val_n": None if val_boards is None else len(val_boards),
    }
    if val_boards is not None:
        data_info["train_val_overlap"] = overlap_counts(train_boards, val_boards)
    save_json(
        {
            "config": cfg,
            "seed": seed,
            "params": count_parameters(model),
            "model_config": model.cfg.to_dict(),
            "data": data_info,
            "engine": engine.to_dict(),
            "device": str(dev)
            + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""),
            "torch": torch.__version__,
            "precision_policy": {
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
        out / "run.json",
    )

    t_start = time.time()
    last_eval: dict = {}
    n_skipped = 0
    window_t0 = time.time()
    window_n = 0
    model.train()
    if dev.type == "cuda":
        warmup_peak_allocated = torch.cuda.max_memory_allocated(dev) / 2**30
        warmup_peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**30
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(step0, end_step):
        idx, batch = pool.sample(tr["batch_size"], seed_ratio)
        if perturb:
            perturb_batch(model, pool, batch, tr, rng, gen_dev)
        c0 = int(rng.integers(0, N - Nc + 1))
        seeds = step_seeds(rng, N)

        def apply_update(gn: torch.Tensor, step: int = step) -> None:
            nonlocal n_skipped
            if bool(torch.isfinite(gn)):  # one synchronisation per iteration, as in the reference
                opt.step()
            else:  # a non-finite gradient: skip the update rather than poison the weights
                n_skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            with torch.no_grad():
                d = ema_decay(tr, step)  # warm-up unless `ema_warmup: false` (as the reference)
                for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)

        x, loss, gn = engine.train_iteration(
            batch, seeds, c0, weights, tr["grad_clip"], apply_update
        )
        batch["x"] = x.detach()
        pool.write_back(idx, batch)
        window_n += 1

        if (step + 1) % tr["log_every"] == 0 or step == step0:
            with torch.no_grad():
                dec = decode_boards(model, task, x, batch["given4"][:, 0], batch["inputs"])
                loss_mask = batch.get("loss_mask")
                mut = ~batch["given4"][:, 0] if loss_mask is None else loss_mask > 0
                correct = ((dec == batch["targets"]) & mut).sum(dim=(1, 2)).float()
                cell_acc = (correct / mut.sum(dim=(1, 2)).clamp_min(1)).mean().item()
                exact = (correct == mut.sum(dim=(1, 2))).float().mean().item()
            now = time.time()
            rec = dict(
                step=step + 1,
                loss=float(loss.item()),
                grad_norm=float(gn),
                lr=sched.get_last_lr()[0],
                batch_cell_acc=cell_acc,
                batch_exact=exact,
                mean_age=float(batch["age"].float().mean().item()),
                skipped_steps=n_skipped,
                elapsed=now - t_start,
                iter_seconds=(now - window_t0) / max(1, window_n),
            )
            if dev.type == "cuda":
                rec["peak_mem_gb"] = max(
                    warmup_peak_allocated, torch.cuda.max_memory_allocated(dev) / 2**30
                )
            window_t0, window_n = now, 0
            with open(log_path, "a") as f:
                f.write(_json_line(rec))
            if not quiet:
                eta = (tr["steps"] - step - 1) * rec["elapsed"] / max(1, step + 1 - step0)
                print(
                    f"step {step + 1:>7d}/{tr['steps']}  loss {rec['loss']:.4f}  "
                    f"cell {cell_acc:.3f}  exact {exact:.3f}  age {rec['mean_age']:.1f}  "
                    f"lr {rec['lr']:.2e}  {rec['iter_seconds']:.3f} s/it  "
                    f"{human_time(rec['elapsed'])} (eta {human_time(eta)})"
                )

        if val_boards is not None and ((step + 1) % tr["eval_every"] == 0 or step + 1 == end_step):
            last_eval = evaluate_quick(ema, task, val_boards, tr["eval"], N, seed)
            last_eval["step"] = step + 1
            with open(log_path, "a") as f:
                f.write(_json_line({"eval": last_eval}))
            if not quiet:
                print(
                    f"  eval@{step + 1}: "
                    + "  ".join(f"{k} {v:.3f}" for k, v in last_eval.items() if k != "step")
                )
            model.train()
            window_t0 = time.time()  # the evaluation is not training time

        if (step + 1) % tr["checkpoint_every"] == 0 or step + 1 == end_step:
            torch.save(
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    opt=opt.state_dict(),
                    sched=sched.state_dict(),
                    pool=pool.state_dict(),
                    rng=rng.bit_generator.state,
                    gen_dev=gen_dev.get_state(),
                    step=step + 1,
                    engine_segment=engine.state.segment,
                ),
                ckpt_path,
            )

    if pause:
        # as nca.train.train: the LR schedule still follows `steps`; no final.pt or summary.json,
        # so the paused run never looks finished, and resuming continues it
        if not quiet:
            print(f"stopped at step {end_step} (stop_at); resume to continue the run")
        return last_eval

    final = dict(
        model=ema.state_dict(),
        model_config=model.cfg.to_dict(),
        config=cfg,
        seed=seed,
        step=end_step,
    )
    if end_step < tr["steps"]:
        # bounded training (`stop_at_step`): mark the intermediate weights as such
        final.update(target_steps=tr["steps"], training_complete=False)
    torch.save(final, out / "final.pt")
    summary = {
        "final_eval": last_eval,
        "train_seconds": time.time() - t_start,
        "steps": end_step,
        "params": count_parameters(model),
        "skipped_steps": n_skipped,
        "engine": engine.to_dict(),
    }
    if stop_at_step is not None:
        summary.update(
            start_step=step0, target_steps=tr["steps"], training_complete=end_step == tr["steps"]
        )
    if dev.type == "cuda":
        summary["train_peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
        summary["warmup_peak_mem_gb"] = warmup_peak_allocated
        summary["peak_mem_gb"] = max(warmup_peak_allocated, summary["train_peak_mem_gb"])
        summary["peak_reserved_gb"] = max(
            warmup_peak_reserved, torch.cuda.max_memory_reserved(dev) / 2**30
        )
    save_json(summary, out / "summary.json")
    return last_eval
