#!/usr/bin/env python
"""Benchmark the engine's backends on one training configuration.

    python scripts/bench_engine.py --config configs/sudoku_ood.yaml --out results/bench/sudoku_ood_pc.json
    python scripts/bench_engine.py --config configs/maze_hard.yaml --variants reference,fused,compiled4 --iters 10
    python scripts/bench_engine.py --config configs/sudoku_ood.yaml --probe        # can torch.compile run here?

A variant is `<backend>[<block>]` with an optional precision suffix: `reference`, `fused`,
`compiled` (fused + torch.compile, block 1), `compiled4` (blocks of 4 steps), `graphs4`
(blocks of 4 with CUDA graphs, `compile_mode=reduce-overhead`), `refcompiled` (reference step
compiled), plus `:fp32` or `:bf16` to force the precision (default: both). Ablations of the
fused pieces: `nopad` (fused without projection-before-padding), `nomix` (without the fused
mixing operator), `noupd` (without the fused update). `--segment q` times a checkpoint segment
(0 = none, default) and `--segments 0,8,4` several.

For every variant the script reports two timings: `train` (rollout + backward + clipping on
the same batch every time: the engine's own cost) and `full` (the complete training
iteration as `train_fast` runs it: replay sampling and perturbations, rollout, backward,
clipping, AdamW, scheduler, EMA, write-back; the number that decides a run's duration), plus
the no-grad rollout time, the first-iteration time (compilation), the peak memory on CUDA,
the FLOPs of one step, and the numerical distance to the reference backend on the same
iteration (with the same checkpoint segment). The timing and the equivalence check are
reported separately: a variant whose timing worked but whose comparison failed keeps its
timing (`ok`) and gets `equivalence_ok: false`. Speed-ups are relative to the fp32 / bf16
`reference` variant with the same segment, or the first reference that ran (`baseline`).
The JSON also records the hardware, the software versions, the commit (and whether the tree
was dirty), the seed and the resolved training configuration. The model initialisation is
seeded with `--seed`, so the same seed gives the same model and batch. Run it with no other
GPU job on the machine; see docs/FAST_ENGINE.md for how to read the file.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.engine.bench import (  # noqa: E402
    DEFAULT_TOL,
    compare_backends,
    count_step_flops,
    hardware_info,
    iteration_seeds,
    make_pool,
    probe_compile,
    time_forward,
    time_full_iterations,
    time_iterations,
    within_tolerance,
)
from nca.engine.engine import Engine, EngineConfig  # noqa: E402
from nca.engine.train import loss_weights, resolve_train_cfg  # noqa: E402
from nca.train import build_run_model, build_task_and_data  # noqa: E402
from nca.utils import get_device, human_time, load_config, save_json, seed_everything  # noqa: E402

VARIANTS = {
    "reference": dict(backend="reference"),
    "refcompiled": dict(backend="reference", compile=True, block=1),
    "fused": dict(backend="fused"),
    "nopad": dict(backend="fused", project_before_pad=False),
    "nomix": dict(backend="fused", fused_mix=False),
    "noupd": dict(backend="fused", fused_update=False),
    "compiled": dict(backend="fused", compile=True, block=1),
    "graphs": dict(backend="fused", compile=True, block=1, compile_mode="reduce-overhead"),
}


def parse_variant(name: str) -> tuple[str, dict, list[bool]]:
    """`compiled4:bf16` -> ("compiled4", {...block=4}, [True]): the precision is returned as
    the list of `amp` values, not in the name (the JSON carries `amp` separately)."""
    full = name
    base, _, prec = name.partition(":")
    digits = ""
    while base and base[-1].isdigit():
        digits = base[-1] + digits
        base = base[:-1]
    if base not in VARIANTS:
        raise SystemExit(f"unknown variant {full!r}; known: {', '.join(VARIANTS)}")
    d = dict(VARIANTS[base])
    if digits:
        d["block"] = int(digits)
        if not d.get("compile"):
            raise SystemExit(f"{full}: only compiled variants take a block size")
    if prec not in ("", "fp32", "bf16"):
        raise SystemExit(f"{full}: precision must be fp32 or bf16")
    amps = {"": [False, True], "fp32": [False], "bf16": [True]}[prec]
    return base + digits, d, amps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="JSON output (default: print only)")
    ap.add_argument(
        "--variants",
        default="reference,fused,compiled,compiled4",
        help="comma-separated variants (see the module docstring)",
    )
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--segments", default="0", help="checkpoint segments to time, e.g. 0,8,4")
    ap.add_argument("--batch", type=int, default=None, help="batch size (default: the config's)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--probe", action="store_true", help="only check torch.compile and exit")
    ap.add_argument("--no-flops", action="store_true")
    ap.add_argument("--no-full", action="store_true", help="skip the full-iteration timing")
    ap.add_argument("--seed", type=int, default=0, help="seeds the model, the batch and the draws")
    args = ap.parse_args()

    dev = get_device(args.device)
    info = hardware_info(dev)
    print("hardware:", json.dumps(info))
    if args.probe:
        p = probe_compile(dev)
        print("torch.compile:", "ok" if p["ok"] else "FAILED", json.dumps(p))
        return

    cfg = load_config(args.config, args.set)
    tr = resolve_train_cfg(cfg)
    if args.batch:
        tr["batch_size"] = args.batch
    task, train_boards, _ = build_task_and_data(cfg["task"], cfg.get("data_root", "data"))
    seed_everything(args.seed)  # the model initialisation, like a training run
    model = build_run_model(cfg).to(dev)
    pool = make_pool(model, task, train_boards, tr["batch_size"], args.seed)
    _, batch = pool.sample(tr["batch_size"], 0.0)
    seeds, c0 = iteration_seeds(tr, args.seed)
    weights = loss_weights(tr["loss_window"], tr["chunk"], dev)
    segments = [int(s) for s in args.segments.split(",")]
    base_tr = dict(tr)

    results = []
    for vname in args.variants.split(","):
        name, d, amps = parse_variant(vname.strip())
        for amp in amps:
            for seg in segments:
                tr_v = {**base_tr, "amp": amp}
                ecfg = EngineConfig(**d, checkpoint_segment=seg or None)
                m = copy.deepcopy(model)
                eng = Engine(m, ecfg, tr_v, dev)
                label = f"{name}:{'bf16' if amp else 'fp32'}" + (f":seg{seg}" if seg else "")
                print(f"\n== {label} ==", flush=True)
                t0 = time.perf_counter()
                pool_v = None
                try:
                    rec = dict(
                        variant=name,
                        engine=ecfg.to_dict(),
                        amp=amp,
                        segment=seg or None,
                        batch_size=tr["batch_size"],
                        rollout=tr["rollout"],
                        chunk=tr["chunk"],
                    )
                    rec["train"] = time_iterations(
                        eng, batch, seeds, c0, weights, args.iters, args.warmup, tr["grad_clip"]
                    )
                    rec["forward"] = time_forward(
                        eng, batch, seeds[: tr["chunk"]], max(1, args.iters // 2), 1
                    )
                    if not args.no_full:
                        # a fresh pool for this variant's model copy, same boards and seed
                        pool_v = make_pool(
                            m, task, train_boards, tr["batch_size"], args.seed, train_cfg=tr_v
                        )
                        rec["full"] = time_full_iterations(
                            eng, pool_v, tr_v, args.iters, args.warmup, args.seed
                        )
                    if not args.no_flops and not ecfg.compile:
                        rec["flops_per_step"] = count_step_flops(eng, batch)
                    rec["ok"] = True
                except Exception as e:  # noqa: BLE001 - report and continue with the next variant
                    rec["ok"] = False
                    rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
                    print("  FAILED:", rec["error"])
                finally:
                    # The pool holds its model and GPU states: do not retain a previous
                    # variant's allocations during the next measurement or equivalence run.
                    del pool_v
                is_reference = ecfg.backend == "reference" and not ecfg.compile
                if rec["ok"] and not is_reference:
                    # equivalence against the reference backend with the *same* segment (same
                    # memory footprint: a reference that would not fit without checkpointing
                    # does not make a working variant fail)
                    try:
                        ref = EngineConfig(backend="reference", checkpoint_segment=seg or None)
                        rec["vs_reference"] = compare_backends(
                            model, ref, ecfg, tr_v, batch, seeds, c0, weights
                        )
                        rec["vs_reference"].pop("grad_per_param", None)
                        v = rec["vs_reference"]
                        tolerance = DEFAULT_TOL["bf16" if amp else "fp32"]
                        rec["equivalence_tolerance"] = tolerance
                        rec["equivalence_ok"], rec["equivalence_error"] = within_tolerance(
                            v, tolerance
                        )
                    except Exception as e:  # noqa: BLE001
                        rec["equivalence_ok"] = False
                        rec["equivalence_error"] = f"{type(e).__name__}: {str(e)[:600]}"
                        print("  equivalence check FAILED (timing kept):", rec["equivalence_error"])
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                rec["wall_seconds"] = time.perf_counter() - t0
                results.append(rec)
                if rec["ok"]:
                    t = rec["train"]
                    line = (
                        f"  iteration {t['seconds_mean'] * 1000:.1f} ms (±{t['seconds_std'] * 1000:.1f}),"
                        f" first {t['first_iteration_seconds']:.2f} s,"
                        f" no-grad {rec['forward']['seconds_per_step'] * 1000:.2f} ms/step"
                    )
                    if "full" in rec:
                        line += f", full {rec['full']['seconds_mean'] * 1000:.1f} ms"
                    if "peak_allocated_gb" in t:
                        line += f", peak {t['peak_allocated_gb']:.2f} GiB"
                    if "vs_reference" in rec:
                        v = rec["vs_reference"]
                        line += f", vs ref: state {v['state_max_abs']:.2e}, grad {v['grad_max_abs']:.2e}"
                    print(line, flush=True)
                del eng, m
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    # speed-ups relative to the reference of the same precision: the one with the same
    # segment when it ran, else the first reference that ran (its label is recorded)
    refs = [r for r in results if r["ok"] and r["variant"] == "reference"]
    for r in results:
        if not r["ok"]:
            continue
        same = [b for b in refs if b["amp"] == r["amp"] and b["segment"] == r["segment"]]
        any_ = [b for b in refs if b["amp"] == r["amp"]]
        b = (same or any_ or [None])[0]
        if b is None:
            continue
        r["baseline"] = f"reference:{'bf16' if b['amp'] else 'fp32'}" + (
            f":seg{b['segment']}" if b["segment"] else ""
        )
        r["speedup_vs_reference"] = b["train"]["seconds_mean"] / r["train"]["seconds_mean"]
        if "full" in r and "full" in b:
            r["speedup_full_vs_reference"] = b["full"]["seconds_mean"] / r["full"]["seconds_mean"]
    out = dict(
        config=args.config,
        overrides=args.set,
        seed=args.seed,
        train=base_tr,
        resolved_config=cfg,
        model_config=model.cfg.to_dict(),
        hardware=info,
        iters=args.iters,
        warmup=args.warmup,
        results=results,
    )
    print("\nsummary:")
    for r in results:
        if r["ok"]:
            su = r.get("speedup_vs_reference")
            suf = r.get("speedup_full_vs_reference")
            print(
                f"  {r['variant']:<12} {'bf16' if r['amp'] else 'fp32'} seg {r['segment'] or '-':<4}"
                f" {r['train']['seconds_mean'] * 1000:8.1f} ms/iter"
                + (f"  x{su:.2f}" if su else "")
                + (f"  full {r['full']['seconds_mean'] * 1000:8.1f} ms" if "full" in r else "")
                + (f"  x{suf:.2f}" if suf else "")
                + ("" if r.get("equivalence_ok", True) else "  EQUIVALENCE FAILED")
            )
    if args.out:
        save_json(out, args.out)
        print("written", args.out, f"({human_time(sum(r['wall_seconds'] for r in results))})")


if __name__ == "__main__":
    main()
