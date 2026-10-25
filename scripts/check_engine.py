#!/usr/bin/env python
"""Numerical equivalence of the engine's backends on real configurations (the "trajectory"
level of the validation, docs/FAST_ENGINE.md): the same iteration, same seeds, same batch,
through the reference backend and through each other variant, with the largest differences
of the final state, the loss and the parameter gradients after 1, 8 and `chunk` steps, and
with checkpoint segments against no checkpointing.

    python scripts/check_engine.py --config configs/sudoku_ood.yaml --batch 8
    python scripts/check_engine.py --config configs/maze_hard.yaml --variants fused,compiled4 --steps 1,8,128
    python scripts/check_engine.py --config configs/sudoku_ood.yaml --train-steps 200 --out results/bench/check_sudoku.json

Every check is judged against a tolerance (`--tol-fp32`, `--tol-bf16`, `--tol-segment`, each
`state,grad_rel`: the state absolutely, the gradients relative to the largest reference
gradient; defaults in `nca.engine.bench.DEFAULT_TOL`), a gradient present on one side only or
a non-finite value fails whatever the tolerance, and the script exits with status 1 when any
check fails. The model initialisation, the batch and the draws are seeded with `--seed`.

`--train-steps n` additionally trains the model for n iterations with each variant from the
same initialisation (CPU-sized data, `--set` to shrink, `--batch` applies here too) and
reports the loss curves, i.e. whether the variants stay close over many updates (they are not
expected to be bitwise equal: fp32 reductions in a different order diverge slowly, bf16
faster; this part is informative, not judged).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.engine.bench import (  # noqa: E402
    DEFAULT_TOL,
    compare_backends,
    hardware_info,
    iteration_seeds,
    make_batch,
    within_tolerance,
)
from nca.engine.engine import EngineConfig  # noqa: E402
from nca.engine.train import loss_weights, resolve_train_cfg, train_fast  # noqa: E402
from nca.train import build_run_model, build_task_and_data  # noqa: E402
from nca.utils import get_device, load_config, save_json, seed_everything  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_engine import parse_variant  # noqa: E402


def _tol(text: str | None, default: dict) -> dict:
    if not text:
        return dict(default)
    a, b = (float(v) for v in text.split(","))
    return dict(state=a, grad_rel=b)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", default="fused,compiled")
    ap.add_argument("--steps", default=None, help="chunk lengths to test (default 1,8,chunk)")
    ap.add_argument("--segments", default="4", help="checkpoint segments to test against none")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seeds the model, the batch and the draws")
    ap.add_argument("--tol-fp32", default=None, metavar="STATE,GRAD_REL")
    ap.add_argument("--tol-bf16", default=None, metavar="STATE,GRAD_REL")
    ap.add_argument("--tol-segment", default=None, metavar="STATE,GRAD_REL")
    args = ap.parse_args()
    tol = dict(
        fp32=_tol(args.tol_fp32, DEFAULT_TOL["fp32"]),
        bf16=_tol(args.tol_bf16, DEFAULT_TOL["bf16"]),
        segment=_tol(args.tol_segment, DEFAULT_TOL["segment"]),
    )

    dev = get_device(args.device)
    cfg = load_config(args.config, args.set)
    tr = resolve_train_cfg(cfg)
    if args.batch:
        tr["batch_size"] = args.batch
    task, train_boards, _ = build_task_and_data(cfg["task"], cfg.get("data_root", "data"))
    seed_everything(args.seed)  # the model initialisation, like a training run
    model = build_run_model(cfg).to(dev)
    batch = make_batch(model, task, train_boards, tr["batch_size"], args.seed)
    steps = [int(s) for s in (args.steps or f"1,8,{tr['chunk']}").split(",")]
    report = dict(
        config=args.config,
        overrides=args.set,
        seed=args.seed,
        train=tr,
        tolerances=tol,
        hardware=hardware_info(dev),
        checks=[],
    )
    failures = []

    def judge(label: str, r: dict, t: dict) -> dict:
        ok, why = within_tolerance(r, t)
        if not ok:
            failures.append(f"{label}: {why}")
        return dict(ok=ok, why=why)

    for vname in args.variants.split(","):
        name, d, amps = parse_variant(vname.strip())
        for amp in amps:
            for n in steps:
                tr_v = {**tr, "amp": amp, "rollout": n, "chunk": n}
                seeds, c0 = iteration_seeds(tr_v, args.seed)
                w = loss_weights(tr["loss_window"], n, dev)
                ref = EngineConfig(backend="reference", checkpoint_segment=None)
                r = compare_backends(
                    model,
                    ref,
                    EngineConfig(**d, checkpoint_segment=None),
                    tr_v,
                    batch,
                    seeds,
                    c0,
                    w,
                )
                r.pop("grad_per_param")
                prec = "bf16" if amp else "fp32"
                verdict = judge(f"{name}:{prec}:{n}", r, tol[prec])
                rec = dict(variant=name, amp=amp, steps=n, **r, **verdict)
                report["checks"].append(rec)
                print(
                    f"{name:<12} {prec} {n:>4} steps: state {r['state_max_abs']:.2e} "
                    f"(max {r['state_max_abs_ref']:.2f}), loss {r['loss_abs']:.2e}, "
                    f"grad {r['grad_max_abs']:.2e} (max {r['grad_max_abs_ref']:.2e})  "
                    + ("PASS" if verdict["ok"] else "FAIL " + verdict["why"])
                )
    # segmented checkpointing against no checkpointing (fused backend)
    for seg in [int(s) for s in args.segments.split(",") if s]:
        for amp in (False, True):
            n = tr["chunk"]
            tr_v = {**tr, "amp": amp, "rollout": n, "chunk": n}
            seeds, c0 = iteration_seeds(tr_v, args.seed)
            w = loss_weights(tr["loss_window"], n, dev)
            a = EngineConfig(backend="fused", checkpoint_segment=None)
            b = EngineConfig(backend="fused", checkpoint_segment=seg)
            r = compare_backends(model, a, b, tr_v, batch, seeds, c0, w)
            r.pop("grad_per_param")
            prec = "bf16" if amp else "fp32"
            verdict = judge(f"fused:seg{seg}:{prec}", r, tol["segment"])
            report["checks"].append(
                dict(variant=f"fused:seg{seg}", amp=amp, steps=n, **r, **verdict)
            )
            print(
                f"{'fused seg' + str(seg):<12} {prec} {n:>4} steps: state {r['state_max_abs']:.2e}, "
                f"loss {r['loss_abs']:.2e}, grad {r['grad_max_abs']:.2e}  "
                + ("PASS" if verdict["ok"] else "FAIL " + verdict["why"])
            )

    if args.train_steps:
        import tempfile

        curves = {}
        for vname in ["reference"] + args.variants.split(","):
            name, d, amps = parse_variant(vname.strip())
            for amp in amps:
                c = json.loads(json.dumps(cfg))
                c.setdefault("train", {})["steps"] = args.train_steps
                if args.batch:
                    c["train"]["batch_size"] = args.batch
                c["train"]["amp"] = amp
                c["train"]["log_every"] = max(1, args.train_steps // 20)
                c["train"]["eval_every"] = args.train_steps
                c["train"]["checkpoint_every"] = args.train_steps
                c["engine"] = {**d, "checkpoint_segment": None}
                with tempfile.TemporaryDirectory() as tmp:
                    res = train_fast(
                        c, tmp, seed=args.seed, device=str(dev), resume=False, quiet=True
                    )
                    losses = [
                        json.loads(line)["loss"]
                        for line in open(Path(tmp) / "log.jsonl")
                        if '"loss"' in line
                    ]
                key = f"{name}:{'bf16' if amp else 'fp32'}"
                curves[key] = dict(losses=losses, final_eval=res)
                print(
                    f"{key:<18} losses {' '.join(f'{v:.4f}' for v in losses[:6])} ... {losses[-1]:.4f}  eval {res}"
                )
        report["training"] = curves
    report["ok"] = not failures
    report["failures"] = failures
    if args.out:
        save_json(report, args.out)
        print("written", args.out)
    if failures:
        print(f"\nFAILED: {len(failures)} check(s) out of tolerance")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print(f"\nPASS: {len(report['checks'])} checks within tolerance")


if __name__ == "__main__":
    main()
