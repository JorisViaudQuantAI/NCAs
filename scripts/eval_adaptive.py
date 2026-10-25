#!/usr/bin/env python
"""Adaptive compute and damage recovery on large mazes (paper Fig. 8).

Three firing rules on the same boards: uniform (p_fire = 0.8 everywhere), adaptive as written in
the paper (p_low = 0.4 once a cell's confidence exceeds 0.95; clamped cells never qualify) and
adaptive_given (clamped cells count as confident). Reports steps and cumulative cell updates
needed for 95% of the boards to be solved, their ratios to the uniform rule (the paper's
1.29x steps / 0.71x updates), with an optional damage event (random discs zeroed) mid-rollout.

The script is incremental: when `--out` exists, a variant already in it is kept if it reached
95% or was run at least `--depth` steps; only the others are (re)computed. `--force` recomputes
everything.

    python scripts/eval_adaptive.py --run runs/maze_ood/seed0 --size 201 --n 20 --depth 20000 \
        --record-every 100 --out results/adaptive/maze201_seed0.json
    python scripts/eval_adaptive.py --run runs/maze_ood/seed0 --size 201 --n 20 --depth 20000 \
        --damage-at 6000 --record-every 100 --out results/adaptive/maze201_damage_seed0.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import add_common, load_run, parse_kv  # noqa: E402
from nca.experiments import ADAPTIVE_VARIANTS, adaptive_compute  # noqa: E402
from nca.utils import Timer, load_json, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(ap)
    ap.add_argument("--size", type=int, default=201)
    ap.add_argument("--depth", type=int, default=20000)
    ap.add_argument(
        "--variants",
        default=",".join(ADAPTIVE_VARIANTS),
        help="comma-separated subset of " + ",".join(ADAPTIVE_VARIANTS),
    )
    ap.add_argument("--force", action="store_true", help="recompute variants already in --out")
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--p-fire", type=float, default=0.8)
    ap.add_argument("--p-low", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--damage-at", type=int, default=None)
    ap.add_argument("--damage-n", type=int, default=3)
    ap.add_argument("--damage-r", type=float, nargs=2, default=[0.05, 0.15])
    args = ap.parse_args()
    model, cfg, seed, task = load_run(args.run, args.device)
    extra = {**cfg["task"].get("test", {}), **parse_kv(args.load)}  # config's test options first
    extra.pop("size", None)
    boards = task.load(
        args.split, cfg.get("data_root", "data"), size=args.size, n=args.n or 20, **extra
    )
    previous = None
    if not args.force and Path(args.out).exists():
        previous = load_json(args.out)
        if previous.get("damage_at") != args.damage_at or previous.get("n_boards") != len(boards):
            previous = None  # a different experiment under the same name: start over
    with Timer() as t:
        res = adaptive_compute(
            model,
            task,
            boards,
            args.depth,
            args.p_fire,
            args.p_low,
            args.tau,
            args.record_every,
            damage_at=args.damage_at,
            damage=dict(n=args.damage_n, r_min=args.damage_r[0], r_max=args.damage_r[1]),
            seed=args.seed if args.seed is not None else seed,
            max_cells=args.max_cells,
            progress=args.progress,
            amp=args.amp,
            variants=tuple(args.variants.split(",")),
            previous=previous,
        )
    res.update(
        run=args.run,
        size=args.size,
        seed=args.seed if args.seed is not None else seed,
        seconds=t.seconds,
    )
    save_json(res, args.out)
    for name, v in res["variants"].items():
        print(
            f"{name:15s} depth {v['depth']:6d}  final exact {v['final_exact']:.3f}  "
            f"steps to 95% {v['steps_to_95']}  cell updates to 95% {v['updates_to_95']}"
        )
    for name, r in res["ratios"].items():
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}x"  # noqa: E731
        print(
            f"{name:15s} vs uniform: steps to 95% {fmt(r['steps_to_95'])}, "
            f"cell updates to 95% {fmt(r['updates_to_95'])}, "
            f"updates per step {fmt(r['updates_per_step'])}  (paper: 1.29x, 0.71x)"
        )


if __name__ == "__main__":
    main()
