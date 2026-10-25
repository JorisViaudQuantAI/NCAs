#!/usr/bin/env python
"""Spatial substrate scaling on Maze-OOD (paper Fig. 4B and the Maze-OOD rows of Table 1): the
NCA trained on 9x9 mazes is run, unchanged, on larger mazes with more steps.

    python scripts/eval_maze_scaling.py --run runs/maze_ood/seed0 --sizes 9 13 17 21 25 29 33 37 59 \
        --n 200 --out results/maze_scaling/maze_ood_seed0.json
    python scripts/eval_maze_scaling.py --run runs/maze_ood/seed0 --table1 --n 100 --out results/maze_scaling/table1_seed0.json

Depth rule: D(size) = 0.3 * size^2 + 20 * size rounded up to 100 (400 for 13x13, 2300 for 59x59,
16200 for 201x201); `--table1` uses the paper's exact D for 13, 59 and 201.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import add_common, load_run, parse_kv  # noqa: E402
from nca.experiments import maze_size_scaling  # noqa: E402
from nca.utils import Timer, save_json  # noqa: E402

TABLE1 = {13: 300, 59: 2000, 201: 13000}


def default_depth(size: int) -> int:
    return int(math.ceil((0.3 * size * size + 20 * size) / 100) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(ap)
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[9, 11, 13, 15, 17, 19, 21, 25, 29, 33, 37, 59]
    )
    ap.add_argument(
        "--table1", action="store_true", help="sizes 13, 59, 201 with the paper's depths"
    )
    ap.add_argument(
        "--depth-scale", type=float, default=1.0, help="multiply the default depth rule"
    )
    args = ap.parse_args()
    model, cfg, seed, task = load_run(args.run, args.device)
    extra = {**cfg["task"].get("test", {}), **parse_kv(args.load)}  # config's test options first
    extra.pop("size", None)
    root = cfg.get("data_root", "data")
    n = args.n or 200

    def load_size(size, n_boards):
        return task.load(args.split, root, size=size, n=n_boards, **extra)

    if args.table1:
        sizes = list(TABLE1)
        depth_for = TABLE1.__getitem__
    else:
        sizes = args.sizes
        depth_for = lambda s: int(default_depth(s) * args.depth_scale)  # noqa: E731
    with Timer() as t:
        res = maze_size_scaling(
            model,
            task,
            load_size,
            sizes,
            depth_for,
            n,
            seed=args.seed if args.seed is not None else seed,
            max_cells=args.max_cells,
            progress=args.progress,
            amp=args.amp,
        )
    res.update(
        run=args.run,
        seed=args.seed if args.seed is not None else seed,
        seconds=t.seconds,
        table1=args.table1,
        load=parse_kv(args.load),  # the board sampling options (seed, exclude_seed, ...)
    )
    save_json(res, args.out)
    for r in res["rows"]:
        print(
            f"size {r['size']:>4d}  D={r['depth']:>6d}  exact {r['exact']:.3f}  valid {r.get('valid', float('nan')):.3f}  "
            f"optimal {r.get('optimal', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
