#!/usr/bin/env python
"""Niche-Capped Diversity Pruning (paper Fig. 6A): accuracy against compute for unpruned
parallel trials and pruning divisors m in {2, 4, 8}, across rollout depths.

    python scripts/eval_pruning.py --run runs/sudoku_extreme/seed0 --n 500 --k 64 \
        --depths 64 128 256 512 --ms 2 4 8 --out results/pruning/sudoku_extreme_seed0.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import add_common, load_boards, load_run, parse_kv  # noqa: E402
from nca.experiments import pruning_pareto  # noqa: E402
from nca.utils import Timer, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(ap)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--ms", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--n-cap", type=int, default=3)
    ap.add_argument(
        "--noise-sigma",
        type=float,
        default=0.1,
        help="paper B.6: sigma 0.1, p_t 0.2, p_s 0.8, first quarter",
    )
    args = ap.parse_args()
    model, cfg, seed, task = load_run(args.run, args.device)
    boards = load_boards(task, cfg, args.split, args.n, parse_kv(args.load))
    noise = (
        None if args.noise_sigma <= 0 else dict(r=0.25, p_t=0.2, p_s=0.8, sigma=args.noise_sigma)
    )
    with Timer() as t:
        res = pruning_pareto(
            model,
            task,
            boards,
            args.depths,
            args.k,
            args.ms,
            args.n_cap,
            noise=noise,
            seed=args.seed if args.seed is not None else seed,
            max_cells=args.max_cells,
            progress=args.progress,
            amp=args.amp,
        )
    res.update(run=args.run, seed=args.seed if args.seed is not None else seed, seconds=t.seconds)
    save_json(res, args.out)
    for r in res["rows"]:
        print(
            f"D={r['depth']:>5d}  m={r['m']}  conf@K {r['conf_k']:.3f}  pass@K {r['pass_k']:.3f}  "
            f"compute {r['compute_units']:.3e} cell-updates/board"
        )


if __name__ == "__main__":
    main()
