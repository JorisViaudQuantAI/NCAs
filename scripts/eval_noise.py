#!/usr/bin/env python
"""Test-time noise sweep (paper Fig. 7B): accuracy against the injected noise magnitude.

    python scripts/eval_noise.py --run runs/sudoku_ood/seed0 --n 500 --depth 256 --k 16 \
        --sigmas 0 0.1 0.2 0.3 0.5 0.7 1.0 --out results/noise/sudoku_ood_seed0.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import add_common, load_boards, load_run, parse_kv  # noqa: E402
from nca.experiments import noise_sweep  # noqa: E402
from nca.utils import Timer, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(ap)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--p-t", type=float, default=0.1)
    ap.add_argument("--p-s", type=float, default=0.2)
    ap.add_argument("--r", type=float, default=0.25)
    args = ap.parse_args()
    model, cfg, seed, task = load_run(args.run, args.device)
    boards = load_boards(task, cfg, args.split, args.n, parse_kv(args.load))
    with Timer() as t:
        res = noise_sweep(
            model,
            task,
            boards,
            args.depth,
            args.k,
            args.sigmas,
            args.p_t,
            args.p_s,
            args.r,
            seed=args.seed if args.seed is not None else seed,
            max_cells=args.max_cells,
            progress=args.progress,
            amp=args.amp,
        )
    res.update(
        run=args.run,
        seed=args.seed if args.seed is not None else seed,
        seconds=t.seconds,
        source=boards.meta.get("source"),
    )
    save_json(res, args.out)
    for r in res["rows"]:
        print(
            f"sigma {r['sigma']:.2f}  K=1 {r['k1']:.3f}  conf@K {r['conf_k']:.3f}  pass@K {r['pass_k']:.3f}"
        )


if __name__ == "__main__":
    main()
