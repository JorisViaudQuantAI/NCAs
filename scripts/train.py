#!/usr/bin/env python
"""Train one NCA from a config file.

    python scripts/train.py --config configs/maze_ood.yaml --seed 0
    python scripts/train.py --config configs/sudoku_ood.yaml --seed 1 --out runs/sudoku_ood/seed1
    python scripts/train.py --config configs/maze_ood.yaml --seed 0 --set train.steps=200

Runs are resumable: rerun the same command after an interruption and it continues from
`checkpoint.pt`. The default output folder is `runs/<config name>/seed<seed>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.train import train  # noqa: E402
from nca.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (default runs/<config>/seed<seed>)")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value"
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = args.out or f"runs/{Path(args.config).stem}/seed{args.seed}"
    res = train(
        cfg, out, seed=args.seed, device=args.device, resume=not args.no_resume, quiet=args.quiet
    )
    print("final:", res)


if __name__ == "__main__":
    main()
