#!/usr/bin/env python
"""Train one NCA with the fast engine (docs/FAST_ENGINE.md).

    python scripts/train_fast.py --config configs/sudoku_ood.yaml --seed 0
    python scripts/train_fast.py --config configs/sudoku_ood.yaml --seed 0 \
        --set engine.backend=reference --set engine.compile=false --out runs/engine/sudoku_ood_reference/seed0
    python scripts/train_fast.py --config configs/maze_ood.yaml --set engine.compile=true --set engine.block=4

Same configs, outputs and resume behaviour as scripts/train.py; the `engine:` section of the
config (or `--set engine.<key>=<value>`) selects the backend, compilation and checkpointing.
The default output folder is `runs/engine/<config name>_<backend>[_compiled]/seed<seed>`, so
engine runs never overwrite the published `runs/<config>/seed<k>` folders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.engine.engine import EngineConfig  # noqa: E402
from nca.engine.train import train_fast  # noqa: E402
from nca.utils import load_config  # noqa: E402


def default_out(config_path: str, cfg: dict, seed: int) -> str:
    e = EngineConfig.from_dict(cfg.get("engine"))
    name = f"{Path(config_path).stem}_{e.backend}" + ("_compiled" if e.compile else "")
    return f"runs/engine/{name}/seed{seed}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (see the module docstring)")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value"
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--disable-tf32", action="store_true", help="match the measured full-FP32 arithmetic policy"
    )
    args = ap.parse_args()
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config, args.set)
    out = args.out or default_out(args.config, cfg, args.seed)
    res = train_fast(
        cfg, out, seed=args.seed, device=args.device, resume=not args.no_resume, quiet=args.quiet
    )
    print("final:", res)


if __name__ == "__main__":
    main()
