#!/usr/bin/env python
"""Animate a rollout (the "spatial chain-of-thought" of paper Fig. 2): a GIF and a strip of
snapshots of the decoded board as the NCA solves one maze or Sudoku.

    python scripts/animate.py --run runs/maze_ood/seed0 --size 37 --depth 1200 --every 10 --out figures/anim/maze37
    python scripts/animate.py --run runs/sudoku_ood/seed0 --depth 256 --every 2 --out figures/anim/sudoku --board 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import load_run, parse_kv  # noqa: E402
from nca.rollout import RolloutConfig, rollout  # noqa: E402
from nca.viz import board_strip, maze_image, save_gif, sudoku_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=None, help="maze size")
    ap.add_argument("--board", type=int, default=0, help="index of the board to animate")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    model, cfg, _, task = load_run(args.run, args.device)
    kw = {**cfg["task"].get("test", {}), **parse_kv(args.load)}  # config's test options first
    kw.pop("n", None)
    if args.size:
        kw["size"] = args.size
    boards = task.load(args.split, cfg.get("data_root", "data"), n=args.board + 1, **kw).subset(
        [args.board]
    )
    rc = RolloutConfig(depth=args.depth, k=1, record_every=args.every, record_boards=1)
    res = rollout(model, task, boards, rc, seed=args.seed)
    trace = res.boards_trace[0]  # (T, H, W)
    solved_at = next(
        (int(s) for s, ok in zip(res.steps, res.trace_exact[0, 0], strict=True) if ok), None
    )
    print(
        f"board {args.board}: solved at step {solved_at} (final exact = {bool(res.trace_exact[0, 0, -1])})"
    )
    H = boards.shape[0]
    if task.name.startswith("maze"):
        scale = max(2, 256 // H)
        frames = [maze_image(t, scale) for t in trace]
    else:
        frames = [sudoku_image(t, boards.given[0], boards.targets[0]) for t in trace]
    save_gif(frames, args.out + ".gif", fps=args.fps)
    idx = np.linspace(0, len(frames) - 1, args.snapshots).round().astype(int)
    board_strip(
        [frames[i] for i in idx],
        [int(res.steps[i]) for i in idx],
        args.out + "_strip",
        title=f"{task.name} ({H}x{boards.shape[1]}), seed {args.seed}",
    )
    print(f"-> {args.out}.gif and {args.out}_strip.png")


if __name__ == "__main__":
    main()
