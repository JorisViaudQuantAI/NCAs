#!/usr/bin/env python
"""Turn the checkpoints of the authors' public code into run folders our eval scripts read.

    git clone https://github.com/paradigms-of-intelligence/reasoning-nca ../reasoning-nca
    python scripts/import_public.py --repo ../reasoning-nca --out runs/public
    python scripts/eval_maze_scaling.py --run runs/public/maze_ood --table1 --n 100 \
        --out results/public/maze_scaling_table1.json

Each task gets `runs/public/<task>/final.pt` and `run.json`, like a finished training run: the
model is `nca.public_ckpt.load_public` (the public code's equations, see `NCAConfig`'s
public-code switches) and the config is ours for that task (data, evaluation), so every
evaluation script runs on these weights exactly as on ours. Nothing is trained.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.public_ckpt import load_public  # noqa: E402
from nca.utils import count_parameters, load_config, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "Maze-OOD": ("maze_ood", "configs/maze_ood.yaml"),
    "Maze-Hard": ("maze_hard", "configs/maze_hard.yaml"),
    "Sudoku-OOD": ("sudoku_ood", "configs/sudoku_ood.yaml"),
    "Sudoku-Extreme": ("sudoku_extreme", "configs/sudoku_extreme.yaml"),
}


def repo_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo", required=True, help="clone of paradigms-of-intelligence/reasoning-nca"
    )
    ap.add_argument("--out", default="runs/public")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    args = ap.parse_args()
    repo = Path(args.repo)
    commit = repo_commit(repo)
    for name in args.tasks:
        slug, cfg_path = TASKS[name]
        model, pc = load_public(repo / "checkpoints" / name)
        cfg = load_config(ROOT / cfg_path)
        cfg["model"] = {"public_checkpoint": name}
        out = Path(args.out) / slug
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                model=model.state_dict(),
                model_config=model.cfg.to_dict(),
                config=cfg,
                seed=int(pc["RANDOM_SEED"]),
                step=int(pc["STEPS_N"]),
            ),
            out / "final.pt",
        )
        save_json(
            {
                "seed": int(pc["RANDOM_SEED"]),
                "steps": int(pc["STEPS_N"]),
                "params": count_parameters(model),
                "public_checkpoint": {"task": name, "repo_commit": commit, "config": pc},
                "data": {"train_source": "official"},
            },
            out / "run.json",
        )
        print(f"{name}: {count_parameters(model):,} parameters -> {out}")


if __name__ == "__main__":
    main()
