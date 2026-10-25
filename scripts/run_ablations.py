#!/usr/bin/env python
"""Training-ingredient ablations (paper Fig. 7A: asynchronous updates, perturbations, sample
replay) and the fixed- vs self-attention comparison (Fig. 13), each variant trained with 3 seeds.

    python scripts/run_ablations.py --config configs/ablations_maze_ood.yaml --seeds 0 1 2
    python scripts/run_ablations.py --config configs/ablations_sudoku_ood.yaml --seeds 0 1 2 --only full no_async

Each variant trains in runs/<config name>/<variant>/seed<k> (resumable) and is then evaluated at
K=1 on the test split (`--eval-depth`, `--eval-n`), writing results/ablations/<task>_<variant>_seed<k>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import load_boards, load_run, parse_kv  # noqa: E402
from nca.engine.train import train_fast  # noqa: E402
from nca.experiments import tts_curves  # noqa: E402
from nca.train import train  # noqa: E402
from nca.utils import load_config, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-depth", type=int, default=None, help="default: 2x the training rollout")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument(
        "--load",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="test-set loading options, e.g. size=13 for Maze-OOD (default: the config's test set)",
    )
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--trainer", choices=("historical", "fast"), default="historical")
    ap.add_argument("--out-root", type=Path, help="new run root, required for the fast trainer")
    ap.add_argument("--results-root", type=Path, help="evaluation JSON directory")
    ap.add_argument(
        "--no-resume", action="store_true", help="start from scratch in empty run folders"
    )
    ap.add_argument("--disable-tf32", action="store_true")
    ap.add_argument(
        "--eval-engine-config", help="optional engine YAML for compiled uniform evaluation"
    )
    args = ap.parse_args()
    if args.trainer == "fast" and args.out_root is None:
        ap.error("--trainer fast requires --out-root to preserve published historical runs")
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    base = load_config(args.config, args.set)
    variants = base.pop("ablations")
    if args.only and set(args.only) - set(variants):
        ap.error("unknown variants: " + ", ".join(sorted(set(args.only) - set(variants))))
    name = Path(args.config).stem
    run_root = args.out_root or Path("runs") / name
    result_root = args.results_root or (
        run_root / "evaluations" if args.out_root is not None else Path("results/ablations")
    )
    trainer = train_fast if args.trainer == "fast" else train
    for vname, overrides in variants.items():
        if args.only and vname not in args.only:
            continue
        for seed in args.seeds:
            out = run_root / vname / f"seed{seed}"
            cfg = load_config(args.config, [f"{k}={v}" for k, v in overrides.items()] + args.set)
            cfg.pop("ablations", None)
            if not args.skip_train:
                if args.no_resume and out.exists() and any(out.iterdir()):
                    ap.error(f"from-scratch output is not empty: {out}")
                print(f"=== {vname} seed {seed} -> {out}")
                trainer(cfg, out, seed=seed, device=args.device, resume=not args.no_resume)
            model, cfg_run, _, task = load_run(str(out), args.device)
            boards = load_boards(task, cfg_run, "test", args.eval_n, parse_kv(args.load))
            depth = args.eval_depth or 2 * cfg_run["train"]["rollout"]
            group = {"sudoku_ood": "level", "sudoku_extreme": "bin"}.get(task.name)
            res = tts_curves(
                model,
                task,
                boards,
                depth,
                args.eval_k,
                max(1, depth // 16),
                seed=seed,
                group_key=group,
                engine=None
                if args.eval_engine_config is None
                else load_config(args.eval_engine_config)["engine"],
            )
            res.update(
                variant=vname,
                seed=seed,
                run=str(out),
                overrides=overrides,
                board_shape=list(boards.shape),
                load=parse_kv(args.load),
                trainer=args.trainer,
            )
            path = result_root / f"{task.name}_{vname}_seed{seed}.json"
            save_json(res, path)
            print(
                f"{vname} seed {seed}: K=1 {res['conf_at_k'][1]:.3f}  conf@{args.eval_k} "
                f"{res['conf_at_k'][min(args.eval_k, res['k'])]:.3f} -> {path}"
            )


if __name__ == "__main__":
    main()
