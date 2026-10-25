"""Archive runs whose model no longer matches their config (and the results computed from them).

The phase scripts resume finished trainings and skip evaluations whose JSON exists, so a change
of architecture in a preset (for example the maze presets on 2026-10-12, see docs/DEVIATIONS.md)
would otherwise be silently ignored. This script compares the `model_config` stored in each
run's `run.json` with the model the config builds today; runs that differ are moved to
`runs/archive/<stamp>/...`, together with the result and figure files given by `--results`
(globs), moved to `results/archive/<stamp>/...` (figures keep their folder under the same
stamp). Nothing is deleted.

    python scripts/archive_stale_runs.py --config configs/maze_ood.yaml configs/ablations_maze_ood.yaml \
        --results "results/maze_scaling/*.json" "results/adaptive/*.json" "results/noise/maze_ood_*.json" \
                  "results/ablations/maze_ood_*.json" "figures/anim/maze37*"
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.train import build_run_model  # noqa: E402
from nca.utils import load_config, load_json  # noqa: E402


def expected_model_config(cfg: dict) -> dict:
    # the same construction as train(), including train-time changes such as no_async
    return build_run_model(cfg).cfg.to_dict()


def run_dirs_and_expected(config_path: str, runs_root: Path) -> list[tuple[Path, dict]]:
    """(run dir, expected model config) for every run the config owns: `runs/<stem>/seed*` or,
    for an ablation config, `runs/<stem>/<variant>/seed*` with the variant's overrides."""
    name = Path(config_path).stem
    base = load_config(config_path)
    variants = base.pop("ablations", None)
    out = []
    if variants is None:
        for d in sorted((runs_root / name).glob("seed*")):
            out.append((d, expected_model_config(base)))
        return out
    for vname, overrides in variants.items():
        cfg = load_config(config_path, [f"{k}={v}" for k, v in overrides.items()])
        cfg.pop("ablations", None)
        for d in sorted((runs_root / name / vname).glob("seed*")):
            out.append((d, expected_model_config(cfg)))
    return out


def stale_runs(configs: list[str], runs_root: Path) -> list[tuple[Path, str]]:
    """Run dirs whose stored model config differs from the expected one, with the reason."""
    stale = []
    for c in configs:
        for d, expected in run_dirs_and_expected(c, runs_root):
            info_path = d / "run.json"
            if not info_path.exists():
                continue
            stored = load_json(info_path).get("model_config", {})
            # json round trip: tuples become lists; keys a run predates are taken as unchanged
            expected = json.loads(json.dumps(expected))
            diff = sorted(k for k in stored if k in expected and stored[k] != expected[k])
            if diff:
                stale.append((d, "model config differs on " + ", ".join(diff)))
    return stale


def move(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"{'would move' if dry_run else 'move'}: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(dst.name + f".{int(time.time())}")
    shutil.move(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", nargs="+", required=True, help="training configs whose runs to check"
    )
    ap.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="globs of result/figure files that depend on those runs",
    )
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    stale = stale_runs(args.config, runs_root)
    if not stale:
        print("no stale run: every run.json matches its config")
        return
    for d, why in stale:
        print(f"stale: {d} ({why})")
        move(d, runs_root / "archive" / args.stamp / d.relative_to(runs_root), args.dry_run)
    for pattern in args.results:
        for f in sorted(glob.glob(pattern)):
            p = Path(f)
            top = p.parts[0]  # results/... or figures/...
            move(p, Path(top) / "archive" / args.stamp / Path(*p.parts[1:]), args.dry_run)
    print(f"archived {len(stale)} run(s) under {runs_root / 'archive' / args.stamp}")


if __name__ == "__main__":
    main()
