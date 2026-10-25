"""scripts/archive_stale_runs.py: only runs whose model really changed are archived."""

import importlib.util
import json
from pathlib import Path

from nca.train import build_run_model
from nca.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = str(ROOT / "configs" / "ablations_maze_ood.yaml")


def _archiver():
    spec = importlib.util.spec_from_file_location(
        "archive_stale_runs", ROOT / "scripts" / "archive_stale_runs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_runs(runs_root: Path) -> None:
    """One run.json per ablation variant, written the way train() writes it."""
    base = load_config(ABLATIONS)
    for vname, overrides in base["ablations"].items():
        cfg = load_config(ABLATIONS, [f"{k}={v}" for k, v in overrides.items()])
        cfg.pop("ablations", None)
        d = runs_root / "ablations_maze_ood" / vname / "seed0"
        d.mkdir(parents=True)
        info = {"model_config": build_run_model(cfg).cfg.to_dict()}
        (d / "run.json").write_text(json.dumps(info))


def test_up_to_date_runs_are_not_stale(tmp_path):
    # no_async stores fire_rate 1.0: it must match, so a second phase run archives nothing
    _write_runs(tmp_path)
    no_async = json.loads(
        (tmp_path / "ablations_maze_ood" / "no_async" / "seed0" / "run.json").read_text()
    )
    assert no_async["model_config"]["fire_rate"] == 1.0
    assert _archiver().stale_runs([ABLATIONS], tmp_path) == []


def test_changed_model_is_stale(tmp_path):
    _write_runs(tmp_path)
    path = tmp_path / "ablations_maze_ood" / "full" / "seed0" / "run.json"
    info = json.loads(path.read_text())
    info["model_config"]["channels"] = info["model_config"]["channels"] + 1
    path.write_text(json.dumps(info))
    stale = _archiver().stale_runs([ABLATIONS], tmp_path)
    assert [d.parent.name for d, _ in stale] == ["full"]
