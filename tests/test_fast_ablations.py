"""The ablation CLI must use the requested engine without replacing published runs."""

import importlib.util
import json
import sys

import pytest
import yaml

from nca.utils import load_config


def cli():
    spec = importlib.util.spec_from_file_location("ablation_cli", "scripts/run_ablations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fast_ablations_require_a_separate_run_root(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["ablations", "--config", "configs/ablations_maze_ood.yaml", "--trainer", "fast"],
    )
    with pytest.raises(SystemExit, match="2"):
        cli().main()


def test_fast_ablation_trains_from_scratch_and_protects_outputs(tmp_path, monkeypatch):
    cfg = load_config("configs/smoke_maze.yaml")
    cfg["data_root"] = str(tmp_path / "data")
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        cfg["task"][split] = dict(size=9, n=n, source="generated", download=False)
    cfg["train"].update(
        steps=2,
        batch_size=2,
        rollout=2,
        chunk=2,
        amp=False,
        log_every=1,
        eval_every=2,
        checkpoint_every=2,
        eval=dict(n=2, depth=2, k=1),
    )
    cfg["engine"] = dict(backend="fused", compile=False, checkpoint_segment=None)
    cfg["ablations"] = {"no_async": {"train.asynchronous": False}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "fresh"
    argv = [
        "ablations",
        "--config",
        str(path),
        "--trainer",
        "fast",
        "--out-root",
        str(out),
        "--seeds",
        "2",
        "--device",
        "cpu",
        "--no-resume",
        "--disable-tf32",
        "--eval-n",
        "2",
        "--eval-depth",
        "2",
        "--eval-k",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod = cli()
    mod.main()
    run = json.loads((out / "no_async/seed2/run.json").read_text())
    assert run["engine"]["config"]["backend"] == "fused"
    assert run["model_config"]["fire_rate"] == 1.0
    assert run["seed"] == 2
    result = json.loads((out / "evaluations/maze_ood_no_async_seed2.json").read_text())
    assert result["trainer"] == "fast"
    checkpoint = out / "no_async/seed2/final.pt"
    before = checkpoint.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        mod.main()
    assert checkpoint.read_bytes() == before


def test_full_budget_presets_keep_controls_comparable():
    for task, steps, count in [("maze_ood", 5000, 4), ("sudoku_ood", 150000, 5)]:
        cfg = load_config(f"configs/budget_optimized/ablations_{task}.yaml")
        assert cfg["train"]["steps"] == steps
        assert len(cfg["ablations"]) == count
        assert cfg["ablations"]["full"] == {}
        if task == "sudoku_ood":
            assert cfg["model"]["overrides"]["norm_fired_only"] is True
