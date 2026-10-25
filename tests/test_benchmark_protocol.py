"""The full timing path must execute the production training recipe."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nca.engine.bench import make_pool, time_full_iterations
from nca.engine.engine import Engine, EngineConfig
from nca.engine.train import resolve_train_cfg, train_fast
from nca.train import build_run_model, build_task_and_data
from nca.utils import load_config, seed_everything


def test_timing_retains_transient_memory_peak_from_later_warmup(monkeypatch):
    spec = importlib.util.spec_from_file_location("extended_bench", "scripts/bench_extended.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    # Model a temporary capture allocation on the second warmup call. It is
    # larger than both the first call and every measured iteration.
    peak = 0
    allocations = iter([1, 7, 2, 3, 4])

    def call():
        nonlocal peak
        peak = max(peak, next(allocations))

    def reset():
        nonlocal peak
        peak = 0

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak * 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: (peak + 1) * 2**20)
    monkeypatch.setattr(bench, "_compiled_graphs", lambda: 0)
    monkeypatch.setattr(bench, "_cuda_graph_recordings", lambda dev: 0)
    result = bench.timed(call, iters=2, warmup=3)
    assert result["first_peak_allocated_gb"] == 1 / 1024
    assert result["warmup_peak_allocated_gb"] == 7 / 1024
    assert result["warmup_peak_reserved_gb"] == 8 / 1024
    assert result["peak_allocated_gb"] == 4 / 1024
    assert result["peak_reserved_gb"] == 5 / 1024


def tiny_config(tmp_path, perturb=True):
    cfg = load_config("configs/smoke_maze.yaml")
    cfg["data_root"] = str(tmp_path / "data")
    cfg["task"]["train"] = dict(size=9, n=16, source="generated", download=False)
    cfg["task"]["val"] = dict(split=None)
    cfg["train"].update(
        steps=3,
        batch_size=2,
        pool_mult=4,
        rollout=4,
        chunk=2,
        augment=True,
        perturb=perturb,
        log_every=3,
        checkpoint_every=3,
    )
    cfg["engine"] = dict(backend="fused", checkpoint_segment=None)
    return cfg


@pytest.mark.parametrize("perturb", [False, True])
def test_full_benchmark_matches_training_batches_and_updates(tmp_path, monkeypatch, perturb):
    cfg = tiny_config(tmp_path, perturb)
    trace = []
    real_iteration = Engine.train_iteration

    def record(self, batch, seeds, c0, weights, grad_clip, apply_update):
        before = batch["x"].detach().clone()
        result = real_iteration(self, batch, seeds, c0, weights, grad_clip, apply_update)
        trace.append(
            (
                before,
                seeds.copy(),
                c0,
                result[0].clone(),
                [p.detach().clone() for p in self.model.parameters()],
            )
        )
        return result

    monkeypatch.setattr(Engine, "train_iteration", record)
    train_fast(cfg, tmp_path / "run", seed=7, device="cpu", quiet=True, resume=False)
    expected = trace[:]
    trace.clear()

    seed_everything(7)
    model = build_run_model(cfg)
    task, boards, _ = build_task_and_data(cfg["task"], cfg["data_root"])
    tr = resolve_train_cfg(cfg)
    pool = make_pool(model, task, boards, 2, seed=7, train_cfg=tr)
    assert pool.size == 8 and pool.augment
    engine = Engine(model, EngineConfig.from_dict(cfg["engine"]), tr, model.device)
    timing = time_full_iterations(engine, pool, tr, iters=3, warmup=0, seed=7)
    assert timing["perturb"] == perturb and timing["skipped_steps"] == 0
    assert len(trace) == len(expected) == 3
    for actual, wanted in zip(trace, expected, strict=True):
        assert torch.equal(actual[0], wanted[0])
        np.testing.assert_array_equal(actual[1], wanted[1])
        assert actual[2] == wanted[2]
        assert torch.equal(actual[3], wanted[3])
        for pa, pb in zip(actual[4], wanted[4], strict=True):
            assert torch.equal(pa, pb)


def test_failed_full_benchmark_restores_model(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    seed_everything(0)
    model = build_run_model(cfg).eval()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    task, boards, _ = build_task_and_data(cfg["task"], cfg["data_root"])
    tr = resolve_train_cfg(cfg)
    pool = make_pool(model, task, boards, 2, train_cfg=tr)
    engine = Engine(model, EngineConfig.from_dict(cfg["engine"]), tr, model.device)
    original = engine.train_iteration
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected after an optimizer update")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "train_iteration", fail_second)
    with pytest.raises(torch.OutOfMemoryError):
        time_full_iterations(engine, pool, tr, iters=3, warmup=0)
    assert not model.training
    assert all(torch.equal(initial[n], p) and p.grad is None for n, p in model.named_parameters())


def test_benchmark_rejects_large_finite_equivalence_error(tmp_path, monkeypatch):
    import json

    import yaml

    spec = importlib.util.spec_from_file_location("bench_cli", Path("scripts/bench_engine.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_config(tmp_path)))
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_engine.py",
            "--config",
            str(cfg_path),
            "--variants",
            "fused:fp32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--no-full",
            "--no-flops",
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(
        cli,
        "compare_backends",
        lambda *a, **k: dict(
            state_max_abs=100.0,
            state_max_abs_ref=1.0,
            grad_max_abs=100.0,
            grad_max_abs_ref=1.0,
            loss_a=1.0,
            loss_abs=0.0,
            nonfinite=[],
            grad_only_a=[],
            grad_only_b=[],
            grad_per_param={},
        ),
    )
    cli.main()
    record = json.loads(out.read_text())["results"][0]
    assert record["ok"] and not record["equivalence_ok"]
    assert "state" in record["equivalence_error"]
