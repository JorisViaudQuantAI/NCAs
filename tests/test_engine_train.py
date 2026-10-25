"""Trajectory- and training-level validation of the fast engine: backends agree on whole
rollouts, segmented checkpointing changes nothing, `train_fast` writes the same files as
`train` and resumes exactly, the update order option behaves as documented, and the
benchmark helpers run (docs/FAST_ENGINE.md, "Validation")."""

import json
import os

import numpy as np
import pytest
import torch

from nca.engine.bench import (
    clone_batch,
    compare_backends,
    count_step_flops,
    iteration_seeds,
    make_batch,
    probe_compile,
    time_iterations,
)
from nca.engine.engine import Engine, EngineConfig
from nca.engine.train import loss_weights, resolve_train_cfg, train_fast
from nca.model import build_model
from nca.tasks import get_task
from nca.train import load_model
from nca.utils import load_config


def _maze_setup(tmp_path, n=8, B=4):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), size=9, n=n, download=False, source="generated")
    model = build_model("maze_ood")
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    batch = make_batch(model, task, boards, B)
    return task, boards, model, batch


def _tr(rollout=8, chunk=8, amp=False, **kw):
    tr = resolve_train_cfg({"train": dict(rollout=rollout, chunk=chunk, amp=amp, **kw)})
    return tr


def test_backends_agree_on_a_rollout(tmp_path):
    task, boards, model, batch = _maze_setup(tmp_path)
    for n in (1, 8):
        tr = _tr(n, n)
        seeds, c0 = iteration_seeds(tr, 0)
        w = loss_weights("all", n, torch.device("cpu"))
        r = compare_backends(
            model,
            EngineConfig(backend="reference", checkpoint_segment=None),
            EngineConfig(backend="fused", checkpoint_segment=None),
            tr,
            batch,
            seeds,
            c0,
            w,
        )
        assert r["state_max_abs"] < 1e-5 and r["grad_max_abs"] < 1e-5 * max(
            1, r["grad_max_abs_ref"]
        )
        assert r["loss_abs"] < 1e-6


def test_sudoku_like_backends_agree(tmp_path):
    task = get_task("sudoku_ood")
    boards = task.load("train", str(tmp_path), n=4, download=False, source="generated")
    model = build_model("sudoku_ood", channels=32, heads=4, norm_groups=4, norm_fired_only=True)
    batch = make_batch(model, task, boards, 4)
    tr = _tr(6, 6)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 6, torch.device("cpu"))
    r = compare_backends(
        model,
        EngineConfig(backend="reference", checkpoint_segment=None),
        EngineConfig(backend="fused", checkpoint_segment=None),
        tr,
        batch,
        seeds,
        c0,
        w,
    )
    assert r["state_max_abs"] < 1e-4 and r["grad_max_abs"] < 1e-4 * max(1, r["grad_max_abs_ref"])


@pytest.mark.parametrize("segment", [1, 3, 4])
def test_segmented_checkpointing_is_exact(tmp_path, segment):
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(8, 8, loss_window=3)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights(3, 8, torch.device("cpu"))
    r = compare_backends(
        model,
        EngineConfig(backend="fused", checkpoint_segment=None),
        EngineConfig(backend="fused", checkpoint_segment=segment),
        tr,
        batch,
        seeds,
        c0,
        w,
    )
    # the recomputed segments see the same draws: identical to the rounding
    assert r["state_max_abs"] == 0.0 and r["loss_abs"] == 0.0
    assert r["grad_max_abs"] < 1e-6


def test_block_size_does_not_change_the_result(tmp_path):
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(7, 7)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 7, torch.device("cpu"))
    r = compare_backends(
        model,
        EngineConfig(backend="fused", block=1, checkpoint_segment=None),
        EngineConfig(backend="fused", block=3, checkpoint_segment=None),  # 3 + 3 + 1 steps
        tr,
        batch,
        seeds,
        c0,
        w,
    )
    assert r["state_max_abs"] == 0.0 and r["grad_max_abs"] == 0.0


def test_update_order_option(tmp_path):
    """With chunk < rollout the steps after the chunk use the old weights when the update is
    applied after the rollout and the new ones otherwise; with chunk == rollout both agree."""
    task, boards, model, batch = _maze_setup(tmp_path)
    w = loss_weights("all", 4, torch.device("cpu"))

    def run(after, rollout):
        import copy

        m = copy.deepcopy(model)
        tr = _tr(rollout, 4)
        eng = Engine(
            m,
            EngineConfig(backend="fused", checkpoint_segment=None, update_after_rollout=after),
            tr,
            m.device,
        )
        seeds, _ = iteration_seeds(tr, 0)
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        calls = []

        def upd(gn):
            calls.append(float(gn))
            opt.step()
            opt.zero_grad(set_to_none=True)

        x, loss, gn = eng.train_iteration(clone_batch(batch), seeds, 0, w, 1e9, upd)
        assert len(calls) == 1
        return x

    assert torch.equal(run(True, 4), run(False, 4))
    assert not torch.equal(run(True, 8), run(False, 8))


def _smoke_cfg(tmp_path, steps, **engine):
    cfg = load_config("configs/smoke_maze.yaml")
    cfg["task"]["train"] = {"size": 9, "n": 64, "source": "generated", "download": False}
    cfg["task"]["val"] = {
        "split": "test",
        "size": 9,
        "n": 8,
        "source": "generated",
        "download": False,
    }
    cfg["data_root"] = str(tmp_path / "data")
    cfg["train"].update(
        steps=steps,
        batch_size=4,
        rollout=6,
        chunk=4,
        log_every=2,
        eval_every=steps,
        checkpoint_every=3,
    )
    cfg["train"]["eval"] = {"n": 8, "depth": 6, "k": 1}
    cfg["engine"] = {"backend": "fused", "checkpoint_segment": 2, **engine}
    return cfg


def test_bounded_training_preserves_schedule_and_exact_resume(tmp_path):
    cfg = _smoke_cfg(tmp_path, 6)
    cfg["train"].update(schedule="cosine", warmup_ratio=0.25)
    cfg["task"]["val"]["split"] = ""  # inherited metadata must not trigger a load
    full, split = tmp_path / "full", tmp_path / "split"
    train_fast(cfg, full, device="cpu", quiet=True, resume=False)
    train_fast(cfg, split, device="cpu", quiet=True, resume=False, stop_at_step=2)
    partial = torch.load(split / "checkpoint.pt", weights_only=False)
    summary = json.loads((split / "summary.json").read_text())
    assert partial["step"] == summary["steps"] == 2
    assert summary["target_steps"] == 6 and not summary["training_complete"]
    with pytest.raises(ValueError, match="after the resumed"):
        train_fast(cfg, split, device="cpu", quiet=True, stop_at_step=2)
    train_fast(cfg, split, device="cpu", quiet=True, stop_at_step=6)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(
        torch.load(full / "checkpoint.pt", weights_only=False),
        torch.load(split / "checkpoint.pt", weights_only=False),
    )
    equal(
        torch.load(full / "final.pt", weights_only=False),
        torch.load(split / "final.pt", weights_only=False),
    )


def test_train_fast_writes_the_usual_files_and_resumes_exactly(tmp_path):
    cfg = _smoke_cfg(tmp_path, 6)
    out_a = tmp_path / "a"
    train_fast(cfg, out_a, seed=0, device="cpu", quiet=True)
    for f in ("final.pt", "run.json", "log.jsonl", "summary.json", "checkpoint.pt"):
        assert (out_a / f).exists()
    run = json.loads((out_a / "run.json").read_text())
    assert run["engine"]["segment"] == 2 and run["engine"]["config"]["backend"] == "fused"
    recs = [json.loads(line) for line in open(out_a / "log.jsonl")]
    assert "iter_seconds" in recs[0]
    model, _ = load_model(out_a / "final.pt")  # the eval scripts' loader reads it
    # interrupted after 3 steps, then resumed: identical weights
    cfg3 = _smoke_cfg(tmp_path, 3)
    out_b = tmp_path / "b"
    train_fast(cfg3, out_b, seed=0, device="cpu", quiet=True)
    cfg6 = _smoke_cfg(tmp_path, 6)
    train_fast(cfg6, out_b, seed=0, device="cpu", quiet=True)
    a = torch.load(out_a / "final.pt", weights_only=False)["model"]
    b = torch.load(out_b / "final.pt", weights_only=False)["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_reference_backend_trains_too(tmp_path):
    cfg = _smoke_cfg(tmp_path, 2, backend="reference", checkpoint_segment=None)
    res = train_fast(cfg, tmp_path / "r", seed=0, device="cpu", quiet=True)
    assert "val_exact" in res


def test_bench_helpers(tmp_path):
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(4, 4)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 4, torch.device("cpu"))
    eng = Engine(model, EngineConfig(backend="fused", checkpoint_segment=None), tr, model.device)
    t = time_iterations(eng, batch, seeds, c0, w, iters=2, warmup=1)
    assert len(t["seconds"]) == 2 and t["seconds_mean"] > 0
    fl = count_step_flops(eng, batch)
    assert fl["forward_backward"] > fl["forward"] > 0
    # the model's weights are untouched by the measurements
    assert all(p.grad is None for p in model.parameters())


def test_autotune_picks_a_segment(tmp_path):
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(8, 8)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 8, torch.device("cpu"))
    eng = Engine(
        model,
        EngineConfig(backend="fused", checkpoint_segment="auto", checkpoint_candidates=(0, 4)),
        tr,
        model.device,
    )
    q = eng.autotune(batch, seeds, c0, w)
    assert q in (None, 4) and len(eng.state.autotune) == 2
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(os.environ.get("NCA_SKIP_COMPILE") == "1", reason="torch.compile disabled")
def test_compiled_blocks_match_eager(tmp_path):
    probe = probe_compile(torch.device("cpu"))
    if not probe["ok"]:
        pytest.skip(f"torch.compile unavailable here: {probe['error']}")
    task, boards, model, batch = _maze_setup(tmp_path, B=2)
    tr = _tr(4, 4)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 4, torch.device("cpu"))
    r = compare_backends(
        model,
        EngineConfig(backend="fused", checkpoint_segment=None),
        EngineConfig(backend="fused", compile=True, block=2, checkpoint_segment=None),
        tr,
        batch,
        seeds,
        c0,
        w,
    )
    assert r["state_max_abs"] < 1e-4 and r["grad_max_abs"] < 1e-4 * max(1, r["grad_max_abs_ref"])


def test_loss_weights():
    dev = torch.device("cpu")
    assert loss_weights("all", 4, dev).tolist() == [1, 1, 1, 1]
    assert loss_weights("last", 4, dev).tolist() == [0, 0, 0, 1]
    assert loss_weights(2, 4, dev).tolist() == [0, 0, 1, 1]
    assert isinstance(np.random.default_rng(0), np.random.Generator)


# ----------------------------------------------------------------------------- review fixes
# Regressions found in code review (2026-10-13): autotune never selects an over-budget
# segment, ARC batches can be re-used after a backward pass, autocast stays inside the BPTT
# chunk, a missing gradient fails the equivalence check, ARC measurements take a clock.


def test_autotune_raises_when_nothing_fits(tmp_path, monkeypatch):
    """A budget nobody fits: an error naming the trials, never a silent over-budget pick."""
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(8, 8)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 8, torch.device("cpu"))
    eng = Engine(
        model,
        EngineConfig(backend="fused", checkpoint_segment="auto", checkpoint_candidates=(0, 4)),
        tr,
        model.device,
    )

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(eng, "train_iteration", oom)
    with pytest.raises(RuntimeError, match="no checkpoint segment fits"):
        eng.autotune(batch, seeds, c0, w)
    assert eng.state.segment is None and len(eng.state.autotune) == 2
    assert all(p.grad is None for p in model.parameters())


def _arc_setup(tmp_path, B=2):
    task = get_task("arc", n_tasks=2)
    boards = task.load("train", str(tmp_path), source="synthetic")
    model = build_model(
        "arc",
        n_tasks=2,
        channels=32,
        c_in=12,
        c_out=12,
        out_offset=12,
        c_task=4,
        heads=2,
        norm_groups=2,
        time_steps=4,
    )
    batch = make_batch(model, task, boards, B)
    return task, boards, model, batch


def test_arc_batch_can_be_rolled_out_twice(tmp_path):
    """The task embedding written into a fresh ARC batch carries an autograd graph that a
    backward pass consumes; `clone_batch` rebuilds it, so autotuning, timing and comparing
    on the same batch work (and the embedding still receives a gradient)."""
    task, boards, model, batch = _arc_setup(tmp_path)
    assert batch["x"].requires_grad
    tr = _tr(4, 4, replay=False, perturb=False, loss="ce")
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 4, torch.device("cpu"))
    eng = Engine(
        model,
        EngineConfig(backend="fused", checkpoint_segment="auto", checkpoint_candidates=(0, 2)),
        tr,
        model.device,
    )
    eng.autotune(batch, seeds, c0, w)  # two candidates, warm-up + timed: four backward passes
    t = time_iterations(eng, batch, seeds, c0, w, iters=2, warmup=1)
    assert len(t["seconds"]) == 2
    r = compare_backends(
        model,
        EngineConfig(backend="reference", checkpoint_segment=None),
        EngineConfig(backend="fused", checkpoint_segment=None),
        tr,
        batch,
        seeds,
        c0,
        w,
    )
    assert "task_embed.weight" in r["grad_per_param"]
    assert not r["grad_only_a"] and not r["grad_only_b"] and not r["nonfinite"]
    assert r["state_max_abs"] < 1e-5
    # the copy has the same values as the original
    c = clone_batch(batch, model)
    assert torch.equal(c["x"].detach(), batch["x"].detach()) and c["x"].requires_grad
    # the fused step's sync-free time embedding equals the reference one
    fl = count_step_flops(eng, batch)
    assert fl["forward"] > 0
    with torch.no_grad():
        ref = Engine(model, EngineConfig(backend="reference"), tr, model.device)
        clk = torch.arange(4)
        a, _ = ref.run_steps(c["x"], c["given4"], c["clamp"], seeds, clk, False)
        b, _ = eng.run_steps(c["x"], c["given4"], c["clamp"], seeds, clk, False)
    assert (a - b).abs().max() < 1e-5


def test_amp_applies_to_the_chunk_only(tmp_path):
    """With `amp` and chunk < rollout the steps outside the chunk run in fp32: the state after
    the rollout equals the one of an fp32 run on the steps before the chunk, and differs from a
    run under autocast everywhere."""
    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(8, 2, amp=True)
    seeds, _ = iteration_seeds(tr, 0)
    eng = Engine(model, EngineConfig(backend="fused", checkpoint_segment=None), tr, model.device)
    c0 = 6
    # the no-grad prefix through the engine...
    x_eng, _ = eng.run_steps(batch["x"], batch["given4"], batch["clamp"], seeds[:c0], None, False)
    # ...equals the fp32 reference step by step
    with torch.no_grad():
        x = batch["x"].detach().clone()
        for d in eng.draws(seeds[:c0], tuple(x.shape)):
            x = eng.step(x, batch["given4"], batch["clamp"], d.fire, d.noise, d.noise_mask, None)
    assert torch.equal(x_eng, x)
    # ...and differs from the same steps under bf16 autocast
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        xa = batch["x"].detach().clone()
        for d in eng.draws(seeds[:c0], tuple(xa.shape)):
            xa = eng.step(xa, batch["given4"], batch["clamp"], d.fire, d.noise, d.noise_mask, None)
    assert not torch.equal(x_eng, xa.float())


def test_compare_backends_reports_a_missing_gradient(tmp_path):
    """A parameter without gradient on one side is a failure, not an ignored key."""
    from nca.engine.bench import DEFAULT_TOL, within_tolerance

    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(4, 4)
    seeds, c0 = iteration_seeds(tr, 0)
    w = loss_weights("all", 4, torch.device("cpu"))
    a = EngineConfig(backend="reference", checkpoint_segment=None)
    r = compare_backends(
        model, a, EngineConfig(backend="fused", checkpoint_segment=None), tr, batch, seeds, c0, w
    )
    ok, why = within_tolerance(r, DEFAULT_TOL["fp32"])
    assert ok, why
    # freeze one parameter of the model used by the second configuration only
    import nca.engine.bench as bench

    orig = bench.Engine

    class Freezing(orig):
        n = 0

        def __init__(self, m, ecfg, tr_, dev):
            Freezing.n += 1
            if Freezing.n == 2:
                next(m.parameters()).requires_grad_(False)
            super().__init__(m, ecfg, tr_, dev)

    bench.Engine = Freezing
    try:
        r = compare_backends(
            model,
            a,
            EngineConfig(backend="fused", checkpoint_segment=None),
            tr,
            batch,
            seeds,
            c0,
            w,
        )
    finally:
        bench.Engine = orig
    assert r["grad_only_a"] and r["grad_max_abs"] == float("inf")
    ok, why = within_tolerance(r, DEFAULT_TOL["fp32"])
    assert not ok and "missing" in why
    assert not within_tolerance(dict(r, nonfinite=["x"]), DEFAULT_TOL["fp32"])[0]


def test_comparison_snapshots_reused_output_storage(tmp_path, monkeypatch):
    """A later graph invocation must not overwrite the reference comparison."""
    with torch.random.fork_rng(devices=[]):
        _, _, model, batch = _maze_setup(tmp_path)
    tr = _tr(1, 1)
    seeds, c0 = iteration_seeds(tr, 0)
    weights = loss_weights("all", 1, model.device)
    storage = torch.empty(())
    calls = 0

    def borrowed_outputs(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        storage.fill_(calls)
        return storage, storage, storage

    monkeypatch.setattr(Engine, "train_iteration", borrowed_outputs)
    cfg = EngineConfig(backend="reference", checkpoint_segment=None)
    result = compare_backends(model, cfg, cfg, tr, batch, seeds, c0, weights)
    assert result["state_max_abs"] == 1
    assert result["loss_a"] == result["grad_norm_a"] == 1
    assert result["loss_b"] == result["grad_norm_b"] == 2


def test_tolerance_rejects_loss_only_mismatch_and_nan():
    from nca.engine.bench import DEFAULT_TOL, within_tolerance

    record = dict(
        state_max_abs=0.0,
        grad_max_abs=0.0,
        grad_max_abs_ref=1.0,
        loss_a=1.0,
        loss_abs=0.0,
        nonfinite=[],
        grad_only_a=[],
        grad_only_b=[],
    )
    assert within_tolerance(record, DEFAULT_TOL["fp32"])[0]
    ok, why = within_tolerance({**record, "loss_abs": 0.01}, DEFAULT_TOL["fp32"])
    assert not ok and "loss" in why
    for key in ("state_max_abs", "grad_max_abs", "loss_abs"):
        assert not within_tolerance({**record, key: float("nan")}, DEFAULT_TOL["fp32"])[0]


def test_time_full_iterations_restores_the_weights(tmp_path):
    from nca.engine.bench import make_pool, time_full_iterations

    task, boards, model, batch = _maze_setup(tmp_path)
    tr = _tr(4, 4, steps=10, batch_size=4)
    pool = make_pool(model, task, boards, 4)
    eng = Engine(model, EngineConfig(backend="fused", checkpoint_segment=None), tr, model.device)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    t = time_full_iterations(eng, pool, tr, iters=2, warmup=1)
    assert len(t["seconds"]) == 2 and t["seconds_mean"] > 0
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())


def test_triton_mix_accepts_normalized_backward_under_both_normalisations():
    """The RTX 4060 campaign found the state returned by `triton_mix` + `normalized_backward`
    corrupted under `norm_fired_only=false`; the cause (the saved normalized state reused by the
    compiled backward) is fixed in `train_iteration` and covered by the CUDA regression
    `test_compiled_normalized_backward_returns_the_final_state`, so the engine accepts the
    combination again under either normalisation. Building the engine does not need CUDA."""
    tr, cpu = _tr(1, 1), torch.device("cpu")
    for fired_only in (False, True):
        model = build_model("sudoku_ood", norm_fired_only=fired_only)
        Engine(
            model, EngineConfig(backend="fused", triton_mix=True, normalized_backward=True), tr, cpu
        )
