"""The rollout engine and the training loop, end to end on tiny problems (CPU, seconds)."""

import numpy as np
import torch

from nca.experiments import (
    adaptive_compute,
    maze_size_scaling,
    noise_sweep,
    pruning_pareto,
    tts_curves,
)
from nca.model import build_model
from nca.rollout import (
    RolloutConfig,
    conf_at_k,
    majority_vote,
    pass_at_k,
    rollout,
    score_candidates,
)
from nca.tasks import get_task
from nca.train import load_model, train
from nca.utils import load_config


def _maze_boards(tmp_path, n=6, size=9):
    task = get_task("maze_ood")
    return task, task.load(
        "test", str(tmp_path), size=size, n=n, download=False, source="generated"
    )


def test_rollout_shapes_and_traces(tmp_path):
    task, boards = _maze_boards(tmp_path)
    model = build_model("maze_ood")
    rc = RolloutConfig(
        depth=10, k=3, record_every=4, record_boards=2, max_cells=500
    )  # forces chunking
    res = rollout(model, task, boards, rc, seed=0)
    assert res.pred.shape == (6, 3, 9, 9) and res.conf.shape == (6, 3)
    assert list(res.steps) == [4, 8, 10]
    assert res.trace_exact.shape == (6, 3, 3) and res.boards_trace.shape == (2, 3, 9, 9)
    # given cells are copied from the inputs
    given = boards.given[:, None].repeat(3, axis=1)
    assert np.array_equal(res.pred[given], np.repeat(boards.inputs[:, None], 3, 1)[given])
    # 80% firing: about 0.8 * cells * steps updates per trial
    cells = 81 * 10
    assert 0.6 * cells < res.cell_updates.mean() < 1.0 * cells
    m = score_candidates(task, res, boards)
    assert set(m) >= {"exact", "valid", "optimal"} and m["exact"].shape == (6, 3)


def test_rollout_is_reproducible(tmp_path):
    task, boards = _maze_boards(tmp_path, n=2)
    model = build_model("maze_ood")
    rc = RolloutConfig(depth=5, k=2)
    a = rollout(model, task, boards, rc, seed=1)
    b = rollout(model, task, boards, rc, seed=1)
    assert np.array_equal(a.pred, b.pred) and np.allclose(a.conf, b.conf)


def test_selection_helpers():
    metric = np.array([[True, False, False], [False, False, True]])
    conf = np.array([[0.1, 0.9, 0.5], [0.2, 0.3, 0.9]])
    alive = np.ones((2, 3), dtype=bool)
    assert conf_at_k(metric, conf, alive, [1, 2, 3]) == {1: 0.5, 2: 0.0, 3: 0.5}
    assert pass_at_k(metric, alive, [1, 3]) == {1: 0.5, 3: 1.0}
    pred = np.array([[[[1]], [[1]], [[2]]]])
    top = majority_vote(pred, np.ones((1, 3), dtype=bool), top=2)
    assert top[0][0][0, 0] == 1 and top[0][1][0, 0] == 2


def test_pruning_and_adaptive_and_damage_run(tmp_path):
    task, boards = _maze_boards(tmp_path, n=3)
    model = build_model("maze_ood")
    rc = RolloutConfig(depth=8, k=4, pruning={"m": 4, "n_cap": 3}, record_every=2)
    res = rollout(model, task, boards, rc, seed=0)
    assert res.alive.sum(axis=1).tolist() == [1, 1, 1]  # 4 -> 2 -> 1 (halved at steps 2, 4, 6)
    # every (board, trial) has a final board and confidence, pruned ones keep their last state
    assert res.pred.shape == (3, 4, 9, 9) and np.isfinite(res.conf).all()
    assert res.trace_exact.shape == (3, 4, 4) and (res.trace_updates[..., -1] > 0).all()
    dead = ~res.alive
    # a trajectory pruned at step 2 stops accumulating updates (its rows are frozen)
    assert (res.cell_updates[dead] < res.cell_updates[res.alive].min()).all()
    rc = RolloutConfig(depth=8, k=1, p_low=0.1, tau=0.5, damage_at=4, record_every=4)
    res = rollout(model, task, boards, rc, seed=0)
    assert res.damage_masks.shape == (3, 9, 9) and res.damage_masks.any()


def test_experiment_functions_return_json_like(tmp_path):
    task, boards = _maze_boards(tmp_path, n=4)
    model = build_model("maze_ood")
    out = tts_curves(model, task, boards, depth=6, k=2, record_every=3, group_key="size")
    assert len(out["depth_curve"]) == 2 and set(out["conf_at_k"]) == {1, 2}
    assert out["source"] == boards.meta.get("source") and out["max_cells"] == 2_000_000
    out = noise_sweep(model, task, boards, depth=4, k=2, sigmas=[0, 0.1], p_t=0.5, p_s=0.5)
    assert [r["sigma"] for r in out["rows"]] == [0, 0.1]
    out = maze_size_scaling(
        model,
        task,
        lambda s, n: task.load(
            "test", str(tmp_path), size=s, n=n, download=False, source="generated"
        ),
        [9, 11],
        lambda s: 4,
        2,
    )
    assert [r["size"] for r in out["rows"]] == [9, 11]
    out = adaptive_compute(
        model, task, boards, depth=6, p_fire=0.8, p_low=0.4, tau=0.95, record_every=3, damage_at=3
    )
    assert set(out["variants"]) == {"uniform", "adaptive", "adaptive_given"}
    assert set(out["ratios"]) == {"adaptive", "adaptive_given"}
    assert (
        out["variants"]["adaptive_given"]["final_updates"]
        < out["variants"]["uniform"]["final_updates"]
    )
    # incremental: a deeper call keeps nothing that fell short of 95% at a smaller depth
    out2 = adaptive_compute(
        model,
        task,
        boards,
        depth=9,
        p_fire=0.8,
        p_low=0.4,
        tau=0.95,
        record_every=3,
        damage_at=3,
        variants=("uniform", "adaptive_given"),
        previous=out,
    )
    assert set(out2["variants"]) == {"uniform", "adaptive_given"}
    for v in out2["variants"].values():
        assert v["steps_to_95"] is not None or v["depth"] == 9
    out = pruning_pareto(model, task, boards, depths=[4], k=4, ms=[2])
    assert [r["m"] for r in out["rows"]] == [1, 2]


def test_training_smoke_and_reload(tmp_path):
    cfg = load_config(
        "configs/smoke_maze.yaml",
        [
            f"data_root={tmp_path}",
            "task.train.n=64",
            "task.val.n=8",
            "train.steps=4",
            "train.batch_size=4",
            "train.rollout=6",
            "train.chunk=4",
            "train.log_every=2",
            "train.eval_every=2",
            "train.checkpoint_every=2",
            "train.eval.n=8",
            "train.eval.depth=6",
            "task.train.download=false",
            "task.val.download=false",
        ],
    )
    out = tmp_path / "run"
    res = train(cfg, out, seed=0, device="cpu", resume=False, quiet=True)
    assert "val_exact" in res and (out / "final.pt").exists() and (out / "checkpoint.pt").exists()
    model, cfg2 = load_model(out / "final.pt", "cpu")
    assert cfg2["task"]["name"] == "maze_ood" and model.cfg.channels == 16
    # resuming continues from the checkpoint and reaches the new step count
    cfg["train"]["steps"] = 6
    train(cfg, out, seed=0, device="cpu", resume=True, quiet=True)
    import json

    lines = [json.loads(line) for line in open(out / "log.jsonl")]
    assert max(d.get("step", 0) for d in lines) == 6


def test_maze_learns_something_quickly(tmp_path):
    """Level 0 of the tutorial: 300 steps on 9x9 mazes (a few seconds in CI). The loss first
    rises while the output channels leave zero, then drops well below its starting value."""
    torch.manual_seed(0)
    cfg = load_config(
        "configs/smoke_maze.yaml",
        [
            f"data_root={tmp_path}",
            "task.train.n=512",
            "task.val.n=32",
            "train.log_every=25",
            "train.eval_every=300",
            "train.checkpoint_every=300",
            "train.eval.n=32",
            "train.eval.depth=48",
            "task.train.download=false",
            "task.val.download=false",
        ],
    )
    train(cfg, tmp_path / "run2", seed=0, device="cpu", resume=False, quiet=True)
    import json

    losses = [
        d["loss"] for d in map(json.loads, open(tmp_path / "run2" / "log.jsonl")) if "loss" in d
    ]
    assert min(losses[-3:]) < 0.6 * losses[0], losses


def test_pruning_compacts_the_gpu_batch(tmp_path, monkeypatch):
    """Paper B.6: pruned trajectories leave the batch, so the model is applied to fewer rows
    after each halving (the external review measured a constant batch before 2026-10-12)."""
    import nca.rollout as R

    task, boards = _maze_boards(tmp_path, n=2)
    model = build_model("maze_ood")
    seen = []
    orig = model.step

    def spy(x, *a, **k):
        seen.append(x.shape[0])
        return orig(x, *a, **k)

    monkeypatch.setattr(model, "step", spy)
    rc = RolloutConfig(depth=8, k=8, pruning={"m": 4, "n_cap": 3}, record_every=2)
    res = R.rollout(model, task, boards, rc, seed=0)
    # 2 boards x 8 trials = 16 rows for steps 1-2, 8 for 3-4, 4 for 5-6, 2 for 7-8
    assert seen == [16, 16, 8, 8, 4, 4, 2, 2]
    assert res.alive.sum(axis=1).tolist() == [1, 1]
    assert res.pred.shape == (2, 8, 9, 9) and np.isfinite(res.conf).all()
    # without pruning nothing changes: the batch stays full
    seen.clear()
    R.rollout(model, task, boards, RolloutConfig(depth=3, k=2), seed=0)
    assert seen == [4, 4, 4]


def test_stop_at_probe_then_resume(tmp_path):
    """`train.stop_at` stops a probe run without final.pt; resuming finishes the same run."""
    import json

    cfg = load_config(
        "configs/smoke_maze.yaml",
        [
            f"data_root={tmp_path}",
            "task.train.n=64",
            "task.val.n=8",
            "train.steps=6",
            "train.stop_at=3",
            "train.batch_size=4",
            "train.rollout=6",
            "train.chunk=4",
            "train.log_every=1",
            "train.eval_every=3",
            "train.checkpoint_every=4",
            "train.eval.n=8",
            "train.eval.depth=6",
            "task.train.download=false",
            "task.val.download=false",
        ],
    )
    out = tmp_path / "probe"
    res = train(cfg, out, seed=0, device="cpu", resume=False, quiet=True)
    assert "val_exact" in res and not (out / "final.pt").exists()
    assert torch.load(out / "checkpoint.pt", weights_only=False)["step"] == 3
    cfg["train"]["stop_at"] = None
    train(cfg, out, seed=0, device="cpu", resume=True, quiet=True)
    assert (out / "final.pt").exists()
    steps = [json.loads(line).get("step", 0) for line in open(out / "log.jsonl")]
    assert max(steps) == 6
