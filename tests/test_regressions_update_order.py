"""Regression tests from code review (2026-10-13):
the weights are updated after the last step of the rollout, and the recovery after damage is
measured after the damage event."""

import numpy as np
import torch

from nca.experiments import adaptive_compute, adaptive_ratios, recovery_point
from nca.model import NCA, build_model
from nca.tasks import get_task
from nca.train import train
from nca.utils import load_config


def _maze(tmp_path, n=3):
    task = get_task("maze_ood")
    return task, task.load("test", str(tmp_path), size=9, n=n, download=False, source="generated")


# 1. one set of weights per rollout -----------------------------------------------------------


def test_the_whole_rollout_uses_the_weights_of_the_start_of_the_iteration(tmp_path, monkeypatch):
    seen = []  # (training mode, a fingerprint of the weights) at every NCA.step call
    original = NCA.step

    def spy(self, *args, **kwargs):
        if self.training:
            seen.append(float(sum(p.detach().abs().sum() for p in self.parameters())))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NCA, "step", spy)
    cfg = load_config(
        "configs/smoke_maze.yaml",
        [
            f"data_root={tmp_path}",
            "task.train.n=16",
            "task.val.n=4",
            "train.steps=3",
            "train.batch_size=4",
            "train.rollout=6",
            "train.chunk=2",  # a chunk in the middle leaves steps after the backward
            "train.log_every=100",
            "train.eval_every=100",
            "train.checkpoint_every=100",
            "train.eval.n=4",
            "train.eval.depth=6",
            "task.train.download=false",
            "task.val.download=false",
        ],
    )
    train(cfg, tmp_path / "run", seed=0, device="cpu", resume=False, quiet=True)
    assert len(seen) == 3 * 6
    per_iteration = [seen[i * 6 : (i + 1) * 6] for i in range(3)]
    for weights in per_iteration:
        assert len(set(weights)) == 1  # no step of an iteration sees updated weights
    assert per_iteration[0][0] != per_iteration[1][0] != per_iteration[2][0]  # but they do move


# 2. recovery measured after the damage event -------------------------------------------------


def test_a_board_solved_before_the_damage_and_never_repaired_is_not_a_recovery():
    steps = np.arange(1, 7)
    upd = np.arange(1, 7) * 10.0
    # solved from the start, destroyed by the damage before step 4 runs, never repaired
    r = recovery_point(steps, [1, 1, 1, 1, 0, 0], upd, damage_at=4)
    assert r["steps_to_95"] is None and r["steps_since_damage_to_95"] is None
    # repaired at step 6: total time 6 (and 60 updates), 2 steps / 20 updates since the damage
    r = recovery_point(steps, [1, 1, 1, 1, 0, 1], upd, damage_at=4)
    assert (r["steps_to_95"], r["updates_to_95"]) == (6, 60.0)
    assert (r["steps_since_damage_to_95"], r["updates_since_damage_to_95"]) == (2, 20.0)
    # without damage the first crossing counts, and no since-damage keys are written
    r = recovery_point(steps, [0, 1, 1, 1, 0, 1], upd)
    assert r == {"steps_to_95": 2, "updates_to_95": 20.0}


def test_ratios_cover_the_time_since_the_damage():
    v = dict(
        uniform=dict(
            steps_to_95=10,
            updates_to_95=100.0,
            steps_since_damage_to_95=4,
            updates_since_damage_to_95=40.0,
            steps=[10],
            final_updates=100.0,
        ),
        adaptive=dict(
            steps_to_95=12,
            updates_to_95=60.0,
            steps_since_damage_to_95=6,
            updates_since_damage_to_95=20.0,
            steps=[12],
            final_updates=60.0,
        ),
    )
    r = adaptive_ratios(v)["adaptive"]
    assert (r["steps_to_95"], r["updates_to_95"]) == (1.2, 0.6)
    assert (r["steps_since_damage_to_95"], r["updates_since_damage_to_95"]) == (1.5, 0.5)


def test_a_curve_that_stops_before_the_damage_is_not_reused(tmp_path):
    task, boards = _maze(tmp_path)
    m = build_model("maze_ood").eval()
    torch.manual_seed(0)
    short = adaptive_compute(m, task, boards, 2, 0.8, 0.4, 0.9, 1, variants=("uniform",))
    short["variants"]["uniform"]["steps_to_95"] = 1  # as an old file could claim
    short["damage_at"] = 4
    # the same experiment with the damage at step 4 and depth 8 cannot reuse a 2-step curve
    res = adaptive_compute(
        m, task, boards, 8, 0.8, 0.4, 0.9, 1, damage_at=4, variants=("uniform",), previous=short
    )
    v = res["variants"]["uniform"]
    assert v["depth"] == 8 and v["steps"] == list(range(1, 9))
    assert "steps_since_damage_to_95" in v
    # a crossing recorded before the event is recomputed too
    stale = dict(res)
    stale["variants"] = {"uniform": dict(v, steps_to_95=3)}
    again = adaptive_compute(
        m, task, boards, 8, 0.8, 0.4, 0.9, 1, damage_at=4, variants=("uniform",), previous=stale
    )
    assert again["variants"]["uniform"]["steps_to_95"] != 3
