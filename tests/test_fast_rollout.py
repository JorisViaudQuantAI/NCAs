"""Evaluation blocks preserve RNG order and stop at every observable event."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from nca.model import build_model
from nca.rollout import RolloutConfig, rollout
from nca.tasks import get_task


def setup(tmp_path, device="cpu", task_name="maze_ood"):
    torch.manual_seed(83)
    task = get_task(task_name)
    args = {"size": 9} if task_name == "maze_ood" else {"workers": 1, "verbose": False}
    boards = task.load("train", str(tmp_path), n=4, source="generated", download=False, **args)
    model = build_model(task_name).to(device)
    with torch.no_grad():
        model.update.fc2.weight.normal_(0, 0.01)
    return task, boards, model


@pytest.mark.parametrize("block", [1, 4, 16])
@pytest.mark.parametrize("events", [False, True])
def test_uniform_blocks_preserve_stream_traces_damage_and_pruning(tmp_path, block, events):
    task, boards, model = setup(tmp_path)
    cfg = RolloutConfig(
        depth=9,
        k=3,
        record_every=4,
        record_boards=2,
        max_cells=2 * 3 * 81,
        noise=dict(r=0.5, p_t=0.5, p_s=0.4, sigma=0.1),
        damage_at=2 if events else None,
        pruning={"m": 3, "n_cap": 2} if events else None,
    )
    ref = asdict(rollout(model, task, boards, cfg, seed=41))
    actual = asdict(rollout(model, task, boards, replace(cfg, engine={"block": block}), seed=41))
    for key, expected in ref.items():
        if expected is None:
            assert actual[key] is None
        elif "conf" in key:
            np.testing.assert_allclose(actual[key], expected, atol=1e-6, rtol=1e-6)
        else:
            np.testing.assert_array_equal(actual[key], expected, err_msg=key)


def test_uniform_blocks_reject_state_dependent_firing(tmp_path):
    task, boards, model = setup(tmp_path)
    with pytest.raises(ValueError, match="uniform firing"):
        rollout(model, task, boards, RolloutConfig(depth=4, p_low=0.1, tau=0.8, engine={}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("task_name", ["maze_ood", "sudoku_ood"])
def test_compiled_uniform_blocks_survive_repeated_evaluations(tmp_path, task_name):
    task, boards, model = setup(tmp_path, "cuda", task_name)
    cfg = RolloutConfig(depth=8, k=2, record_every=4)
    for seed in (0, 1, 2):
        ref = rollout(model, task, boards, cfg, seed=seed)
        actual = rollout(
            model,
            task,
            boards,
            replace(cfg, engine={"compile": True, "block": 4, "compile_mode": "reduce-overhead"}),
            seed=seed,
        )
        np.testing.assert_array_equal(actual.pred, ref.pred)
        np.testing.assert_array_equal(actual.cell_updates, ref.cell_updates)
        np.testing.assert_allclose(actual.conf, ref.conf, atol=1e-5, rtol=1e-5)
