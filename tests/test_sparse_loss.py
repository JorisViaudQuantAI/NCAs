"""Skipping zero-weight readouts must retain the full rollout's gradients."""

import numpy as np
import pytest
import torch

from nca.engine.bench import clone_batch, compare_backends, make_batch
from nca.engine.engine import Engine, EngineConfig
from nca.engine.train import loss_weights, resolve_train_cfg
from nca.model import build_model
from nca.tasks import get_task


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("window", ["last", 3, "all"])
@pytest.mark.parametrize("microbatch", [None, 1])
def test_sparse_loss_keeps_state_loss_and_parameter_gradients(
    tmp_path, recursive, window, microbatch
):
    torch.manual_seed(37)
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), source="generated", size=5, n=4)
    model = build_model("maze_ood")
    with torch.no_grad():
        model.update.fc2.weight.normal_(0, 0.001)
    tr = resolve_train_cfg({"train": dict(rollout=12, chunk=8, amp=False, loss_window=window)})
    batch = make_batch(model, task, boards, 2, 0)
    common = dict(backend="reference", checkpoint_segment=2, checkpoint_recursive=recursive)
    a = EngineConfig(**common)
    b = EngineConfig(**common, skip_zero_loss=True, microbatch=microbatch)
    result = compare_backends(
        model, a, b, tr, batch, np.arange(12), 2, loss_weights(window, 8, torch.device("cpu"))
    )
    assert result["state_max_abs"] < 1e-6
    assert result["loss_abs"] < 1e-6
    assert result["grad_max_abs"] < 1e-6


def test_sparse_loss_tracks_changed_weights_and_keeps_differentiable_zero(tmp_path):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), source="generated", size=5, n=4)
    model = build_model("maze_ood")
    tr = resolve_train_cfg({"train": dict(rollout=4, chunk=4, amp=False)})
    batch = make_batch(model, task, boards, 2, 0)
    a = Engine(model, EngineConfig(backend="reference", checkpoint_segment=None), tr, model.device)
    b = Engine(
        model,
        EngineConfig(backend="reference", checkpoint_segment=2, skip_zero_loss=True),
        tr,
        model.device,
    )
    weights = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for replacement in [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]:
        weights.copy_(torch.tensor(replacement))
        model.zero_grad(set_to_none=True)
        _, la, _ = a.train_iteration(clone_batch(batch, model), np.arange(4), 0, weights, 1.0, None)
        grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        _, lb, _ = b.train_iteration(clone_batch(batch, model), np.arange(4), 0, weights, 1.0, None)
        torch.testing.assert_close(la, lb)
        for p, g in zip(model.parameters(), grads, strict=True):
            if g is not None:
                torch.testing.assert_close(p.grad, g)
