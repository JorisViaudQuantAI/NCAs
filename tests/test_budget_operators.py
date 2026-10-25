"""Execution changes retain the model, stochastic stream and independent replay storage."""

import copy

import numpy as np
import pytest
import torch

from nca.engine.bench import make_pool
from nca.engine.ops import linear_projection
from nca.engine.step import StepOptions, make_step
from nca.model import build_model
from nca.tasks import get_task


@pytest.mark.parametrize("padding", ["wrap", "zero", "wall"])
def test_linear_value_projection_preserves_forward_and_gradients(padding):
    torch.manual_seed(93)
    model = build_model(
        "sudoku_ood",
        channels=32,
        heads=4,
        norm_groups=4,
        padding=padding,
        norm_fired_only=True,
    ).double()
    with torch.no_grad():
        model.update.fc2.weight.normal_(0, 0.02)
    inputs = torch.zeros(2, 9, 9, dtype=torch.long)
    given = torch.rand(2, 9, 9) < 0.3
    x, given4, clamp = model.init_state(inputs, given)
    x = x.detach().requires_grad_(True)
    fire = (torch.rand(2, 1, 9, 9) < 0.7).double()
    ref = model.step(x, given4, clamp, fire)
    fast = make_step(model, StepOptions(linear_value=True, linear_mlp=True))
    out = fast(x, given4, clamp, fire, None, None, None)
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    upstream = torch.randn_like(out)
    wrt = (x, *model.parameters())
    expected = torch.autograd.grad(ref, wrt, upstream, allow_unused=True)
    actual = torch.autograd.grad(out, wrt, upstream, allow_unused=True)
    for a, b in zip(actual, expected, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_linear_projection_keeps_autocast_dtype():
    layer = torch.nn.Conv2d(8, 12, 1)
    x = torch.randn(2, 8, 3, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, expected = linear_projection(x, layer), layer(x)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.02)


def test_replay_gather_owns_storage_and_preserves_rng(tmp_path):
    model = build_model("maze_ood")
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), n=16, size=9, source="generated", download=False)
    pool = make_pool(model, task, boards, 4, 17)
    rng = copy.deepcopy(pool.rng)
    expected_idx = rng.choice(pool.size, size=4, replace=False)
    expected_state = pool.x[expected_idx].clone()
    idx, batch = pool.sample(4, 0)
    np.testing.assert_array_equal(idx, expected_idx)
    torch.testing.assert_close(batch["x"], expected_state, rtol=0, atol=0)
    assert pool.rng.bit_generator.state == rng.bit_generator.state
    batch["x"].add_(1)
    torch.testing.assert_close(pool.x[idx], expected_state, rtol=0, atol=0)
    pool.write_back(idx, batch)
    torch.testing.assert_close(pool.x[idx], expected_state + 1, rtol=0, atol=0)
    assert bool((pool.age[idx] == 1).all())
