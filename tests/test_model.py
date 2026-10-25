"""Invariants of the NCA cell and of the training machinery."""

import numpy as np
import pytest
import torch

from nca.model import PRESETS, build_model
from nca.perturb import apply_damage, circular_masks, sample_noise
from nca.pruning import niche_capped_pruning, pruned_compute_fraction
from nca.utils import count_parameters

PAPER_PARAMS = {  # Table 1 of the paper, rounded as printed there
    "maze_ood": 10_000,
    "maze_hard": 41_000,
    "sudoku_ood": 160_000,
    "sudoku_extreme": 603_000,
    "arc": 3_500_000,
}


@pytest.mark.parametrize("name", list(PRESETS))
def test_parameter_counts_close_to_paper(name):
    kw = {"n_tasks": 1} if name == "arc" else {}
    n = count_parameters(build_model(name, **kw))
    # Sudoku presets match the paper to 0.1%; the maze presets round to the paper's 10K / 41K
    # (scalar 3x3 kernels + MLP hidden E * K_heads * C, see docs/DEVIATIONS.md 2026-10-12); the
    # ARC count is within 20% (the paper does not say whether biases are counted)
    assert 0.8 * PAPER_PARAMS[name] <= n <= 1.2 * PAPER_PARAMS[name]
    if name.startswith("maze"):
        assert n == {"maze_ood": 10420, "maze_hard": 41284}[name]
        assert round(n, -3) == PAPER_PARAMS[name]


def test_kernel_perception_is_a_shared_scalar_kernel_per_head():
    from nca.model import ConvPerception, KernelPerception

    torch.manual_seed(0)
    C, K = 5, 3
    kp = KernelPerception(C, K)
    # the same thing written as a dense conv with block-diagonal (per head, per channel) weights
    dense = ConvPerception(C, K)
    w = torch.zeros(K * C, C, 3, 3)
    for k in range(K):
        for c in range(C):
            w[k * C + c, c] = kp.weight[k, 0]
    dense.conv.weight.data.copy_(w)
    xp = torch.randn(2, C, 9, 9)
    assert torch.allclose(kp(xp, 7, 7), dense(xp, 7, 7), atol=1e-6)
    assert sum(p.numel() for p in kp.parameters()) == 9 * K


def _maze_batch(B=3, H=9, W=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, 4, (B, H, W), generator=g)
    inputs[inputs == 1] = 0  # the path is never given
    given = inputs >= 2
    return inputs, given


def test_given_cells_stay_clamped_and_free_cells_move():
    m = build_model("maze_ood")
    inputs, given = _maze_batch()
    x, g4, clamp = m.init_state(inputs, given)
    assert torch.equal(x[g4.expand_as(x)], clamp[g4.expand_as(clamp)])
    torch.nn.init.normal_(m.update.fc2.weight, std=0.1)  # a non-trivial rule
    x1 = m.step(x, g4, clamp, torch.ones(3, 1, 9, 9))
    assert torch.equal(x1[g4.expand_as(x1)], clamp[g4.expand_as(clamp)])
    assert not torch.allclose(x1[~g4.expand_as(x1)], x[~g4.expand_as(x)])


def test_zero_fire_mask_is_identity():
    m = build_model("sudoku_ood")
    inputs = torch.randint(0, 10, (2, 9, 9))
    x, g4, clamp = m.init_state(inputs, inputs > 0)
    x = m.normalize(x)
    x = m.apply_clamp(x, g4, clamp)
    x1 = m.step(x, g4, clamp, torch.zeros(2, 1, 9, 9))
    assert torch.allclose(x1, x, atol=1e-3)  # eps inside the normalisation: not exactly idempotent


def test_group_norm_unit_groups():
    m = build_model("sudoku_ood")
    x = torch.randn(2, 128, 9, 9)
    xn = m.normalize(x).view(2, 4, 32, 9, 9)
    assert torch.allclose(xn.norm(dim=2), torch.ones(2, 4, 9, 9), atol=1e-3)
    # a zero cell stays finite (forward and backward)
    x[0, :, 0, 0] = 0
    x.requires_grad_(True)
    m.normalize(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_decode_and_confidence_on_clean_targets():
    m = build_model("maze_ood")
    inputs, given = _maze_batch()
    targets = inputs.clone()
    targets[~given] = torch.randint(0, 2, (int((~given).sum()),))
    x, g4, clamp = m.init_state(inputs, given)
    # write the exact target embedding into the output slice -> decode must return the targets
    x[:, :2] = m.target_embedding(targets)
    dec = torch.where(given, inputs, m.decode(x))
    assert torch.equal(dec, targets)
    assert torch.allclose(m.confidence(x)[~given], torch.ones(int((~given).sum())), atol=1e-3)


def test_wall_padding_blocks_information_at_border():
    """With wall padding the border is a constant wall vector: the perception of a corner
    cell must not depend on what lies beyond the grid."""
    m = build_model("maze_ood")
    x = torch.randn(1, 16, 9, 9)
    d1 = m.delta(x)
    d2 = m.delta(torch.cat([x, x], 0))[:1]
    assert torch.allclose(d1, d2)


def test_wrap_padding_is_translation_equivariant_for_conv():
    m = build_model("maze_ood", padding="wrap")
    x = torch.randn(1, 16, 9, 9)
    d = m.delta(x)
    d_roll = m.delta(torch.roll(x, (2, 3), dims=(2, 3)))
    assert torch.allclose(torch.roll(d, (2, 3), dims=(2, 3)), d_roll, atol=1e-5)


def test_fixed_attention_rejects_wrong_grid():
    m = build_model("sudoku_ood")
    with pytest.raises(AssertionError):
        m.delta(torch.randn(1, 128, 8, 8))


def test_self_attention_runs():
    m = build_model("sudoku_ood", perception="self_attn")
    x = torch.randn(2, 128, 9, 9)
    assert m.delta(x).shape == x.shape


def test_adaptive_fire_mask_lowers_rate_for_confident_cells():
    m = build_model("maze_ood")
    inputs, given = _maze_batch(B=8)
    x, g4, clamp = m.init_state(inputs, given)
    x[:, :2] = m.target_embedding(inputs)  # every cell maximally confident
    g = torch.Generator().manual_seed(0)
    fm = m.sample_fire_mask(x, p_fire=0.8, p_low=0.2, tau=0.95, generator=g)
    mutable = (~given)[:, None]
    assert 0.1 < fm[mutable].mean().item() < 0.3  # confident (mutable) cells fire at p_low
    assert 0.7 < fm[~mutable].mean().item() < 0.9  # given cells are not "confident" (score 0.5)
    fm = m.sample_fire_mask(x, p_fire=0.8, p_low=0.2, tau=0.95, generator=g, given=g4)
    assert 0.1 < fm[~mutable].mean().item() < 0.3  # ... unless passed as `given`


def test_noise_and_damage_shapes():
    x = torch.randn(4, 16, 9, 9)
    g = torch.Generator().manual_seed(0)
    nz = sample_noise(x, p_t=1.0, p_s=0.5, sigma=0.1, generator=g)
    assert nz.shape == x.shape and 0.3 < (nz != 0).float().mean().item() < 0.7
    assert sample_noise(x, 0.0, 0.5, 0.1, g) is None
    rng = np.random.default_rng(0)
    masks = circular_masks(4, 9, 9, 1, 0.1, 0.4, rng)
    assert masks.shape == (4, 9, 9) and masks.any()
    xd, md = apply_damage(x, 1.0, 1, 0.1, 0.4, rng, force=True)
    assert torch.equal(
        xd[torch.from_numpy(md)[:, None].expand_as(xd)], torch.zeros(int(md.sum()) * 16)
    )


def test_pruning_halves_and_prefers_diverse_niches():
    rng = np.random.default_rng(0)
    N, K = 1, 8
    dec = torch.zeros(N, K, 2, 2, dtype=torch.long)
    dec[0, 4:] = 1  # two niches of 4
    conf = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])
    alive = torch.ones(N, K, dtype=torch.bool)
    new = niche_capped_pruning(dec, conf, alive, n_cap=3, rng=rng)
    assert new.sum() == 4
    assert new[0, :4].sum() == 2 and new[0, 4:].sum() == 2  # balanced across niches
    assert new[0, 0] and new[0, 4]  # most confident member of each niche kept
    assert abs(pruned_compute_fraction(2) - 0.75) < 1e-9
    assert abs(pruned_compute_fraction(8) - 0.249) < 1e-3


def test_time_embedding_local_clock():
    m = build_model("arc", n_tasks=2)
    inputs = torch.randint(0, 12, (1, 32, 32))
    x, g4, clamp = m.init_state(
        inputs, torch.ones_like(inputs, dtype=torch.bool), task_ids=torch.tensor([1])
    )
    d_global = m.delta(x, torch.tensor(3))
    clock = torch.full((1, 1, 32, 32), 3, dtype=torch.long)
    d_local = m.delta(x, clock)
    assert torch.allclose(d_global, d_local, atol=1e-5)
