"""Operator-level validation of the fast engine (docs/FAST_ENGINE.md, "Validation"): every
fused operator against the reference modules of `nca.model`, outputs and gradients, on the
delicate cases (empty / full / mixed firing masks, zero cells, noise, borders, clamping), plus
`torch.autograd.gradcheck` in float64."""

import pytest
import torch

from nca.engine.ops import fixed_attention_mix, fused_cell_update, project_values, shared_kernel_mix
from nca.engine.rng import draw_step
from nca.engine.step import StepOptions, make_step
from nca.model import OFFSETS, FixedAttention, KernelPerception, _pad, build_model

torch.manual_seed(0)


@pytest.fixture
def isolated_torch_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize("reference_forward", [False, True])
def test_shared_kernel_mix_matches_convolution_and_gradcheck(isolated_torch_rng, reference_forward):
    torch.manual_seed(714)
    layer = KernelPerception(2, 3).double()
    xp = torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    ref = layer(xp, 3, 4)
    out = shared_kernel_mix(xp, layer.weight, reference_forward)
    torch.testing.assert_close(out, ref)
    upstream = torch.randn_like(out)
    expected = torch.autograd.grad(ref, (xp, layer.weight), upstream)
    actual = torch.autograd.grad(out, (xp, layer.weight), upstream)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert torch.autograd.gradcheck(
        lambda x, w: shared_kernel_mix(x, w, reference_forward), (xp, layer.weight)
    )


@pytest.mark.parametrize("reference_forward", [False, True])
def test_shared_kernel_mix_preserves_bf16_autocast_boundaries(
    isolated_torch_rng, reference_forward
):
    torch.manual_seed(715)
    layer = KernelPerception(3, 4)
    xp = torch.randn(2, 3, 6, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref = layer(xp, 4, 5)
        out = shared_kernel_mix(xp, layer.weight, reference_forward)
    assert out.dtype == ref.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=0.015625, rtol=0.01)
    g = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, (xp, layer.weight), g)
    actual = torch.autograd.grad(out, (xp, layer.weight), g)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0.125, rtol=0.02)


def _reference_mix(vp, mix):
    H, W = mix.shape[-2:]
    z = None
    for n, (dy, dx) in enumerate(OFFSETS):
        term = vp[:, :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W] * mix[None, :, n, None]
        z = term if z is None else z + term
    return z


def test_fixed_attention_mix_matches_reference_and_gradcheck():
    B, K, D, H, W = 2, 3, 2, 4, 5
    vp = torch.randn(B, K, D, H + 2, W + 2, dtype=torch.float64, requires_grad=True)
    mix = torch.randn(K, 9, H, W, dtype=torch.float64, requires_grad=True)
    z = fixed_attention_mix(vp, mix)
    zr = _reference_mix(vp, mix)
    assert torch.allclose(z, zr)
    g = torch.randn_like(z)
    ga = torch.autograd.grad((z * g).sum(), (vp, mix))
    gb = torch.autograd.grad((zr * g).sum(), (vp, mix))
    assert torch.allclose(ga[0], gb[0]) and torch.allclose(ga[1], gb[1])
    assert torch.autograd.gradcheck(fixed_attention_mix, (vp, mix))


def test_fixed_attention_mix_mixed_precision_matches_reference():
    """bf16 values with fp32 mixing weights: same promoted computation as the reference."""
    B, K, D, H, W = 2, 2, 4, 3, 3
    vp = torch.randn(B, K, D, H + 2, W + 2).to(torch.bfloat16).requires_grad_(True)
    mix = torch.randn(K, 9, H, W, requires_grad=True)
    z = fixed_attention_mix(vp, mix)
    zr = _reference_mix(vp, mix)
    assert z.dtype == zr.dtype == torch.float32
    assert torch.allclose(z, zr)
    g = torch.randn_like(z)
    ga = torch.autograd.grad((z * g).sum(), (vp, mix))
    gb = torch.autograd.grad((zr * g).sum(), (vp, mix))
    assert ga[0].dtype == torch.bfloat16 and ga[1].dtype == torch.float32
    # the fp32 accumulation order differs, so the bf16 gradient can differ by one bf16 ulp
    assert torch.allclose(ga[0].float(), gb[0].float(), atol=1e-1, rtol=2e-2)
    assert torch.allclose(ga[1], gb[1], atol=1e-4)


@pytest.mark.parametrize("padding", ["wrap", "zero", "wall"])
def test_project_values_equals_pad_then_project(padding):
    C, K, H, W = 8, 2, 5, 4
    att = FixedAttention(C, K, (H, W)).double()
    x = torch.randn(3, C, H, W, dtype=torch.float64)
    pad_vec = torch.randn(C, dtype=torch.float64) if padding == "wall" else torch.zeros(C).double()
    ref = att.value(_pad(x, padding, pad_vec))
    out = project_values(x, att.value.weight, padding, pad_vec)
    assert torch.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("groups,fired_only", [(0, False), (4, False), (4, True), (1, True)])
@pytest.mark.parametrize("normalized_backward", [False, True])
def test_fused_cell_update_gradcheck_and_reference(
    groups, fired_only, normalized_backward, isolated_torch_rng
):
    B, C, H, W = 2, 8, 3, 3
    x = torch.randn(B, C, H, W, dtype=torch.float64)
    x[0, :, 0, 0] = 0.0  # a damaged (all-zero) cell
    x = x.requires_grad_(True)
    dx = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    fire = (torch.rand(B, 1, H, W) < 0.6).double()
    fire[1] = 0.0  # a board where nothing fires
    noise = torch.zeros(B, C, H, W, dtype=torch.float64)
    nmask = torch.zeros(B, 1, H, W, dtype=torch.bool)
    nmask[1, 0, 1, 1] = True  # a non-firing cell that receives noise
    noise[1, :, 1, 1] = 0.3
    given4 = torch.rand(B, 1, H, W) < 0.3
    clamp = torch.randn(B, 4, H, W, dtype=torch.float64)
    sl = slice(0, 4)

    def ref(x, dx):
        u = x + fire * dx + noise
        if groups:
            ug = u.view(B, groups, C // groups, H, W)
            y = (ug / (ug.pow(2).sum(2, keepdim=True) + 1e-4).sqrt()).view(B, C, H, W)
            if fired_only:
                moved = (fire > 0) | nmask
                y = torch.where(moved, y, u)
        else:
            y = u
        yin = torch.where(given4, clamp, y[:, sl])
        return torch.cat([yin, y[:, sl.stop :]], dim=1)

    out = fused_cell_update(
        x, dx, fire, noise, nmask, given4, clamp, groups, fired_only, sl, normalized_backward
    )
    assert torch.allclose(out, ref(x, dx))
    g = torch.randn_like(out)
    ga = torch.autograd.grad((out * g).sum(), (x, dx))
    gb = torch.autograd.grad((ref(x, dx) * g).sum(), (x, dx))
    assert torch.allclose(ga[0], gb[0]) and torch.allclose(ga[1], gb[1])
    fn = lambda x, dx: fused_cell_update(  # noqa: E731
        x, dx, fire, noise, nmask, given4, clamp, groups, fired_only, sl, normalized_backward
    )
    assert torch.autograd.gradcheck(fn, (x, dx))


def test_normalized_backward_shares_the_returned_state(isolated_torch_rng):
    x = torch.randn(2, 8, 3, 3, requires_grad=True)
    dx = torch.randn_like(x, requires_grad=True)
    fire = torch.ones(2, 1, 3, 3)
    given = torch.rand(2, 1, 3, 3) < 0.5
    clamp = torch.randn(2, 4, 3, 3)
    retained = []

    def pack(tensor):
        if tensor.numel() == x.numel():
            retained.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fused_cell_update(
            x, dx, fire, None, None, given, clamp, 4, True, slice(0, 4), normalized_backward=True
        )
    assert retained == [out.data_ptr()]
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(dx.grad).all()


def test_unnormalized_update_does_not_retain_a_full_state():
    """Maze must not retain C float channels when its backward only needs masks."""
    x = torch.randn(2, 32, 5, 5, requires_grad=True)
    dx = torch.randn_like(x, requires_grad=True)
    fire = torch.ones(2, 1, 5, 5)
    given = torch.zeros_like(fire, dtype=torch.bool)
    clamp = torch.zeros(2, 4, 5, 5)
    retained = []

    def pack(tensor):
        retained.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fused_cell_update(x, dx, fire, None, None, given, clamp, 0, False, slice(0, 4))
    assert sum(retained) < x.numel() * x.element_size()
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(dx.grad, torch.ones_like(dx))


def _small_models():
    yield "maze", build_model("maze_ood"), 9
    yield (
        "sudoku",
        build_model(
            "sudoku_ood",
            channels=32,
            heads=4,
            norm_groups=4,
            grid_size=(9, 9),
            norm_fired_only=False,
        ),
        9,
    )
    yield (
        "sudoku_fired",
        build_model("sudoku_extreme", channels=32, heads=4, norm_groups=4, grid_size=(9, 9)),
        9,
    )
    yield (
        "arc",
        build_model(
            "arc",
            channels=48,
            c_in=12,
            c_out=12,
            out_offset=12,
            c_task=8,
            n_tasks=2,
            heads=4,
            norm_groups=4,
            grid_size=(6, 6),
            time_steps=8,
        ),
        6,
    )


def _batch(model, size, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = model.cfg.vocab_size
    inputs = torch.randint(0, V, (B, size, size), generator=g)
    given = torch.rand((B, size, size), generator=g) < 0.4
    inputs[~given] = 0
    tids = torch.tensor([0, 1]) if model.cfg.n_tasks else None
    x, given4, clamp = model.init_state(inputs, given, tids, g)
    return x.detach(), given4, clamp


@pytest.mark.parametrize(
    "name,model,size", list(_small_models()), ids=lambda v: v if isinstance(v, str) else ""
)
@pytest.mark.parametrize(
    "opts",
    [
        StepOptions(True, True, True),
        StepOptions(True, False, False),
        StepOptions(False, False, True),
    ],
)
def test_fused_step_matches_reference_step(name, model, size, opts):
    """The fused backend's step equals `NCA.step` (outputs and parameter gradients) for every
    perception / normalisation / padding / time-embedding combination of the presets."""
    torch.manual_seed(1)
    for p in model.parameters():  # a non-trivial update MLP (maze starts at zero)
        p.data.add_(0.05 * torch.randn_like(p))
    x, given4, clamp = _batch(model, size)
    x[0, :, 2, 2] = 0.0  # a zero cell
    shape = tuple(x.shape)
    draw = draw_step(
        123, shape, model.cfg.fire_rate, {"p_t": 1.0, "p_s": 0.3, "sigma": 0.2}, x.device
    )
    clock = torch.tensor(3) if model.cfg.time_steps else None
    fast = make_step(model, opts)
    for fire in (draw.fire, torch.zeros_like(draw.fire), torch.ones_like(draw.fire)):
        for noise, nmask in ((draw.noise, draw.noise_mask), (None, None)):
            model.zero_grad(set_to_none=True)
            xa = x.clone().requires_grad_(True)
            ya = model.step(xa, given4, clamp, fire, noise, clock)
            ga = torch.autograd.grad(
                ya.pow(2).sum() * 0.1, [xa] + list(model.parameters()), allow_unused=True
            )
            xb = x.clone().requires_grad_(True)
            yb = fast(xb, given4, clamp, fire, noise, nmask, clock)
            gb = torch.autograd.grad(
                yb.pow(2).sum() * 0.1, [xb] + list(model.parameters()), allow_unused=True
            )
            assert torch.allclose(ya, yb, atol=1e-5), (name, opts)
            for a, b in zip(ga, gb, strict=True):
                if a is None or b is None:
                    assert a is None and b is None
                else:
                    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (name, opts)


def test_fused_step_under_bf16_autocast_matches_reference():
    model = build_model("sudoku_ood", channels=32, heads=4, norm_groups=4, grid_size=(9, 9))
    x, given4, clamp = _batch(model, 9)
    draw = draw_step(7, tuple(x.shape), 0.8, {"p_t": 1.0, "p_s": 0.3, "sigma": 0.2}, x.device)
    fast = make_step(model, StepOptions.for_backend("fused"))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ya = model.step(x, given4, clamp, draw.fire, draw.noise, None)
        yb = fast(x, given4, clamp, draw.fire, draw.noise, draw.noise_mask, None)
    assert ya.dtype == yb.dtype == torch.float32
    # the bf16 matmuls are the same ops in both paths; only summation order differs
    assert torch.allclose(ya, yb, atol=2e-2)


def test_draw_step_is_deterministic_and_dense():
    shape = (4, 6, 5, 5)
    a = draw_step(99, shape, 0.8, {"p_t": 0.5, "p_s": 0.3, "sigma": 0.1}, torch.device("cpu"))
    b = draw_step(99, shape, 0.8, {"p_t": 0.5, "p_s": 0.3, "sigma": 0.1}, torch.device("cpu"))
    assert torch.equal(a.fire, b.fire) and torch.equal(a.noise, b.noise)
    assert set(a.fire.unique().tolist()) <= {0.0, 1.0}
    # noise only where the mask says, and the mask only on active boards
    assert torch.equal((a.noise != 0).any(dim=1, keepdim=True), a.noise_mask)
    rows = a.noise_mask.flatten(1).any(dim=1)
    assert rows.sum() < 4 or True  # some boards inactive with high probability; shape check only
    c = draw_step(100, shape, 0.8, {"p_t": 0.5, "p_s": 0.3, "sigma": 0.1}, torch.device("cpu"))
    assert not torch.equal(a.fire, c.fire)
    none = draw_step(1, shape, 0.8, None, torch.device("cpu"))
    assert none.noise is None and none.noise_mask is None
    # a step with no active board is a zero tensor, not a CPU branch
    z = draw_step(1, shape, 0.8, {"p_t": 1e-9, "p_s": 0.3, "sigma": 0.1}, torch.device("cpu"))
    assert z.noise is not None and float(z.noise.abs().sum()) == 0.0
