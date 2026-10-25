"""Selected-cell perception must preserve boundaries, filter order and gradients."""

import pytest
import torch

from nca.engine.ops import selected_sobel_torch
from nca.engine.step import StepOptions, make_step
from nca.model import SobelPerception, _pad
from nca.train import build_run_model
from nca.utils import load_config


@pytest.mark.parametrize("padding", ["wrap", "zero"])
@pytest.mark.parametrize("firing", [0.0, 0.5, 1.0])
def test_selected_sobel_matches_dense_features_and_state_gradients(padding, firing):
    torch.manual_seed(42)
    x = torch.randn(2, 5, 3, 7, dtype=torch.float64, requires_grad=True)
    rows = torch.nonzero(torch.rand(2 * 3 * 7) < firing).flatten()
    dense = SobelPerception(5).double()(_pad(x, padding, None), 3, 7)
    ref = dense.permute(0, 2, 3, 1).reshape(-1, 15)[rows]
    out = selected_sobel_torch(x, rows, padding)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(out)
    a = torch.autograd.grad(ref, x, upstream, retain_graph=True)[0]
    b = torch.autograd.grad(out, x, upstream)[0]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("padding", ["wrap", "zero"])
@pytest.mark.parametrize("checkpoint_mlp", [False, True])
def test_sparse_sobel_step_preserves_parameter_gradients(padding, checkpoint_mlp):
    torch.manual_seed(41)
    cfg = load_config("configs/visual_sudoku_compute.yaml", [])
    cfg["model"]["overrides"].update(
        channels=8, expansion=2, zero_init_update=False, padding=padding
    )
    model = build_run_model(cfg).double()
    inp = torch.rand(2, 5, 7)
    x, given, clamp = model.init_state(inp, torch.rand_like(inp) > 0.4)
    x = x.double().requires_grad_()
    clamp = clamp.double()
    fire = (torch.rand(2, 1, 5, 7) > 0.5).double()
    ref = model.step(x, given, clamp, fire)
    step = make_step(
        model, StepOptions(sparse_mlp=True, sparse_sobel=True, checkpoint_mlp=checkpoint_mlp)
    )
    out = step(x, given, clamp, fire, None, None, None)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    params = (x, *model.parameters())
    upstream = torch.randn_like(x)
    a = torch.autograd.grad(ref, params, upstream, retain_graph=True)
    b = torch.autograd.grad(out, params, upstream)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("padding", ["wrap", "zero"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("firing", [0.0, 0.5, 1.0])
def test_cuda_selected_sobel_matches_cudnn(padding, amp, firing):
    from nca.engine.triton_sobel import selected_sobel_cuda

    torch.manual_seed(42)
    x = torch.randn(2, 5, 7, 11, device="cuda", requires_grad=True)
    rows = torch.nonzero(torch.rand(2 * 7 * 11, device="cuda") < firing).flatten()
    perceive = SobelPerception(5).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        dense = perceive(_pad(x, padding, None), 7, 11)
        out = selected_sobel_cuda(x, rows, padding)
    ref = dense.permute(0, 2, 3, 1).reshape(-1, 15)[rows]
    atol, rtol = (0.016, 0.008) if amp else (2e-6, 2e-6)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    upstream = torch.randn_like(out)
    a = torch.autograd.grad(ref, x, upstream, retain_graph=True)[0]
    b = torch.autograd.grad(out, x, upstream)[0]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_changing_fired_cell_count_does_not_compile_new_sobel_kernels():
    from nca.engine.bench import _custom_kernel_variants
    from nca.engine.triton_sobel import selected_sobel_cuda

    x = torch.randn(1, 8, 9, 13, device="cuda")
    selected_sobel_cuda(x, torch.arange(3, device="cuda"), "wrap")
    before = _custom_kernel_variants()
    for n in (4, 5, 16, 32, 33, 64, 65, 117):
        rows = torch.arange(n, device="cuda")
        out = selected_sobel_cuda(x, rows, "wrap")
        torch.testing.assert_close(out, selected_sobel_torch(x, rows, "wrap"))
    assert _custom_kernel_variants() == before
