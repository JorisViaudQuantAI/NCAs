"""GPU-only regressions for kernels and graph-owned output lifetime."""

import copy

import pytest
import torch

from nca.engine.bench import clone_batch, iteration_seeds, make_batch
from nca.engine.engine import Engine, EngineConfig
from nca.engine.ops import fixed_attention_mix, shared_kernel_mix
from nca.engine.train import loss_weights, resolve_train_cfg
from nca.model import _pad, build_model
from nca.tasks import get_task

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("channels", [8, 24, 32, 128])
@pytest.mark.parametrize("expansion", [2, 4])
def test_triton_mlp_matches_bf16_linear_forward_and_gradients(channels, expansion):
    from nca.engine.ops import linear_cell_update
    from nca.engine.triton_mlp import fused_mlp_update
    from nca.model import MLPUpdate

    torch.manual_seed(921)
    layer = MLPUpdate(channels, channels, expansion * channels, zero_init=False).cuda()
    x = torch.randn(8, channels, 9, 9, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = linear_cell_update(x, layer, "mlp")
        out = fused_mlp_update(x, layer)
    torch.testing.assert_close(out, ref, atol=0.008, rtol=0.02)
    upstream = torch.randn_like(out) / out.numel()
    wrt = (x, *layer.parameters())
    expected = torch.autograd.grad(ref, wrt, upstream)
    actual = torch.autograd.grad(out, wrt, upstream)
    for a, b in zip(actual, expected, strict=True):
        relative = (a - b).float().norm() / b.float().norm().clamp_min(1e-12)
        assert float(relative) < 0.02


def test_triton_mlp_checkpoint_recomputation(tmp_path):
    from nca.engine.bench import DEFAULT_TOL, compare_backends, within_tolerance

    torch.manual_seed(924)
    task = get_task("sudoku_ood")
    boards = task.load(
        "train", str(tmp_path), n=4, source="generated", download=False, workers=1, verbose=False
    )
    model = build_model(
        "sudoku_ood", channels=32, heads=4, norm_groups=4, norm_fired_only=True
    ).cuda()
    with torch.no_grad():
        model.update.fc2.weight.normal_(0, 0.001)
    tr = resolve_train_cfg({"train": dict(rollout=8, chunk=8, amp=True, loss="mse")})
    seeds, c0 = iteration_seeds(tr, 0)
    result = compare_backends(
        model,
        EngineConfig(backend="fused", triton_mlp=True, checkpoint_segment=None),
        EngineConfig(backend="fused", triton_mlp=True, checkpoint_segment=4),
        tr,
        make_batch(model, task, boards, 4, 0),
        seeds,
        c0,
        loss_weights("all", 8, torch.device("cuda")),
    )
    # Segments change the FP32 loss-summation order. Use the existing segment
    # protocol (1e-6), rather than requiring a bitwise-identical scalar loss.
    assert result["state_max_abs"] == 0
    ok, why = within_tolerance(result, DEFAULT_TOL["segment"])
    assert ok, (why, result)


@pytest.mark.parametrize("fired_only", [False, True])
def test_compiled_normalized_backward_returns_the_final_state(tmp_path, fired_only):
    """The fused update saves its output under normalized_backward; the compiled backward
    must not hand back that buffer as the final state (it came back overwritten with
    gradient-sized values with triton_mix and norm_fired_only=false)."""
    from nca.engine.bench import DEFAULT_TOL, compare_backends, within_tolerance

    torch.manual_seed(925)
    task = get_task("sudoku_ood")
    boards = task.load(
        "train", str(tmp_path), n=8, source="generated", download=False, workers=1, verbose=False
    )
    # the published Sudoku-OOD size: the reuse did not show on a 32-channel model
    model = build_model("sudoku_ood", norm_fired_only=fired_only).cuda()
    tr = resolve_train_cfg({"train": dict(rollout=8, chunk=8, amp=False, loss="mse")})
    seeds, c0 = iteration_seeds(tr, 0)
    result = compare_backends(
        model,
        EngineConfig(backend="reference", checkpoint_segment=None),
        EngineConfig(
            backend="fused",
            compile=True,
            block=4,
            triton_mix=True,
            normalized_backward=True,
            checkpoint_segment=None,
        ),
        tr,
        make_batch(model, task, boards, 8, 0),
        seeds,
        c0,
        loss_weights("all", 8, torch.device("cuda")),
    )
    ok, why = within_tolerance(result, DEFAULT_TOL["fp32"])
    assert ok, (why, result)


@pytest.mark.parametrize("amp", [False, True])
def test_iteration_graph_refreshes_inputs_rng_and_parameters(tmp_path, amp):
    torch.manual_seed(731)
    task = get_task("sudoku_ood")
    boards = task.load(
        "train", str(tmp_path), n=8, source="generated", download=False, workers=1, verbose=False
    )
    original = build_model(
        "sudoku_ood",
        channels=32,
        heads=4,
        norm_groups=4,
        norm_fired_only=True,
    ).cuda()
    with torch.no_grad():
        original.update.fc2.weight.normal_(0, 0.001)
    tr = resolve_train_cfg({"train": dict(rollout=8, chunk=8, amp=amp, perturb=True, loss="mse")})
    config = EngineConfig(backend="fused", compile=True, block=2, checkpoint_segment=None)
    from dataclasses import replace

    ref = Engine(copy.deepcopy(original), config, tr, original.device)
    fast = Engine(
        copy.deepcopy(original), replace(config, iteration_graph=True), tr, original.device
    )
    opt_ref = torch.optim.AdamW(ref.model.parameters(), lr=1e-4)
    opt_fast = torch.optim.AdamW(fast.model.parameters(), lr=1e-4)
    weights = loss_weights("all", 8, original.device)
    for seed in (0, 1, 2, 3, 4):
        batch = make_batch(original, task, boards, 4, seed)
        seeds, _ = iteration_seeds(tr, seed)
        xr, lr, gr = ref.train_iteration(clone_batch(batch), seeds, 0, weights, 1.0, None)
        xf, lf, gf = fast.train_iteration(clone_batch(batch), seeds, 0, weights, 1.0, None)
        atol, rtol = (0.01, 0.02) if amp else (1e-5, 1e-4)
        for a, b in [(xf, xr), (lf, lr), (gf, gr)]:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        for pr, pf in zip(ref.model.parameters(), fast.model.parameters(), strict=True):
            if pr.grad is not None:
                torch.testing.assert_close(pf.grad, pr.grad, atol=atol, rtol=rtol)
        opt_ref.step()
        opt_fast.step()
        opt_ref.zero_grad(set_to_none=True)
        opt_fast.zero_grad(set_to_none=True)


@pytest.mark.parametrize("amp", [False, True])
def test_owned_microbatch_gradients_survive_cuda_graph_replay(tmp_path, amp):
    torch.manual_seed(72)
    task = get_task("arc", n_tasks=2, rearc_per_task=0)
    boards = task.load("train", str(tmp_path), source="synthetic", download=False)
    original = build_model(
        "arc",
        channels=32,
        c_in=12,
        c_out=12,
        c_task=4,
        out_offset=12,
        heads=2,
        norm_groups=2,
        n_tasks=2,
        time_steps=8,
        expansion=2,
    ).cuda()
    tr = resolve_train_cfg({"train": dict(rollout=8, chunk=8, amp=amp, loss="ce", perturb=False)})
    ref = Engine(
        copy.deepcopy(original),
        EngineConfig(backend="reference", microbatch=2, checkpoint_segment=None),
        tr,
        original.device,
    )
    fast = Engine(
        copy.deepcopy(original),
        EngineConfig(
            backend="fused",
            compile=True,
            block=2,
            compile_mode="reduce-overhead",
            microbatch=2,
            clone_microbatch_grads=True,
            checkpoint_segment=None,
        ),
        tr,
        original.device,
    )
    weights = loss_weights(2, 8, original.device)
    # Exercise warm-up, capture and later replays across parameter updates.
    for seed in (0, 1, 2, 3):
        batch = make_batch(original, task, boards, 4, seed)
        seeds, _ = iteration_seeds(tr, seed)
        ref.model.zero_grad(set_to_none=True)
        fast.model.zero_grad(set_to_none=True)
        xr, lr, _ = ref.train_iteration(clone_batch(batch, ref.model), seeds, 0, weights, 1e9, None)
        xf, lf, _ = fast.train_iteration(
            clone_batch(batch, fast.model), seeds, 0, weights, 1e9, None
        )
        atol, rtol = (0.02, 0.03) if amp else (1e-5, 1e-4)
        torch.testing.assert_close(xf, xr, atol=atol, rtol=rtol)
        torch.testing.assert_close(lf, lr, atol=atol, rtol=rtol)
        with torch.no_grad():
            for pr, pf in zip(ref.model.parameters(), fast.model.parameters(), strict=True):
                if pr.grad is not None:
                    torch.testing.assert_close(pf.grad, pr.grad, atol=atol, rtol=rtol)
                    pr.add_(pr.grad, alpha=-1e-4)
                    pf.add_(pf.grad, alpha=-1e-4)


@pytest.fixture(autouse=True)
def full_fp32_precision():
    """Use the same explicit precision policy as the benchmark worker."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(allow_tf32=False, benchmark=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("split_k", [1, 2, 8])
def test_wrapped_triton_values_and_gradients(dtype, channels_last, split_k):
    from nca.engine.triton_ops import wrapped_attention_mix_cuda

    with torch.random.fork_rng():
        torch.manual_seed(191)
        B, K, D, H, W = 3, 2, 4, 5, 7
        v = torch.randn(B, K * D, H, W, device="cuda", dtype=dtype)
        if channels_last:
            v = v.contiguous(memory_format=torch.channels_last)
        v = v.view(B, K, D, H, W).requires_grad_(True)
        a = torch.randn(K, 9, H, W, device="cuda", requires_grad=True)
        padded = _pad(v.reshape(B, K * D, H, W), "wrap", None)
        ref = fixed_attention_mix(padded.view(B, K, D, H + 2, W + 2), a)
        out = wrapped_attention_mix_cuda(v, a, split_k=split_k)
        torch.testing.assert_close(out, ref, atol=3e-6, rtol=1e-5)
        g = torch.randn_like(out)
        gr = torch.autograd.grad(ref, (v, a), g)
        go = torch.autograd.grad(out, (v, a), g)
        torch.testing.assert_close(
            go[0],
            gr[0],
            atol=0.0625 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        )
        torch.testing.assert_close(go[1], gr[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "compile_options",
    [
        {"compile_mode": "reduce-overhead"},
        {"compile_backward_only": True},
        {"compile": False, "compile_update": True},
    ],
)
def test_compiled_cross_inference_training_boundaries(tmp_path, compile_options):
    with torch.random.fork_rng():
        torch.manual_seed(192)
        task = get_task("maze_ood")
        boards = task.load("train", str(tmp_path), size=9, n=8, source="generated")
        model = build_model("maze_ood").cuda()
        with torch.no_grad():
            model.update.fc2.weight.normal_(0, 0.001)
            model.update.fc2.bias.normal_(0, 0.001)
        batch = make_batch(model, task, boards, 2, seed=0)
        tr = resolve_train_cfg({"train": {"rollout": 12, "chunk": 4, "amp": False}})
        ref = Engine(
            copy.deepcopy(model),
            EngineConfig(backend="reference", checkpoint_segment=None),
            tr,
            model.device,
        )
        fast = Engine(
            copy.deepcopy(model),
            EngineConfig(
                backend="fused",
                block=2,
                checkpoint_segment=None,
                **{"compile": True, **compile_options},
            ),
            tr,
            model.device,
        )
        weights = loss_weights("all", 4, model.device)
        for seed, c0 in [(0, 1), (1, 3), (2, 1)]:
            seeds, _ = iteration_seeds(tr, seed)
            ref.model.zero_grad(set_to_none=True)
            fast.model.zero_grad(set_to_none=True)
            xr, lr, _ = ref.train_iteration(
                clone_batch(batch, ref.model), seeds, c0, weights, 1e9, None
            )
            xf, lf, _ = fast.train_iteration(
                clone_batch(batch, fast.model), seeds, c0, weights, 1e9, None
            )
            torch.testing.assert_close(xf, xr, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(lf, lr, atol=1e-6, rtol=1e-5)
            for pr, pf in zip(ref.model.parameters(), fast.model.parameters(), strict=True):
                if pr.grad is not None:
                    torch.testing.assert_close(pf.grad, pr.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("amp", [False, True])
def test_shared_kernel_triton_backward_preserves_convolution_forward(amp):
    with torch.random.fork_rng():
        torch.manual_seed(193)
        x = torch.randn(3, 4, 7, 9, device="cuda", requires_grad=True)
        weight = torch.randn(2, 1, 3, 3, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            ref = torch.nn.functional.conv2d(x.reshape(12, 1, 7, 9), weight)
            ref = ref.view(3, 4, 2, 5, 7).transpose(1, 2).reshape(3, 8, 5, 7)
            out = shared_kernel_mix(x, weight, reference_forward=True, triton_backward=True)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        g = torch.randn_like(out)
        gr = torch.autograd.grad(ref, (x, weight), g)
        go = torch.autograd.grad(out, (x, weight), g)
        for actual, expected in zip(go, gr, strict=True):
            torch.testing.assert_close(
                actual, expected, atol=0.0625 if amp else 1e-5, rtol=0.02 if amp else 1e-5
            )


@pytest.mark.parametrize("padding", ["zero", "wrap"])
@pytest.mark.parametrize("amp", [False, True])
def test_attention_triton_padding_outputs_and_gradients(padding, amp):
    from nca.engine.triton_ops import wrapped_attention_mix_cuda
    from nca.model import _pad

    torch.manual_seed(412)
    v = torch.randn(
        2,
        2,
        4,
        5,
        6,
        device="cuda",
        dtype=torch.bfloat16 if amp else torch.float32,
        requires_grad=True,
    )
    a = torch.randn(2, 9, 5, 6, device="cuda", requires_grad=True)
    vp = _pad(v.reshape(2, 8, 5, 6), padding, None).reshape(2, 2, 4, 7, 8)
    ref = fixed_attention_mix(vp, a)
    out = wrapped_attention_mix_cuda(v, a, padding)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    g = torch.randn_like(out)
    ga = torch.autograd.grad(out, (v, a), g)
    gb = torch.autograd.grad(ref, (v, a), g)
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y, atol=0.125 if amp else 2e-5, rtol=0.02 if amp else 2e-5)


@pytest.mark.parametrize("arc", [False, True])
def test_inference_graphs_reuse_new_draws_and_keep_returned_outputs(arc):
    import numpy as np

    from nca.engine.inference import InferenceEngine

    torch.manual_seed(731)
    if arc:
        model = build_model(
            "arc",
            channels=32,
            c_in=12,
            c_out=12,
            out_offset=12,
            c_task=4,
            heads=2,
            norm_groups=2,
            n_tasks=1,
            time_steps=4,
            grid_size=(5, 5),
        ).cuda()
    else:
        model = build_model("maze_ood").cuda()
        with torch.no_grad():
            model.update.fc2.weight.normal_(0, 0.001)
    inputs = torch.zeros(1, 5, 5, device="cuda", dtype=torch.long)
    ids = torch.zeros(1, device="cuda", dtype=torch.long) if arc else None
    x, g, clamp = model.init_state(inputs, torch.ones_like(inputs, dtype=torch.bool), ids)
    batch = dict(x=x, given4=g, clamp=clamp)
    fast = InferenceEngine(
        model,
        EngineConfig(backend="fused", compile=True, block=4, compile_mode="reduce-overhead"),
        amp=True,
    )
    reference = InferenceEngine(model, EngineConfig(backend="reference"), amp=True)
    retained = []
    for seed in [0, 19, 87]:
        seeds = np.arange(seed, seed + 9)  # Exercises a partial final block too.
        expected, expected_clock = reference(batch, seeds)
        actual, clock = fast(batch, seeds)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0)
        if arc:
            assert torch.equal(clock, expected_clock)
        for earlier, snapshot in retained:
            torch.testing.assert_close(earlier, snapshot, atol=0, rtol=0)
        retained.append((actual, actual.clone()))
        if clock is not None:
            retained.append((clock, clock.clone()))
