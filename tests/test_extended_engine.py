"""Regression coverage for pixel states, recursive gradients, ARC clocks and TTT IDs."""

import numpy as np
import pytest
import torch

from nca.engine.bench import compare_backends, make_batch
from nca.engine.engine import Engine, EngineConfig
from nca.engine.inference import InferenceEngine
from nca.engine.rng import draw_step
from nca.engine.train import loss_weights, resolve_train_cfg
from nca.model import build_model
from nca.tasks import get_task
from nca.tasks.visual_sudoku import PixelBoards
from nca.train import build_run_model
from nca.ttt import augmentation_boards
from nca.utils import load_config


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_microbatch_rejects_cuda_graph_gradient_accumulation(mode):
    with pytest.raises(ValueError, match="use compile_mode=null"):
        EngineConfig(compile=True, compile_mode=mode, microbatch=4)


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_graph_checkpoint_combination_fails_before_compilation(mode, monkeypatch):
    def unexpected_compile(*args, **kwargs):
        raise AssertionError("unsupported configuration must fail before compilation")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    config = EngineConfig(
        backend="reference", compile=True, compile_mode=mode, checkpoint_segment=2
    )
    # Device metadata is enough: the constructor must not allocate CUDA storage.
    with pytest.raises(ValueError, match="checkpoint recomputation"):
        Engine(build_model("maze_ood"), config, {"rollout": 8, "chunk": 8}, torch.device("cuda"))


@pytest.mark.parametrize("microbatch", [None, 1])
def test_fire_cache_preserves_checkpointed_rollout_gradients_and_clears_afterward(
    tmp_path, microbatch
):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), source="generated", size=5, n=4)
    model = build_model("maze_ood")
    with torch.no_grad():
        model.update.fc2.weight.normal_(0, 0.001)
    tr = resolve_train_cfg({"train": dict(rollout=8, chunk=8, amp=False, perturb=False)})
    batch = make_batch(model, task, boards, 2, 0)
    a = EngineConfig(backend="reference", checkpoint_segment=2)
    b = EngineConfig(
        backend="reference", checkpoint_segment=2, microbatch=microbatch, cache_fire=True
    )
    weights = loss_weights("all", 8, torch.device("cpu"))
    result = compare_backends(model, a, b, tr, batch, np.arange(8), 0, weights)
    assert result["state_max_abs"] < 1e-6
    assert result["loss_abs"] < 1e-6
    assert result["grad_max_abs"] < 1e-6
    engine = Engine(model, b, tr, torch.device("cpu"))
    engine.train_iteration(batch, np.arange(8), 0, weights, 1.0, None)
    assert engine._fire_cache is None


def test_fire_cache_released_on_error_and_noise_draws_are_not_cached(tmp_path, monkeypatch):
    import nca.engine.engine as engine_module

    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), source="generated", size=5, n=4)
    model = build_model("maze_ood")
    tr = resolve_train_cfg({"train": dict(rollout=4, chunk=4, perturb=False, amp=False)})
    engine = Engine(
        model, EngineConfig(cache_fire=True, checkpoint_segment=2), tr, torch.device("cpu")
    )
    batch = make_batch(model, task, boards, 2, 0)
    original = engine_module.draw_step
    called = []

    def draw(*args, **kwargs):
        called.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "draw_step", draw)
    engine.train_iteration(batch, np.arange(4), 0, torch.ones(4), 1.0, None)
    assert len(called) == 4  # backward recomputation reuses only the masks
    engine.tr["perturb"] = True
    called.clear()
    engine.train_iteration(batch, np.arange(4), 0, torch.ones(4), 1.0, None)
    assert len(called) > 4
    engine.tr["perturb"] = False

    def fail(*args, **kwargs):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(engine_module, "draw_step", fail)
    with pytest.raises(RuntimeError, match="probe failure"):
        engine.train_iteration(batch, np.arange(4), 0, torch.ones(4), 1.0, None)
    assert engine._fire_cache is None


@pytest.mark.parametrize("microbatch,mode", [(4, None), (None, "reduce-overhead")])
def test_validated_compilation_choices_remain_available(microbatch, mode):
    config = EngineConfig(compile=True, compile_mode=mode, microbatch=microbatch)
    assert config.microbatch == microbatch
    assert config.compile_mode == mode


def test_pixel_boards_preserve_fractional_values_across_pool_operations():
    x = np.array([[[0.25, 0.75], [0.125, 1.0]]], np.float32)
    b = PixelBoards(x, x + 0.1, np.ones_like(x, bool))
    for out in [b.subset([0]), b.repeat(2), b.concat(b)]:
        assert out.inputs.dtype == np.float32
        assert out.inputs[0, 0, 0] == 0.25
        assert out.targets[0, 0, 0] == pytest.approx(0.35)


def test_pixel_model_clamps_input_and_has_differentiable_distinct_output():
    cfg = load_config("configs/visual_sudoku_compute.yaml", [])
    cfg["model"]["overrides"].update(channels=8, expansion=2)
    model = build_run_model(cfg)
    inp = torch.rand(2, 5, 5)
    x, g, c = model.init_state(inp, torch.ones_like(inp, dtype=torch.bool))
    assert torch.equal(x[:, 0], inp)
    assert not torch.equal(x[:, 1], inp)
    assert torch.equal(model.target_embedding(inp)[:, 0], inp)
    x = model.step(x, g, c, torch.ones_like(g, dtype=torch.float32))
    x[:, 1].sum().backward()
    assert model.update.fc2.weight.grad.abs().sum() > 0
    assert torch.equal(x[:, 0], inp)


def test_recursive_checkpoint_preserves_loss_state_and_parameter_gradients(tmp_path):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), source="generated", size=5, n=4)
    model = build_model("maze_ood")
    tr = resolve_train_cfg({"train": dict(rollout=9, chunk=9, amp=False)})
    batch = make_batch(model, task, boards, 2, 0)
    a = EngineConfig(backend="reference", checkpoint_segment=None)
    b = EngineConfig(backend="reference", checkpoint_segment=2, checkpoint_recursive=True)
    result = compare_backends(
        model, a, b, tr, batch, np.arange(9), 0, loss_weights("all", 9, torch.device("cpu"))
    )
    assert result["state_max_abs"] == 0
    assert result["loss_abs"] < 1e-6
    assert result["grad_max_abs"] < 1e-5


def test_inference_arc_local_clocks_equal_explicit_firing_loop():
    torch.manual_seed(52)
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
        grid_size=(4, 4),
    )
    inputs = torch.zeros(1, 4, 4, dtype=torch.long)
    x, g, c = model.init_state(
        inputs, torch.ones_like(inputs, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
    )
    batch = dict(x=x, given4=g, clamp=c)
    clock = torch.zeros_like(g, dtype=torch.long)
    ref = x.detach().clone()
    with torch.no_grad():
        for seed in range(7):
            d = draw_step(seed, tuple(x.shape), model.cfg.fire_rate, None, x.device)
            ref = model.step(ref, g, c, d.fire, None, clock)
            clock = clock + d.fire.long()
    out, actual = InferenceEngine(model, EngineConfig(backend="reference", block=3))(
        batch, np.arange(7)
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert torch.equal(actual, clock)


def test_ttt_gives_each_offline_transform_its_own_embedding_id():
    x = np.arange(6).reshape(2, 3)
    b = augmentation_boards([(x, x + 1), (x, x + 2)], seed=6)
    assert len(b) == 102
    assert len(np.unique(b.meta["task_id"])) == 51
    assert np.array_equal(b.meta["pairs"][0][0], x)
    assert np.array_equal(b.meta["color_perm"][0], np.arange(10))
    assert len(set(b.meta["dihedral_k"])) == 6
    with pytest.raises(ValueError):
        augmentation_boards([(x, None)])


def test_inference_noise_ends_inside_a_compiled_block():
    torch.manual_seed(91)
    model = build_model("maze_ood", channels=16, c_in=8, c_out=8, expansion=2)
    inp = torch.zeros(1, 5, 5, dtype=torch.long)
    x, given, clamp = model.init_state(inp, torch.zeros_like(inp, dtype=torch.bool))
    noise = dict(r=0.25, p_t=1.0, p_s=1.0, sigma=0.15)
    ref = x.detach().clone()
    with torch.no_grad():
        for t in range(8):
            d = draw_step(
                t, tuple(x.shape), model.cfg.fire_rate, noise if t < 2 else None, x.device
            )
            ref = model.step(ref, given, clamp, d.fire, d.noise)
    out, _ = InferenceEngine(model, EngineConfig(backend="reference", block=3), noise=noise)(
        dict(x=x, given4=given, clamp=clamp), np.arange(8)
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_ttt_initialization_preserves_shared_weights_and_creates_new_embeddings():
    from nca.ttt import prepare_ttt_model

    model = build_model(
        "arc",
        channels=32,
        c_in=12,
        c_out=12,
        out_offset=12,
        c_task=4,
        heads=2,
        norm_groups=2,
        n_tasks=3,
        time_steps=4,
    )
    adapted = prepare_ttt_model(model, seed=12)
    assert model.cfg.n_tasks == 3
    assert adapted.cfg.n_tasks == 51
    assert adapted.task_embed.weight.shape == (51, 4)
    for name, value in model.state_dict().items():
        if name != "task_embed.weight":
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)


def test_ttt_updates_model_and_task_embeddings_with_fresh_augmented_batches():
    from nca.engine.bench import make_pool
    from nca.tasks.arc import ARCTask
    from nca.ttt import TTTSession

    task = ARCTask(n_tasks=51, online_scale=True)
    boards = augmentation_boards([(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))])
    model = build_model(
        "arc",
        channels=32,
        c_in=12,
        c_out=12,
        out_offset=12,
        c_task=4,
        heads=2,
        norm_groups=2,
        n_tasks=51,
        time_steps=4,
    )
    tr = resolve_train_cfg(
        {
            "train": dict(
                batch_size=2,
                rollout=4,
                chunk=4,
                loss_window=4,
                loss="ce",
                perturb=False,
                replay=False,
                augment=True,
                amp=False,
            )
        }
    )
    engine = Engine(
        model, EngineConfig(backend="fused", checkpoint_segment=2), tr, torch.device("cpu")
    )
    pool = make_pool(model, task, boards, 2, seed=4, train_cfg=tr)
    session = TTTSession(engine, pool, steps=100)
    old = model.task_embed.weight.detach().clone()
    old_update = model.update.fc2.weight.detach().clone()
    for _ in range(2):
        _, loss, gn = session.iteration()
        assert torch.isfinite(loss) and torch.isfinite(gn)
    assert not torch.equal(old, model.task_embed.weight)
    assert not torch.equal(old_update, model.update.fc2.weight)
    assert session.opt.param_groups[0]["weight_decay"] == 0


@pytest.mark.parametrize("arc", [False, True])
def test_microbatch_preserves_draws_masked_loss_and_shared_parameter_gradients(tmp_path, arc):
    torch.manual_seed(29)
    if arc:
        task = get_task("arc", n_tasks=2)
        boards = task.load("train", str(tmp_path), source="synthetic", download=False)
        model = build_model(
            "arc",
            channels=32,
            c_in=12,
            c_out=12,
            out_offset=12,
            c_task=4,
            heads=2,
            norm_groups=2,
            n_tasks=2,
            time_steps=4,
        )
    else:
        task = get_task("maze_ood")
        boards = task.load("train", str(tmp_path), source="generated", size=5, n=8)
        model = build_model("maze_ood")
    tr = resolve_train_cfg(
        {"train": dict(rollout=4, chunk=4, amp=False, loss="ce" if arc else "mse")}
    )
    batch = make_batch(model, task, boards, 5, seed=6)
    if arc:
        # Unequal numbers of valid cells: simply averaging micro-batch losses is incorrect.
        batch["loss_mask"][0, :, 8:] = 0
        batch["loss_mask"][1, :, 16:] = 0
    a = EngineConfig(backend="reference", checkpoint_segment=2)
    b = EngineConfig(backend="reference", checkpoint_segment=2, microbatch=2)
    result = compare_backends(
        model, a, b, tr, batch, np.arange(4), 0, loss_weights("all", 4, torch.device("cpu"))
    )
    assert result["state_max_abs"] < 1e-6
    assert result["loss_abs"] < 1e-6
    assert result["grad_max_abs"] < 1e-6


@pytest.mark.parametrize("firing", [0.0, 0.5, 1.0])
def test_sparse_pixel_mlp_matches_dense_states_and_all_gradients(firing):
    from nca.engine.step import StepOptions, make_step

    cfg = load_config("configs/visual_sudoku_compute.yaml", [])
    cfg["model"]["overrides"].update(channels=8, expansion=2, zero_init_update=False)
    model = build_run_model(cfg).double()
    inp = torch.rand(2, 7, 7)
    x, g, c = model.init_state(inp, torch.ones_like(inp, dtype=torch.bool))
    x = x.double().requires_grad_()
    c = c.double()
    fire = (torch.rand(2, 1, 7, 7) < firing).double()
    ref = model.step(x, g, c, fire)
    out = make_step(model, StepOptions(sparse_mlp=True))(x, g, c, fire, None, None, None)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    params = (x, *model.parameters())
    upstream = torch.randn_like(x)
    a = torch.autograd.grad(ref, params, upstream, retain_graph=True)
    b = torch.autograd.grad(out, params, upstream)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-10, rtol=1e-10)


def test_ttt_reference_and_fused_follow_same_online_batches_and_adam_updates():
    import copy

    from nca.engine.bench import make_pool
    from nca.tasks.arc import ARCTask
    from nca.ttt import TTTSession

    torch.manual_seed(72)
    task = ARCTask(n_tasks=51)
    x = np.array([[1, 2], [3, 4]])
    boards = augmentation_boards([(x, x + 1), (x, x + 2)], seed=3)
    original = build_model(
        "arc",
        channels=32,
        c_in=12,
        c_out=12,
        out_offset=12,
        c_task=4,
        heads=2,
        norm_groups=2,
        n_tasks=51,
        time_steps=4,
    )
    tr = resolve_train_cfg(
        {
            "train": dict(
                batch_size=2,
                rollout=4,
                chunk=4,
                loss_window=4,
                loss="ce",
                perturb=False,
                replay=False,
                augment=True,
                amp=False,
            )
        }
    )
    sessions = []
    for backend in ["reference", "fused"]:
        model = copy.deepcopy(original)
        e = Engine(
            model, EngineConfig(backend=backend, checkpoint_segment=2), tr, torch.device("cpu")
        )
        sessions.append(
            TTTSession(e, make_pool(model, task, boards, 2, seed=9, train_cfg=tr), steps=100)
        )
    for _ in range(3):
        a = sessions[0].iteration()
        b = sessions[1].iteration()
        torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[1], b[1], atol=1e-5, rtol=1e-5)
    for a, b in zip(
        sessions[0].engine.model.parameters(), sessions[1].engine.model.parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("online_scale", [False, True])
def test_ttt_online_placement_preserves_each_offline_transform(online_scale):
    from nca.tasks.arc import ARCTask, transform_grid, unrender

    original = np.array([[1, 2, 3], [4, 5, 6]])
    target = np.array([[2, 3, 4], [5, 6, 7]])
    boards = augmentation_boards([(original, target)], seed=17)
    task = ARCTask(n_tasks=51, online_scale=online_scale)
    rendered = task.augment(boards, np.random.default_rng(32))
    np.testing.assert_array_equal(rendered.meta["task_id"], boards.meta["task_id"])
    np.testing.assert_array_equal(rendered.meta["dihedral_k"], boards.meta["dihedral_k"])
    np.testing.assert_array_equal(rendered.meta["color_perm"], boards.meta["color_perm"])
    for i in range(51):
        m = rendered.meta
        actual_input = unrender(rendered.inputs[i], m["scale"][i], m["offset"][i], m["in_shape"][i])
        actual_target = unrender(
            rendered.targets[i], m["scale"][i], m["offset"][i], m["out_shape"][i]
        )
        k, perm = boards.meta["dihedral_k"][i], boards.meta["color_perm"][i]
        np.testing.assert_array_equal(actual_input, transform_grid(original, k, perm))
        np.testing.assert_array_equal(actual_target, transform_grid(target, k, perm))
