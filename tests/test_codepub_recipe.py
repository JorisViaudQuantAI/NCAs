"""The opt-in switches that reproduce the training recipe of the authors' public code (betas,
init_lr_ratio, loss_reduce, ema_warmup, damage fill / geometry, batch_noise, swap_source,
chunk_start / writeback) and `train.stop_at` in the fast engine. Every switch defaults to the
historical behaviour; these tests check what each one does when it is set (CPU, seconds)."""

import json

import numpy as np
import pytest
import torch

from nca.engine.bench import iteration_seeds, make_batch
from nca.engine.engine import Engine, EngineConfig
from nca.engine.train import loss_weights, train_fast
from nca.engine.train import resolve_train_cfg as resolve_engine_cfg
from nca.model import build_model
from nca.perturb import apply_batch_noise, apply_damage, circular_masks
from nca.pool import ReplayPool
from nca.tasks import get_task
from nca.train import ema_decay, lr_at, perturb_batch, resolve_train_cfg, step_loss, train
from nca.utils import load_config


def _cfg(tmp_path, steps=2, rollout=6, chunk=4, **train_kw):
    cfg = load_config(
        "configs/smoke_maze.yaml",
        [
            f"data_root={tmp_path / 'data'}",
            "task.train.n=64",
            "task.val.n=8",
            f"train.steps={steps}",
            "train.batch_size=4",
            f"train.rollout={rollout}",
            f"train.chunk={chunk}",
            "train.log_every=1",
            f"train.eval_every={steps}",
            f"train.checkpoint_every={steps}",
            "train.eval.n=8",
            "train.eval.depth=6",
            "task.train.download=false",
            "task.val.download=false",
        ],
    )
    cfg["train"].update(train_kw)
    return cfg


def _engine_cfg(cfg):
    return {**cfg, "engine": {"backend": "fused", "checkpoint_segment": None}}


def _pool_setup(tmp_path, preset="maze_ood", B=4, **overrides):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), size=9, n=16, download=False, source="generated")
    model = build_model(preset, **overrides)
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    pool = ReplayPool(model, task, boards, 4 * B, rng, gen, augment=False)
    return model, pool, rng, gen


def _ckpt(path):
    return torch.load(path / "checkpoint.pt", weights_only=False)


# ---------------------------------------------------------------- learning rate, loss, EMA


def test_lr_warmup_from_init_ratio_then_constant():
    total = 100
    for schedule in ("cosine", "constant"):
        cfg = dict(warmup_ratio=0.1, schedule=schedule, cosine_floor=1.0, init_lr_ratio=10.0)
        assert lr_at(0, total, cfg) == pytest.approx(10.0)
        assert lr_at(5, total, cfg) == pytest.approx(10.0 + (1 - 10.0) * 5 / 10)
        assert [lr_at(s, total, cfg) for s in (10, 50, 99)] == pytest.approx([1.0] * 3)
    # init_lr_ratio 1: flat; None: the historical warmup from 1 / warm
    flat = dict(warmup_ratio=0.02, schedule="constant", init_lr_ratio=1.0)
    assert all(lr_at(s, 1000, flat) == 1.0 for s in range(0, 1000, 7))
    ours = dict(warmup_ratio=0.1, schedule="constant", init_lr_ratio=None)
    assert lr_at(0, total, ours) == pytest.approx(0.1) and lr_at(9, total, ours) == 1.0


def test_loss_reduce_mean_is_sum_over_output_channels():
    model = build_model("maze_ood")
    x = torch.randn(3, model.cfg.channels, 9, 9)
    targets = torch.randint(0, 2, (3, 9, 9))
    mask = (torch.rand(3, 9, 9) < 0.5).float()
    c_out = model.cfg.c_out
    for m in (None, mask):
        s = step_loss(model, x, targets, "mse", m)
        mean = step_loss(model, x, targets, "mse_mean", m)
        assert torch.allclose(mean, s / c_out, rtol=1e-6)
    tr = resolve_train_cfg({"train": {"loss_reduce": "mean"}})
    assert tr["loss"] == "mse_mean"
    assert resolve_train_cfg({})["loss"] == "mse"
    with pytest.raises(ValueError, match="MSE"):
        resolve_train_cfg({"train": {"loss": "ce", "loss_reduce": "mean"}})
    with pytest.raises(ValueError, match="loss_reduce"):
        resolve_train_cfg({"train": {"loss_reduce": "max"}})


def test_engine_loss_honours_loss_reduce(tmp_path):
    task = get_task("maze_ood")
    boards = task.load("train", str(tmp_path), size=9, n=8, download=False, source="generated")
    model = build_model("maze_ood")
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    batch = make_batch(model, task, boards, 4)
    w = loss_weights("all", 4, torch.device("cpu"))
    losses = {}
    for reduce in ("sum", "mean"):
        tr = resolve_engine_cfg({"train": dict(rollout=4, chunk=4, loss_reduce=reduce)})
        seeds, c0 = iteration_seeds(tr, 0)
        eng = Engine(
            model, EngineConfig(backend="fused", checkpoint_segment=None), tr, model.device
        )
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        _, losses[reduce], _ = eng.train_iteration(b, seeds, c0, w, 1.0, None)
        model.zero_grad(set_to_none=True)
    assert torch.allclose(losses["mean"], losses["sum"] / model.cfg.c_out, rtol=1e-5)


def test_ema_decay_switch():
    tr = dict(ema=0.999, ema_warmup=True)
    assert ema_decay(tr, 0) == pytest.approx(0.1) and ema_decay(tr, 10**6) == 0.999
    tr["ema_warmup"] = False
    assert ema_decay(tr, 0) == 0.999 and ema_decay(tr, 10**6) == 0.999


@pytest.mark.parametrize("trainer", ["reference", "engine"])
def test_ema_without_warmup_keeps_the_initial_weights_at_decay_one(tmp_path, trainer):
    """ema = 1 without warm-up: the EMA never moves; with the warm-up it does at step 0."""

    def ema_after(steps, warmup):
        cfg = _cfg(tmp_path, steps=steps, rollout=4, chunk=2, ema=1.0, ema_warmup=warmup)
        out = tmp_path / f"{trainer}_{steps}_{warmup}"
        if trainer == "engine":
            train_fast(_engine_cfg(cfg), out, device="cpu", resume=False, quiet=True)
        else:
            train(cfg, out, device="cpu", resume=False, quiet=True)
        return _ckpt(out)["ema"]

    a, b = ema_after(1, False), ema_after(2, False)
    assert all(torch.equal(a[k], b[k]) for k in a)
    c, d = ema_after(1, True), ema_after(2, True)
    assert not all(torch.equal(c[k], d[k]) for k in c)


def test_extra_ema_follows_the_same_trajectory(tmp_path):
    """An extra EMA without warm-up equals the main EMA of a run trained with ema_warmup: false,
    and leaves the trained weights and the main EMA unchanged."""
    extra = {"const": {"ema_warmup": False}}
    base = _cfg(tmp_path, steps=3, rollout=4, chunk=2)
    train(base, tmp_path / "a", device="cpu", resume=False, quiet=True)
    train(
        _cfg(tmp_path, steps=3, rollout=4, chunk=2, ema_extra=extra),
        tmp_path / "b",
        device="cpu",
        resume=False,
        quiet=True,
    )
    train(
        _cfg(tmp_path, steps=3, rollout=4, chunk=2, ema_warmup=False),
        tmp_path / "c",
        device="cpu",
        resume=False,
        quiet=True,
    )

    def w(run, name):
        return torch.load(tmp_path / run / f"{name}.pt", weights_only=False)["model"]

    for name in ("final", "final_raw"):
        assert all(torch.equal(w("a", name)[k], w("b", name)[k]) for k in w("a", name))
    assert all(torch.equal(w("b", "final_const")[k], w("c", "final")[k]) for k in w("c", "final"))
    with pytest.raises(ValueError, match="ema_extra"):
        train_fast(
            _engine_cfg(_cfg(tmp_path, steps=1, ema_extra=extra)),
            tmp_path / "d",
            device="cpu",
            resume=False,
            quiet=True,
        )


# ---------------------------------------------------------------- batch preparation


def test_damage_official_geometry_and_fill(tmp_path):
    rng = np.random.default_rng(0)
    H = W = 21
    masks = circular_masks(200, H, W, 4, 0.1, 0.4, rng, geometry="official")
    assert masks.any()
    # centres in [-0.5, 0.5] and r < 0.4 on a [-1, 1] grid: |coordinate| >= 0.9 is never hit
    coords = np.linspace(-1, 1, H)
    far = np.abs(coords) >= 0.9
    assert not masks[:, far, :].any() and not masks[:, :, far].any()
    # one disc reproduces the formula of the public make_circle_masks
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    m = circular_masks(1, H, W, 1, 0.1, 0.4, rng1, geometry="official")[0]
    cy, cx = rng2.uniform(-0.5, 0.5, size=2)
    r = rng2.uniform(0.1, 0.4)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    np.testing.assert_array_equal(m, (yy - cy) ** 2 + (xx - cx) ** 2 < r**2)
    # fill value inside the discs, untouched outside
    x = torch.randn(4, 8, H, W)
    xd, md = apply_damage(
        x, 1.0, 4, 0.1, 0.4, np.random.default_rng(1), force=True, fill=1.0, geometry="official"
    )
    mt = torch.from_numpy(md)[:, None].expand_as(x)
    assert md.any() and torch.all(xd[mt] == 1.0) and torch.equal(xd[~mt], x[~mt])


def test_damage_fill_keeps_clamped_channels(tmp_path):
    model, pool, rng, gen = _pool_setup(tmp_path)
    tr = resolve_train_cfg(
        {
            "train": dict(
                batch_size=4,
                swap=dict(p=0.0),
                damage=dict(p=1.0, n=4, r_min=0.3, r_max=0.4, fill=1.0, geometry="official"),
            )
        }
    )
    _, batch = pool.sample(4, 0.0)
    perturb_batch(model, pool, batch, tr, rng, gen)
    x, g4, clamp = batch["x"], batch["given4"], batch["clamp"]
    sl = model.cfg.in_slice
    assert torch.equal(x, model.apply_clamp(x, g4, clamp))
    assert torch.equal(x[:, sl][g4.expand_as(clamp)], clamp[g4.expand_as(clamp)])
    # (the maze input slice is the whole state: the fill shows on the free cells)
    filled = (x == 1.0).all(dim=1) & ~g4[:, 0]
    assert filled.any()


def test_batch_noise_leaves_given_input_channels_untouched(tmp_path):
    model, pool, rng, gen = _pool_setup(tmp_path)
    _, batch = pool.sample(4, 0.0)
    x, g4 = batch["x"], batch["given4"]
    sl = model.cfg.in_slice
    xn = apply_batch_noise(x, g4, sl, 1.0, 1.0, 0.15, torch.Generator().manual_seed(0))
    gin = g4.expand(-1, sl.stop - sl.start, -1, -1)
    in_ch = torch.zeros(1, x.shape[1], 1, 1, dtype=torch.bool)
    in_ch[:, sl] = True
    frozen = (g4 & in_ch).expand_as(x)
    assert frozen.any() and (~frozen).any()
    assert torch.equal(xn[frozen], x[frozen])
    assert (xn[~frozen] != x[~frozen]).all()  # p = ps = 1: everything else is noised
    # p = 0: nothing changes
    assert torch.equal(apply_batch_noise(x, g4, sl, 0.0, 1.0, 0.15, gen), x)
    # in the batch preparation: noise, renormalise, clamp
    tr = resolve_train_cfg(
        {"train": dict(batch_size=4, swap=dict(p=0.0), damage=dict(p=0.0), batch_noise={})}
    )
    assert tr["batch_noise"] == dict(p=0.1, ps=0.2, sigma=0.15)
    tr["batch_noise"] = dict(p=1.0, ps=0.5, sigma=0.15)
    before = batch["x"].clone()
    perturb_batch(model, pool, batch, tr, rng, gen)
    assert torch.equal(batch["x"][:, sl][gin], before[:, sl][gin])
    assert not torch.equal(batch["x"], before)


def test_batch_noise_renormalises_normalised_models(tmp_path):
    model, pool, rng, gen = _pool_setup(tmp_path, channels=16, norm_groups=2)
    tr = resolve_train_cfg(
        {
            "train": dict(
                batch_size=4,
                swap=dict(p=0.0),
                damage=dict(p=0.0),
                batch_noise=dict(p=1.0, ps=1.0, sigma=0.5),
            )
        }
    )
    _, batch = pool.sample(4, 0.0)
    x0, g4, clamp = batch["x"].clone(), batch["given4"], batch["clamp"]
    probe = torch.Generator().manual_seed(0)
    probe.set_state(gen.get_state())
    noisy = apply_batch_noise(x0, g4, model.cfg.in_slice, 1.0, 1.0, 0.5, probe)
    perturb_batch(model, pool, batch, tr, rng, gen)
    assert torch.equal(batch["x"], model.apply_clamp(model.normalize(noisy), g4, clamp))
    assert not torch.allclose(batch["x"], model.apply_clamp(noisy, g4, clamp))


def test_swap_source_pool_copies_a_pool_board(tmp_path):
    model, pool, rng, gen = _pool_setup(tmp_path)
    pool.x.add_(torch.randn_like(pool.x))  # make the pool states distinguishable
    pool.x.copy_(model.apply_clamp(pool.x, pool.given4, pool.clamp))
    pool.age.fill_(7)
    _, batch = pool.sample(4, 0.0)
    before = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
    which = np.array([0, 2])
    probe = np.random.default_rng()
    probe.bit_generator.state = pool.rng.bit_generator.state
    src = probe.integers(0, pool.size, size=len(which))
    pool.swap_from_pool(batch, which)
    sl = model.cfg.in_slice
    for i, j in zip(which, src, strict=True):
        for k, store in (
            ("inputs", pool.inputs),
            ("targets", pool.targets),
            ("given4", pool.given4),
            ("clamp", pool.clamp),
        ):
            assert torch.equal(batch[k][i], store[j]), k
        g = pool.given4[j].expand(sl.stop - sl.start, -1, -1)
        assert torch.equal(batch["x"][i, sl][g], pool.x[j, sl][g])
        assert torch.equal(batch["x"][i, sl][~g], before["x"][i, sl][~g])  # state kept
        assert torch.equal(batch["x"][i, sl.stop :], before["x"][i, sl.stop :])
        assert batch["age"][i] == 0
    for i in (1, 3):  # not swapped
        assert torch.equal(batch["x"][i], before["x"][i]) and batch["age"][i] == 7
    with pytest.raises(ValueError, match="swap_source"):
        resolve_train_cfg({"train": {"swap_source": "dataset"}})


@pytest.mark.parametrize("trainer", ["reference", "engine"])
def test_public_recipe_runs_in_both_trainers(tmp_path, trainer):
    cfg = _cfg(
        tmp_path,
        steps=4,
        betas=[0.9, 0.95],
        weight_decay=1e-4,
        warmup_ratio=0.5,
        init_lr_ratio=2.0,
        loss_reduce="mean",
        ema_warmup=False,
        damage=dict(p=0.5, n=4, fill=0.0, geometry="official"),
        batch_noise=dict(p=0.5, ps=0.3, sigma=0.15),
        swap_source="pool",
        swap=dict(p=0.5),
    )
    out = tmp_path / trainer
    if trainer == "engine":
        train_fast(_engine_cfg(cfg), out, device="cpu", resume=False, quiet=True)
    else:
        train(cfg, out, device="cpu", resume=False, quiet=True)
    ck = _ckpt(out)
    assert tuple(ck["opt"]["param_groups"][0]["betas"]) == (0.9, 0.95)
    recs = [json.loads(line) for line in open(out / "log.jsonl")]
    lrs = [r["lr"] for r in recs if "lr" in r]
    # warm = 2 steps from 2 x lr to lr; the first record shows the LR of the second step
    assert lrs[0] == pytest.approx(cfg["train"]["lr"] * (2.0 + (1 - 2.0) * 1 / 2))
    assert all(np.isfinite(r["loss"]) for r in recs if "loss" in r)
    assert (out / "final.pt").exists()


# ---------------------------------------------------------------- truncated BPTT variants


def test_per_sample_chunks_and_chunk_end_writeback_run(tmp_path):
    cfg = _cfg(tmp_path, steps=2, rollout=8, chunk=3, chunk_start="per_sample")
    for writeback, window in (("rollout_end", "all"), ("chunk_end", "last"), ("chunk_end", 2)):
        cfg["train"].update(writeback=writeback, loss_window=window)
        out = tmp_path / f"ps_{writeback}_{window}"
        train(cfg, out, device="cpu", resume=False, quiet=True)
        recs = [json.loads(line) for line in open(out / "log.jsonl")]
        assert all(np.isfinite(r["loss"]) for r in recs if "loss" in r)
        assert _ckpt(out)["step"] == 2
    with pytest.raises(ValueError, match="chunk_start"):
        resolve_train_cfg({"train": {"chunk_start": "random"}})


def test_per_sample_chunk_end_is_shared_rollout_end_when_chunk_is_rollout(tmp_path):
    """N == Nc: every board's chunk is the whole rollout, so the per-sample code must give the
    loss, the gradient and the pool states of the shared one (up to summation order)."""
    base = _cfg(tmp_path, steps=1, rollout=5, chunk=5)
    other = _cfg(tmp_path, steps=1, rollout=5, chunk=5)
    other["train"].update(chunk_start="per_sample", writeback="chunk_end")
    train(base, tmp_path / "a", device="cpu", resume=False, quiet=True)
    train(other, tmp_path / "b", device="cpu", resume=False, quiet=True)
    a, b = _ckpt(tmp_path / "a"), _ckpt(tmp_path / "b")
    assert torch.allclose(a["pool"]["x"], b["pool"]["x"], atol=1e-6)
    for k in ("inputs", "targets", "given4", "age"):
        assert torch.equal(a["pool"][k], b["pool"][k]), k
    assert a["rng"] == b["rng"]
    la, lb = (
        [json.loads(line) for line in open(tmp_path / d / "log.jsonl") if '"loss"' in line][0]
        for d in ("a", "b")
    )
    assert lb["loss"] == pytest.approx(la["loss"], rel=1e-5)
    assert lb["grad_norm"] == pytest.approx(la["grad_norm"], rel=1e-4)


def test_engine_rejects_reference_only_tbptt_options(tmp_path):
    for kw in (dict(chunk_start="per_sample"), dict(writeback="chunk_end")):
        cfg = _engine_cfg(_cfg(tmp_path, **kw))
        with pytest.raises(ValueError, match="reference trainer only"):
            train_fast(cfg, tmp_path / "x", device="cpu", resume=False, quiet=True)
    assert resolve_engine_cfg({"train": {"betas": [0.9, 0.95]}})["betas"] == [0.9, 0.95]


# ---------------------------------------------------------------- stop_at in the fast engine


def test_engine_honours_train_stop_at(tmp_path):
    full, probe = tmp_path / "full", tmp_path / "probe"
    cfg = _engine_cfg(_cfg(tmp_path, steps=4))
    cfg["train"]["checkpoint_every"] = 10
    train_fast(cfg, full, device="cpu", resume=False, quiet=True)
    cfg["train"]["stop_at"] = 2
    train_fast(cfg, probe, device="cpu", resume=False, quiet=True)
    assert _ckpt(probe)["step"] == 2
    assert not (probe / "final.pt").exists() and not (probe / "summary.json").exists()
    with pytest.raises(ValueError, match="stop_at"):
        train_fast(cfg, probe, device="cpu", quiet=True)  # nothing left before stop_at
    cfg["train"]["stop_at"] = None
    train_fast(cfg, probe, device="cpu", quiet=True)
    a = torch.load(full / "final.pt", weights_only=False)["model"]
    b = torch.load(probe / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert json.loads((probe / "summary.json").read_text())["steps"] == 4


def test_per_sample_gradient_is_the_mean_of_the_boards_own_chunks(tmp_path):
    """Deterministic steps (fire rate 1, no noise): the per-sample loss and gradient are the
    means of what a shared rollout gives each board alone with its own chunk start, and the
    chunk-end write-back is each board's state at the end of its own chunk."""
    from nca.train import _rollout_per_sample, _rollout_shared

    model, pool, _, gen = _pool_setup(tmp_path, fire_rate=1.0)
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    _, batch = pool.sample(2, 0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    c0s = np.array([0, 2])

    def run(fn, rows, c0, rollout=6, **kw):
        tr = resolve_train_cfg(
            {"train": dict(rollout=rollout, chunk=3, perturb=False, grad_clip=1e9, **kw)}
        )
        b = {k: batch[k][rows] for k in ("x", "given4", "clamp", "targets")}
        x, loss, _ = fn(
            model, opt, b["x"], b["given4"], b["clamp"], b["targets"], None, c0, tr, gen, False
        )
        return x, loss, [p.grad.clone() for p in model.parameters()]

    x_ps, loss_ps, g_ps = run(_rollout_per_sample, [0, 1], c0s, writeback="chunk_end")
    singles = [run(_rollout_shared, [i], int(c0s[i])) for i in (0, 1)]
    assert torch.allclose(loss_ps, (singles[0][1] + singles[1][1]) / 2, rtol=1e-5)
    for k, g in enumerate(g_ps):
        ref = (singles[0][2][k] + singles[1][2][k]) / 2
        assert torch.allclose(g, ref, rtol=1e-4, atol=1e-7)
    # a shared rollout that stops at the end of board i's chunk ends in its written-back state
    ends = [run(_rollout_shared, [i], int(c0s[i]), rollout=int(c0s[i]) + 3)[0] for i in (0, 1)]
    assert torch.allclose(x_ps, torch.cat(ends), atol=1e-6)
