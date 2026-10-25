"""ARC through the training and evaluation machinery, on CPU with synthetic tasks (seconds).

Regression tests for the ARC review of 2026-10-12: replay pool states carrying the task
embedding's graph (backward through the graph twice), the clamp mask `given` used as the
evaluation mask (ARC predictions overwritten by the inputs), `score_candidates` dropping the
ARC meta, the gradient-checkpoint heuristic ignoring the grid size, the loss mask missing from
the pool checkpoint, the step embedding of paper eq. B.3-(1) and the majority-vote selection.
"""

import json
import warnings

import numpy as np
import pytest
import torch

from nca.experiments import majority_pass_at_m, tts_curves
from nca.model import build_model
from nca.pool import ReplayPool
from nca.rollout import RolloutConfig, RolloutResult, decode_boards, rollout, score_candidates
from nca.tasks import arc as A
from nca.tasks import get_task
from nca.train import auto_grad_checkpoint, graph_span, resolve_train_cfg, train
from nca.utils import load_config

TINY_MODEL = dict(
    channels=32, c_in=12, c_out=12, out_offset=12, c_task=4, heads=2, norm_groups=2, time_steps=4
)


def _arc(tmp_path, n_tasks=2, split="train"):
    task = get_task("arc", n_tasks=n_tasks, online_dihedral=True, online_color=True)
    return task, task.load(split, str(tmp_path), source="synthetic")


def _tiny_model(n_tasks=2, **over):
    return build_model("arc", n_tasks=n_tasks, **{**TINY_MODEL, **over})


def _tiny_cfg(tmp_path, **over):
    base = dict(steps=3, batch_size=2, rollout=4, chunk=4, loss_window=2, log_every=1, eval_every=3)
    base.update(over)
    return load_config(
        "configs/arc_small.yaml",
        [
            f"data_root={tmp_path}",
            "task.args.n_tasks=2",
            "task.args.rearc_per_task=0",
            "task.train.source=synthetic",
            "task.val.source=synthetic",
            "model.overrides.n_tasks=2",
            *[f"model.overrides.{k}={v}" for k, v in TINY_MODEL.items()],
            *[f"train.{k}={v}" for k, v in base.items()],
            "train.checkpoint_every=3",
            "train.eval.n=2",
            "train.eval.depth=4",
        ],
    )


# ---------------------------------------------------------------- 1. pool states and the graph
def test_pool_stores_detached_states_and_task_embedding_gets_a_gradient(tmp_path):
    task, boards = _arc(tmp_path)
    model = _tiny_model()
    rng = np.random.default_rng(0)
    pool = ReplayPool(model, task, boards, 4, rng, None, task_ids=boards.meta["task_id"])
    assert not pool.x.requires_grad
    for _ in range(2):  # the second backward used to fail: the pool kept the first graph alive
        idx, batch = pool.sample(2, seed_ratio=1.0)
        assert batch["x"].requires_grad
        # the fresh rows carry the *current* embedding
        te = model.task_embed(batch["task_ids"])[:, :, None, None]
        assert torch.allclose(batch["x"][:, model.cfg.task_slice], te.expand(-1, -1, 32, 32))
        model.task_embed.weight.grad = None
        batch["x"].sum().backward()
        assert model.task_embed.weight.grad is not None and model.task_embed.weight.grad.any()
        batch["x"] = batch["x"].detach()
        pool.write_back(idx, batch)
        assert not pool.x.requires_grad
    # replayed rows (not fresh) keep the evolved task slice of the pool, detached
    with torch.no_grad():
        pool.x[:, model.cfg.task_slice] = 7.0
    _, batch = pool.sample(2, seed_ratio=0.5)
    assert torch.all(batch["x"][1, model.cfg.task_slice] == 7.0)
    assert not torch.all(batch["x"][0, model.cfg.task_slice] == 7.0)


def test_arc_training_runs_several_steps_and_scores_the_output(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    out = tmp_path / "run"
    res = train(cfg, out, seed=0, device="cpu", resume=False, quiet=True)
    recs = [json.loads(line) for line in open(out / "log.jsonl")]
    steps = [r for r in recs if "loss" in r]
    assert [r["step"] for r in steps] == [1, 2, 3]
    # 2. the batch metric looks at the NCA output, not at the (all given) inputs
    assert all(r["batch_exact"] == 0.0 for r in steps)
    assert all(0.0 < r["batch_cell_acc"] < 1.0 for r in steps)
    assert set(res) >= {"val_exact", "val_exact_with_border", "val_cell_acc", "val_conf"}
    # 5. the checkpoint restores the pool with its loss mask, and the run resumes
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["pool"]["loss_mask"].shape == (2, 32, 32)
    cfg["train"]["steps"] = 4
    train(cfg, out, seed=0, device="cpu", resume=True, quiet=True)
    recs = [json.loads(line) for line in open(out / "log.jsonl")]
    assert max(r.get("step", 0) for r in recs) == 4


# ---------------------------------------------------------------- 2. clamp mask vs eval mask
def test_arc_rollout_reads_predictions_from_the_output_slice(tmp_path):
    task, boards = _arc(tmp_path, split="test")
    assert boards.given.all() and not task.fill_given
    mask = task.eval_mask(boards)
    assert np.array_equal(mask, boards.meta["loss_mask"]) and mask.any() and not mask.all()
    model = _tiny_model()
    rc = RolloutConfig(depth=4, k=2, record_every=2)
    res = rollout(model, task, boards, rc, seed=0, task_ids=boards.meta["task_id"])
    # predictions are not a copy of the inputs, and the trace is a real accuracy
    assert not np.array_equal(res.pred[:, 0], boards.inputs)
    assert np.all(res.trace_cell_acc < 1.0)
    m = score_candidates(task, res, boards)
    assert np.array_equal(res.trace_exact[:, :, -1], m["exact_with_border"])
    # a perfect state decodes to the target on every scored cell: trace exact is True
    x = torch.zeros(len(boards), model.cfg.channels, 32, 32)
    x[:, model.cfg.out_slice] = -5.0
    # mlp head: make the logit of the target token win through the readout
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[:, :, 0, 0] = torch.eye(12)
    x[:, model.cfg.out_slice] = (
        torch.nn.functional.one_hot(torch.from_numpy(boards.targets), 12)
        .permute(0, 3, 1, 2)
        .float()
    )
    given = torch.from_numpy(boards.given)
    dec = decode_boards(model, task, x, given, torch.from_numpy(boards.inputs))
    assert np.array_equal(dec.numpy(), boards.targets)
    # boards without a target are never exact and have an empty eval mask
    _, t3 = task.ttt_variants(
        {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]},
        n_variants=1,
    )
    assert not task.eval_mask(t3).any() and not task.metrics(t3.targets, t3)["exact"].any()


@pytest.mark.parametrize("name", ["maze_ood", "sudoku_ood"])
def test_maze_and_sudoku_scoring_is_unchanged(name, tmp_path):
    """Given cells are clamped AND excluded from the metric, exactly as before: the evaluation
    mask is `~given` and the decoded prediction copies the inputs on given cells."""
    task = get_task(name)
    kw = dict(size=9, n=4) if name == "maze_ood" else dict(n=4, seed=0)
    boards = task.load("test", str(tmp_path), download=False, source="generated", **kw)
    assert task.fill_given
    assert np.array_equal(task.eval_mask(boards), ~boards.given)
    model = build_model(name)
    inputs, given = torch.from_numpy(boards.inputs), torch.from_numpy(boards.given)
    x, g4, clamp = model.init_state(inputs, given)
    x = x + 0.3 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    dec = decode_boards(model, task, x, given, inputs)
    assert torch.equal(dec, torch.where(given, inputs, model.decode(x)))
    res = rollout(model, task, boards, RolloutConfig(depth=3, k=2, record_every=3), seed=0)
    rep = np.repeat(boards.inputs[:, None], 2, 1)
    g = np.repeat(boards.given[:, None], 2, 1)
    assert np.array_equal(res.pred[g], rep[g])
    m = score_candidates(task, res, boards)
    assert m["exact"].shape == (4, 2)
    assert np.array_equal(res.trace_exact[:, :, -1], m["exact"])


# ---------------------------------------------------------------- 3. meta along K
def test_score_candidates_repeats_arc_meta(tmp_path):
    task, boards = _arc(tmp_path, split="test")
    rep = boards.repeat(3)
    assert len(rep) == 3 * len(boards)
    assert np.array_equal(rep.meta["task_id"], np.repeat(boards.meta["task_id"], 3))
    assert rep.meta["loss_mask"].shape == (3 * len(boards), 32, 32)
    assert rep.meta["pairs"].dtype == object and len(rep.meta["pairs"]) == 3 * len(boards)
    assert rep.meta["source"] == "synthetic" and rep.meta["n_tasks"] == boards.meta["n_tasks"]
    model = _tiny_model()
    res = rollout(
        model, task, boards, RolloutConfig(depth=2, k=3), seed=0, task_ids=boards.meta["task_id"]
    )
    m = score_candidates(task, res, boards)  # used to raise KeyError: 'has_target'
    assert set(m) == {"exact", "exact_with_border", "cell_acc"}
    assert all(v.shape == (len(boards), 3) for v in m.values())
    # the ground truth as prediction scores 1 on every trial
    res.pred[:] = boards.targets[:, None]
    m = score_candidates(task, res, boards)
    assert m["exact"].all() and np.all(m["cell_acc"] == 1.0)


# ---------------------------------------------------------------- 4. checkpoint heuristic
def test_auto_grad_checkpoint_counts_cells():
    assert not auto_grad_checkpoint(100, 64, 16, (9, 9))  # Maze-OOD
    assert not auto_grad_checkpoint(128, 32, 32, (30, 30))  # Maze-Hard
    assert auto_grad_checkpoint(96, 128, 256, (9, 9))  # Sudoku-Extreme
    assert auto_grad_checkpoint(64, 32, 128, (32, 32))  # arc_small: 32 x 32 cells count
    cfg = load_config("configs/arc_small.yaml")
    assert cfg["train"]["grad_checkpoint"] is True
    assert load_config("configs/sudoku_ood.yaml")["train"]["grad_checkpoint"] is False


def test_per_sample_chunks_count_the_whole_rollout():
    tr = resolve_train_cfg({"train": {"rollout": 256, "chunk": 128, "chunk_start": "per_sample"}})
    assert graph_span(tr) == 256
    assert auto_grad_checkpoint(graph_span(tr), 32, 32, (30, 30))  # Maze-Hard, public-code TBPTT
    tr = resolve_train_cfg({"train": {"rollout": 256, "chunk": 128, "writeback": "chunk_end"}})
    assert graph_span(tr) == 128


# ---------------------------------------------------------------- 5. loss mask in the checkpoint
def test_pool_state_dict_keeps_the_loss_mask(tmp_path):
    task, boards = _arc(tmp_path)
    model = _tiny_model()
    rng = np.random.default_rng(0)
    pool = ReplayPool(model, task, boards, 3, rng, None, task_ids=boards.meta["task_id"])
    d = pool.state_dict()
    assert torch.equal(d["loss_mask"], (pool.targets != A.PAD).float())
    other = ReplayPool(
        model, task, boards, 3, np.random.default_rng(1), None, task_ids=boards.meta["task_id"]
    )
    assert not torch.equal(other.loss_mask, pool.loss_mask)
    other.load_state_dict(d)
    assert torch.equal(other.loss_mask, pool.loss_mask)
    assert torch.equal(other.loss_mask, (other.targets != A.PAD).float())
    # an old checkpoint without the key still loads (with a warning)
    old = {k: v for k, v in d.items() if k != "loss_mask"}
    third = ReplayPool(
        model, task, boards, 3, np.random.default_rng(2), None, task_ids=boards.meta["task_id"]
    )
    with pytest.warns(UserWarning, match="loss_mask"):
        third.load_state_dict(old)
    assert torch.equal(third.targets, pool.targets)
    # a maze pool has no loss mask and loads silently
    mtask = get_task("maze_ood")
    mb = mtask.load("test", str(tmp_path), size=9, n=4, download=False, source="generated")
    mpool = ReplayPool(build_model("maze_ood"), mtask, mb, 2, rng, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mpool.load_state_dict({k: v for k, v in mpool.state_dict().items() if k != "loss_mask"})


# ---------------------------------------------------------------- 6. step embedding, eq. B.3-(1)
def test_step_embedding_enters_the_hidden_state_of_firing_cells():
    model = _tiny_model(norm_groups=0)
    cfg = model.cfg
    with torch.no_grad():  # a zero update rule: the state only moves by the step embedding
        model.update.fc2.weight.zero_()
        model.update.fc2.bias.zero_()
    inputs = torch.randint(0, 12, (2, 32, 32))
    given = torch.ones(2, 32, 32, dtype=torch.bool)
    x, g4, clamp = model.init_state(inputs, given, torch.tensor([0, 1]))
    x = x.detach()
    hid = cfg.hid_slice
    te = model.time_embed.weight[2]
    # global clock, every cell fires: the hidden slice absorbs e_step(t), nothing else moves
    x1 = model.step(x, g4, clamp, torch.ones(2, 1, 32, 32), clock=torch.tensor(2))
    assert torch.allclose(x1[:, hid], x[:, hid] + te[None, :, None, None])
    assert torch.equal(x1[:, : hid.start], x[:, : hid.start])
    assert torch.equal(x1[:, hid.stop :], x[:, hid.stop :])
    # a cell that does not fire is frozen
    fm = torch.zeros(2, 1, 32, 32)
    fm[:, :, :16] = 1.0
    x2 = model.step(x, g4, clamp, fm, clock=torch.tensor(2))
    assert torch.equal(x2[:, :, 16:], x[:, :, 16:])
    assert torch.allclose(x2[:, hid, :16], x[:, hid, :16] + te[None, :, None, None])
    # local clocks index the table per cell (clamped to the last entry)
    clock = torch.zeros(2, 1, 32, 32, dtype=torch.long)
    clock[0] = 1
    clock[1] = 99
    x3 = model.step(x, g4, clamp, torch.ones(2, 1, 32, 32), clock=clock)
    assert torch.allclose(x3[0, hid], x[0, hid] + model.time_embed.weight[1][:, None, None])
    assert torch.allclose(x3[1, hid], x[1, hid] + model.time_embed.weight[3][:, None, None])
    # the embedding reaches the sensed state too (the proposed update depends on the clock)
    model = _tiny_model(norm_groups=0)
    d1 = model.delta(x, torch.tensor(0))
    d2 = model.delta(x, torch.tensor(3))
    assert not torch.allclose(d1, d2)


# ---------------------------------------------------------------- 6. majority vote selection
def test_majority_vote_selection_maps_back_to_the_canonical_frame(tmp_path):
    task, boards = _arc(tmp_path, split="test")
    boards = task.augment(boards, np.random.default_rng(0))  # scales, symmetries, colours
    N = len(boards)
    K = 4
    pred = np.zeros((N, K, 32, 32), dtype=np.int64)
    pred[:, 0] = boards.targets  # trial 0 right, trials 1-3 agree on a wrong grid
    wrong = boards.targets.copy()
    for n in range(N):
        oi, oj = boards.meta["offset"][n]
        wrong[n, oi, oj] = (wrong[n, oi, oj] + 1) % 10
    pred[:, 1:] = wrong[:, None]
    alive = np.ones((N, K), dtype=bool)
    res = RolloutResult(pred, np.zeros((N, K)), alive, np.zeros((N, K)), np.array([]), *[None] * 6)
    out = majority_pass_at_m(res, boards, [1, 2, 4], top=2)
    assert out[1] == {1: 1.0, 2: 1.0}  # the first trial alone
    assert out[2] == {1: 1.0, 2: 1.0}  # a 1-1 tie: trial 0 comes first
    assert out[4] == {1: 0.0, 2: 1.0}  # outvoted 3-1: second attempt
    # pruned trials do not vote
    alive[:, 1:] = False
    assert majority_pass_at_m(res, boards, [4], top=2)[4] == {1: 1.0, 2: 1.0}
    # boards without a target never pass
    boards.meta["has_target"][:] = False
    assert majority_pass_at_m(res, boards, [1], top=2)[1] == {1: 0.0, 2: 0.0}


def test_tts_curves_select_option(tmp_path):
    task, boards = _arc(tmp_path, split="test")
    model = _tiny_model()
    out = tts_curves(model, task, boards, depth=2, k=2, record_every=2, select="majority")
    assert out["select"] == "majority" and set(out["majority_pass_at_k"]) == {1, 2}
    assert set(out["majority_pass_at_k"][2]) == {1, 2}
    assert "conf_at_k" in out
    out = tts_curves(model, task, boards, depth=2, k=2, record_every=2)
    assert out["select"] == "conf" and "majority_pass_at_k" not in out
    mtask = get_task("maze_ood")
    mb = mtask.load("test", str(tmp_path), size=9, n=2, download=False, source="generated")
    with pytest.raises(ValueError, match="ARC"):
        tts_curves(
            build_model("maze_ood"), mtask, mb, depth=2, k=1, record_every=2, select="majority"
        )
    with pytest.raises(ValueError):
        tts_curves(model, task, boards, depth=2, k=1, record_every=2, select="vote")
