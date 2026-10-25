"""Tests of the ARC benchmark module (rendering, transforms, synthetic tasks, TTT variants).

Pure numpy, offline: the official data is never downloaded here (hand-written task files laid
out like the ARC-AGI repository stand in for it).
"""

import json
import os

import numpy as np
import pytest

from nca.tasks import arc as A
from nca.tasks.base import Boards


def _random_grid(rng, h, w):
    return rng.integers(0, 10, size=(h, w)).astype(np.int64)


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_render_unrender_round_trip(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        hi, wi = (int(v) for v in rng.integers(1, 31, size=2))
        ho, wo = (int(v) for v in rng.integers(1, 31, size=2))
        inp, out = _random_grid(rng, hi, wi), _random_grid(rng, ho, wo)
        scales = A.valid_scales([inp.shape, out.shape])
        assert scales[0] == 1 and all(s * max(hi, wi, ho, wo) <= 32 for s in scales)
        for scale in scales:
            x, y, mask = A.render_pair(inp, out, scale=scale, offset=None, rng=rng)
            # the offset is shared: both grids start where the first non-pad/border cell of x is
            oi, oj = np.argwhere(x < A.PAD)[0]
            assert np.array_equal(A.unrender(x, scale, (oi, oj), inp.shape), inp)
            assert np.array_equal(A.unrender(y, scale, (oi, oj), out.shape), out)
            assert np.array_equal(mask, y != A.PAD)
            # border ring = perimeter of the scaled target, clipped to the canvas
            i0, i1 = max(oi - 1, 0), min(oi + scale * ho + 1, 32)
            j0, j1 = max(oj - 1, 0), min(oj + scale * wo + 1, 32)
            expected = (i1 - i0) * (j1 - j0) - scale * scale * ho * wo
            assert (y == A.BORDER).sum() == expected
            assert (y < A.PAD).sum() == scale * scale * ho * wo
            # pad everywhere outside grid + ring
            box = np.zeros((32, 32), dtype=bool)
            box[i0:i1, j0:j1] = True
            assert np.all(y[~box] == A.PAD)
            xbox = np.zeros((32, 32), dtype=bool)
            xbox[
                max(oi - 1, 0) : min(oi + scale * hi + 1, 32),
                max(oj - 1, 0) : min(oj + scale * wi + 1, 32),
            ] = True
            assert np.all(x[~xbox] == A.PAD) and np.all(x[xbox] != A.PAD)


def test_render_explicit_placement_and_nearest_neighbour():
    g = np.array([[1, 2], [3, 4]])
    x, y, mask = A.render_pair(g, g, scale=3, offset=(5, 7))
    assert x.shape == (32, 32) and np.array_equal(x, y)
    assert np.all(y[5:8, 7:10] == 1) and np.all(y[8:11, 10:13] == 4)
    assert y[4, 6] == A.BORDER and y[11, 13] == A.BORDER and y[4, 14] == A.PAD
    assert mask.sum() == 8 * 8
    with pytest.raises(ValueError):
        A.render_grid(np.zeros((30, 30)), scale=2)
    # clipped ring: a 30 x 30 grid at offset 0 keeps only the bottom / right part of its ring
    _, y, _ = A.render_pair(np.zeros((30, 30)), np.zeros((30, 30)), scale=1, offset=(0, 0))
    assert (y == A.BORDER).sum() == 30 + 30 + 1


# ----------------------------------------------------------------------------------------------
# transforms
# ----------------------------------------------------------------------------------------------


def test_dihedral_and_color_inverses():
    rng = np.random.default_rng(3)
    g = _random_grid(rng, 5, 7)
    seen = set()
    for k in range(8):
        t = A.dihedral(g, k)
        assert np.array_equal(A.dihedral_inverse(t, k), g)
        seen.add(t.tobytes() + bytes(t.shape))
        for k2 in range(8):
            both = A.dihedral(A.dihedral(g, k), k2)
            assert np.array_equal(both, A.dihedral(g, A.compose_dihedral(k, k2)))
    assert len(seen) == 8
    for _ in range(5):
        perm = A.random_color_perm(rng)
        assert perm[0] == 0 and sorted(perm.tolist()) == list(range(10))
        p = A.permute_colors(g, perm)
        assert np.array_equal(A.permute_colors(p, A.inverse_perm(perm)), g)
        assert np.array_equal(p == 0, g == 0)
        t = A.transform_grid(g, 5, perm)
        assert np.array_equal(A.inverse_transform_grid(t, 5, perm), g)
    # pad / border tokens survive a colour permutation
    canvas = A.render_grid(g, scale=1, offset=(2, 2))
    pc = A.permute_colors(canvas, A.random_color_perm(rng))
    assert np.array_equal(pc >= A.PAD, canvas >= A.PAD) and np.array_equal(
        pc[pc >= 10], canvas[canvas >= 10]
    )


# ----------------------------------------------------------------------------------------------
# synthetic tasks, loading, augmentation, metrics
# ----------------------------------------------------------------------------------------------


def test_synthetic_rules_are_deterministic_and_valid():
    a = A.synthetic_tasks(4, seed=0)
    b = A.synthetic_tasks(4, seed=0)
    assert [n for n, _ in a] == [
        "synthetic_fill_enclosed",
        "synthetic_recolor_largest",
        "synthetic_mirror_h",
        "synthetic_fill_enclosed_1",
    ]
    for (_, ta), (_, tb) in zip(a, b, strict=True):
        for s in ("train", "test"):
            assert len(ta[s]) == len(tb[s])
            for pa, pb in zip(ta[s], tb[s], strict=True):
                assert np.array_equal(pa["input"], pb["input"])
                assert np.array_equal(pa["output"], pb["output"])
                assert pa["input"].max() <= 9 and pa["output"].shape[0] <= 30
    _, mirror = a[2]
    p = mirror["train"][0]
    assert p["output"].shape == (p["input"].shape[0], 2 * p["input"].shape[1])


def test_load_synthetic_and_metrics(tmp_path):
    task = A.ARCTask(n_tasks=3, online_dihedral=True, online_color=True)
    boards = task.load("train", str(tmp_path), source="synthetic")
    assert boards.shape == (32, 32) and boards.given.all()
    assert set(boards.meta["task_id"].tolist()) == {0, 1, 2}
    assert boards.meta["source"] == "synthetic" and len(boards.meta["task_names"]) == 3
    assert boards.meta["loss_mask"].shape == (len(boards), 32, 32)
    assert np.array_equal(boards.meta["loss_mask"], boards.targets != A.PAD)
    assert boards.meta["pairs"].dtype == object and len(boards.meta["pairs"]) == len(boards)
    assert set(np.unique(boards.inputs)) <= set(range(12))
    test = task.load("test", str(tmp_path), source="synthetic")
    assert set(test.meta["task_id"].tolist()) == {0, 1, 2} and len(test) == 6
    # metrics on the ground truth
    m = task.metrics(boards.targets, boards)
    assert m["exact"].all() and m["exact_with_border"].all() and np.all(m["cell_acc"] == 1.0)
    assert task.main_metric in m
    # corrupt one target-grid cell of board 0 -> exact False, cell_acc < 1
    pred = boards.targets.copy()
    oi, oj = boards.meta["offset"][0]
    pred[0, oi, oj] = (pred[0, oi, oj] + 1) % 10
    m = task.metrics(pred, boards)
    assert not m["exact"][0] and not m["exact_with_border"][0] and m["cell_acc"][0] < 1.0
    assert m["exact"][1:].all()
    # corrupt a pad cell -> no effect; corrupt a border cell -> only exact_with_border
    pred = boards.targets.copy()
    pad_cells = np.argwhere(boards.targets[0] == A.PAD)
    pred[0][tuple(pad_cells[-1])] = 3
    m = task.metrics(pred, boards)
    assert m["exact"][0] and m["exact_with_border"][0] and m["cell_acc"][0] == 1.0
    pred = boards.targets.copy()
    border_cells = np.argwhere(boards.targets[0] == A.BORDER)
    pred[0][tuple(border_cells[0])] = A.PAD
    m = task.metrics(pred, boards)
    assert m["exact"][0] and not m["exact_with_border"][0]
    # subsampling with n keeps metas aligned
    sub = task.load("train", str(tmp_path), n=4, seed=1, source="synthetic")
    assert len(sub) == 4 and len(sub.meta["pairs"]) == 4 and sub.meta["loss_mask"].shape[0] == 4


def test_augment_re_renders_from_original_pairs(tmp_path):
    task = A.ARCTask(n_tasks=3, online_dihedral=True, online_color=True)
    boards = task.load("train", str(tmp_path), source="synthetic")
    rng = np.random.default_rng(0)
    sub = boards.subset(np.array([0, 2, 5, 1]))
    aug = task.augment(sub, rng)
    assert isinstance(aug, Boards) and aug.shape == (32, 32) and len(aug) == 4
    assert np.array_equal(aug.meta["task_id"], sub.meta["task_id"])
    assert np.array_equal(aug.meta["loss_mask"], aug.targets != A.PAD)
    assert aug.meta["pairs"] is not sub.meta["pairs"]
    meta = aug.meta
    for i in range(4):
        inp, out = sub.meta["pairs"][i]
        assert np.array_equal(meta["pairs"][i][0], inp) and np.array_equal(meta["pairs"][i][1], out)
        k, perm, s, o = (
            meta["dihedral_k"][i],
            meta["color_perm"][i],
            meta["scale"][i],
            meta["offset"][i],
        )
        got = A.map_back(aug.targets[i], s, o, meta["out_shape"][i], k, perm)
        assert np.array_equal(got, out)
        got_in = A.map_back(aug.inputs[i], s, o, meta["in_shape"][i], k, perm)
        assert np.array_equal(got_in, inp)
        assert tuple(meta["out_shape"][i]) == A.dihedral(out, int(k)).shape
    # over many draws, scales > 1 and non-trivial symmetries / permutations do appear
    big = task.augment(boards, rng)
    assert (big.meta["scale"] > 1).any() and (big.meta["dihedral_k"] > 0).any()
    assert (big.meta["color_perm"] != A.IDENTITY_PERM).any()
    assert task.metrics(big.targets, big)["exact"].all()
    # with everything off, augment is the scale-1 / offset-0 rendering
    plain = A.ARCTask(n_tasks=3, online_scale=False).augment(boards, rng)
    assert np.array_equal(plain.inputs, boards.inputs) and np.array_equal(
        plain.targets, boards.targets
    )


# ----------------------------------------------------------------------------------------------
# official-style files, TTT variants
# ----------------------------------------------------------------------------------------------

_TASKS = {
    "00000001": {  # "add one to every colour" on a 2 x 2 -> 2 x 2
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[1, 2], [3, 4]]},
            {"input": [[5, 5], [0, 0]], "output": [[6, 6], [1, 1]]},
        ],
        "test": [{"input": [[7, 0], [0, 7]], "output": [[8, 1], [1, 8]]}],
    },
    "00000002": {  # transpose: shapes differ between input and output
        "train": [
            {"input": [[1, 2, 3]], "output": [[1], [2], [3]]},
            {"input": [[4, 0, 0], [0, 5, 0]], "output": [[4, 0], [0, 5], [0, 0]]},
            {"input": [[9]], "output": [[9]]},
        ],
        "test": [{"input": [[1, 1, 2, 2]], "output": [[1], [1], [2], [2]]}],
    },
    "00000003": {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]}, {"input": [[3]]}],
    },
}


def _write_tasks(root):
    d = os.path.join(root, "arc", "raw", "ARC-AGI-master", "data", "training")
    os.makedirs(d)
    for name, t in _TASKS.items():
        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump(t, f)
    return d


def test_task_files_and_official_load(tmp_path):
    root = str(tmp_path)
    assert A.list_tasks(root) == []
    with pytest.raises(FileNotFoundError):
        A.fetch_arc(root, download=False)
    d = _write_tasks(root)
    assert (
        A.list_tasks(root, "training") == sorted(_TASKS) and A.list_tasks(root, "evaluation") == []
    )
    assert A.fetch_arc(root, download=False) == os.path.dirname(d)
    t = A.load_task_json(A.task_path(root, "training", "00000002"))
    assert len(t["train"]) == 3 and t["train"][1]["output"].shape == (3, 2)
    assert t["train"][0]["input"].dtype == np.int64
    assert A.load_task_json(A.task_path(root, "training", "00000003"))["test"][1]["output"] is None
    task = A.ARCTask(n_tasks=2, task_seed=0)
    boards = task.load("train", root, download=False)
    assert boards.meta["source"] == "official" and len(boards.meta["task_names"]) == 2
    assert set(boards.meta["task_id"].tolist()) == {0, 1}
    assert all(n in _TASKS for n in boards.meta["task_name"])
    # the seeded selection is stable and task_ids overrides it
    again = task.load("train", root, download=False)
    assert again.meta["task_names"] == boards.meta["task_names"]
    explicit = task.load("test", root, download=False, task_ids=["00000003", "00000001"])
    assert explicit.meta["task_names"] == ("00000003", "00000001") and len(explicit) == 3
    assert explicit.meta["has_target"].tolist() == [True, False, True]
    m = task.metrics(explicit.targets, explicit)
    assert m["exact"].tolist() == [True, False, True]
    # grids are rendered at scale 1 / offset 0: the input grid sits at the top-left corner
    i = list(explicit.meta["task_name"]).index("00000001")
    assert np.array_equal(explicit.inputs[i][:2, :2], [[7, 0], [0, 7]])
    assert explicit.inputs[i][2, 2] == A.BORDER and explicit.inputs[i][3, 3] == A.PAD
    with pytest.raises(FileNotFoundError):
        task.load("train", root, download=False, task_ids=["deadbeef"])
    with pytest.raises(ValueError):
        task.load("val", root)


def test_ttt_variants(tmp_path):
    rng = np.random.default_rng(0)
    task = A.ARCTask()
    train, test = task.ttt_variants(_TASKS["00000002"], n_variants=6, rng=rng, task_name="t2")
    assert len(train) == 6 * 3 and len(test) == 6
    assert (
        set(train.meta["task_id"].tolist()) == set(range(6)) == set(test.meta["task_id"].tolist())
    )
    assert np.array_equal(train.meta["variant"], train.meta["task_id"])
    assert train.meta["dihedral_k"][0] == 0 and np.array_equal(
        train.meta["color_perm"][0], A.IDENTITY_PERM
    )
    assert (
        len(
            {
                (int(k), tuple(p))
                for k, p in zip(test.meta["dihedral_k"], test.meta["color_perm"], strict=True)
            }
        )
        == 6
    )
    assert train.given.all() and test.given.all() and test.meta["has_target"].all()
    assert test.meta["test_index"].tolist() == [0] * 6
    orig = np.array(_TASKS["00000002"]["test"][0]["output"])
    for i in range(len(test)):
        m = test.meta
        back = A.map_back(
            test.targets[i],
            m["scale"][i],
            m["offset"][i],
            m["out_shape"][i],
            m["dihedral_k"][i],
            m["color_perm"][i],
        )
        assert np.array_equal(back, orig)
        # the same transform was applied to the input of the pair
        inp = A.map_back(
            test.inputs[i],
            m["scale"][i],
            m["offset"][i],
            m["in_shape"][i],
            m["dihedral_k"][i],
            m["color_perm"][i],
        )
        assert np.array_equal(inp, np.array(_TASKS["00000002"]["test"][0]["input"]))
    # online augmentation composes on top of the variant transform and still maps back
    aug = A.ARCTask(online_dihedral=True, online_color=True).augment(train, rng)
    assert np.array_equal(aug.meta["base_dihedral_k"], train.meta["base_dihedral_k"])
    for i in range(len(aug)):
        m = aug.meta
        back = A.map_back(
            aug.targets[i],
            m["scale"][i],
            m["offset"][i],
            m["out_shape"][i],
            m["dihedral_k"][i],
            m["color_perm"][i],
        )
        assert np.array_equal(back, train.meta["pairs"][i][1])
    assert task.metrics(aug.targets, aug)["exact"].all()
    # hidden test outputs are allowed
    _, t3 = task.ttt_variants(_TASKS["00000003"], n_variants=2, rng=rng)
    assert t3.meta["has_target"].tolist() == [True, False, True, False]
    assert np.all(t3.targets[1] == A.PAD) and not t3.meta["loss_mask"][1].any()


def test_registry():
    from nca.tasks import get_task

    t = get_task("arc", n_tasks=5)
    assert isinstance(t, A.ARCTask) and t.vocab_size == 12 and t.output_tokens == tuple(range(12))
    assert len(t.token_names) == 12 and t.token_names[A.PAD] == "pad"
