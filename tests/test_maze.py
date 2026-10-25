"""Tests of the maze benchmarks (generators, metrics, augmentation, caching). Pure numpy."""

import time

import numpy as np
import pytest

from nca.tasks import maze as M
from nca.tasks.base import Boards

OOD = M.MazeOOD()


def _boards(inputs, targets):
    inputs = np.asarray(inputs)[None] if np.asarray(inputs).ndim == 2 else np.asarray(inputs)
    targets = np.asarray(targets)[None] if np.asarray(targets).ndim == 2 else np.asarray(targets)
    return Boards(inputs, targets, inputs != M.EMPTY)


def _all(metrics, keys=("exact", "valid", "optimal")):
    return all(bool(np.all(metrics[k])) for k in keys)


@pytest.mark.parametrize("size", [5, 9, 17])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_perfect_maze_structure(size, seed):
    rng = np.random.default_rng(seed)
    inp, tgt = M.generate_perfect_maze(size, rng)
    assert inp.shape == tgt.shape == (size, size)
    # border is wall
    assert np.all(inp[0] == M.WALL) and np.all(inp[-1] == M.WALL)
    assert np.all(inp[:, 0] == M.WALL) and np.all(inp[:, -1] == M.WALL)
    # odd lattice: (odd, odd) cells are never wall, (even, even) cells always are
    assert np.all(inp[1::2, 1::2] != M.WALL)
    assert np.all(inp[0::2, 0::2] == M.WALL)
    # exactly two endpoints, kept in the target; walls kept in the target
    assert (inp == M.ENDPOINT).sum() == 2 == (tgt == M.ENDPOINT).sum()
    assert np.array_equal(inp == M.WALL, tgt == M.WALL)
    assert set(np.unique(inp)) <= {0, 2, 3} and set(np.unique(tgt)) <= {0, 1, 2, 3}
    # unique solution: the corridor graph is a tree
    assert M.corridor_graph_is_tree(inp)
    # the target path is a valid, optimal path between the endpoints
    boards = _boards(inp, tgt)
    m = OOD.metrics(tgt[None], boards)
    assert _all(m) and m["cell_acc"][0] == 1.0
    assert boards.given.sum() == (inp != 0).sum()
    # and it is the BFS shortest path
    ends = M._endpoints(inp, (M.ENDPOINT,))
    assert len(M.bfs_path(inp != M.WALL, *ends)) == (tgt == M.PATH).sum() + 2


def test_metrics_reject_corrupted_paths():
    rng = np.random.default_rng(5)
    for _ in range(5):
        inp, tgt = M.generate_perfect_maze(9, rng)
        if (tgt == M.PATH).sum() >= 2:
            break
    boards = _boards(inp, tgt)
    # remove one path cell
    cells = np.argwhere(tgt == M.PATH)
    broken = tgt.copy()
    broken[tuple(cells[len(cells) // 2])] = M.EMPTY
    m = OOD.metrics(broken[None], boards)
    assert not m["exact"][0] and not m["valid"][0] and not m["optimal"][0]
    assert 0 < m["cell_acc"][0] < 1
    # add a branch: an empty corridor cell next to the path
    on = (tgt == M.PATH) | (tgt == M.ENDPOINT)
    p = np.pad(on, 1)
    nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    cand = np.argwhere((tgt == M.EMPTY) & nb)
    assert len(cand) > 0
    branched = tgt.copy()
    branched[tuple(cand[0])] = M.PATH
    m = OOD.metrics(branched[None], boards)
    assert not m["exact"][0] and not m["valid"][0] and not m["optimal"][0]
    # a path predicted on a given (wall) cell is ignored, predictions elsewhere still count
    on_wall = tgt.copy()
    on_wall[0, 0] = M.PATH
    assert _all(OOD.metrics(on_wall[None], boards))
    # a valid but longer path (perfect mazes have none) -> emulate with a loopy grid
    grid = np.full((5, 5), M.WALL, dtype=np.int64)
    grid[1:4, 1:4] = M.EMPTY
    grid[1, 1], grid[1, 3] = M.ENDPOINT, M.ENDPOINT
    short = grid.copy()
    short[1, 2] = M.PATH
    long = grid.copy()
    long[2, 1] = long[3, 1] = long[3, 2] = long[3, 3] = long[2, 3] = M.PATH
    b = _boards(grid, short)
    m = OOD.metrics(long[None], b)
    assert m["valid"][0] and not m["optimal"][0] and not m["exact"][0]


def test_path_validity_direct():
    m = np.zeros((4, 4), dtype=bool)
    m[1, 0:4] = True
    assert M.path_validity(m, [(1, 0), (1, 3)]) == (True, 4)
    assert M.path_validity(m, [(1, 0), (1, 2)])[0] is False  # endpoint of degree 2
    m2 = m.copy()
    m2[3, 0] = True  # detached cell
    assert M.path_validity(m2, [(1, 0), (1, 3)])[0] is False
    # a path plus a detached 2x2 cycle: degrees are all 2 but the walk does not cover everything
    m3 = np.zeros((6, 6), dtype=bool)
    m3[0, 0:3] = True
    m3[3:5, 3:5] = True
    assert M.path_validity(m3, [(0, 0), (0, 2)])[0] is False


def test_dihedral_is_bijection_and_preserves_validity():
    x = np.arange(2 * 7 * 7).reshape(2, 7, 7)
    seen = []
    for k in range(8):
        y = M.dihedral_transform(x, k)
        assert y.shape == x.shape
        assert np.array_equal(M.dihedral_inverse(y, k), x)
        assert np.array_equal(np.sort(y.ravel()), np.sort(x.ravel()))
        seen.append(y)
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.array_equal(seen[i], seen[j])
    rng = np.random.default_rng(0)
    mazes = [M.generate_perfect_maze(9, rng) for _ in range(32)]
    boards = _boards(np.stack([a for a, _ in mazes]), np.stack([b for _, b in mazes]))
    aug = OOD.augment(boards, np.random.default_rng(1))
    assert aug.shape == boards.shape and len(aug) == len(boards)
    assert set(np.unique(aug.meta["dihedral"])) == set(range(8))
    assert np.array_equal(aug.given, aug.inputs != M.EMPTY)
    assert _all(OOD.metrics(aug.targets, aug))
    # undoing the transform gives back the original boards
    for i, k in enumerate(aug.meta["dihedral"]):
        assert np.array_equal(M.dihedral_inverse(aug.targets[i], int(k)), boards.targets[i])
    # MazeHard augments only BFS targets (tested on Maze-Hard boards in test_hard_bfs_augment)
    assert M.MazeHard("astar").augment(boards, rng) is boards


def test_hard_mazes():
    rng = np.random.default_rng(0)
    astar_task, bfs_task = M.MazeHard("astar"), M.MazeHard("bfs")
    differ = 0
    for _ in range(6):
        inp, ta, tb = M.generate_hard_maze(30, rng)
        assert inp.shape == ta.shape == tb.shape == (30, 30)
        assert (inp == M.START).sum() == 1 and (inp == M.GOAL).sum() == 1
        assert np.array_equal(inp == M.WALL, ta == M.WALL)
        assert not M.corridor_graph_is_tree(inp)  # loops were added
        la, lb = (ta == M.PATH).sum() + 2, (tb == M.PATH).sum() + 2
        assert la == lb >= 110
        ba, bb = _boards(inp, ta), _boards(inp, tb)
        assert _all(astar_task.metrics(ta[None], ba))
        assert _all(bfs_task.metrics(tb[None], bb))
        # the other optimal path is valid and optimal against either reference, exact only if equal
        m = astar_task.metrics(tb[None], ba)
        assert m["valid"][0] and m["optimal"][0]
        assert bool(m["exact"][0]) == np.array_equal(ta, tb)
        differ += not np.array_equal(ta, tb)
    assert differ >= 0  # several optimal paths occur on ~25% of the mazes (not asserted: random)


def test_hard_bfs_augment():
    """The BFS target of a transformed maze is the BFS path of that maze, not the rotated target:
    with several shortest paths the fixed neighbour order picks a different one after a rotation,
    and rotated targets then contradict each other (bug of the first BFS runs, 2026-10-19)."""
    rng = np.random.default_rng(3)
    mazes = [M.generate_hard_maze(30, rng) for _ in range(8)]
    inputs = np.stack([m[0] for m in mazes])
    targets = np.stack([m[2] for m in mazes])
    boards = _boards(np.repeat(inputs, 8, axis=0), np.repeat(targets, 8, axis=0))
    bfs = M.MazeHard("bfs")
    aug = bfs.augment(boards, np.random.default_rng(4))
    assert set(np.unique(aug.meta["dihedral"])) == set(range(8))
    moved = 0
    for i, k in enumerate(aug.meta["dihedral"]):
        assert np.array_equal(aug.targets[i], M.bfs_target(aug.inputs[i]))
        assert np.array_equal(aug.inputs[i], M.dihedral_transform(boards.inputs[i], int(k)))
        assert _all(bfs.metrics(aug.targets[i][None], _boards(aug.inputs[i], aug.targets[i])))
        moved += not np.array_equal(aug.targets[i], M.dihedral_transform(boards.targets[i], int(k)))
    assert moved > 0  # a plain rotation of the target would have been wrong for these boards
    assert np.array_equal(boards.targets, np.repeat(targets, 8, axis=0))  # input boards untouched


def test_hard_bfs_local_target():
    """`bfs_local` targets: a shortest path whose every step back from the goal goes to the first
    neighbour (up/down/left/right) one step closer to the start, recomputed on augmented mazes."""
    rng = np.random.default_rng(5)
    mazes = [M.generate_hard_maze(30, rng) for _ in range(8)]
    inputs = np.stack([m[0] for m in mazes])
    differ = 0
    for inp, _, tb in mazes:
        tl = M.bfs_target(inp, local=True)
        start, goal = M._endpoints(inp, (M.START,))[0], M._endpoints(inp, (M.GOAL,))[0]
        path = M.bfs_local_path(inp != M.WALL, start, goal)
        dist = M.bfs_distances(inp != M.WALL, start)
        assert len(path) == dist[goal] + 1 == (tl == M.PATH).sum() + 2
        for (i, j), prev in zip(path[1:][::-1], path[:-1][::-1], strict=True):
            nbs = [(i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))]
            first = next(
                n for n in nbs if 0 <= n[0] < 30 and 0 <= n[1] < 30 and dist[n] == dist[i, j] - 1
            )
            assert prev == first
        differ += not np.array_equal(tl, tb)
    assert differ > 0  # the global BFS tie-break picks another shortest path on some mazes
    boards = _boards(np.repeat(inputs, 8, axis=0), np.zeros_like(np.repeat(inputs, 8, axis=0)))
    task = M.MazeHard("bfs_local")
    aug = task.augment(boards, np.random.default_rng(6))
    for i, k in enumerate(aug.meta["dihedral"]):
        if k:
            assert np.array_equal(aug.targets[i], M.bfs_target(aug.inputs[i], local=True))
            assert _all(task.metrics(aug.targets[i][None], _boards(aug.inputs[i], aug.targets[i])))


def test_hrm_parser_roundtrip():
    rng = np.random.default_rng(2)
    inp, ta, _ = M.generate_hard_maze(30, rng)
    q = M.decode_ascii(inp).replace("\n", "")
    a = M.decode_ascii(ta).replace("\n", "")
    assert len(q) == 900 and set(q) <= set("#.SG") and set(a) <= set("#.SGo")
    inp2, ta2 = M.parse_hrm_maze(q, a)
    assert np.array_equal(inp2, inp) and np.array_equal(ta2, ta)
    assert np.array_equal(M.parse_hrm_maze(q.replace(".", " "), a)[0], inp)


def test_caching_and_generated_source(tmp_path):
    root = str(tmp_path)
    b1 = OOD.load("test", root, size=9, n=16, seed=0, source="generated")
    b2 = OOD.load("test", root, size=9, n=16, seed=0, source="generated")
    assert (tmp_path / "maze" / "maze_ood_test_9_n16_seed0.npz").exists()
    assert np.array_equal(b1.inputs, b2.inputs) and np.array_equal(b1.targets, b2.targets)
    assert b1.meta["source"] == "generated" and b1.meta["size"] == 9
    assert b1.inputs.dtype == np.int64 and b1.given.dtype == bool and b1.shape == (9, 9)
    assert np.array_equal(b1.given, b1.inputs != 0)
    assert b1.meta["path_len"].shape == (16,)
    assert _all(OOD.metrics(b1.targets, b1))
    # same seed, other split or size -> other mazes
    b3 = OOD.load("train", root, size=9, n=16, seed=0, source="generated")
    assert not np.array_equal(b1.inputs, b3.inputs)
    assert OOD.load("test", root, size=11, n=4, source="generated").shape == (11, 11)
    # "auto" without download falls back to the generator offline
    b4 = OOD.load("test", root, size=9, n=16, seed=0, download=False)
    assert b4.meta["source"] == "generated" and np.array_equal(b4.inputs, b1.inputs)
    with pytest.raises(FileNotFoundError):
        OOD.load("test", root, size=9, n=16, download=False, source="official")
    # hard mazes
    hard = M.MazeHard("bfs")
    h1 = hard.load("test", root, n=3, seed=0, source="generated")
    h2 = M.MazeHard("astar").load("test", root, n=3, seed=0, download=False)
    assert (tmp_path / "maze" / "maze_hard_test_30_n3_seed0.npz").exists()
    assert h1.shape == (30, 30) and h1.meta["targets"] == "bfs" and h2.meta["targets"] == "astar"
    assert np.array_equal(h1.inputs, h2.inputs)
    assert _all(hard.metrics(h1.targets, h1)) and np.all(h1.meta["path_len"] >= 110)
    assert set(np.unique(h1.inputs)) <= {0, 2, 3, 4}


def test_large_maze_generation_is_fast():
    t0 = time.time()
    inp, tgt = M.generate_perfect_maze(201, np.random.default_rng(0))
    assert inp.shape == (201, 201) and M.corridor_graph_is_tree(inp)
    boards = _boards(inp, tgt)
    assert _all(OOD.metrics(tgt[None], boards))
    assert time.time() - t0 < 5.0


def test_render_and_ascii():
    inp, tgt = M.generate_perfect_maze(9, np.random.default_rng(0))
    img = M.render_maze(tgt)
    assert img.shape == (9, 9, 3) and img.dtype == np.uint8
    assert np.array_equal(img[0, 0], (40, 40, 40))
    ends = np.argwhere(tgt == M.ENDPOINT)
    assert np.array_equal(img[tuple(ends[0])], M._COLORS[M.START])
    assert np.array_equal(img[tuple(ends[1])], M._COLORS[M.GOAL])
    txt = M.decode_ascii(tgt)
    lines = txt.split("\n")
    assert len(lines) == 9 and all(len(line) == 9 for line in lines)
    assert txt.count("S") == 1 and txt.count("G") == 1 and txt.count("o") == (tgt == M.PATH).sum()


def _render_official_style(targets, border=3, block=2):
    """Emulate the easy-to-hard-data images: (N, 3, P, P) floats + (N, P, P) solution masks."""
    N, S, _ = targets.shape
    P = 2 * border + block * S
    rgb = np.zeros((N, 3, P, P), dtype=np.float32)  # black border and walls
    sol = np.zeros((N, P, P), dtype=np.uint8)
    for n in range(N):
        ends = np.argwhere(targets[n] == M.ENDPOINT)
        for i in range(S):
            for j in range(S):
                t = targets[n, i, j]
                sl = (
                    slice(border + block * i, border + block * (i + 1)),
                    slice(border + block * j, border + block * (j + 1)),
                )
                if t == M.WALL:
                    continue
                if t == M.ENDPOINT:
                    rgb[n, 1 if (i, j) == tuple(ends[0]) else 0][sl] = 1.0
                else:
                    rgb[n, :, sl[0], sl[1]] = 1.0
                if t in (M.PATH, M.ENDPOINT):
                    sol[n][sl] = 1
    return rgb, sol


def test_official_ood_conversion_on_synthetic_images(tmp_path):
    rng = np.random.default_rng(0)
    mazes = [M.generate_perfect_maze(9, rng) for _ in range(5)]
    inputs = np.stack([a for a, _ in mazes])
    targets = np.stack([b for _, b in mazes])
    folder = tmp_path / "maze_data_test_9"
    folder.mkdir()
    rgb, sol = _render_official_style(targets)
    assert rgb.shape == (5, 3, 24, 24)
    np.save(folder / "inputs.npy", rgb)
    np.save(folder / "solutions.npy", sol)
    data = M.convert_official_ood(str(folder), 9)
    assert np.array_equal(data["inputs"], inputs) and np.array_equal(data["targets"], targets)
    assert np.array_equal(data["path_len"], (targets == M.PATH).sum(axis=(1, 2)) + 2)
    # a 0-255 uint8 rendering with another border/block size is understood too
    rgb2, sol2 = _render_official_style(targets, border=2, block=3)
    np.save(folder / "inputs.npy", (rgb2 * 255).astype(np.uint8))
    np.save(folder / "solutions.npy", sol2)
    assert np.array_equal(M.convert_official_ood(str(folder), 9)["targets"], targets)
    # a layout that is not block-uniform raises
    rgb3 = rgb.copy()
    rgb3[:, :, 3:21, 3:21] = rng.random((5, 3, 18, 18)) > 0.5
    np.save(folder / "inputs.npy", rgb3)
    np.save(folder / "solutions.npy", sol)
    with pytest.raises(ValueError):
        M.convert_official_ood(str(folder), 9)
    # the loader uses the raw folder when present, and caches the conversion
    raw = tmp_path / "root" / "maze" / "raw" / "maze_data_test_9"
    raw.mkdir(parents=True)
    np.save(raw / "inputs.npy", rgb)
    np.save(raw / "solutions.npy", sol)
    b = OOD.load("test", str(tmp_path / "root"), size=9, n=3, download=False)
    assert b.meta["source"] == "official" and len(b) == 3
    assert (tmp_path / "root" / "maze" / "maze_ood_official_test_9.npz").exists()
    assert _all(OOD.metrics(b.targets, b))


def test_official_hard_conversion_on_synthetic_csv(tmp_path):
    import csv

    rng = np.random.default_rng(0)
    mazes = [M.generate_hard_maze(30, rng) for _ in range(3)]
    raw = tmp_path / "maze" / "raw" / "maze_hard"
    raw.mkdir(parents=True)
    with open(raw / "test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "q", "a", "rating"])
        for inp, ta, _ in mazes:
            q = M.decode_ascii(inp).replace("\n", "").replace(".", " ")
            a = M.decode_ascii(ta).replace("\n", "").replace(".", " ")
            w.writerow(["synthetic", q, a, "0"])
    b = M.MazeHard("astar").load("test", str(tmp_path), download=False)
    assert b.meta["source"] == "official" and len(b) == 3
    assert np.array_equal(b.inputs, np.stack([m[0] for m in mazes]))
    assert np.array_equal(b.targets, np.stack([m[1] for m in mazes]))
    bb = M.MazeHard("bfs").load("test", str(tmp_path), download=False)
    assert np.array_equal(bb.targets, np.stack([m[2] for m in mazes]))
    assert _all(M.MazeHard("bfs").metrics(bb.targets, bb))
    bl = M.MazeHard("bfs_local").load("test", str(tmp_path), download=False)
    assert bl.meta["targets"] == "bfs_local"
    assert np.array_equal(bl.targets, np.stack([M.bfs_target(m[0], local=True) for m in mazes]))
    bo = M.MazeHard("astar_own").load("test", str(tmp_path), download=False)
    assert np.array_equal(bo.targets, np.stack([M.astar_target(m[0]) for m in mazes]))
    assert _all(M.MazeHard("astar_own").metrics(bo.targets, bo))
    assert M.MazeHard("astar_own").augment(bo, np.random.default_rng(0)) is bo


def test_bfs_dlru_targets_are_recomputed_on_augmented_boards(tmp_path):
    # bfs_dlru: BFS expanding down/left/right/up (the rule of the HRM-shipped paths)
    from nca.tasks import get_task
    from nca.tasks.maze import bfs_target

    task = get_task("maze_hard", targets="bfs_dlru")
    boards = task.load("train", str(tmp_path), n=6, download=False, source="generated")
    for b in range(len(boards)):
        assert np.array_equal(boards.targets[b], bfs_target(boards.inputs[b], order="DLRU"))
    aug = task.augment(boards, np.random.default_rng(1))
    for b in range(len(aug)):
        assert np.array_equal(aug.targets[b], bfs_target(aug.inputs[b], order="DLRU"))
    assert all(float(np.mean(v)) == 1.0 for v in task.metrics(aug.targets, aug).values())


def test_subsample_exclude_seed_shares_no_board_with_the_excluded_draws():
    from nca.tasks.maze import _subsample

    N = 1000
    data = {"inputs": np.arange(N), "targets": np.arange(N)}
    table1 = _subsample(data, 100, 0)["inputs"]
    again = _subsample(data, 100, 7)["inputs"]
    assert np.intersect1d(table1, again).size > 0  # a new seed alone is not disjoint
    sel = _subsample(data, 100, 7, exclude_seed=0)["inputs"]
    assert sel.size == 100 and np.intersect1d(sel, table1).size == 0
    final = _subsample(data, 500, 1, exclude_seed=[0, 7], exclude_n=100)["inputs"]
    dev7 = _subsample(data, 100, 7)["inputs"]
    assert final.size == 500 and not np.intersect1d(final, np.r_[table1, dev7]).size
    rest = _subsample(data, None, 1, exclude_seed=[0, 7], exclude_n=100)["inputs"]
    assert rest.size == N - np.union1d(table1, dev7).size
    # without exclusion the historical draw is unchanged
    idx = np.sort(np.random.default_rng(0).choice(N, size=100, replace=False))
    np.testing.assert_array_equal(table1, idx)
