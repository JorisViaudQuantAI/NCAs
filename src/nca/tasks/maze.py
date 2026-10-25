"""Maze benchmarks: Maze-OOD (Bansal et al., 2022) and Maze-Hard (Wang et al., 2025, HRM).

Both benchmarks are grids of discrete tokens (see `base.py`). The NCA receives the walls and the
two endpoints and must mark the solution path on every other cell.

Token vocabularies
------------------
* Maze-OOD, V = 4: ``0 empty, 1 path, 2 wall, 3 endpoint`` (both endpoints share token 3).
* Maze-Hard, V = 5: ``0 empty, 1 path, 2 wall, 3 start, 4 goal``.

`inputs` = walls + endpoints (0 elsewhere), `given = inputs != 0`, `targets` = inputs with the
solution path written as token 1 on the corridor cells it crosses.

Data sources (``source="auto"`` tries them in this order and prints which one it used)
-------------------------------------------------------------------------------------
1. A cached ``.npz`` under ``<root>/maze/``.
2. The official data.
   * Maze-OOD: ``https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_{split}_{size}.tar.gz``
     (python package ``easy-to-hard-data``). Each archive holds ``inputs.npy`` of shape
     ``(N, 3, P, P)``, an RGB rendering with a pixel border and square pixel blocks per maze cell
     (expected ``P = 2 * size + 6``: 3-pixel border, 2x2 pixels per cell; walls black, corridors
     white, endpoints green/red), and ``solutions.npy`` ``(N, P, P)``, a binary mask of the path.
     The border width and block size are *inferred* from the images and checked (every block must
     be uniform, every converted maze must have a valid path); a layout we do not understand raises.
     Train = 50K mazes of size 9, test = 10K mazes per size in 9..37 (step 2), 10K of size 59,
     1K of size 201. We subsample the test sets (default 1000 per size) to keep evaluation cheap.
   * Maze-Hard: Hugging Face dataset ``sapientinc/Maze-30x30-Hard-1k`` (``train.csv`` /
     ``test.csv``, columns ``source,q,a,rating``; ``q`` and ``a`` are 900-character strings over
     the charset ``"# SGo"``: ``#`` wall, space (or ``.``) empty, ``S`` start, ``G`` goal and ``o``
     the A* path in the answer). 1000 train / 1000 test, path length >= 110, mazes often admit
     several optimal paths. BFS targets are recomputed from the puzzle with our own BFS.
3. The local generators (always available, seeded, reproducible):
   * `generate_perfect_maze`: randomized iterative DFS on a k x k lattice (k = (size-1)//2),
     rendered on the (2k+1) x (2k+1) grid; the corridor graph is a tree, so the path between the
     two random endpoints is unique (as in Maze-OOD).
   * `generate_hard_maze`: a 31 x 31 perfect maze with a random fraction of its interior walls
     removed (loops appear), cropped to 30 x 30, endpoints chosen so that the shortest path has at
     least 110 cells; targets by A* (Manhattan, deterministic tie-break) and by BFS. This only
     *approximates* the HRM data (whose generator we could not reproduce exactly).

Cache files (``<root>/maze/``)
------------------------------
* generated: ``maze_ood_{split}_{size}_n{n}_seed{seed}.npz``,
  ``maze_hard_{split}_30_n{n}_seed{seed}.npz`` (keys ``inputs``, ``targets`` or
  ``targets_astar`` + ``targets_bfs``, ``path_len``).
* official (full conversion, subsampled at load time): ``maze_ood_official_{split}_{size}.npz``,
  ``maze_hard_official_{split}_30.npz``. Raw downloads are kept under ``<root>/maze/raw/``.

Everything here is pure numpy + stdlib so that the data pipeline works without torch.
"""

from __future__ import annotations

import csv
import heapq
import os
import shutil
import tarfile
import time
import urllib.request
from collections import deque

import numpy as np

from nca.tasks.base import Boards, Task

# ----------------------------------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------------------------------

EMPTY, PATH, WALL, ENDPOINT = 0, 1, 2, 3  # Maze-OOD
START, GOAL = 3, 4  # Maze-Hard (3 = start, 4 = goal)

OOD_URL = "https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_{split}_{size}.tar.gz"
HARD_URL = "https://huggingface.co/datasets/sapientinc/Maze-30x30-Hard-1k/resolve/main/{split}.csv"
OFFICIAL_OOD_SIZES = tuple(range(9, 39, 2)) + (59, 201)

# neighbour order used by every search here: up, down, left, right
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _log(msg: str) -> None:
    print(f"[maze] {msg}", flush=True)


def maze_dir(root: str) -> str:
    d = os.path.join(root, "maze")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------
# Shortest paths (pure python on flat indices of a wall-padded grid; fast enough for 201 x 201)
# ----------------------------------------------------------------------------------------------


def _padded(passable: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad with one ring of blocked cells so neighbour lookups need no bounds checks."""
    p = np.pad(passable.astype(bool), 1, constant_values=False)
    return p.reshape(-1), p.shape[1]


def bfs_distances(passable: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """BFS over the 4-neighbourhood from `src`; -1 where unreachable. `passable` is bool (H, W)."""
    H, W = passable.shape
    flat, Wp = _padded(passable)
    steps = (-Wp, Wp, -1, 1)
    dist = np.full(flat.shape[0], -1, dtype=np.int64)
    s = (src[0] + 1) * Wp + src[1] + 1
    if not flat[s]:
        raise ValueError("source is not a passable cell")
    dist[s] = 0
    q = deque([s])
    passable_list = flat.tolist()
    d = dist  # numpy writes are fine here
    while q:
        c = q.popleft()
        dc = d[c] + 1
        for st in steps:
            nb = c + st
            if passable_list[nb] and d[nb] < 0:
                d[nb] = dc
                q.append(nb)
    return dist.reshape(H + 2, Wp)[1:-1, 1:-1]


def bfs_path(
    passable: np.ndarray, src: tuple[int, int], dst: tuple[int, int], order: str = "UDLR"
) -> list | None:
    """Shortest path (list of (i, j), endpoints included) by BFS; neighbours are expanded in
    `order` (letters U, D, L, R). With order "DLRU" this reproduces the paths shipped with the
    HRM Maze-Hard data on all 1000 training mazes (scripts/compare_hard_targets.py).

    Returns None when `dst` is unreachable.
    """
    flat, Wp = _padded(passable)
    step_of = {"U": -Wp, "D": Wp, "L": -1, "R": 1}
    steps = tuple(step_of[c] for c in order)
    passable_list = flat.tolist()
    s = (src[0] + 1) * Wp + src[1] + 1
    t = (dst[0] + 1) * Wp + dst[1] + 1
    if not (passable_list[s] and passable_list[t]):
        return None
    parent = [-1] * len(passable_list)
    parent[s] = s
    q = deque([s])
    found = s == t
    while q and not found:
        c = q.popleft()
        for st in steps:
            nb = c + st
            if passable_list[nb] and parent[nb] < 0:
                parent[nb] = c
                if nb == t:
                    found = True
                    break
                q.append(nb)
    if not found:
        return None
    return _unwind(parent, s, t, Wp)


def bfs_local_path(passable: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> list | None:
    """Shortest path chosen by a local rule: BFS distances from `src`, then walk back from `dst`,
    each step to the first neighbour one step closer to `src` in the order up/down/left/right.

    Unlike `bfs_path`, whose parent pointers depend on the global discovery order, each cell's
    choice here only depends on the distances of its four neighbours. Returns None when `dst`
    is unreachable.
    """
    dist = bfs_distances(passable, src)
    if dist[dst] < 0:
        return None
    H, W = passable.shape
    path = [dst]
    i, j = dst
    while dist[i, j] > 0:
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and dist[a, b] == dist[i, j] - 1:
                i, j = a, b
                break
        path.append((i, j))
    return path[::-1]


def astar_path(passable: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> list | None:
    """Shortest path by A* with the Manhattan heuristic.

    Deterministic tie-break: smallest f, then smallest h, then earliest insertion; neighbours are
    expanded in the order up, down, left, right. Returns None when unreachable.
    """
    flat, Wp = _padded(passable)
    steps = (-Wp, Wp, -1, 1)
    passable_list = flat.tolist()
    s = (src[0] + 1) * Wp + src[1] + 1
    t = (dst[0] + 1) * Wp + dst[1] + 1
    if not (passable_list[s] and passable_list[t]):
        return None
    ti, tj = divmod(t, Wp)

    def h(c: int) -> int:
        i, j = divmod(c, Wp)
        return abs(i - ti) + abs(j - tj)

    g = {s: 0}
    parent = [-1] * len(passable_list)
    parent[s] = s
    closed = bytearray(len(passable_list))
    counter = 0
    heap = [(h(s), h(s), counter, s)]
    while heap:
        f, _, _, c = heapq.heappop(heap)
        if closed[c]:
            continue
        if c == t:
            return _unwind(parent, s, t, Wp)
        closed[c] = 1
        gc = g[c] + 1
        for st in steps:
            nb = c + st
            if not passable_list[nb] or closed[nb]:
                continue
            if nb not in g or gc < g[nb]:
                g[nb] = gc
                parent[nb] = c
                hn = h(nb)
                counter += 1
                heapq.heappush(heap, (gc + hn, hn, counter, nb))
    return None


def _unwind(parent: list, s: int, t: int, Wp: int) -> list:
    out = []
    c = t
    while True:
        i, j = divmod(c, Wp)
        out.append((i - 1, j - 1))
        if c == s:
            break
        c = parent[c]
    out.reverse()
    return out


# ----------------------------------------------------------------------------------------------
# Path validity (used by the metrics and to check converted official data)
# ----------------------------------------------------------------------------------------------


def path_validity(path_mask: np.ndarray, endpoints: list[tuple[int, int]]) -> tuple[bool, int]:
    """Is `path_mask` (bool (H, W), endpoints included) one simple 4-connected path between the
    two endpoints? Returns (valid, number of cells of the path).

    Conditions: both endpoints are in the mask with exactly one path-neighbour, every other path
    cell has exactly two, and walking from one endpoint reaches the other after visiting every
    cell of the mask (so no detached cycles).
    """
    m = np.asarray(path_mask, dtype=bool)
    total = int(m.sum())
    if len(endpoints) != 2:
        return False, total
    (ai, aj), (bi, bj) = endpoints
    if not (m[ai, aj] and m[bi, bj]) or (ai, aj) == (bi, bj):
        return False, total
    p = np.pad(m, 1).astype(np.int8)
    deg = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    deg_on = deg[m]
    if deg[ai, aj] != 1 or deg[bi, bj] != 1:
        return False, total
    # exactly two cells of degree 1 (the endpoints), everything else of degree 2
    if (deg_on == 1).sum() != 2 or np.any(deg_on > 2) or np.any(deg_on == 0):
        return False, total
    # walk from a to b; with the degree conditions the walk is forced
    H, W = m.shape
    flat = p.reshape(-1).tolist()
    Wp = W + 2
    steps = (-Wp, Wp, -1, 1)
    prev, cur = -1, (ai + 1) * Wp + aj + 1
    target = (bi + 1) * Wp + bj + 1
    visited = 1
    while cur != target:
        nxt = -1
        for st in steps:
            nb = cur + st
            if flat[nb] and nb != prev:
                nxt = nb
                break
        if nxt < 0:  # cannot happen given the degree checks, kept for safety
            return False, total
        prev, cur = cur, nxt
        visited += 1
        if visited > total:
            return False, total
    return visited == total, total


def corridor_graph_is_tree(tokens: np.ndarray) -> bool:
    """True when the non-wall cells form a tree (connected, |E| = |V| - 1): every pair of
    corridor cells is then joined by exactly one simple path (the Maze-OOD property)."""
    free = np.asarray(tokens) != WALL
    V = int(free.sum())
    if V == 0:
        return False
    E = int((free[:-1] & free[1:]).sum() + (free[:, :-1] & free[:, 1:]).sum())
    if E != V - 1:
        return False
    src = tuple(int(x) for x in np.argwhere(free)[0])
    return int((bfs_distances(free, src) >= 0).sum()) == V


# ----------------------------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------------------------


def _carve_tree(k: int, rng: np.random.Generator) -> np.ndarray:
    """Perfect maze on a k x k lattice by iterative randomized DFS. Returns uint8 (2k+1, 2k+1)."""
    n = k * k
    grid = np.full((2 * k + 1, 2 * k + 1), WALL, dtype=np.uint8)
    grid[1::2, 1::2] = EMPTY
    if n == 1:
        return grid
    visited = bytearray(n)
    start = int(rng.integers(n))
    visited[start] = 1
    stack = [start]
    rand = rng.random(n).tolist()  # one draw per carved edge (n - 1 edges)
    t = 0
    while stack:
        c = stack[-1]
        i, j = divmod(c, k)
        nbrs = []
        if i > 0 and not visited[c - k]:
            nbrs.append(c - k)
        if i < k - 1 and not visited[c + k]:
            nbrs.append(c + k)
        if j > 0 and not visited[c - 1]:
            nbrs.append(c - 1)
        if j < k - 1 and not visited[c + 1]:
            nbrs.append(c + 1)
        if not nbrs:
            stack.pop()
            continue
        nxt = nbrs[int(rand[t] * len(nbrs))]
        t += 1
        visited[nxt] = 1
        i2, j2 = divmod(nxt, k)
        grid[i + i2 + 1, j + j2 + 1] = EMPTY
        stack.append(nxt)
    return grid


def _write_path(grid: np.ndarray, path: list, endpoint_tokens: tuple[int, int]) -> np.ndarray:
    target = grid.copy()
    for i, j in path[1:-1]:
        target[i, j] = PATH
    (ai, aj), (bi, bj) = path[0], path[-1]
    target[ai, aj] = endpoint_tokens[0]
    target[bi, bj] = endpoint_tokens[1]
    return target


def generate_perfect_maze(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One Maze-OOD style maze of odd `size` >= 5.

    Returns ``(inputs, target)``, both uint8 ``(size, size)`` with tokens 0/1/2/3. The corridor
    graph is a spanning tree of the lattice (odd coordinates are lattice cells, the border is all
    wall), the two endpoints are distinct random corridor cells and the path between them is
    unique (found by BFS).
    """
    if size < 5 or size % 2 == 0:
        raise ValueError(f"size must be odd and >= 5, got {size}")
    k = (size - 1) // 2
    grid = _carve_tree(k, rng)
    free = np.argwhere(grid == EMPTY)
    a, b = rng.choice(len(free), size=2, replace=False)
    src, dst = tuple(int(x) for x in free[a]), tuple(int(x) for x in free[b])
    path = bfs_path(grid != WALL, src, dst)
    assert path is not None
    inputs = grid.copy()
    inputs[src] = ENDPOINT
    inputs[dst] = ENDPOINT
    return inputs, _write_path(grid, path, (ENDPOINT, ENDPOINT))


def generate_hard_maze(
    size: int = 30,
    rng: np.random.Generator | None = None,
    min_path_len: int = 110,
    extra_openings: float = 0.07,
    max_tries: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Maze-Hard style maze. Returns ``(inputs, target_astar, target_bfs)``, uint8 (size, size).

    A perfect ``(size+1) x (size+1)`` maze is generated, a fraction `extra_openings` of its
    *removable* interior walls (walls separating two lattice cells) is knocked out so that loops
    appear, and the grid is cropped to ``size x size`` by dropping the last row and column. The
    start is a random corridor cell and the goal a random corridor cell at BFS distance
    >= ``min_path_len - 1`` (so the shortest path has >= `min_path_len` cells, endpoints
    included); a maze with no such pair is discarded and a new one generated. The A* target uses
    the Manhattan heuristic with a deterministic tie-break, the BFS target the neighbour order
    up/down/left/right; when several optimal paths exist the two may differ.

    This approximates the HRM data (Wang et al., 2025) whose exact generator is not reproduced.
    """
    rng = np.random.default_rng() if rng is None else rng
    if size % 2 != 0:
        raise ValueError("size must be even (a (size+1) x (size+1) perfect maze is cropped)")
    k = size // 2
    for _ in range(max_tries):
        grid = _carve_tree(k, rng)
        # removable walls: interior cells between two lattice cells, still wall
        cand = []
        for i in range(1, 2 * k):
            for j in range(1, 2 * k):
                if (i + j) % 2 == 1 and grid[i, j] == WALL:
                    cand.append((i, j))
        n_open = int(round(extra_openings * len(cand)))
        if n_open > 0:
            for idx in rng.choice(len(cand), size=n_open, replace=False):
                grid[cand[idx]] = EMPTY
        grid = np.ascontiguousarray(grid[:size, :size])
        passable = grid != WALL
        free = np.argwhere(passable)
        for _ in range(5):
            src = tuple(int(x) for x in free[rng.integers(len(free))])
            dist = bfs_distances(passable, src)
            far = np.argwhere(dist >= min_path_len - 1)
            if len(far) == 0:
                continue
            dst = tuple(int(x) for x in far[rng.integers(len(far))])
            p_astar = astar_path(passable, src, dst)
            p_bfs = bfs_path(passable, src, dst)
            assert p_astar is not None and p_bfs is not None
            assert len(p_astar) == len(p_bfs) == int(dist[dst]) + 1
            inputs = grid.copy()
            inputs[src] = START
            inputs[dst] = GOAL
            return (
                inputs,
                _write_path(grid, p_astar, (START, GOAL)),
                _write_path(grid, p_bfs, (START, GOAL)),
            )
    raise RuntimeError("could not generate a hard maze satisfying min_path_len")


def _dataset_rng(kind: str, split: str, size: int, seed: int) -> np.random.Generator:
    """Independent streams for every (kind, split, size, seed) so that train and test differ."""
    kind_id = {"ood": 1, "hard": 2}[kind]
    split_id = {"train": 1, "val": 2, "test": 3}.get(split, 9)
    return np.random.default_rng([seed, kind_id, split_id, size])


def make_dataset(
    kind: str, split: str, n: int, size: int, seed: int = 0, verbose: bool = True, **gen_kwargs
) -> dict[str, np.ndarray]:
    """Generate `n` mazes with the local generator. `kind` is "ood" or "hard".

    Returns a dict of arrays: ``inputs`` (n, size, size) uint8, ``path_len`` (n,) and
    ``targets`` (ood) or ``targets_astar`` + ``targets_bfs`` (hard).
    """
    if kind not in ("ood", "hard"):
        raise ValueError(f"unknown kind {kind!r}")
    rng = _dataset_rng(kind, split, size, seed)
    inputs = np.empty((n, size, size), dtype=np.uint8)
    path_len = np.empty(n, dtype=np.int64)
    out: dict[str, np.ndarray] = {"inputs": inputs, "path_len": path_len}
    if kind == "ood":
        out["targets"] = np.empty_like(inputs)
    else:
        out["targets_astar"] = np.empty_like(inputs)
        out["targets_bfs"] = np.empty_like(inputs)
    t0 = time.time()
    report_every = max(1, n // 10) if n >= 5000 or size >= 59 else n + 1
    for i in range(n):
        if kind == "ood":
            inp, tgt = generate_perfect_maze(size, rng)
            out["targets"][i] = tgt
            path_len[i] = int((tgt == PATH).sum()) + 2
        else:
            inp, ta, tb = generate_hard_maze(size, rng, **gen_kwargs)
            out["targets_astar"][i] = ta
            out["targets_bfs"][i] = tb
            path_len[i] = int((ta == PATH).sum()) + 2
        inputs[i] = inp
        if verbose and (i + 1) % report_every == 0 and i + 1 < n:
            _log(f"generated {i + 1}/{n} {kind} mazes of size {size} ({time.time() - t0:.1f}s)")
    if verbose:
        _log(f"generated {n} {kind} mazes of size {size} in {time.time() - t0:.1f}s")
    return out


def cache_path(root: str, kind: str, split: str, size: int, n: int, seed: int) -> str:
    return os.path.join(maze_dir(root), f"maze_{kind}_{split}_{size}_n{n}_seed{seed}.npz")


def official_cache_path(root: str, kind: str, split: str, size: int) -> str:
    return os.path.join(maze_dir(root), f"maze_{kind}_official_{split}_{size}.npz")


def _load_or_generate(
    root: str, kind: str, split: str, size: int, n: int, seed: int, **gen_kwargs
) -> dict[str, np.ndarray]:
    path = cache_path(root, kind, split, size, n, seed)
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data = make_dataset(kind, split, n, size, seed, **gen_kwargs)
    np.savez_compressed(path, **data)
    _log(f"cached {path}")
    return data


# ----------------------------------------------------------------------------------------------
# Official data: download + conversion
# ----------------------------------------------------------------------------------------------


_UNAVAILABLE: set[str] = set()  # URLs whose download failed in this process (no retry)


def _download(url: str, dest: str, timeout: float = 60.0) -> None:
    if url in _UNAVAILABLE:
        raise ConnectionError(f"{url} already failed to download in this process")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    _log(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        _UNAVAILABLE.add(url)
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, dest)


def _extract_tar(archive: str, dest: str) -> None:
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # python < 3.12 without the filter argument
            tf.extractall(dest)


def _classify_rgb(img: np.ndarray) -> np.ndarray:
    """RGB (N, 3, P, P) -> classes (N, P, P): 0 corridor, 1 wall, 2 green, 3 red; -1 unknown."""
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    r, g, b = x[:, 0] > 0.5, x[:, 1] > 0.5, x[:, 2] > 0.5
    cls = np.full(r.shape, -1, dtype=np.int8)
    cls[r & g & b] = 0
    cls[~r & ~g & ~b] = 1
    cls[~r & g & ~b] = 2
    cls[r & ~g & ~b] = 3
    return cls


def _infer_pixel_layout(cls: np.ndarray, sol: np.ndarray, size: int) -> tuple[int, int]:
    """Find (border, block) such that P = 2 * border + block * size and every block is uniform."""
    P = cls.shape[-1]
    candidates = [
        (b, (P - 2 * b) // size)
        for b in range(0, P // 2 + 1)
        if (P - 2 * b) % size == 0 and (P - 2 * b) // size >= 1
    ]
    for b, c in candidates:
        idx = b + c * np.arange(size)
        ok = True
        for arr in (cls, sol):
            tok = arr[:, idx][:, :, idx]
            block = np.repeat(np.repeat(tok, c, axis=1), c, axis=2)
            if not np.array_equal(block, arr[:, b : b + c * size, b : b + c * size]):
                ok = False
                break
        if ok:
            return b, c
    raise ValueError(
        f"cannot understand the pixel layout of the official mazes (P={P}, size={size}); "
        f"tried (border, block) = {candidates}"
    )


def convert_official_ood(folder: str, size: int, chunk: int = 64) -> dict[str, np.ndarray]:
    """Convert ``inputs.npy`` / ``solutions.npy`` of an `easy-to-hard-data` maze folder into
    token grids ``(N, size, size)``. Raises ValueError when the layout is not understood."""
    x = np.load(os.path.join(folder, "inputs.npy"), mmap_mode="r")
    y = np.load(os.path.join(folder, "solutions.npy"), mmap_mode="r")
    if (
        x.ndim != 4
        or x.shape[1] != 3
        or x.shape[2] != x.shape[3]
        or y.shape != (x.shape[0],) + x.shape[2:]
    ):
        raise ValueError(f"unexpected official array shapes inputs={x.shape} solutions={y.shape}")
    N, P = x.shape[0], x.shape[2]
    if P != 2 * size + 6:
        _log(f"warning: official images are {P}px wide, expected {2 * size + 6} for size {size}")
    probe = min(N, 8)
    cls0 = _classify_rgb(np.asarray(x[:probe]))
    if (cls0 < 0).any():
        raise ValueError("official maze images contain pixels that are not black/white/green/red")
    border, block = _infer_pixel_layout(cls0, (np.asarray(y[:probe]) > 0).astype(np.int8), size)
    _log(f"official pixel layout: border={border}px, block={block}px per cell")
    idx = border + block * np.arange(size)
    inputs = np.empty((N, size, size), dtype=np.uint8)
    targets = np.empty_like(inputs)
    path_len = np.empty(N, dtype=np.int64)
    class_to_token = np.array([EMPTY, WALL, ENDPOINT, ENDPOINT], dtype=np.uint8)
    for s in range(0, N, chunk):
        cls = _classify_rgb(np.asarray(x[s : s + chunk]))
        if (cls < 0).any():
            raise ValueError(
                "official maze images contain pixels that are not black/white/green/red"
            )
        tok = class_to_token[cls[:, idx][:, :, idx]]
        sol = np.asarray(y[s : s + chunk])[:, idx][:, :, idx] > 0
        tgt = tok.copy()
        tgt[(tok == EMPTY) & sol] = PATH
        inputs[s : s + chunk] = tok
        targets[s : s + chunk] = tgt
    bad = 0
    for i in range(N):
        valid, length = path_validity(
            (targets[i] == PATH) | (inputs[i] == ENDPOINT), _endpoints(inputs[i], (ENDPOINT,))
        )
        path_len[i] = length
        bad += not valid
    if bad > 0.01 * N:
        raise ValueError(
            f"{bad}/{N} converted official mazes do not have a valid path; layout misunderstood?"
        )
    if bad:
        _log(f"warning: {bad}/{N} converted official mazes have an invalid path")
    return {"inputs": inputs, "targets": targets, "path_len": path_len}


def _fetch_official_ood(root: str, split: str, size: int, download: bool) -> dict[str, np.ndarray]:
    cache = official_cache_path(root, "ood", split, size)
    if os.path.exists(cache):
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    if size not in OFFICIAL_OOD_SIZES or (split == "train" and size != 9):
        raise FileNotFoundError(f"no official Maze-OOD {split} set of size {size}")
    raw = os.path.join(maze_dir(root), "raw")
    folder = os.path.join(raw, f"maze_data_{split}_{size}")
    if not os.path.exists(os.path.join(folder, "inputs.npy")):
        if not download:
            raise FileNotFoundError(f"{folder} not found and download=False")
        archive = os.path.join(raw, f"maze_data_{split}_{size}.tar.gz")
        if not os.path.exists(archive):
            _download(OOD_URL.format(split=split, size=size), archive)
        _extract_tar(archive, raw)
        if not os.path.exists(os.path.join(folder, "inputs.npy")):
            raise FileNotFoundError(f"archive did not contain {folder}/inputs.npy")
    data = convert_official_ood(folder, size)
    np.savez_compressed(cache, **data)
    _log(f"cached {cache}")
    return data


def parse_hrm_maze(q: str, a: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one HRM text puzzle (`q`) and answer (`a`) into token grids (inputs, target_astar)."""
    L = len(q)
    n = int(round(L**0.5))
    if n * n != L or len(a) != L:
        raise ValueError(f"HRM puzzle of length {L} is not a square grid (answer length {len(a)})")
    table = {"#": WALL, " ": EMPTY, ".": EMPTY, "S": START, "G": GOAL}
    try:
        inputs = np.array([table[ch] for ch in q], dtype=np.uint8).reshape(n, n)
    except KeyError as e:
        raise ValueError(
            f"unexpected character {e} in HRM puzzle (charset '# SGo' expected)"
        ) from e
    unknown = set(a) - set("# .SGo")
    if unknown:
        raise ValueError(f"unexpected characters {sorted(unknown)} in HRM answer")
    on_path = np.frombuffer(a.encode(), dtype=np.uint8).reshape(n, n) == ord("o")
    target = inputs.copy()
    target[(inputs == EMPTY) & on_path] = PATH
    return inputs, target


def convert_official_hard(csv_path: str) -> dict[str, np.ndarray]:
    """Convert an HRM ``train.csv`` / ``test.csv`` into token grids with A* and BFS targets."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.strip().lower() for h in header]
        qi, ai = (cols.index("q"), cols.index("a")) if "q" in cols and "a" in cols else (1, 2)
        rows = [(row[qi], row[ai]) for row in reader if len(row) > max(qi, ai)]
    if not rows:
        raise ValueError(f"no puzzles found in {csv_path}")
    inputs, t_astar, t_bfs, path_len = [], [], [], []
    bad, not_shortest = 0, 0
    for q, a in rows:
        inp, tgt = parse_hrm_maze(q, a)
        if (inp == START).sum() != 1 or (inp == GOAL).sum() != 1:
            raise ValueError("HRM maze without exactly one start and one goal")
        ends = _endpoints(inp, (START,)) + _endpoints(inp, (GOAL,))  # start first, then goal
        valid, length = path_validity((tgt == PATH) | (inp == START) | (inp == GOAL), ends)
        bad += not valid
        p = bfs_path(inp != WALL, ends[0], ends[1])
        if p is None:
            raise ValueError("HRM maze with unreachable goal")
        not_shortest += len(p) != length
        inputs.append(inp)
        t_astar.append(tgt)
        t_bfs.append(_write_path(inp, p, (START, GOAL)))
        path_len.append(len(p))
    N = len(rows)
    if bad > 0.01 * N:
        raise ValueError(
            f"{bad}/{N} official Maze-Hard answers are not valid paths; format misunderstood?"
        )
    if bad or not_shortest:
        _log(
            f"warning: official Maze-Hard: {bad}/{N} invalid answers, {not_shortest}/{N} not shortest"
        )
    return {
        "inputs": np.stack(inputs),
        "targets_astar": np.stack(t_astar),
        "targets_bfs": np.stack(t_bfs),
        "path_len": np.array(path_len, dtype=np.int64),
    }


def _fetch_official_hard(root: str, split: str, download: bool) -> dict[str, np.ndarray]:
    cache = official_cache_path(root, "hard", split, 30)
    if os.path.exists(cache):
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    raw = os.path.join(maze_dir(root), "raw", "maze_hard")
    csv_path = os.path.join(raw, f"{split}.csv")
    if not os.path.exists(csv_path):
        if not download:
            raise FileNotFoundError(f"{csv_path} not found and download=False")
        _download(HARD_URL.format(split=split), csv_path)
    data = convert_official_hard(csv_path)
    np.savez_compressed(cache, **data)
    _log(f"cached {cache}")
    return data


def _subsample(
    data: dict[str, np.ndarray], n: int | None, seed: int, exclude_seed=None, exclude_n=None
) -> dict[str, np.ndarray]:
    """`n` boards drawn with `seed`. `exclude_seed` (one seed or a list) first removes the boards
    of the draws of `exclude_n` (default `n`) boards with those seeds, so that a final sample
    shares no board with the samples already used (the Table 1 sample: n=100, seed 0)."""
    N = len(data["inputs"])
    if exclude_seed is None:
        if n is None or n >= N:
            return data
        idx = np.random.default_rng(seed).choice(N, size=n, replace=False)
    else:
        seeds = [exclude_seed] if np.isscalar(exclude_seed) else list(exclude_seed)
        m = exclude_n or n
        if m is None:
            raise ValueError("exclude_seed needs exclude_n or n: the size of the excluded draws")
        used = [np.random.default_rng(s).choice(N, size=min(m, N), replace=False) for s in seeds]
        pool = np.setdiff1d(np.arange(N), np.concatenate(used))
        size = len(pool) if n is None else min(n, len(pool))
        idx = np.random.default_rng(seed).choice(pool, size=size, replace=False)
    idx = np.sort(idx)
    return {k: v[idx] for k, v in data.items()}


# ----------------------------------------------------------------------------------------------
# Dihedral augmentation
# ----------------------------------------------------------------------------------------------


def dihedral_transform(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry `k` in 0..7 of the square to the last two axes: rotate by 90*(k%4) degrees,
    then transpose when k >= 4."""
    out = np.rot90(arr, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.swapaxes(out, -2, -1)
    return np.ascontiguousarray(out)


def dihedral_inverse(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.swapaxes(arr, -2, -1) if k >= 4 else arr
    return np.ascontiguousarray(np.rot90(out, -(k % 4), axes=(-2, -1)))


def augment_dihedral(boards: Boards, rng: np.random.Generator) -> Boards:
    """Random dihedral symmetry per board (the same one for inputs, targets and given)."""
    H, W = boards.shape
    if H != W:
        raise ValueError("dihedral augmentation needs square boards")
    ks = rng.integers(8, size=len(boards))
    inputs, targets, given = boards.inputs.copy(), boards.targets.copy(), boards.given.copy()
    for k in range(1, 8):
        sel = np.nonzero(ks == k)[0]
        if len(sel) == 0:
            continue
        inputs[sel] = dihedral_transform(boards.inputs[sel], k)
        targets[sel] = dihedral_transform(boards.targets[sel], k)
        given[sel] = dihedral_transform(boards.given[sel], k)
    meta = dict(boards.meta)
    meta["dihedral"] = ks
    return Boards(inputs, targets, given, meta)


# ----------------------------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------------------------


def _endpoints(tokens: np.ndarray, endpoint_tokens: tuple[int, ...]) -> list[tuple[int, int]]:
    pts = np.argwhere(np.isin(tokens, endpoint_tokens))
    return [tuple(int(x) for x in p) for p in pts]


def bfs_target(inputs: np.ndarray, local: bool = False, order: str = "UDLR") -> np.ndarray:
    """Maze-Hard BFS target of one board ``(H, W)``: the `bfs_path` (or `bfs_local_path` when
    `local`) from START to GOAL written over the input tokens."""
    ends = _endpoints(inputs, (START,)) + _endpoints(inputs, (GOAL,))
    if local:
        p = bfs_local_path(inputs != WALL, ends[0], ends[1])
    else:
        p = bfs_path(inputs != WALL, ends[0], ends[1], order)
    if p is None:
        raise ValueError("maze with unreachable goal")
    return _write_path(inputs, p, (START, GOAL))


def astar_target(inputs: np.ndarray) -> np.ndarray:
    """Our `astar_path` from START to GOAL of one board ``(H, W)``, written over the input tokens."""
    ends = _endpoints(inputs, (START,)) + _endpoints(inputs, (GOAL,))
    p = astar_path(inputs != WALL, ends[0], ends[1])
    if p is None:
        raise ValueError("maze with unreachable goal")
    return _write_path(inputs, p, (START, GOAL))


class _MazeTask(Task):
    endpoint_tokens: tuple[int, ...] = (ENDPOINT,)

    def _metrics(self, pred: np.ndarray, boards: Boards) -> dict[str, np.ndarray]:
        pred = np.asarray(pred)
        mutable = ~boards.given
        N = len(boards)
        exact = np.all((pred == boards.targets) | ~mutable, axis=(1, 2))
        cell_acc = self.cell_accuracy(pred, boards)
        valid = np.zeros(N, dtype=bool)
        optimal = np.zeros(N, dtype=bool)
        is_end = np.isin(boards.inputs, self.endpoint_tokens)
        pred_path = ((pred == PATH) & mutable) | is_end
        ref_len = (boards.targets == PATH).sum(axis=(1, 2)) + is_end.sum(axis=(1, 2))
        for i in range(N):
            ends = [tuple(int(x) for x in p) for p in np.argwhere(is_end[i])]
            v, length = path_validity(pred_path[i], ends)
            valid[i] = v
            optimal[i] = v and length == ref_len[i]
        return {"exact": exact, "cell_acc": cell_acc, "valid": valid, "optimal": optimal}

    def metrics(self, pred: np.ndarray, boards: Boards) -> dict[str, np.ndarray]:
        """ "exact": all mutable cells match the reference; "cell_acc": fraction of mutable cells
        correct; "valid": the predicted path cells + endpoints form one simple 4-connected path
        between the endpoints; "optimal": valid and as short as the reference path."""
        return self._metrics(pred, boards)


class MazeOOD(_MazeTask):
    """Maze-OOD: train on 9 x 9 mazes, test on larger ones (up to 201 x 201)."""

    name = "maze_ood"
    vocab_size = 4
    output_tokens = (EMPTY, PATH)
    token_names = ("empty", "path", "wall", "endpoint")
    main_metric = "exact"
    endpoint_tokens = (ENDPOINT,)

    def load(
        self,
        split: str,
        root: str,
        size: int = 9,
        n: int | None = None,
        seed: int = 0,
        download: bool = True,
        source: str = "auto",
        exclude_seed=None,
        exclude_n: int | None = None,
        wall_ring: bool = False,
        **kwargs,
    ) -> Boards:
        """`wall_ring`: surround each maze with a ring of given walls, the format of the authors'
        public code (its `[1:-1:2]` sampling keeps one border pixel per side: a size-9 maze is an
        11 x 11 grid). Their checkpoint relies on it: with wrap-around padding and no ring,
        opposite corridors become neighbours (0% exact instead of 100% on 9 x 9)."""
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if source not in ("auto", "official", "generated"):
            raise ValueError("source must be 'auto', 'official' or 'generated'")
        n_default = 50000 if split == "train" else 1000
        data, used = None, None
        if source in ("auto", "official"):
            try:
                data = _subsample(
                    _fetch_official_ood(root, split, size, download),
                    n,
                    seed,
                    exclude_seed,
                    exclude_n,
                )
                used = "official"
            except Exception as e:  # noqa: BLE001 - any failure falls back to the generator
                if source == "official":
                    raise
                _log(
                    f"official Maze-OOD {split} size {size} unavailable ({type(e).__name__}: {e}); "
                    f"using the local generator"
                )
        if data is None:
            data = _load_or_generate(root, "ood", split, size, n_default if n is None else n, seed)
            used = "generated"
        inputs, targets = data["inputs"].astype(np.int64), data["targets"].astype(np.int64)
        if wall_ring:
            inputs, targets = (
                np.pad(a, ((0, 0), (1, 1), (1, 1)), constant_values=WALL) for a in (inputs, targets)
            )
        meta = {"source": used, "size": int(size), "path_len": data["path_len"]}
        if wall_ring:
            meta["wall_ring"] = True
        return Boards(inputs, targets, inputs != EMPTY, meta)

    def augment(self, boards: Boards, rng: np.random.Generator) -> Boards:
        return augment_dihedral(boards, rng)


class MazeHard(_MazeTask):
    """Maze-Hard: 30 x 30 mazes with loops, shortest paths of >= 110 cells (HRM benchmark)."""

    name = "maze_hard"
    vocab_size = 5
    output_tokens = (EMPTY, PATH)
    token_names = ("empty", "path", "wall", "start", "goal")
    main_metric = "exact"
    endpoint_tokens = (START, GOAL)

    def __init__(self, targets: str = "astar") -> None:
        # bfs_local: shortest path picked by a local tie-break (`bfs_local_path`), computed here
        # because the paper does not say how its BFS breaks ties (DEVIATIONS 2026-10-19)
        # astar_own: our `astar_path` instead of the HRM-shipped paths (a control for our own targets)
        # bfs_dlru: BFS expanding neighbours down/left/right/up, the rule that reproduces the
        # HRM-shipped paths exactly; with the dihedral augmentation recomputed on each board it
        # is our reading of the paper's BFS targets (DEVIATIONS 2026-10-19)
        if targets not in ("astar", "astar_own", "bfs", "bfs_local", "bfs_dlru"):
            raise ValueError(
                "targets must be 'astar', 'astar_own', 'bfs', 'bfs_local' or 'bfs_dlru'"
            )
        self.targets = targets

    def load(
        self,
        split: str,
        root: str,
        n: int | None = None,
        seed: int = 0,
        download: bool = True,
        source: str = "auto",
        exclude_seed=None,
        exclude_n: int | None = None,
        **kwargs,
    ) -> Boards:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if source not in ("auto", "official", "generated"):
            raise ValueError("source must be 'auto', 'official' or 'generated'")
        data, used = None, None
        if source in ("auto", "official"):
            try:
                data = _subsample(
                    _fetch_official_hard(root, split, download), n, seed, exclude_seed, exclude_n
                )
                used = "official"
            except Exception as e:  # noqa: BLE001
                if source == "official":
                    raise
                _log(
                    f"official Maze-Hard {split} unavailable ({type(e).__name__}: {e}); "
                    f"using the local generator"
                )
        if data is None:
            data = _load_or_generate(root, "hard", split, 30, 1000 if n is None else n, seed)
            used = "generated"
        inputs = data["inputs"].astype(np.int64)
        if self.targets == "bfs_local":
            targets = np.stack([bfs_target(x, local=True) for x in inputs]).astype(np.int64)
        elif self.targets == "astar_own":
            targets = np.stack([astar_target(x) for x in inputs]).astype(np.int64)
        elif self.targets == "bfs_dlru":
            targets = np.stack([bfs_target(x, order="DLRU") for x in inputs]).astype(np.int64)
        else:
            targets = data[f"targets_{self.targets}"].astype(np.int64)
        meta = {"source": used, "size": 30, "targets": self.targets, "path_len": data["path_len"]}
        return Boards(inputs, targets, inputs != EMPTY, meta)

    def augment(self, boards: Boards, rng: np.random.Generator) -> Boards:
        # A* targets are orientation dependent (tie-break), so the paper only augments BFS targets.
        # The BFS target is one shortest path chosen by a fixed neighbour order, which depends on the
        # orientation too, so it is recomputed on each transformed maze ("computed on the fly", A.1)
        # instead of being rotated with it: a rotated target is the BFS path of the rotated maze
        # for only 89 of the first 200 official training mazes.
        if self.targets in ("astar", "astar_own"):
            return boards
        out = augment_dihedral(boards, rng)
        for i in np.nonzero(out.meta["dihedral"])[0]:
            out.targets[i] = bfs_target(
                out.inputs[i],
                local=self.targets == "bfs_local",
                order="DLRU" if self.targets == "bfs_dlru" else "UDLR",
            )
        return out


# ----------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------

_COLORS = {
    EMPTY: (255, 255, 255),
    PATH: (70, 130, 230),
    WALL: (40, 40, 40),
    START: (60, 180, 75),
    GOAL: (220, 50, 50),
}


def _second_endpoint_as_goal(tokens: np.ndarray) -> np.ndarray:
    """For Maze-OOD grids (two token-3 endpoints, no token 4) mark the second endpoint as GOAL
    so that it is drawn/printed differently."""
    t = np.asarray(tokens).copy()
    if not (t == GOAL).any():
        ends = np.argwhere(t == ENDPOINT)
        if len(ends) == 2:
            t[tuple(ends[1])] = GOAL
    return t


def render_maze(tokens: np.ndarray) -> np.ndarray:
    """RGB picture ``(H, W, 3)`` uint8: wall dark, empty white, path blue, endpoints green / red."""
    t = _second_endpoint_as_goal(tokens)
    lut = np.zeros((max(_COLORS) + 1, 3), dtype=np.uint8)
    for k, c in _COLORS.items():
        lut[k] = c
    return lut[np.clip(t, 0, len(lut) - 1)]


ASCII = ".o#SG"


def decode_ascii(tokens: np.ndarray) -> str:
    """Text rendering, one char per cell: ``.`` empty, ``o`` path, ``#`` wall, ``S``/``G`` endpoints."""
    t = _second_endpoint_as_goal(tokens)
    return "\n".join("".join(ASCII[int(v)] for v in row) for row in t)
