"""Sudoku benchmarks of "Reasoning with Neural Cellular Automata" (Appendix A.2).

Two tasks share this module:

* **Sudoku-OOD** (`SudokuOOD`, name ``"sudoku_ood"``): the AKOrN setup (Miyato et al., 2024).
  Train = the 9K "easy" SAT-Net boards (Wang et al., 2019; 31-42 clues, i.e. 39-50 empty cells)
  + 1K validation boards; test = the RRN dataset (Palm et al., 2017): 18 difficulty levels defined
  by the number of empty cells (47 to 64 empty cells, i.e. 34 down to 17 clues), 1K boards per
  level (18K boards).
* **Sudoku-Extreme** (`SudokuExtreme`, name ``"sudoku_extreme"``): the HRM setup (Wang et al.,
  2025). Train = 1000 boards sub-sampled from a 3.8M-board csv; test = 422,780 boards with a
  difficulty ``rating`` = number of backtracks of the tdoku solver (0 to 289).

Board format
------------
Boards are ``(N, 9, 9)`` int64 token grids with vocabulary V = 10: token 0 = empty, tokens 1-9 =
digits. ``inputs`` holds the clues (0 elsewhere), ``targets`` the full solution, ``given`` is True
on clue cells. The NCA predicts a digit on every empty cell; token 0 is never predicted, so
``output_tokens = (1, ..., 9)``.

String format: 81 characters, row-major, ``'0'`` or ``'.'`` for an empty cell (both accepted
on input; ``board_to_string`` writes ``'0'``).

Data sources (tried in this order when ``source="auto"``)
----------------------------------------------------------
1. A cached ``.npz`` under ``<root>/sudoku/`` named
   ``sudoku_{kind}_{split}_n{n}_seed{seed}_{official|generated}.npz`` (an "official" cache is
   preferred over a "generated" one). Each file stores ``inputs``/``targets`` (uint8), ``given``
   (bool) and the meta arrays ``n_empty``, ``difficulty`` (backtracks of *our* solver, see below)
   plus ``rating`` (tdoku backtracks) when the official data provide it.
2. The official files, downloaded into ``<root>/sudoku/raw/`` when ``download=True``:

   * SAT-Net: ``https://powei.tw/sudoku.zip`` -> ``features.pt`` / ``labels.pt``, torch tensors of
     one-hot boards ``(10000, 9, 9, 9)``; the first 9000 are the train split and the last 1000 the
     validation split (the SAT-Net / AKOrN convention). Reading them needs ``torch``, which is
     imported lazily and only inside the loader; without torch the file is skipped.
   * RRN (DGL mirror): ``https://data.dgl.ai/dataset/sudoku-hard.zip`` -> ``train.csv``,
     ``valid.csv``, ``test.csv``; each line is ``puzzle,solution`` with two 81-character digit
     strings (``0`` = empty). The parser is defensive: it looks for two 81-character fields on
     every line and ignores headers and malformed lines.
   * Sudoku-Extreme (Hugging Face ``sapientinc/sudoku-extreme``):
     ``https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/{train,test}.csv``
     with header ``source,question,answer,rating`` (``.`` or ``0`` for empty). ``train.csv`` is
     ~3.8M lines, so only its first ``EXTREME_TRAIN_HEAD_LINES`` lines are streamed and the train
     boards are sub-sampled from that head.

   These hosts may be unreachable (or the file formats may have changed): every failure prints a
   notice and the loader falls back to the local generator.
3. The local generator (always available, pure Python + numpy):

   * ``solve``: backtracking with candidate bitmasks, MRV cell choice and naked-singles
     propagation. Its ``backtracks`` count (number of guesses that had to be undone before the
     first solution was found, with candidates tried in increasing order) is our difficulty proxy.
     It is similar in spirit to the tdoku rating used by Sudoku-Extreme but it is **not** the same
     scale (tdoku propagates much more and counts differently), hence the separate names
     ``difficulty`` (ours) and ``rating`` (official).
   * ``random_solution``: solves the empty board with shuffled candidates.
   * ``generate_puzzle``: removes clues from a random full grid in random order while the puzzle
     keeps a unique solution. One pass yields a *minimal* puzzle (no clue can be removed any more)
     with typically 22-26 clues; 17-21 clues are rare, so generation is best effort: the lowest
     reachable clue count is accepted. Boards that go *below* the requested number of clues are
     refilled with random clues from the solution, so every level that is reachable at all is
     produced exactly. For the Sudoku-OOD test set the achieved distribution over the 18 levels
     is printed and stored in ``meta["level"]`` (= actual number of empty cells).

Train boards are generated with a uniform number of clues in ``[31, 42]`` (the SAT-Net range);
Sudoku-Extreme generated boards use ``[22, 30]`` clues and are binned by our solver's backtracks
(``meta["difficulty_scale"] = "backtracks(ours)"``) instead of the tdoku rating.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .base import Boards, Task

# ----------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------

ALL_DIGITS = 0x3FE  # bits 1..9 set
_ROW = [i // 9 for i in range(81)]
_COL = [i % 9 for i in range(81)]
_BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]
_DIGITS = list(range(1, 10))

SATNET_URL = "https://powei.tw/sudoku.zip"
RRN_URL = "https://data.dgl.ai/dataset/sudoku-hard.zip"
EXTREME_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/{split}.csv"
EXTREME_TRAIN_HEAD_LINES = 100_000  # lines of the 3.8M-line train.csv that are streamed
DOWNLOAD_TIMEOUT = 30.0

OOD_TRAIN_CLUES = (31, 42)  # inclusive range of clues of the SAT-Net training boards
OOD_TEST_LEVELS = tuple(range(47, 65))  # number of empty cells of the 18 RRN levels
EXTREME_TRAIN_CLUES = (22, 30)  # inclusive range of clues for generated Sudoku-Extreme boards
#: difficulty bins of Figure 12 of the paper (inclusive rating ranges)
EXTREME_BINS = ((0, 0), (1, 5), (6, 11), (12, 16), (17, 28), (29, 53), (54, 10**9))
EXTREME_BIN_NAMES = ("0", "1-5", "6-11", "12-16", "17-28", "29-53", "54+")

#: a standard puzzle (Wikipedia's example) and its solution, used by the tests / demos
EXAMPLE_PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
EXAMPLE_SOLUTION = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)


# ----------------------------------------------------------------------------------------------
# board helpers
# ----------------------------------------------------------------------------------------------


def string_to_board(s: str) -> np.ndarray:
    """81-character string (``0`` or ``.`` = empty) -> ``(9, 9)`` int64 array."""
    s = s.strip()
    if len(s) != 81:
        raise ValueError(f"expected 81 characters, got {len(s)}")
    return np.array([0 if ch in ".0" else int(ch) for ch in s], dtype=np.int64).reshape(9, 9)


def board_to_string(tokens: np.ndarray) -> str:
    """``(9, 9)`` (or 81) token array -> 81-character string with ``0`` for empty cells."""
    flat = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if flat.size != 81:
        raise ValueError(f"expected 81 cells, got {flat.size}")
    return "".join(str(int(v)) for v in flat)


def pretty(tokens: np.ndarray) -> str:
    """ASCII grid with 3x3 box separators; empty cells are shown as ``.``."""
    g = np.asarray(tokens, dtype=np.int64).reshape(9, 9)
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        cells = [str(v) if v else "." for v in g[r]]
        lines.append(
            " ".join(cells[0:3]) + " | " + " ".join(cells[3:6]) + " | " + " ".join(cells[6:9])
        )
    return "\n".join(lines)


def is_valid_solution(grid: np.ndarray) -> bool:
    """True when ``grid`` ``(9, 9)`` is a complete grid satisfying all row/column/box constraints."""
    g = np.asarray(grid, dtype=np.int64).reshape(9, 9)
    if g.min() < 1 or g.max() > 9:
        return False
    full = set(range(1, 10))
    for k in range(9):
        if set(g[k, :].tolist()) != full or set(g[:, k].tolist()) != full:
            return False
        box = g[3 * (k // 3) : 3 * (k // 3) + 3, 3 * (k % 3) : 3 * (k % 3) + 3]
        if set(box.reshape(-1).tolist()) != full:
            return False
    return True


def is_consistent(grid: np.ndarray) -> bool:
    """True when the non-empty cells of ``grid`` do not violate any constraint (0 = empty)."""
    g = np.asarray(grid, dtype=np.int64).reshape(-1)
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i in range(81):
        v = int(g[i])
        if v:
            b = 1 << v
            if (rows[_ROW[i]] | cols[_COL[i]] | boxes[_BOX[i]]) & b:
                return False
            rows[_ROW[i]] |= b
            cols[_COL[i]] |= b
            boxes[_BOX[i]] |= b
    return True


def _to_cells(board) -> list[int]:
    return [int(v) for v in np.asarray(board, dtype=np.int64).reshape(-1)]


# ----------------------------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------------------------


def solve(board, count_limit: int = 2, rng=None) -> tuple[int, np.ndarray | None, int]:
    """Backtracking solver with candidate bitmasks, MRV and naked-singles propagation.

    Parameters
    ----------
    board : ``(9, 9)`` or 81-long array/list, 0 = empty.
    count_limit : stop once this many solutions have been found (2 = uniqueness check).
    rng : optional ``numpy.random.Generator`` or ``random.Random``; when given, the candidate
        digits of every guess are tried in random order (used to sample random full grids).

    Returns
    -------
    ``(n_solutions, solution, backtracks)``: the number of solutions found (capped at
    ``count_limit``), the first solution as a ``(9, 9)`` int64 array (``None`` if there is none)
    and the number of guesses that had to be undone (the difficulty proxy; with
    ``count_limit=1`` and ``rng=None`` it counts the failed guesses before the first solution).
    """
    cells = _to_cells(board)
    if len(cells) != 81:
        raise ValueError("board must have 81 cells")
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(cells):
        if v:
            b = 1 << v
            if (rows[_ROW[i]] | cols[_COL[i]] | boxes[_BOX[i]]) & b:
                return 0, None, 0
            rows[_ROW[i]] |= b
            cols[_COL[i]] |= b
            boxes[_BOX[i]] |= b
    shuffle = None
    if rng is not None:
        shuffle = rng.shuffle
    state = [0, None, 0]  # n_found, first solution (list), backtracks
    _search(cells, rows, cols, boxes, count_limit, shuffle, state)
    sol = None if state[1] is None else np.array(state[1], dtype=np.int64).reshape(9, 9)
    return state[0], sol, state[2]


def _search(cells, rows, cols, boxes, count_limit, shuffle, state) -> None:
    placed = []
    while True:
        best, best_mask, best_n = -1, 0, 10
        for i in range(81):
            if cells[i] == 0:
                mask = ALL_DIGITS & ~(rows[_ROW[i]] | cols[_COL[i]] | boxes[_BOX[i]])
                n = mask.bit_count()
                if n < best_n:
                    if n == 0:  # dead end
                        best = -2
                        break
                    best, best_mask, best_n = i, mask, n
                    if n == 1:
                        break
        if best == -2:
            break
        if best == -1:  # no empty cell left: a solution
            state[0] += 1
            if state[1] is None:
                state[1] = list(cells)
            break
        if best_n == 1:  # naked single: propagate without guessing
            d = best_mask.bit_length() - 1
            cells[best] = d
            rows[_ROW[best]] |= best_mask
            cols[_COL[best]] |= best_mask
            boxes[_BOX[best]] |= best_mask
            placed.append((best, best_mask))
            continue
        cands = [d for d in _DIGITS if best_mask >> d & 1]
        if shuffle is not None:
            shuffle(cands)
        r, c, bx = _ROW[best], _COL[best], _BOX[best]
        for d in cands:
            b = 1 << d
            cells[best] = d
            rows[r] |= b
            cols[c] |= b
            boxes[bx] |= b
            _search(cells, rows, cols, boxes, count_limit, shuffle, state)
            rows[r] ^= b
            cols[c] ^= b
            boxes[bx] ^= b
            cells[best] = 0
            if state[0] >= count_limit:
                break
            state[2] += 1  # the guess is undone
        break
    for i, b in placed:
        cells[i] = 0
        rows[_ROW[i]] ^= b
        cols[_COL[i]] ^= b
        boxes[_BOX[i]] ^= b


def difficulty(board) -> int:
    """Our difficulty proxy: backtracks of ``solve(board, count_limit=1)`` (deterministic order)."""
    return solve(board, count_limit=1)[2]


def random_solution(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-ish random complete grid: solve the empty board with shuffled candidates."""
    _, sol, _ = solve([0] * 81, count_limit=1, rng=rng)
    assert sol is not None
    return sol


def _remove_clues(puzzle: list[int], order, n_clues: int) -> int:
    """Remove clues of ``puzzle`` in place following ``order`` while the solution stays unique.

    Stops as soon as ``n_clues`` clues remain. Returns the number of clues left.
    """
    clues = sum(1 for v in puzzle if v)
    for i in order:
        if clues <= n_clues:
            break
        v = puzzle[i]
        if not v:
            continue
        puzzle[i] = 0
        # cheap shortcut: if the cell is a naked single, the puzzle is still unique
        used = 0
        for j in range(81):
            if puzzle[j] and (_ROW[j] == _ROW[i] or _COL[j] == _COL[i] or _BOX[j] == _BOX[i]):
                used |= 1 << puzzle[j]
        if (ALL_DIGITS & ~used).bit_count() == 1:
            clues -= 1
            continue
        n_sol, _, _ = solve(puzzle, count_limit=2)
        if n_sol == 1:
            clues -= 1
        else:
            puzzle[i] = v
    return clues


def generate_puzzle(
    rng: np.random.Generator, n_clues: int, max_tries: int = 10, exact: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a uniquely-solvable puzzle with ``n_clues`` clues.

    Starts from a random full grid and removes clues in random order while the solution stays
    unique. Each try removes as many clues as possible (a *minimal* puzzle, typically 22-26
    clues); a new grid is drawn until ``n_clues`` is reached or ``max_tries`` tries are spent.

    With ``exact=True`` (default) a ``RuntimeError`` is raised when ``n_clues`` could not be
    reached. With ``exact=False`` the try with the fewest clues is returned (best effort, used
    for the hardest levels of the test set). Boards never have *fewer* than ``n_clues`` clues.

    Returns ``(puzzle, solution)`` as ``(9, 9)`` int64 arrays.
    """
    if not 17 <= n_clues <= 81:
        raise ValueError("n_clues must be in [17, 81]")
    best = None
    for _ in range(max_tries):
        sol = random_solution(rng)
        puzzle = sol.reshape(-1).tolist()
        order = rng.permutation(81).tolist()
        clues = _remove_clues(puzzle, order, n_clues)
        if clues == n_clues:
            return np.array(puzzle, dtype=np.int64).reshape(9, 9), sol
        if best is None or clues < best[0]:
            best = (clues, puzzle, sol)
    if exact:
        raise RuntimeError(
            f"could not reach {n_clues} clues in {max_tries} tries (best: {best[0]})"
        )
    clues, puzzle, sol = best
    return np.array(puzzle, dtype=np.int64).reshape(9, 9), sol


def _refill(puzzle: np.ndarray, solution: np.ndarray, n_clues: int, rng: np.random.Generator):
    """Add random clues (from the solution) to ``puzzle`` until it has exactly ``n_clues``."""
    puzzle = puzzle.copy().reshape(-1)
    sol = solution.reshape(-1)
    empty = np.flatnonzero(puzzle == 0)
    need = n_clues - int((puzzle != 0).sum())
    if need > 0:
        add = rng.choice(empty, size=need, replace=False)
        puzzle[add] = sol[add]
    return puzzle.reshape(9, 9)


def generate_board(seed: int, n_clues: int, max_tries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort generation for one board (worker-friendly: seeded by an int).

    Aims at the *lowest* reachable clue count up to ``max_tries`` grids and refills with random
    clues up to ``n_clues`` when a try went lower, so the result has exactly ``n_clues`` clues
    whenever that count was reachable, and the lowest reached count otherwise.
    """
    rng = np.random.default_rng(seed)
    puzzle, sol = generate_puzzle(rng, n_clues, max_tries=max_tries, exact=False)
    return _refill(puzzle, sol, n_clues, rng), sol


def _gen_worker(args):
    seed, n_clues, max_tries = args
    puzzle, sol = generate_board(seed, n_clues, max_tries)
    return puzzle, sol, difficulty(puzzle)


def generate_boards(
    clue_targets, seed: int, max_tries: int = 3, workers: int | None = None, verbose: bool = True
) -> Boards:
    """Generate one board per entry of ``clue_targets`` (deterministic given ``seed``).

    Board ``k`` is generated from the ``k``-th child of ``SeedSequence(seed)``, so the result does
    not depend on the number of ``workers`` (a ``multiprocessing`` pool is used when
    ``workers > 1``; ``None`` picks ``min(cpu_count, 8)`` for 64+ boards).
    """
    clue_targets = [int(c) for c in clue_targets]
    n = len(clue_targets)
    seeds = np.random.SeedSequence(seed).spawn(n)
    jobs = [
        (int(s.generate_state(1)[0]), c, max_tries)
        for s, c in zip(seeds, clue_targets, strict=True)
    ]
    if workers is None:
        workers = 1 if n < 64 else max(1, min(os.cpu_count() or 1, 8))
    results = []
    t0 = time.time()
    last = t0

    def progress(k):
        nonlocal last
        if verbose and n >= 100 and (time.time() - last > 5.0 or k == n):
            last = time.time()
            print(
                f"[sudoku] generated {k}/{n} boards ({last - t0:.0f} s)",
                file=sys.stderr,
                flush=True,
            )

    if workers > 1 and n > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(workers) as pool:
            for k, res in enumerate(pool.imap(_gen_worker, jobs, chunksize=4), 1):
                results.append(res)
                progress(k)
    else:
        for k, job in enumerate(jobs, 1):
            results.append(_gen_worker(job))
            progress(k)
    inputs = np.stack([r[0] for r in results]) if results else np.zeros((0, 9, 9), np.int64)
    targets = np.stack([r[1] for r in results]) if results else np.zeros((0, 9, 9), np.int64)
    diff = np.array([r[2] for r in results], dtype=np.int64)
    return _make_boards(inputs, targets, {"difficulty": diff, "source": "generated"})


# ----------------------------------------------------------------------------------------------
# boards construction, caching
# ----------------------------------------------------------------------------------------------


def _make_boards(inputs: np.ndarray, targets: np.ndarray, meta: dict | None = None) -> Boards:
    inputs = np.asarray(inputs, dtype=np.int64).reshape(-1, 9, 9)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1, 9, 9)
    given = inputs != 0
    meta = dict(meta or {})
    meta["n_empty"] = (~given).sum(axis=(1, 2)).astype(np.int64)
    return Boards(inputs, targets, given, meta)


GENERATOR_VERSION = 2
"""Bumped whenever the generator's output changes, so that stale caches are never reused:
v2 (2026-10-12) = one random stream per split (the v1 caches leak solutions between splits)."""


def cache_path(
    root: str, kind: str, split: str, n: int | str, seed: int, source: str, sampling: str = ""
) -> Path:
    """``<root>/sudoku/sudoku_{kind}_{split}_n{n}_seed{seed}_{source}.npz``; generated caches
    carry the generator version: ``..._generated_v2.npz`` (the unversioned v1 files are ignored)."""
    tag = f"generated_v{GENERATOR_VERSION}" if source == "generated" else source
    if sampling:  # e.g. "strat" / "uniform" for the Sudoku-Extreme test sample
        tag = f"{tag}_{sampling}"
    return Path(root) / "sudoku" / f"sudoku_{kind}_{split}_n{n}_seed{seed}_{tag}.npz"


def save_boards(path: Path, boards: Boards) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "inputs": boards.inputs.astype(np.uint8),
        "targets": boards.targets.astype(np.uint8),
        "given": boards.given,
    }
    for k, v in boards.meta.items():
        if isinstance(v, np.ndarray):
            arrays["meta_" + k] = v
        else:
            arrays["metastr_" + k] = np.array(str(v))
    np.savez_compressed(path, **arrays)


def load_boards(path: Path) -> Boards:
    with np.load(path, allow_pickle=False) as f:
        meta = {}
        for k in f.files:
            if k.startswith("meta_"):
                meta[k[5:]] = f[k]
            elif k.startswith("metastr_"):
                meta[k[8:]] = str(f[k])
        return Boards(f["inputs"].astype(np.int64), f["targets"].astype(np.int64), f["given"], meta)


def default_n(kind: str, split: str) -> int:
    """Default number of generated boards of a split."""
    if kind == "ood":
        return {"train": 9000, "val": 1000, "test": 100 * len(OOD_TEST_LEVELS)}[split]
    return 1000


def _find_cache(root, kind, split, n, seed, source, sampling=""):
    sources = ("official", "generated") if source == "auto" else (source,)
    for s in sources:
        n_name = n if n is not None else ("all" if s == "official" else default_n(kind, split))
        p = cache_path(root, kind, split, n_name, seed, s, sampling)
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------------------------------
# official data
# ----------------------------------------------------------------------------------------------


def _notice(msg: str) -> None:
    print(f"[sudoku] {msg}", file=sys.stderr, flush=True)


def _download(url: str, dest: Path, max_lines: int | None = None) -> Path | None:
    """Download ``url`` to ``dest`` (optionally only its first ``max_lines`` text lines)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        _notice(f"downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "nca-reasoning/0.1"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as f:
            if max_lines is None:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            else:
                for k, line in enumerate(resp):
                    if k >= max_lines:
                        break
                    f.write(line)
        os.replace(tmp, dest)
        return dest
    except Exception as e:  # noqa: BLE001 - any network / IO failure falls back to the generator
        _notice(f"download failed ({type(e).__name__}: {e}); falling back to the local generator")
        if tmp.exists():
            tmp.unlink()
        return None


def _parse_pair_lines(text: str):
    """Yield ``(puzzle, solution)`` 81-char strings from csv-like lines (defensive)."""
    for row in csv.reader(io.StringIO(text)):
        fields = [
            f.strip()
            for f in row
            if len(f.strip()) == 81 and all(c in "0123456789." for c in f.strip())
        ]
        if len(fields) >= 2:
            yield fields[0], fields[1]


def _boards_from_strings(pairs, meta: dict | None = None, check: bool = True) -> Boards:
    inputs, targets = [], []
    for q, a in pairs:
        qi, ai = string_to_board(q), string_to_board(a)
        if check and (not is_valid_solution(ai) or not np.all((qi == 0) | (qi == ai))):
            continue
        inputs.append(qi)
        targets.append(ai)
    if not inputs:
        raise ValueError("no valid board found")
    return _make_boards(np.stack(inputs), np.stack(targets), meta)


def _read_zip_member(zip_path: Path, name_suffix: str) -> bytes | None:
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.filename.endswith(name_suffix) and not info.is_dir():
                return z.read(info)
    return None


def load_rrn_official(root: str, split: str = "test", download: bool = True) -> Boards | None:
    """The RRN dataset (DGL mirror). ``split`` in ``{"train", "valid", "test"}``."""
    raw = Path(root) / "sudoku" / "raw" / "sudoku-hard.zip"
    if not raw.exists():
        if not download or _download(RRN_URL, raw) is None:
            return None
    try:
        data = _read_zip_member(raw, f"{split}.csv")
        if data is None:
            raise ValueError(f"{split}.csv not found in the archive")
        boards = _boards_from_strings(
            _parse_pair_lines(data.decode("utf-8", "replace")), {"source": "official:rrn"}
        )
        return boards
    except Exception as e:  # noqa: BLE001
        _notice(
            f"could not read the RRN archive ({type(e).__name__}: {e}); falling back to the local generator"
        )
        return None


def load_satnet_official(root: str, split: str = "train", download: bool = True) -> Boards | None:
    """The SAT-Net boards (needs torch). ``split``: "train" (first 9000) or "val" (last 1000)."""
    raw = Path(root) / "sudoku" / "raw" / "satnet-sudoku.zip"
    if not raw.exists():
        if not download or _download(SATNET_URL, raw) is None:
            return None
    try:
        import torch  # noqa: F401 - lazy, optional
    except ImportError:
        _notice(
            "torch is not installed: cannot read the SAT-Net .pt files; falling back to the local generator"
        )
        return None
    try:
        import torch

        feats = _read_zip_member(raw, "features.pt")
        labels = _read_zip_member(raw, "labels.pt")
        if feats is None or labels is None:
            raise ValueError("features.pt / labels.pt not found in the archive")
        x = torch.load(io.BytesIO(feats), map_location="cpu", weights_only=True)
        y = torch.load(io.BytesIO(labels), map_location="cpu", weights_only=True)
        inputs = _onehot_to_tokens(np.asarray(x))
        targets = _onehot_to_tokens(np.asarray(y))
        n_train = 9000
        if split == "train":
            inputs, targets = inputs[:n_train], targets[:n_train]
        elif split == "val":
            inputs, targets = inputs[n_train:], targets[n_train:]
        else:
            raise ValueError(split)
        return _make_boards(inputs, targets, {"source": "official:satnet"})
    except Exception as e:  # noqa: BLE001
        _notice(
            f"could not read the SAT-Net archive ({type(e).__name__}: {e}); falling back to the local generator"
        )
        return None


def _onehot_to_tokens(x: np.ndarray) -> np.ndarray:
    """``(N, 9, 9, 9)`` / ``(N, 81, 9)`` / ``(N, 729)`` one-hot (all-zero = empty) -> ``(N, 9, 9)``."""
    x = x.reshape(x.shape[0], 81, 9)
    tokens = x.argmax(axis=-1) + 1
    tokens[x.sum(axis=-1) == 0] = 0
    return tokens.reshape(-1, 9, 9)


def load_extreme_official(root: str, split: str = "test", download: bool = True) -> Boards | None:
    """Sudoku-Extreme csv (Hugging Face). Returns boards with ``meta["rating"]``."""
    name = "test.csv" if split == "test" else f"train_head{EXTREME_TRAIN_HEAD_LINES}.csv"
    raw = Path(root) / "sudoku" / "raw" / "sudoku-extreme" / name
    if not raw.exists():
        url = EXTREME_URL.format(split="test" if split == "test" else "train")
        max_lines = None if split == "test" else EXTREME_TRAIN_HEAD_LINES
        if not download or _download(url, raw, max_lines=max_lines) is None:
            return None
    try:
        inputs, targets, ratings = [], [], []
        with open(raw, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("empty csv")
            cols = [h.strip().lower() for h in header]
            qi = cols.index("question") if "question" in cols else 1
            ai = cols.index("answer") if "answer" in cols else 2
            ri = cols.index("rating") if "rating" in cols else 3
            for row in reader:
                if len(row) <= max(qi, ai):
                    continue
                q, a = row[qi].strip(), row[ai].strip()
                if len(q) != 81 or len(a) != 81:
                    continue
                inputs.append(string_to_board(q))
                targets.append(string_to_board(a))
                try:
                    ratings.append(int(float(row[ri])))
                except (ValueError, IndexError):
                    ratings.append(-1)
        if not inputs:
            raise ValueError("no board parsed")
        boards = _make_boards(
            np.stack(inputs), np.stack(targets), {"source": "official:sudoku-extreme"}
        )
        boards.meta["rating"] = np.array(ratings, dtype=np.int64)
        return boards
    except Exception as e:  # noqa: BLE001
        _notice(
            f"could not read {raw.name} ({type(e).__name__}: {e}); falling back to the local generator"
        )
        return None


# ----------------------------------------------------------------------------------------------
# dataset assembly
# ----------------------------------------------------------------------------------------------


def _subsample(boards: Boards, n: int | None, rng: np.random.Generator) -> Boards:
    if n is None or n >= len(boards):
        return boards
    return boards.subset(np.sort(rng.choice(len(boards), size=n, replace=False)))


def _spread_over_levels(boards: Boards, n: int | None, rng: np.random.Generator) -> Boards:
    """Take ``n`` boards spread evenly over the 18 RRN levels (by number of empty cells)."""
    if n is None:
        return boards
    levels = boards.meta["n_empty"]
    per = _split_counts(n, len(OOD_TEST_LEVELS))
    idx = []
    for lvl, k in zip(OOD_TEST_LEVELS, per, strict=True):
        cand = np.flatnonzero(levels == lvl)
        idx.extend(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
    rest = n - len(idx)
    if rest > 0:  # some levels are short: fill with remaining boards of any level
        others = np.setdiff1d(np.arange(len(boards)), np.array(idx, dtype=np.int64))
        idx.extend(rng.choice(others, size=min(rest, len(others)), replace=False).tolist())
    return boards.subset(np.sort(np.array(idx, dtype=np.int64)))


def _split_counts(n: int, k: int) -> list[int]:
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def bin_index(values: np.ndarray, bins=EXTREME_BINS) -> np.ndarray:
    """Index of the difficulty bin of every value (see ``EXTREME_BINS``)."""
    values = np.asarray(values)
    out = np.full(values.shape, len(bins) - 1, dtype=np.int64)
    for i, (lo, hi) in enumerate(bins):
        out[(values >= lo) & (values <= hi)] = i
    return out


def _stratify(boards: Boards, key: str, n: int | None, rng: np.random.Generator) -> Boards:
    """Take ~``n/len(bins)`` boards per difficulty bin (bins over ``meta[key]``).

    The equal quotas make the per-bin curves comparable but over-weight the rare bins, so every
    selected board also gets ``meta["weight"]`` = (share of its bin in ``boards``) / (number of
    selected boards of that bin): a weighted mean over the sample estimates the mean over the
    full benchmark (`tts_curves` reports it as ``conf_at_k_weighted``)."""
    bins = bin_index(boards.meta[key])
    boards.meta["bin"] = bins
    if n is None or n >= len(boards):
        return boards
    share = np.bincount(bins, minlength=len(EXTREME_BINS)) / len(bins)
    quota = _split_counts(n, len(EXTREME_BINS))
    idx = []
    for b, k in enumerate(quota):
        cand = np.flatnonzero(bins == b)
        idx.extend(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
    rest = n - len(idx)
    if rest > 0:
        others = np.setdiff1d(np.arange(len(boards)), np.array(idx, dtype=np.int64))
        idx.extend(rng.choice(others, size=min(rest, len(others)), replace=False).tolist())
    out = boards.subset(np.sort(np.array(idx, dtype=np.int64)))
    sel = np.bincount(out.meta["bin"], minlength=len(EXTREME_BINS))
    out.meta["weight"] = share[out.meta["bin"]] / np.maximum(sel[out.meta["bin"]], 1)
    return out


def _add_difficulty(boards: Boards, verbose: bool = True) -> Boards:
    if "difficulty" in boards.meta:
        return boards
    t0 = time.time()
    diff = np.zeros(len(boards), dtype=np.int64)
    for k, b in enumerate(boards.inputs):
        diff[k] = difficulty(b)
        if verbose and k % 20000 == 19999:
            _notice(f"rated {k + 1}/{len(boards)} boards ({time.time() - t0:.0f} s)")
    boards.meta["difficulty"] = diff
    return boards


def make_dataset(
    kind: str,
    split: str,
    n: int | None = None,
    seed: int = 0,
    root: str = "data",
    download: bool = True,
    source: str = "auto",
    stratify: bool = True,
    workers: int | None = None,
    max_tries: int = 3,
    verbose: bool = True,
) -> Boards:
    """Build (or load from cache) a Sudoku dataset.

    ``kind`` in ``{"ood", "extreme"}``, ``split`` in ``{"train", "val", "test"}``, ``source`` in
    ``{"auto", "official", "generated"}``. ``n=None`` means "the default size" for generated data
    (9000 / 1000 / 1800 for OOD, 1000 / 1000 for Extreme) and "everything" for official data.
    The result is cached as ``<root>/sudoku/sudoku_{kind}_{split}_n{N}_seed{seed}_{source}.npz``
    with ``N`` the actual number of boards. ``workers`` and ``max_tries`` (grids tried per board
    before accepting the lowest clue count reached) only concern the generator.
    """
    if kind not in ("ood", "extreme"):
        raise ValueError(kind)
    # "dev" (Sudoku-OOD only): RRN's own validation file, out of distribution like the test set but
    # disjoint from it, for choosing recipes without tuning on the test set (2026-10-21)
    if split not in ("train", "val", "test", "dev") or (split == "dev" and kind != "ood"):
        raise ValueError(split)
    if split == "dev" and source == "generated":
        raise ValueError("the dev split exists only for the official RRN data")
    if source not in ("auto", "official", "generated"):
        raise ValueError(source)
    # the Sudoku-Extreme test sample depends on `stratify`, so it is part of the cache identity
    sampling = (
        ("strat_v2" if stratify else "uniform") if (kind, split) == ("extreme", "test") else ""
    )
    cached = _find_cache(root, kind, split, n, seed, source, sampling)
    if cached is not None:
        if verbose:
            _notice(f"loading {cached}")
        return load_boards(cached)

    rng = np.random.default_rng(seed)
    boards = None
    used = None
    if source in ("auto", "official"):
        boards = _official(kind, split, root, download, n, stratify, rng, verbose)
        if boards is not None:
            used = "official"
        elif source == "official":
            raise RuntimeError(f"official data unavailable for sudoku_{kind}/{split}")
    if boards is None:
        boards = _generated(kind, split, n, seed, stratify, workers, max_tries, verbose)
        used = "generated"
    boards = _add_difficulty(boards, verbose)
    boards.meta["source"] = str(boards.meta.get("source", used))
    n_name = "all" if (n is None and used == "official") else len(boards)
    path = cache_path(root, kind, split, n_name, seed, used, sampling)
    save_boards(path, boards)
    if verbose:
        _notice(f"saved {len(boards)} boards to {path}")
    return boards


def _official(kind, split, root, download, n, stratify, rng, verbose):
    if kind == "ood":
        if split in ("test", "dev"):
            boards = load_rrn_official(root, "test" if split == "test" else "valid", download)
            if boards is not None:
                boards = _spread_over_levels(boards, n, rng)
                boards.meta["level"] = boards.meta["n_empty"].copy()
            return boards
        boards = load_satnet_official(root, split, download)
        return None if boards is None else _subsample(boards, n, rng)
    boards = load_extreme_official(root, split, download)
    if boards is None:
        return None
    if split == "test":
        boards.meta["difficulty_scale"] = "rating(tdoku)"
        return _stratify(boards, "rating", n, rng) if stratify else _subsample(boards, n, rng)
    boards = _subsample(boards, n if n is not None else 1000, rng)
    boards.meta["bin"] = bin_index(boards.meta["rating"])
    return boards


def split_seed(kind: str, split: str, seed: int) -> int:
    """One independent random stream per (benchmark, split, seed). Before 2026-10-12 every split
    was generated from the same `seed`, and since board `k` always starts from the `k`-th child
    of that seed, train and test shared most of their solution grids (see docs/DEVIATIONS.md)."""
    kind_id = ("ood", "extreme").index(kind)
    split_id = ("train", "val", "test").index(split)
    return int(np.random.SeedSequence([int(seed), 7 + kind_id, 11 + split_id]).generate_state(1)[0])


def _generated(kind, split, n, seed, stratify, workers, max_tries, verbose):
    seed = split_seed(kind, split, seed)
    rng = np.random.default_rng(seed)
    if kind == "ood":
        if split == "test":
            n = 100 * len(OOD_TEST_LEVELS) if n is None else int(n)
            per = _split_counts(n, len(OOD_TEST_LEVELS))
            clues = [
                81 - lvl for lvl, k in zip(OOD_TEST_LEVELS, per, strict=True) for _ in range(k)
            ]
            boards = generate_boards(
                clues, seed, max_tries=max_tries, workers=workers, verbose=verbose
            )
            boards.meta["level"] = boards.meta["n_empty"].copy()
            boards.meta["level_target"] = 81 - np.array(clues, dtype=np.int64)
            if verbose:
                achieved = {
                    int(lvl): int((boards.meta["level"] == lvl).sum()) for lvl in OOD_TEST_LEVELS
                }
                _notice(f"achieved test levels (empty cells -> count): {achieved}")
            return boards
        n = (9000 if split == "train" else 1000) if n is None else int(n)
        clues = rng.integers(OOD_TRAIN_CLUES[0], OOD_TRAIN_CLUES[1] + 1, size=n)
        return generate_boards(clues, seed, max_tries=max_tries, workers=workers, verbose=verbose)
    # extreme
    n = 1000 if n is None else int(n)
    pool_n = 2 * n if (split == "test" and stratify) else n
    clues = rng.integers(EXTREME_TRAIN_CLUES[0], EXTREME_TRAIN_CLUES[1] + 1, size=pool_n)
    boards = generate_boards(clues, seed, max_tries=max_tries, workers=workers, verbose=verbose)
    boards.meta["difficulty_scale"] = "backtracks(ours)"
    if split == "test" and stratify:
        boards = _stratify(boards, "difficulty", n, rng)
    else:
        boards.meta["bin"] = bin_index(boards.meta["difficulty"])
    return boards


# ----------------------------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------------------------


def sudoku_symmetry(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Sample one Sudoku-invariant symmetry: ``(row_perm, col_perm, transpose, digit_perm)``."""

    def perm_within_bands() -> np.ndarray:
        bands = rng.permutation(3)
        return np.concatenate([3 * b + rng.permutation(3) for b in bands])

    digit_perm = np.concatenate([[0], rng.permutation(9) + 1])
    return perm_within_bands(), perm_within_bands(), bool(rng.integers(2)), digit_perm


def apply_symmetry(
    grid: np.ndarray, row_perm, col_perm, transpose: bool, digit_perm=None
) -> np.ndarray:
    g = np.asarray(grid)
    if transpose:
        g = g.T
    g = g[row_perm][:, col_perm]
    if digit_perm is not None:
        g = np.asarray(digit_perm)[g]
    return g


# ----------------------------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------------------------


class SudokuOOD(Task):
    """Sudoku-OOD: train on easy SAT-Net-like boards, test on the 18 RRN difficulty levels."""

    name = "sudoku_ood"
    kind = "ood"
    vocab_size = 10
    output_tokens = tuple(range(1, 10))
    token_names = ("empty",) + tuple(str(d) for d in range(1, 10))
    main_metric = "exact"

    def load(
        self,
        split: str,
        root: str,
        n: int | None = None,
        seed: int = 0,
        download: bool = True,
        source: str = "auto",
        **kwargs,
    ) -> Boards:
        """``split`` in ``{"train", "val", "test"}``; see ``make_dataset`` for the other options.

        ``n`` is the total number of boards (``None`` = default size for generated data, all for
        official data); for the test split it is spread evenly over the 18 levels.
        """
        return make_dataset(
            self.kind, split, n=n, seed=seed, root=root, download=download, source=source, **kwargs
        )

    def augment(self, boards: Boards, rng: np.random.Generator) -> Boards:
        """Apply an independent random Sudoku symmetry to every board (inputs/targets/given)."""
        inputs = np.empty_like(boards.inputs)
        targets = np.empty_like(boards.targets)
        given = np.empty_like(boards.given)
        for k in range(len(boards)):
            rp, cp, tr, dp = sudoku_symmetry(rng)
            inputs[k] = apply_symmetry(boards.inputs[k], rp, cp, tr, dp)
            targets[k] = apply_symmetry(boards.targets[k], rp, cp, tr, dp)
            given[k] = apply_symmetry(boards.given[k], rp, cp, tr)
        return Boards(inputs, targets, given, dict(boards.meta))

    def metrics(self, pred: np.ndarray, boards: Boards) -> dict[str, np.ndarray]:
        """``exact`` (all mutable cells right), ``valid`` (filled grid satisfies all constraints),
        ``cell_acc`` (fraction of mutable cells right)."""
        pred = np.asarray(pred)
        out = super().metrics(pred, boards)
        filled = np.where(boards.given, boards.inputs, pred)
        out["valid"] = np.array([is_valid_solution(g) for g in filled], dtype=bool)
        out["cell_acc"] = self.cell_accuracy(pred, boards)
        return out


class SudokuExtreme(SudokuOOD):
    """Sudoku-Extreme: 1000 training boards, test boards binned by tdoku rating (Figure 12)."""

    name = "sudoku_extreme"
    kind = "extreme"

    def load(
        self,
        split: str,
        root: str,
        n: int | None = None,
        seed: int = 0,
        download: bool = True,
        source: str = "auto",
        stratify: bool = True,
        **kwargs,
    ) -> Boards:
        """``split`` in ``{"train", "test"}`` (train default 1000, test default 1000 stratified
        over the 7 difficulty bins of ``EXTREME_BINS``; ``n=None`` = all 422,780 official test
        boards when available). meta: ``source``, ``n_empty``, ``rating`` (official tdoku
        backtracks) and/or ``difficulty`` (our solver's backtracks), ``bin``,
        ``difficulty_scale``."""
        if split == "val":
            raise ValueError(
                "sudoku_extreme has no validation split (use a held-out part of train)"
            )
        return make_dataset(
            self.kind,
            split,
            n=n,
            seed=seed,
            root=root,
            download=download,
            source=source,
            stratify=stratify,
            **kwargs,
        )


__all__ = [
    "EXAMPLE_PUZZLE",
    "EXAMPLE_SOLUTION",
    "EXTREME_BINS",
    "EXTREME_BIN_NAMES",
    "OOD_TEST_LEVELS",
    "SudokuExtreme",
    "SudokuOOD",
    "apply_symmetry",
    "bin_index",
    "board_to_string",
    "cache_path",
    "difficulty",
    "generate_board",
    "generate_boards",
    "generate_puzzle",
    "is_consistent",
    "is_valid_solution",
    "load_boards",
    "make_dataset",
    "pretty",
    "random_solution",
    "save_boards",
    "solve",
    "string_to_board",
    "sudoku_symmetry",
]
