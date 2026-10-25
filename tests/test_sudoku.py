"""Tests of the Sudoku benchmark module (offline, small sizes; whole file < ~30 s)."""

import numpy as np
import pytest

from nca.tasks import sudoku as S
from nca.tasks.base import Boards


def test_solver_known_puzzle():
    puzzle = S.string_to_board(S.EXAMPLE_PUZZLE)
    n, sol, _ = S.solve(puzzle, count_limit=2)
    assert n == 1
    assert S.board_to_string(sol) == S.EXAMPLE_SOLUTION
    assert S.is_valid_solution(sol)
    assert np.all((puzzle == 0) | (puzzle == sol))


def test_solver_empty_board_has_many_solutions():
    n, sol, _ = S.solve(np.zeros((9, 9), np.int64), count_limit=2)
    assert n == 2
    assert S.is_valid_solution(sol)


def test_solver_rejects_inconsistent_board():
    g = S.string_to_board(S.EXAMPLE_PUZZLE)
    g[0, 1] = 5  # duplicate 5 in the first row
    assert S.solve(g)[0] == 0
    assert not S.is_consistent(g)


def test_random_solution_is_valid_and_seeded():
    a = S.random_solution(np.random.default_rng(1))
    b = S.random_solution(np.random.default_rng(1))
    c = S.random_solution(np.random.default_rng(2))
    assert S.is_valid_solution(a)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_puzzle_30_clues_unique():
    puzzle, sol = S.generate_puzzle(np.random.default_rng(0), n_clues=30)
    assert int((puzzle != 0).sum()) == 30
    assert np.all((puzzle == 0) | (puzzle == sol))
    n, found, _ = S.solve(puzzle, count_limit=2)
    assert n == 1
    assert np.array_equal(found, sol)


def test_generate_board_refills_to_exact_count():
    puzzle, sol = S.generate_board(seed=3, n_clues=40)
    assert int((puzzle != 0).sum()) == 40
    assert S.solve(puzzle, count_limit=2)[0] == 1


def test_string_round_trip_and_pretty():
    g = S.string_to_board(S.EXAMPLE_PUZZLE)
    assert S.board_to_string(g) == S.EXAMPLE_PUZZLE
    dotted = S.EXAMPLE_PUZZLE.replace("0", ".")
    assert np.array_equal(S.string_to_board(dotted), g)
    p = S.pretty(g)
    assert p.count("\n") == 10 and "------+-------+------" in p and p.startswith("5 3 . | . 7 .")
    with pytest.raises(ValueError):
        S.string_to_board("123")


def _boards_from_pairs(pairs):
    inputs = np.stack([S.string_to_board(q) for q, _ in pairs])
    targets = np.stack([S.string_to_board(a) for _, a in pairs])
    return Boards(inputs, targets, inputs != 0, {})


def test_augmentation_preserves_validity_uniqueness_and_given():
    task = S.SudokuOOD()
    rng = np.random.default_rng(0)
    puzzle, sol = S.generate_puzzle(rng, n_clues=32)
    boards = Boards(puzzle[None], sol[None], (puzzle != 0)[None], {"n_empty": np.array([49])})
    for _ in range(5):
        aug = task.augment(boards, rng)
        assert aug.shape == (9, 9) and len(aug) == 1
        assert S.is_valid_solution(aug.targets[0])
        assert np.array_equal(aug.given[0], aug.inputs[0] != 0)
        assert np.all((aug.inputs[0] == 0) | (aug.inputs[0] == aug.targets[0]))
        assert aug.given.sum() == boards.given.sum()
        n, found, _ = S.solve(aug.inputs[0], count_limit=2)
        assert n == 1 and np.array_equal(found, aug.targets[0])
    # a non-trivial symmetry actually changes the board (with overwhelming probability)
    assert not np.array_equal(aug.targets[0], sol)


def test_metrics():
    task = S.SudokuOOD()
    puzzle = S.string_to_board(S.EXAMPLE_PUZZLE)
    sol = S.string_to_board(S.EXAMPLE_SOLUTION)
    boards = Boards(puzzle[None], sol[None], (puzzle != 0)[None], {})
    m = task.metrics(sol[None], boards)
    assert set(m) >= {"exact", "valid", "cell_acc"}
    assert bool(m["exact"][0]) and bool(m["valid"][0]) and m["cell_acc"][0] == 1.0
    # swap two mutable digits of the solution
    wrong = sol.copy()
    mutable = np.argwhere(puzzle == 0)
    (a, b) = mutable[0], mutable[1]
    wrong[a[0], a[1]], wrong[b[0], b[1]] = sol[b[0], b[1]], sol[a[0], a[1]]
    assert wrong[a[0], a[1]] != sol[a[0], a[1]]
    m = task.metrics(wrong[None], boards)
    assert not bool(m["exact"][0]) and not bool(m["valid"][0])
    assert 0 < m["cell_acc"][0] < 1
    # a prediction that is garbage on given cells is ignored there (given cells are filled in)
    garbage = sol.copy()
    garbage[puzzle != 0] = 0
    m = task.metrics(garbage[None], boards)
    assert bool(m["exact"][0]) and bool(m["valid"][0])


def test_generated_ood_train_and_cache(tmp_path):
    task = S.SudokuOOD()
    b = task.load("train", str(tmp_path), n=20, seed=0, download=False, source="generated")
    assert len(b) == 20 and b.shape == (9, 9)
    assert b.inputs.dtype == np.int64 and b.given.dtype == bool
    assert b.meta["source"] == "generated"
    assert np.all((81 - b.meta["n_empty"] >= 31) & (81 - b.meta["n_empty"] <= 42))
    assert np.array_equal(b.meta["n_empty"], (~b.given).sum(axis=(1, 2)))
    assert b.meta["difficulty"].shape == (20,)
    for p, s in zip(b.inputs, b.targets, strict=True):
        assert S.is_valid_solution(s) and S.solve(p, 2)[0] == 1
    path = S.cache_path(str(tmp_path), "ood", "train", 20, 0, "generated")
    assert path.exists() and path.name == "sudoku_ood_train_n20_seed0_generated_v2.npz"
    b2 = task.load("train", str(tmp_path), n=20, seed=0, download=False, source="generated")
    assert np.array_equal(b.inputs, b2.inputs) and np.array_equal(b.targets, b2.targets)
    assert np.array_equal(b.given, b2.given) and np.array_equal(
        b.meta["difficulty"], b2.meta["difficulty"]
    )
    assert b2.meta["source"] == "generated"
    # "auto" finds the generated cache when nothing official is there
    b3 = task.load("train", str(tmp_path), n=20, seed=0, download=False, source="auto")
    assert np.array_equal(b.inputs, b3.inputs)


def test_generated_ood_test_levels(tmp_path):
    task = S.SudokuOOD()
    b = task.load("test", str(tmp_path), n=36, seed=0, download=False, source="generated")
    assert len(b) == 36
    assert np.array_equal(b.meta["level"], b.meta["n_empty"])
    targets = b.meta["level_target"]
    assert sorted(set(targets.tolist())) == list(S.OOD_TEST_LEVELS)
    # every reachable level is hit exactly; the hardest ones are best effort (never harder)
    assert np.all(b.meta["n_empty"] <= targets)
    assert np.all(b.meta["n_empty"][targets <= 56] == targets[targets <= 56])
    for p in b.inputs:
        assert S.solve(p, 2)[0] == 1


def test_generated_extreme(tmp_path):
    task = S.SudokuExtreme()
    b = task.load("test", str(tmp_path), n=14, seed=0, download=False, source="generated")
    assert len(b) == 14
    assert b.meta["difficulty_scale"] == "backtracks(ours)"
    assert np.array_equal(b.meta["bin"], S.bin_index(b.meta["difficulty"]))
    tr = task.load("train", str(tmp_path), n=10, seed=1, download=False, source="generated")
    assert len(tr) == 10 and "bin" in tr.meta
    with pytest.raises(ValueError):
        task.load("val", str(tmp_path), n=5, download=False, source="generated")


def test_bin_index():
    assert S.bin_index(np.array([0, 1, 5, 6, 11, 12, 16, 17, 28, 29, 53, 54, 289])).tolist() == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
    ]


def test_official_source_without_download_raises(tmp_path):
    with pytest.raises(RuntimeError):
        S.SudokuOOD().load("test", str(tmp_path), n=18, download=False, source="official")


def test_task_attributes():
    t = S.SudokuOOD()
    assert t.name == "sudoku_ood" and t.vocab_size == 10 and t.output_tokens == tuple(range(1, 10))
    assert len(t.token_names) == 10 and t.main_metric == "exact"
    assert S.SudokuExtreme().name == "sudoku_extreme"


def test_generated_splits_use_distinct_streams(tmp_path):
    """Before 2026-10-12 every split was generated from the same seed, so board k of the test
    split had the solution of board k of the train split (found in code review)."""
    from nca.tasks.base import overlap_counts

    task = S.SudokuOOD()
    tr = task.load("train", str(tmp_path), n=12, seed=0, download=False, source="generated")
    te = task.load("test", str(tmp_path), n=12, seed=0, download=False, source="generated")
    assert S.split_seed("ood", "train", 0) != S.split_seed("ood", "test", 0)
    assert S.split_seed("ood", "train", 0) != S.split_seed("extreme", "train", 0)
    assert overlap_counts(tr, te) == dict(puzzles=0, solutions=0)
    assert not np.array_equal(tr.targets, te.targets)
    ex = S.SudokuExtreme()
    etr = ex.load("train", str(tmp_path), n=12, seed=0, download=False, source="generated")
    ete = ex.load("test", str(tmp_path), n=12, seed=0, download=False, source="generated")
    assert overlap_counts(etr, ete) == dict(puzzles=0, solutions=0)
    # the helper itself: a split against itself overlaps completely
    assert overlap_counts(tr, tr) == dict(puzzles=12, solutions=12)


def test_unversioned_generated_cache_is_ignored(tmp_path):
    """A v1 cache (same seed for every split) must not be picked up (second review, P2)."""
    task = S.SudokuOOD()
    b = task.load("train", str(tmp_path), n=12, seed=0, download=False, source="generated")
    v2 = S.cache_path(str(tmp_path), "ood", "train", 12, 0, "generated")
    v1 = v2.with_name("sudoku_ood_train_n12_seed0_generated.npz")
    v2.rename(v1)  # pretend only an old cache exists
    assert S._find_cache(str(tmp_path), "ood", "train", 12, 0, "auto") is None
    b2 = task.load("train", str(tmp_path), n=12, seed=0, download=False, source="generated")
    assert v2.exists() and np.array_equal(b.inputs, b2.inputs)
