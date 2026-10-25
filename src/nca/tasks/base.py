"""Common interface between the benchmarks (maze, sudoku, arc) and the NCA code.

A benchmark is a set of *boards*. A board is a small grid of discrete tokens. The NCA receives
the *input* tokens (clues, walls, endpoints), everything else being unknown, and must produce
the *target* tokens on the unknown cells.

Conventions shared by every task
--------------------------------
* Tokens are integers in `range(vocab_size)`; token 0 always means "empty / unknown".
* `inputs[n, i, j]` is the token given to the NCA at cell (i, j) of board n (0 if unknown).
* `targets[n, i, j]` is the full solution. On given cells it equals the input token.
* `given[n, i, j]` is True where the input is given (those cells are *frozen* in the NCA state:
  their input channels never change during a rollout).
* Metrics are computed on the cells of `Task.eval_mask` only, per board: by default the
  *mutable* cells (`~given`), whose prediction the NCA has to produce. ARC departs from this:
  there `given` is True everywhere (the input slice is read-only on the whole canvas) and the
  prediction lives in a separate output slice, so every cell carrying a target is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Boards:
    """A batch of boards of one size. All arrays are `(N, H, W)`."""

    inputs: np.ndarray  # int64, tokens given to the NCA (0 = unknown)
    targets: np.ndarray  # int64, full solution
    given: np.ndarray  # bool, True where `inputs` is given (frozen cells)
    meta: dict = field(default_factory=dict)  # per-board extra info (e.g. "difficulty": (N,))

    def __post_init__(self) -> None:
        assert self.inputs.shape == self.targets.shape == self.given.shape, "shape mismatch"
        assert self.inputs.ndim == 3, "boards must be (N, H, W)"
        self.inputs = self.inputs.astype(np.int64)
        self.targets = self.targets.astype(np.int64)
        self.given = self.given.astype(bool)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.inputs.shape[1], self.inputs.shape[2]

    def subset(self, idx) -> Boards:
        idx = np.asarray(idx)
        meta = {
            k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(self),) else v)
            for k, v in self.meta.items()
        }
        return type(self)(self.inputs[idx], self.targets[idx], self.given[idx], meta)

    def repeat(self, k: int) -> Boards:
        """Every board `k` times in a row (board n becomes rows n*k ... n*k+k-1), the per-board
        meta arrays (first dimension N) repeated along with it, scalar meta kept."""
        n = len(self)
        meta = {}
        for key, v in self.meta.items():
            per_board = isinstance(v, np.ndarray) and v.shape[:1] == (n,)
            meta[key] = np.repeat(v, k, axis=0) if per_board else v
        return type(self)(
            np.repeat(self.inputs, k, axis=0),
            np.repeat(self.targets, k, axis=0),
            np.repeat(self.given, k, axis=0),
            meta,
        )

    def concat(self, other: Boards) -> Boards:
        meta = {}
        for k in self.meta:
            a, b = self.meta[k], other.meta.get(k)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                meta[k] = np.concatenate([a, b])
            else:
                meta[k] = a
        return type(self)(
            np.concatenate([self.inputs, other.inputs]),
            np.concatenate([self.targets, other.targets]),
            np.concatenate([self.given, other.given]),
            meta,
        )


class Task:
    """What the NCA code needs to know about a benchmark.

    Subclasses set the class attributes and implement `load`, `augment` and `metrics`.
    """

    name: str = "task"
    vocab_size: int = 2
    #: tokens the NCA may predict on mutable cells (a subset of the vocabulary; the others are
    #: input-only tokens such as walls). Their embeddings live in the output slice.
    output_tokens: tuple[int, ...] = (0, 1)
    #: human-readable names, index = token
    token_names: tuple[str, ...] = ("empty", "one")
    #: the metric used as "accuracy" in tables (a key of `metrics()`'s result)
    main_metric: str = "exact"
    #: True when the decoded prediction of a given cell is its input token (maze, sudoku: the
    #: readout slice lies inside the clamped input slice, so the NCA cannot predict anything
    #: else there and input-only tokens such as walls are not even decodable). False when the
    #: readout is a separate slice and every cell's prediction is read from the NCA (ARC).
    fill_given: bool = True

    def load(self, split: str, root: str, **kwargs) -> Boards:
        """Return the boards of a split ("train", "val", "test", ...).

        Implementations should (1) look for a cached `.npz` under `root`, (2) try to download /
        convert the official data, (3) fall back to the local generator, and say which one they
        used (a `meta["source"]` string).
        """
        raise NotImplementedError

    def augment(self, boards: Boards, rng: np.random.Generator) -> Boards:
        """Dataset augmentation applied to fresh samples when they enter the replay pool.

        Must return boards with the same shape. Default: identity.
        """
        return boards

    def eval_mask(self, boards: Boards) -> np.ndarray:
        """`(N, H, W)` bool: the cells whose prediction is scored (metrics, accuracy traces).

        This is distinct from `boards.given`, the cells whose *input channels* are clamped.
        Default: the mutable cells `~given`. ARC overrides it (every cell with a target).
        """
        return ~boards.given

    def metrics(self, pred: np.ndarray, boards: Boards) -> dict[str, np.ndarray]:
        """Per-board metrics from predicted tokens `pred` `(N, H, W)`.

        Every task returns at least "exact": True when all scored cells match the target.
        """
        mask = self.eval_mask(boards)
        exact = np.array(
            [np.all(p[m] == t[m]) for p, t, m in zip(pred, boards.targets, mask, strict=True)]
        )
        return {"exact": exact}

    def cell_accuracy(self, pred: np.ndarray, boards: Boards) -> np.ndarray:
        """Fraction of scored cells correct, per board."""
        mask = self.eval_mask(boards)
        correct = (pred == boards.targets) & mask
        return correct.sum(axis=(1, 2)) / np.maximum(mask.sum(axis=(1, 2)), 1)


def exact_match(pred: np.ndarray, boards: Boards) -> np.ndarray:
    return Task.metrics(Task(), pred, boards)["exact"]


def _row_keys(a: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(a.astype(np.int64))}


def overlap_counts(a: Boards, b: Boards) -> dict[str, int]:
    """How many boards of `b` share a puzzle (`inputs`) or a solution (`targets`) with a board
    of `a`. Used to check that a generated test split does not leak into the train split."""
    if a.shape != b.shape:
        return dict(puzzles=0, solutions=0)
    puz, sol = _row_keys(a.inputs), _row_keys(a.targets)
    return dict(
        puzzles=int(sum(r.tobytes() in puz for r in b.inputs)),
        solutions=int(sum(r.tobytes() in sol for r in b.targets)),
    )
