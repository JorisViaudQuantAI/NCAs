"""Fixed token embeddings (paper B.1, "Task Embedding").

Discrete tokens v in {0, ..., V-1} are mapped to fixed (non-learned) vectors e_v in R^{C_in}.

* `onehot`: e_v is the v-th basis vector. Used for the maze tasks, where the output slice is the
  first C_out channels: the two predictable tokens (empty, path) then occupy exactly the output
  slice, and input-only tokens (wall, endpoints) are orthogonal to it.
* `orthonormal`: rows of a random orthogonal matrix (fixed seed). Used for Sudoku and ARC, where
  every channel group gets a non-zero component (needed by the per-group normalisation).
* `semantic`: the maze embeddings of the authors' public code (reasoning-nca, `reset_params`):
  every entry is -1 except the first `c_out` coordinates, which place each token on its own
  point of the readout space (Maze-OOD, c_out = 2: empty (0, -1), path (0, 1), wall (1, 0),
  endpoint (-1, 0); Maze-Hard, c_out = 3: empty (0, 0, 0), path (0, 0, 1), wall (1, 0, -1),
  start (0, -1, 1), goal (0, 1, 1)). Unlike `onehot`, walls and endpoints have distinct readout
  coordinates, so a readout over the whole vocabulary can name them, and a clamped cell sits
  exactly on its own embedding (confidence 1 with the L2 head). The PDF describes orthogonal
  embeddings instead; see docs/DEVIATIONS.md.
"""

from __future__ import annotations

import numpy as np
import torch

SEMANTIC_HEADS = {
    (4, 2): [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)],
    (5, 3): [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, -1.0), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0)],
}


def make_embeddings(
    vocab_size: int, dim: int, kind: str = "onehot", seed: int = 0, c_out: int | None = None
) -> torch.Tensor:
    """Return a `(vocab_size, dim)` float tensor of token embeddings (orthonormal rows except for
    `semantic`, which needs `c_out`)."""
    if vocab_size > dim:
        raise ValueError(f"need dim >= vocab_size, got dim={dim} < V={vocab_size}")
    if kind == "onehot":
        e = torch.zeros(vocab_size, dim)
        e[torch.arange(vocab_size), torch.arange(vocab_size)] = 1.0
        return e
    if kind == "semantic":
        head = SEMANTIC_HEADS.get((vocab_size, c_out))
        if head is None:
            raise ValueError(f"no semantic embedding for vocab {vocab_size}, c_out {c_out}")
        e = torch.full((vocab_size, dim), -1.0)
        e[:, :c_out] = torch.tensor(head)
        return e
    if kind == "orthonormal":
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        return torch.tensor(q[:vocab_size], dtype=torch.float32)
    raise ValueError(f"unknown embedding kind {kind!r}")
