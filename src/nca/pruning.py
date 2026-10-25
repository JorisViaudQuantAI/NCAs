"""Niche-Capped Diversity Pruning (paper Section 3.3 and Appendix B.6).

When `K` stochastic trials run in parallel on the same board, many of them converge to the same
intermediate board. At `m - 1` equidistant checkpoints the ensemble is halved:

1. decode the current board of every alive trajectory;
2. group trajectories with an identical decoded board into *niches*;
3. keep at most `n_cap` trajectories per niche (the most confident ones);
4. halve the ensemble (to floor(alive / 2)), taking survivors round-robin across niches so that
   as many distinct hypotheses as possible stay represented.

Compute then scales as (2 - 2^-(m-1)) / m times the unpruned K x D (25%, 53%, 75% savings for
m = 2, 4, 8).
"""

from __future__ import annotations

import numpy as np
import torch


def niche_capped_pruning(
    decoded: torch.Tensor,
    conf: torch.Tensor,
    alive: torch.Tensor,
    n_cap: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """decoded (N, K, H, W) long, conf (N, K), alive (N, K) bool -> new alive (N, K) bool."""
    N, K = conf.shape
    dec = decoded.cpu().numpy()
    cf = conf.cpu().numpy()
    al = alive.cpu().numpy().copy()
    for n in range(N):
        idx = np.nonzero(al[n])[0]
        if len(idx) <= 1:
            continue
        target = len(idx) // 2
        niches: dict[bytes, list[int]] = {}
        for k in idx:
            niches.setdefault(dec[n, k].tobytes(), []).append(int(k))
        # cap each niche by confidence
        capped = []
        for members in niches.values():
            members = sorted(members, key=lambda k: -cf[n, k])
            capped.append(members[:n_cap])
        # round-robin across niches (random niche order, most confident member first)
        order = rng.permutation(len(capped))
        survivors: list[int] = []
        depth = 0
        while len(survivors) < target:
            added = False
            for j in order:
                if depth < len(capped[j]):
                    survivors.append(capped[j][depth])
                    added = True
                    if len(survivors) >= target:
                        break
            if not added:
                break
            depth += 1
        new = np.zeros(K, dtype=bool)
        new[survivors] = True
        al[n] = new
    return torch.from_numpy(al).to(alive.device)


def pruned_compute_fraction(m: int) -> float:
    """Fraction of the unpruned K x D compute used with m halvings (paper B.6)."""
    return (2 - 2 ** (-(m - 1))) / m
