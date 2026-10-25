"""Training (and test-time) perturbations: state noise, damage, target swap (paper B.2).

* **State noise** (during a rollout): with temporal probability `p_t` (per sample and step),
  Gaussian noise N(0, sigma^2) is added to a random subset of cells (spatial probability `p_s`).
  Clamped input channels are reset after every step, so they are effectively excluded.
* **Damage** (before a rollout, or at a chosen test-time step): with probability `p`, `n`
  circular patches of radius r ~ U(r_min, r_max) x grid size reset the cell state to zero
  (`fill`, `geometry: official` for the discs of the public code's `make_circle_masks`).
* **Target swap** (before a rollout): with probability `p`, the board of a pool sample (clues,
  targets, frozen cells) is replaced by a fresh one while its state is kept, so the NCA has to
  notice the change and re-solve (`pool.py: ReplayPool.replace_boards`, `swap_from_pool`).
* **Batch noise** (public code, opt-in, before a rollout): with probability `p` per board,
  N(0, sigma^2) noise on a random `ps` fraction of the cells, every channel except the input
  channels of the given cells (`apply_batch_noise`; the trainer renormalises and clamps).
"""

from __future__ import annotations

import numpy as np
import torch


def sample_noise(
    x: torch.Tensor,
    p_t: float,
    p_s: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Noise tensor for one step, or None when no sample is hit this step."""
    if sigma <= 0 or p_t <= 0 or p_s <= 0:
        return None
    B, C, H, W = x.shape
    dev = x.device
    active = (torch.rand((B, 1, 1, 1), generator=generator, device=dev) < p_t).to(x.dtype)
    if active.sum() == 0:
        return None
    cells = (torch.rand((B, 1, H, W), generator=generator, device=dev) < p_s).to(x.dtype)
    noise = torch.randn(x.shape, generator=generator, device=dev, dtype=x.dtype) * sigma
    return noise * active * cells


def circular_masks(
    B: int,
    H: int,
    W: int,
    n: int,
    r_min: float,
    r_max: float,
    rng: np.random.Generator,
    hit: np.ndarray | None = None,
    geometry: str = "ours",
) -> np.ndarray:
    """(B, H, W) bool masks: union of `n` random discs per board, radius relative to min(H, W).
    `hit` selects which boards get damaged (default: all).

    `geometry: official` draws the discs of the public code's `make_circle_masks` instead:
    cell coordinates `linspace(-1, 1)` along each axis, centres uniform in [-0.5, 0.5]^2,
    radius r ~ U(r_min, r_max) in those units and the disc `(y - cy)^2 + (x - cx)^2 < r^2`."""
    masks = np.zeros((B, H, W), dtype=bool)
    if geometry == "official":
        ys = np.linspace(-1.0, 1.0, H)[:, None]
        xs = np.linspace(-1.0, 1.0, W)[None, :]
        for b in range(B):
            if hit is not None and not hit[b]:
                continue
            for _ in range(n):
                cy, cx = rng.uniform(-0.5, 0.5, size=2)
                r = rng.uniform(r_min, r_max)
                masks[b] |= (ys - cy) ** 2 + (xs - cx) ** 2 < r**2
        return masks
    if geometry != "ours":
        raise ValueError(f"unknown damage geometry {geometry!r} (ours | official)")
    yy, xx = np.mgrid[0:H, 0:W]
    scale = min(H, W)
    for b in range(B):
        if hit is not None and not hit[b]:
            continue
        for _ in range(n):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            r = rng.uniform(r_min, r_max) * scale
            masks[b] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    return masks


def apply_damage(
    x: torch.Tensor,
    p: float,
    n: int,
    r_min: float,
    r_max: float,
    rng: np.random.Generator,
    force: bool = False,
    fill: float = 0.0,
    geometry: str = "ours",
) -> tuple[torch.Tensor, np.ndarray]:
    """Set the state to `fill` (default zero) inside random discs (`circular_masks`, `geometry`)
    for a fraction `p` of the boards (`force`: all). Returns the new state and the (B, H, W)
    damage masks (for figures)."""
    B, C, H, W = x.shape
    hit = np.ones(B, dtype=bool) if force else rng.random(B) < p
    if not hit.any():
        return x, np.zeros((B, H, W), dtype=bool)
    masks = circular_masks(B, H, W, n, r_min, r_max, rng, hit, geometry)
    m = torch.from_numpy(masks).to(x.device)[:, None]
    return x.masked_fill(m, fill), masks


def apply_batch_noise(
    x: torch.Tensor,
    given4: torch.Tensor,
    in_slice: slice,
    p: float,
    ps: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The public code's noise on a prepared batch: each board with probability `p`, each of its
    cells with probability `ps`, gets N(0, sigma^2) on every channel except the input channels
    (`in_slice`) of the given cells (`given4`, (B, 1, H, W) bool). Returns the noisy state, to
    be renormalised and clamped by the caller (`train.py: perturb_batch`)."""
    B, C, H, W = x.shape
    dev = x.device
    boards = torch.rand((B, 1, 1, 1), generator=generator, device=dev) < p
    cells = torch.rand((B, 1, H, W), generator=generator, device=dev) < ps
    noise = torch.randn(x.shape, generator=generator, device=dev, dtype=x.dtype) * sigma
    in_ch = torch.zeros((1, C, 1, 1), dtype=torch.bool, device=dev)
    in_ch[:, in_slice] = True
    keep = boards & cells & ~(given4.to(dev) & in_ch)
    return x + noise * keep.to(x.dtype)


def swap_indices(B: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of the samples whose board is swapped this iteration."""
    return np.nonzero(rng.random(B) < p)[0]
