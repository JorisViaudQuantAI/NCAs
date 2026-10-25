"""Counter-based randomness for the fast engine.

The reference training loop draws the firing mask and the state noise of every step from one
running GPU generator. That is fine as long as every step is executed exactly once, but it
breaks two of the engine's optimisations:

* `sample_noise` decides *on the CPU* whether any board receives noise this step
  (`active.sum() == 0`), which forces a GPU synchronisation 48 times per Sudoku-OOD iteration
  for a branch that is taken with probability 0.9^256 ~ 2e-12;
* segmented gradient checkpointing re-runs a segment of steps during the backward pass, and the
  re-run must see exactly the same masks and noise as the forward pass.

The engine therefore gives every step of every training iteration its own 62-bit seed, drawn
once per iteration from the run's NumPy generator (which is already part of the checkpoint).
`draw_step` regenerates the step's firing mask and noise from that seed alone, on the device,
without any CPU branch: the noise is `randn * sigma * active * cells`, exactly the reference
distribution, and a step where no board is active simply gets a zero noise tensor. Regenerating
a step is cheap (three uniform draws and one Gaussian draw), so a checkpointed segment never
has to store its noise tensors (48 dense Sudoku-OOD noise tensors would take 486 MiB in fp32).

The random *stream* differs from the reference loop's (which skips draws on inactive steps),
so a run of the engine and a run of `nca.train.train` with the same seed are statistically,
not bitwise, equivalent. All engine backends share this module, so they are bitwise comparable
with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SEED_BITS = 62


def step_seeds(rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` independent step seeds for one training iteration (int64, in [0, 2**62))."""
    return rng.integers(0, 2**SEED_BITS, size=n, dtype=np.int64)


@dataclass
class StepDraw:
    """The random tensors of one NCA step."""

    fire: torch.Tensor  # (B, 1, H, W) float in {0, 1}
    noise: torch.Tensor | None  # (B, C, H, W) dense noise (zero where not applied) or None
    noise_mask: torch.Tensor | None  # (B, 1, H, W) bool: the cells that received noise


def draw_step(
    seed: int,
    shape: tuple[int, int, int, int],
    p_fire: float,
    noise: dict | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> StepDraw:
    """Regenerate the firing mask and the noise of one step from its seed.

    `noise` is the training config's `{p_t, p_s, sigma}` (None or sigma <= 0: no noise). The
    draws happen in a fixed order (fire, active, cells, gaussian) so that every backend sees the
    same tensors for the same seed.
    """
    B, C, H, W = shape
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    u = torch.rand((B, 1, H, W), generator=gen, device=device)
    fire = (u < p_fire).to(dtype)
    if noise is None or noise["sigma"] <= 0 or noise["p_t"] <= 0 or noise["p_s"] <= 0:
        return StepDraw(fire, None, None)
    active = torch.rand((B, 1, 1, 1), generator=gen, device=device) < noise["p_t"]
    cells = torch.rand((B, 1, H, W), generator=gen, device=device) < noise["p_s"]
    mask = active & cells  # (B, 1, H, W) bool, no CPU branch
    g = torch.randn((B, C, H, W), generator=gen, device=device, dtype=dtype)
    return StepDraw(fire, g * (noise["sigma"] * mask.to(dtype)), mask)
