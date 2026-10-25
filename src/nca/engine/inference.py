"""Compiled forward-only rollouts, including ARC's local firing clocks."""

import torch

from .rng import draw_step
from .step import draws_to_lists, make_step


def inference_block(step, x, given4, clamp, fires, noises, masks, clock, amp):
    for i, fire in enumerate(fires):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            x = step(
                x,
                given4,
                clamp,
                fire,
                None if noises is None else noises[i],
                None if masks is None else masks[i],
                clock,
            )
        x = x.float()
        if clock is not None:
            clock = clock + fire.long()
    return x, clock


class InferenceEngine:
    """No loss/backward; precision and noise are explicit, independent of training."""

    def __init__(self, model, config, amp=False, noise=None):
        self.model, self.config, self.amp, self.noise = model, config, amp, noise
        self.step = make_step(model, config.step_options())
        self.block = inference_block
        if config.channels_last:
            model.to(memory_format=torch.channels_last)
        if config.compile:
            self.block = torch.compile(inference_block, dynamic=False, mode=config.compile_mode)

    @torch.no_grad()
    def __call__(self, batch, seeds):
        x = batch["x"].detach().clone()
        if self.config.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        clock = (
            torch.zeros_like(batch["given4"], dtype=torch.long)
            if self.model.cfg.time_steps
            else None
        )
        if self.config.compile_mode in ("reduce-overhead", "max-autotune"):
            torch.compiler.cudagraph_mark_step_begin()
        for i in range(0, len(seeds), self.config.block):
            ds = [
                draw_step(
                    int(s),
                    tuple(x.shape),
                    self.model.cfg.fire_rate,
                    self.noise
                    if self.noise is not None and i + j < self.noise.get("r", 1.0) * len(seeds)
                    else None,
                    x.device,
                )
                for j, s in enumerate(seeds[i : i + self.config.block])
            ]
            fires, noises, masks = draws_to_lists(ds)
            x, clock = self.block(
                self.step, x, batch["given4"], batch["clamp"], fires, noises, masks, clock, self.amp
            )
            if self.config.compile_mode in ("reduce-overhead", "max-autotune"):
                x = x.clone()
                clock = None if clock is None else clock.clone()
        return x, clock


def uniform_inference_block(step, x, given4, clamp, fires, noises, amp):
    masks = [None if noise is None else (noise != 0).any(1, keepdim=True) for noise in noises]
    return inference_block(step, x, given4, clamp, fires, noises, masks, None, amp)[0]


class UniformRolloutBlocks:
    """Compiled blocks for the evaluation pipeline, using its original RNG draws.

    The caller stops blocks at trace, pruning and damage boundaries. Uniform
    firing is independent of the intermediate states, so drawing a block ahead
    preserves the generator stream. Adaptive firing and ARC clocks stay on the
    historical path until separately implemented and validated.
    """

    def __init__(self, model, config, amp=False):
        self.config, self.amp = config, amp
        self.step = make_step(model, config.step_options())
        self.block = uniform_inference_block
        self.graphs = config.compile and config.compile_mode in ("reduce-overhead", "max-autotune")
        if config.compile:
            self.block = torch.compile(
                uniform_inference_block, dynamic=False, mode=config.compile_mode
            )

    def __call__(self, x, given4, clamp, fires, noises):
        if self.config.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.graphs:
            torch.compiler.cudagraph_mark_step_begin()
        x = self.block(self.step, x, given4, clamp, fires, noises, self.amp)
        return x.clone() if self.graphs else x
