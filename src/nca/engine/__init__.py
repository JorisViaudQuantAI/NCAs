"""Fast training engine: the NCA of `nca.model` with fused operators, block compilation and
segmented checkpointing behind one interface (docs/FAST_ENGINE.md)."""

from .engine import Engine, EngineConfig, EngineState
from .ops import fixed_attention_mix, fused_cell_update, project_values
from .rng import StepDraw, draw_step, step_seeds
from .step import StepOptions, make_step, run_block

__all__ = [
    "Engine",
    "EngineConfig",
    "EngineState",
    "StepDraw",
    "StepOptions",
    "draw_step",
    "fixed_attention_mix",
    "fused_cell_update",
    "make_step",
    "project_values",
    "run_block",
    "step_seeds",
]
