"""Reasoning with Neural Cellular Automata: a single-GPU replication.

Package layout (see docs/PAPER_MAP.md for the paper <-> code correspondence):

    nca.embeddings   fixed orthonormal token embeddings            (paper B.1, "Task Embedding")
    nca.model        the NCA cell: state slices, perception, update, firing mask, normalisation, heads
    nca.pool         sample replay buffer                           (paper B.2)
    nca.perturb      training perturbations: noise, damage, target swap  (paper B.2)
    nca.train        BPTT training loop with EMA, AdamW, schedules  (paper B.2, Table 3)
    nca.rollout      inference: depth D, K parallel trials, noise, adaptive firing, damage  (paper B.4)
    nca.pruning      Niche-Capped Diversity Pruning                 (paper B.6)
    nca.tasks        benchmarks: maze, sudoku, arc                  (paper Appendix A)
    nca.viz          figures and rollout animations
"""

__version__ = "0.1.0"
