"""ARC per-task adaptation: 51 distinct task IDs and joint weight/embedding updates.

This implements the compute loop, not official ARC accuracy evaluation or ensemble voting.
Only demonstration pairs are accepted; test targets must never be included.
"""

import copy

import numpy as np
import torch

from .engine.rng import step_seeds
from .engine.train import loss_weights
from .tasks.arc import make_boards, random_color_perm


def augmentation_boards(demonstrations, seed=0):
    rng = np.random.default_rng(seed)
    variants = [(0, np.arange(10))]
    variants += [(k, random_color_perm(rng)) for k in range(1, 6) for _ in range(10)]
    pairs, tids, names, ks, perms = [], [], [], [], []
    for tid, (k, perm) in enumerate(variants):
        for inp, target in demonstrations:
            if target is None:
                raise ValueError("TTT requires demonstration targets, not test examples")
            pairs.append((inp, target))
            tids.append(tid)
            names.append(f"ttt-{tid}")
            ks.append(k)
            perms.append(perm)
    return make_boards(
        pairs,
        tids,
        names,
        dihedral_k=ks,
        color_perm=perms,
        extra_meta={
            "source": "TTT demonstration augmentations",
            # ARCTask.augment re-renders the original pairs. Preserve each
            # embedding ID's offline frame when applying online placement.
            "base_dihedral_k": np.asarray(ks, dtype=np.int64),
            "base_color_perm": np.asarray(perms, dtype=np.int64),
        },
    )


class TTTSession:
    """One task per session. Caller loads pretrained common weights and fresh task embeddings."""

    def __init__(self, engine, pool, steps=10000, lr=1e-4):
        if engine.model.cfg.n_tasks != 51:
            raise ValueError("ARC TTT requires 51 freshly initialized task embeddings")
        self.engine, self.pool = engine, pool
        self.opt = torch.optim.Adam(engine.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, steps)
        self.weights = loss_weights(16, engine.Nc, engine.device)

    def iteration(self):
        _, batch = self.pool.sample(self.engine.tr["batch_size"], 1.0)
        seeds = step_seeds(self.pool.rng, self.engine.N)
        self.opt.zero_grad(set_to_none=True)

        def update(gn):
            if not bool(torch.isfinite(gn)):
                raise FloatingPointError("non-finite TTT gradient norm")
            self.opt.step()
            self.scheduler.step()

        return self.engine.train_iteration(batch, seeds, 0, self.weights, 1.0, update)


def prepare_ttt_model(pretrained, seed=0):
    """Copy pretrained shared weights; initialize 51 new task embeddings for one eval task."""
    model = copy.deepcopy(pretrained)
    model.cfg = copy.deepcopy(pretrained.cfg)
    model.cfg.n_tasks = 51
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model.task_embed = torch.nn.Embedding(51, model.cfg.c_task, device=model.device)
        torch.nn.init.normal_(model.task_embed.weight, std=model.cfg.sigma_init)
    return model
