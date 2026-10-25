"""Sample replay buffer (paper B.2, "Training pipeline").

The pool holds `M` evolving grid states. Every training iteration draws a batch of `B` states,
replaces a fraction `r_seed` of them by fresh boards (age 0), unrolls the NCA on the batch and
writes the final states back. A state therefore accumulates rollouts over many iterations
(its `age`), which lets the NCA experience long trajectories without backpropagating through
them, and teaches it to keep a solved board stable.

The pool only ever stores *detached* tensors. ARC states carry the learned task embedding in
their task slice: `fresh()` detaches it, and `sample()` writes the *current* embedding (with
its autograd history) into the fresh boards of a batch, so the embedding receives a gradient
within the step without the pool keeping a stale graph alive across steps.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .model import NCA
from .tasks.base import Boards, Task


class ReplayPool:
    def __init__(
        self,
        model: NCA,
        task: Task,
        boards: Boards,
        size: int,
        rng: np.random.Generator,
        generator: torch.Generator | None,
        augment: bool = True,
        task_ids: np.ndarray | None = None,
    ):
        self.model = model
        self.task = task
        self.boards = boards
        self.rng = rng
        self.generator = generator
        self.augment = augment
        self.task_ids = task_ids  # (len(boards),) or None
        self.size = size
        self.device = model.device
        fresh = self.fresh(size)
        self.x = fresh["x"]
        self.given4 = fresh["given4"]
        self.clamp = fresh["clamp"]
        self.inputs = fresh["inputs"]
        self.targets = fresh["targets"]
        self.tids = fresh["task_ids"]
        self.loss_mask = fresh["loss_mask"]
        self.age = torch.zeros(size, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ fresh boards
    def fresh(self, n: int) -> dict[str, torch.Tensor | None]:
        """`n` fresh (augmented) boards with freshly initialised states."""
        idx = self.rng.integers(0, len(self.boards), size=n)
        sub = self.boards.subset(idx)
        if self.augment:
            sub = self.task.augment(sub, self.rng)
        inputs = torch.from_numpy(sub.inputs).to(self.device)
        targets = torch.from_numpy(sub.targets).to(self.device)
        given = torch.from_numpy(sub.given).to(self.device)
        tids = None
        if self.task_ids is not None:
            tids = torch.from_numpy(self.task_ids[idx]).to(self.device)
        x, given4, clamp = self.model.init_state(inputs, given, tids, self.generator)
        x = x.detach()  # the task embedding's graph is re-attached per batch in `sample`
        lm = sub.meta.get("loss_mask")
        loss_mask = None if lm is None else torch.from_numpy(np.asarray(lm)).to(self.device).float()
        return dict(
            x=x,
            given4=given4,
            clamp=clamp,
            inputs=inputs,
            targets=targets,
            task_ids=tids,
            loss_mask=loss_mask,
        )

    # ------------------------------------------------------------------ batches
    def sample(self, batch_size: int, seed_ratio: float) -> tuple[np.ndarray, dict]:
        """Draw `batch_size` pool entries; replace `round(seed_ratio * batch_size)` of them by
        fresh boards. Returns the pool indices and a dict of batch tensors (clones)."""
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        # index_select already owns its storage. Reuse one device index instead
        # of copying the same NumPy index to CUDA for every field, then cloning
        # every advanced-indexing result a second time.
        device_idx = torch.as_tensor(idx, device=self.device)
        n_fresh = int(round(seed_ratio * batch_size))
        batch = dict(
            x=self.x.index_select(0, device_idx),
            given4=self.given4.index_select(0, device_idx),
            clamp=self.clamp.index_select(0, device_idx),
            inputs=self.inputs.index_select(0, device_idx),
            targets=self.targets.index_select(0, device_idx),
            task_ids=None if self.tids is None else self.tids.index_select(0, device_idx),
            loss_mask=None
            if self.loss_mask is None
            else self.loss_mask.index_select(0, device_idx),
            age=self.age.index_select(0, device_idx),
        )
        if n_fresh > 0:
            f = self.fresh(n_fresh)
            for k in ("x", "given4", "clamp", "inputs", "targets"):
                batch[k][:n_fresh] = f[k]
            if batch["task_ids"] is not None:
                batch["task_ids"][:n_fresh] = f["task_ids"]
            if batch["loss_mask"] is not None:
                batch["loss_mask"][:n_fresh] = f["loss_mask"]
            batch["age"][:n_fresh] = 0
            if batch["task_ids"] is not None:
                # fresh boards start from the current task embedding (differentiable); replayed
                # states keep the task slice their rollouts evolved
                rows = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                rows[:n_fresh] = True
                batch["x"] = self.model.write_task_embedding(batch["x"], batch["task_ids"], rows)
        return idx, batch

    def replace_boards(self, batch: dict, which: np.ndarray) -> None:
        """Target swap: give the samples `which` a fresh board but keep their state."""
        if len(which) == 0:
            return
        f = self.fresh(len(which))
        w = torch.from_numpy(which).to(self.device)
        for k in ("given4", "clamp", "inputs", "targets"):
            batch[k][w] = f[k]
        if batch["task_ids"] is not None:
            batch["task_ids"][w] = f["task_ids"]
        if batch["loss_mask"] is not None:
            batch["loss_mask"][w] = f["loss_mask"]
        batch["x"] = self.model.apply_clamp(batch["x"], batch["given4"], batch["clamp"])

    def swap_from_pool(self, batch: dict, which: np.ndarray) -> None:
        """Target swap of the public code (`train.swap_source: pool`): each sample of `which`
        takes the board of a pool entry drawn uniformly (with replacement) from the whole pool
        (inputs, targets, given cells, clamp values, loss mask, task id) and that entry's
        input-channel state on its given cells; the rest of the sample's state is kept and its
        age restarts at 0."""
        if len(which) == 0:
            return
        src = torch.as_tensor(self.rng.integers(0, self.size, size=len(which)), device=self.device)
        w = torch.as_tensor(which, device=self.device)
        for k, store in (
            ("given4", self.given4),
            ("clamp", self.clamp),
            ("inputs", self.inputs),
            ("targets", self.targets),
        ):
            batch[k][w] = store[src]
        if batch["task_ids"] is not None:
            batch["task_ids"][w] = self.tids[src]
        if batch["loss_mask"] is not None:
            batch["loss_mask"][w] = self.loss_mask[src]
        batch["age"][w] = 0
        sl = self.model.cfg.in_slice
        x = batch["x"]
        x_in = x[:, sl].clone()
        x_in[w] = torch.where(batch["given4"][w], self.x[src][:, sl], x_in[w])
        batch["x"] = torch.cat([x[:, : sl.start], x_in, x[:, sl.stop :]], dim=1)
        batch["x"] = self.model.apply_clamp(batch["x"], batch["given4"], batch["clamp"])

    def write_back(self, idx: np.ndarray, batch: dict) -> None:
        """Store the post-rollout states (and possibly swapped boards) back into the pool."""
        idx = torch.as_tensor(idx, device=self.device)
        with torch.no_grad():
            self.x[idx] = batch["x"].detach()
        self.given4[idx] = batch["given4"]
        self.clamp[idx] = batch["clamp"]
        self.inputs[idx] = batch["inputs"]
        self.targets[idx] = batch["targets"]
        if self.tids is not None:
            self.tids[idx] = batch["task_ids"]
        if self.loss_mask is not None:
            self.loss_mask[idx] = batch["loss_mask"]
        self.age[idx] = batch["age"] + 1

    def state_dict(self) -> dict:
        return dict(
            x=self.x,
            given4=self.given4,
            clamp=self.clamp,
            inputs=self.inputs,
            targets=self.targets,
            tids=self.tids,
            loss_mask=self.loss_mask,
            age=self.age,
        )

    def load_state_dict(self, d: dict) -> None:
        if "loss_mask" not in d and self.loss_mask is not None:
            # checkpoints written before the loss mask was saved: the pool keeps the mask of the
            # boards it was built with, which does not match the restored canvases. Harmless
            # when every batch is fresh (ARC, `replay: false`), wrong for replayed states.
            warnings.warn(
                "pool checkpoint without 'loss_mask': the restored states keep a freshly drawn "
                "loss mask that does not match their boards",
                stacklevel=2,
            )
        for k, v in d.items():
            setattr(self, k, None if v is None else v.to(self.device))
