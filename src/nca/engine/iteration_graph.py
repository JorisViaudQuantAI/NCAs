"""One CUDA Graph for a complete differentiable rollout and backward.

Replay/augmentation, random draws, AdamW, EMA and the scheduler remain outside
the graph. All draws and batch tensors are copied into owned static buffers on
every iteration; no random masks or stale parameter casts are reused.
"""

from dataclasses import replace

import numpy as np
import torch


class IterationGraph:
    def __init__(self, owner, batch, seeds, weights, grad_clip):
        from .engine import Engine, clone_batch

        if owner.device.type != "cuda":
            raise ValueError("iteration_graph requires CUDA")
        if owner.N != owner.Nc or owner.model.cfg.n_tasks or owner.model.cfg.time_steps:
            raise ValueError("iteration_graph requires chunk=rollout without task/step embeddings")
        if owner.state.segment or owner.cfg.microbatch or owner._cuda_graphs:
            raise ValueError(
                "iteration_graph requires no checkpointing/microbatching/nested CUDA graphs"
            )
        self.inner = Engine(
            owner.model, replace(owner.cfg, iteration_graph=False), owner.tr, owner.device
        )
        self.batch = clone_batch(batch)
        self.weights = weights.detach().clone()
        self.grad_clip = grad_clip
        self.shape = tuple(batch["x"].shape)
        self.draws = owner.draws(seeds, self.shape)
        self.capture_seeds = np.arange(owner.N)
        static_draws = self.draws
        self.inner.draws = lambda ids, shape: [static_draws[int(i)] for i in ids]
        self.parameters = [p for p in owner.model.parameters() if p.requires_grad]
        self.gradients = [torch.zeros_like(p) for p in self.parameters]
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=owner.device)
        stream.wait_stream(torch.cuda.current_stream(owner.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._bind_gradients()
                self._core()
        torch.cuda.current_stream(owner.device).wait_stream(stream)
        torch.cuda.synchronize(owner.device)
        self._bind_gradients()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = self._core()
        torch.cuda.current_stream(owner.device).wait_stream(stream)

    def _bind_gradients(self):
        # External optimizers may use zero_grad(set_to_none=True).
        for parameter, gradient in zip(self.parameters, self.gradients, strict=True):
            parameter.grad = gradient

    def _core(self):
        for gradient in self.gradients:
            gradient.zero_()
        return self.inner.train_iteration(
            self.batch, self.capture_seeds, 0, self.weights, self.grad_clip, None
        )

    def __call__(self, owner, batch, seeds, c0, weights, grad_clip, apply_update):
        if c0 != 0 or tuple(batch["x"].shape) != self.shape or grad_clip != self.grad_clip:
            raise ValueError("iteration_graph cannot change batch shape, chunk start or clipping")
        for key, target in self.batch.items():
            source = batch[key]
            if torch.is_tensor(target):
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError(f"iteration_graph input shape/dtype changed: {key}")
                target.copy_(source)
        self.weights.copy_(weights)
        for target, source in zip(self.draws, owner.draws(seeds, self.shape), strict=True):
            target.fire.copy_(source.fire)
            if target.noise is not None:
                target.noise.copy_(source.noise)
                target.noise_mask.copy_(source.noise_mask)
        self._bind_gradients()
        self.graph.replay()
        state, loss, norm = (value.detach().clone() for value in self.outputs)
        if apply_update is not None:
            apply_update(norm)
        return state, loss, norm
