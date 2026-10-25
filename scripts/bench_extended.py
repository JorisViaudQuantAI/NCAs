"""Timing helpers for forward-only inference and ARC test-time adaptation."""

import time

import numpy as np
import torch

from nca.engine.bench import _compiled_graphs, _cuda_graph_recordings, _custom_kernel_variants
from nca.engine.engine import EngineConfig
from nca.engine.inference import InferenceEngine
from nca.engine.rng import step_seeds
from nca.ttt import TTTSession


def timed(fn, iters, warmup):
    samples = []
    first = None
    before = captures = 0
    custom_before = _custom_kernel_variants()
    memory0 = {}
    warmup_memory = {}
    for i in range(warmup + iters):
        if i == warmup:
            before, captures = _compiled_graphs(), _cuda_graph_recordings(torch.device("cuda"))
            custom_before = _custom_kernel_variants()
            warmup_memory = dict(
                warmup_peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                warmup_peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
            )
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if first is None:
            first = elapsed
            memory0 = dict(
                first_peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                first_peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
            )
        if i >= warmup:
            samples.append(elapsed)
        del result
    return dict(
        seconds=samples,
        seconds_mean=float(np.mean(samples)),
        seconds_std=float(np.std(samples)),
        seconds_min=min(samples),
        iters=iters,
        warmup=warmup,
        first_iteration_seconds=first,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
        timed_compiled_graphs=_compiled_graphs() - before,
        timed_custom_kernel_variants=_custom_kernel_variants() - custom_before,
        timed_cuda_graph_recordings=_cuda_graph_recordings(torch.device("cuda")) - captures,
        **memory0,
        **warmup_memory,
    )


def inference(model, ecfg, batch, spec):
    # K independent trajectories are represented by batch rows. No pruning or voting here.
    seeds = step_seeds(np.random.default_rng(spec.get("seed", 0)), spec["depth"])
    noise = spec.get("inference_noise")
    runner = InferenceEngine(model, ecfg, spec["precision"] == "bf16", noise)
    if spec["mode"] == "validate_inference":
        ref = InferenceEngine(
            model,
            EngineConfig(backend="reference", checkpoint_segment=None),
            spec["precision"] == "bf16",
            noise,
        )
        a, ca = ref(batch, seeds)
        a = a.detach().clone()
        ca = None if ca is None else ca.clone()
        b, cb = runner(batch, seeds)
        err = float((a - b).abs().max())
        tol = 2e-2 if spec["precision"] == "bf16" else 1e-4
        ok = bool(torch.isfinite(b).all()) and err <= tol
        clocks_equal = ca is None or bool(torch.equal(ca, cb))
        with torch.no_grad():
            if model.cfg.head == "pixels":
                readout_error = float((model.decode(a) - model.decode(b)).abs().max())
                disagreement = None
            else:
                readout_error = float((model.scores(a) - model.scores(b)).abs().max())
                disagreement = float((model.decode(a) != model.decode(b)).float().mean())
        ok = ok and readout_error <= tol
        return dict(
            ok=ok and clocks_equal,
            checks=[
                dict(
                    ok=ok and clocks_equal,
                    state_max_abs=err,
                    tolerance=tol,
                    clocks_equal=clocks_equal,
                    readout_max_abs=readout_error,
                    decoded_disagreement_fraction=disagreement,
                )
            ],
            inference=dict(
                depth=len(seeds),
                batch=batch["x"].shape[0],
                local_clocks=bool(model.cfg.time_steps),
                noise=noise,
            ),
        )

    def call():
        x, _ = runner(batch, seeds)
        return model.decode(x)

    return dict(
        ok=True,
        measurement=timed(call, spec.get("iters", 5), spec.get("warmup", 3)),
        inference=dict(
            depth=len(seeds),
            batch=batch["x"].shape[0],
            local_clocks=bool(model.cfg.time_steps),
            noise=noise,
            includes="fixed initial state clone, RNG, rollout, GPU readout; no CPU transfer/voting",
        ),
    )


def ttt(engine, pool, spec):
    session = TTTSession(engine, pool)
    return dict(
        ok=True,
        measurement=timed(session.iteration, spec.get("iters", 5), spec.get("warmup", 3)),
        ttt=dict(
            steps_per_task=10000,
            variants=51,
            optimizer="Adam",
            lr=1e-4,
            weight_decay=0,
            ema=False,
            pretrained_checkpoint=spec.get("checkpoint"),
            data="synthetic demonstration pairs; timing only, no accuracy claim",
        ),
    )
