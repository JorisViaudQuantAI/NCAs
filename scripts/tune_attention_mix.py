"""Bounded CUDA microbenchmark of the wrapped attention weight-gradient tiles.

This diagnoses one operator; end-to-end timings decide whether to keep a tile.
All inputs are generated with seed 321 and the actual Sudoku batch/grid shape.
"""

import statistics

import torch
import triton

from nca.engine.triton_ops import _wrap_mix_dweights


def run():
    torch.manual_seed(321)
    B, K, D, H, W = 256, 16, 8, 9, 9
    results = []
    for channels_last in (False, True):
        v = torch.randn(B, K * D, H, W, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, K * D, H, W, device="cuda")
        if channels_last:
            v = v.contiguous(memory_format=torch.channels_last)
            g = g.contiguous(memory_format=torch.channels_last)
        v, g = v.view(B, K, D, H, W), g.view(B, K, D, H, W)
        ref = torch.stack(
            [
                (torch.roll(v.float(), (1 - n // 3, 1 - n % 3), (-2, -1)) * g).sum((0, 2))
                for n in range(9)
            ],
            dim=1,
        )
        da = torch.empty_like(ref)
        for R in (32, 64, 128, 256):
            for P in (4, 8, 16, 32):

                def invoke(v=v, g=g, da=da, R=R, P=P):
                    _wrap_mix_dweights[(K * 9, triton.cdiv(H * W, P))](
                        v,
                        g,
                        da,
                        B,
                        K,
                        D,
                        H,
                        W,
                        *v.stride(),
                        *g.stride(),
                        R,
                        P,
                        enable_fp_fusion=False,
                    )

                for _ in range(5):
                    invoke()
                torch.cuda.synchronize()
                error = (da - ref).abs().max().item()
                valid = torch.allclose(da, ref, atol=1e-4, rtol=1e-5)
                times = []
                for _ in range(3):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(30):
                        invoke()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) / 30)
                results.append(
                    dict(
                        channels_last=channels_last,
                        R=R,
                        P=P,
                        milliseconds=times,
                        mean_ms=statistics.mean(times),
                        max_abs_error=error,
                        ok=valid,
                    )
                )
    return dict(shape=[B, K, D, H, W], seed=321, trials=results, ok=all(r["ok"] for r in results))
