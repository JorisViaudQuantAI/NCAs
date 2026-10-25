"""Selected-cell Sobel kernels; no dense perception or atomic gradient scatter."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["R"])
def _selected_forward(
    X,
    ROWS,
    Y,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    R,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    WRAP: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    pos = tl.load(ROWS + r, r < R, 0)
    b, i, j = pos // (H * W), (pos // W) % H, pos % W
    valid = (c[:, None] < C) & (r[None, :] < R)
    sx = tl.full((BC, BR), 0, tl.float32)
    sy = tl.full((BC, BR), 0, tl.float32)
    center = tl.full((BC, BR), 0, tl.float32)
    for dy in tl.static_range(3):
        for dx in tl.static_range(3):
            ni, nj = i + dy - 1, j + dx - 1
            inside = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
            if WRAP:
                ni, nj = (ni + H) % H, (nj + W) % W
                inside = r < R
            v = tl.load(
                X + b[None, :] * S0 + c[:, None] * S1 + ni[None, :] * S2 + nj[None, :] * S3,
                valid & inside[None, :],
                0,
            ).to(tl.float32)
            sx = sx + v * ((dx - 1) * (2 if dy == 1 else 1) / 8)
            sy = sy + v * ((dy - 1) * (2 if dx == 1 else 1) / 8)
            if dy == 1 and dx == 1:
                center = v
    out = r[None, :] * (3 * C) + 3 * c[:, None]
    tl.store(Y + out, center, valid)
    tl.store(Y + out + 1, sx, valid)
    tl.store(Y + out + 2, sy, valid)


@triton.jit
def _selected_backward(
    G,
    MAP,
    DX,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    WRAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < B * C * H * W
    c, pos = q % C, q // C
    b, i, j = pos // (H * W), (pos // W) % H, pos % W
    acc = tl.full((BLOCK,), 0, tl.float32)
    for dy in tl.static_range(3):
        for dx in tl.static_range(3):
            oi, oj = i + 1 - dy, j + 1 - dx
            inside = valid & (oi >= 0) & (oi < H) & (oj >= 0) & (oj < W)
            if WRAP:
                oi, oj = (oi + H) % H, (oj + W) % W
                inside = valid
            row = tl.load(MAP + b * H * W + oi * W + oj, inside, -1)
            active = inside & (row >= 0)
            off = row * (3 * C) + 3 * c
            gx = tl.load(G + off + 1, active, 0).to(tl.float32)
            gy = tl.load(G + off + 2, active, 0).to(tl.float32)
            acc = acc + gx * ((dx - 1) * (2 if dy == 1 else 1) / 8)
            acc = acc + gy * ((dy - 1) * (2 if dx == 1 else 1) / 8)
            if dy == 1 and dx == 1:
                acc = acc + tl.load(G + off, active, 0).to(tl.float32)
    tl.store(DX + q, acc, valid)


class _SelectedSobel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, wrap):
        B, C, H, W = x.shape
        out = torch.empty((rows.numel(), 3 * C), device=x.device, dtype=x.dtype)
        if rows.numel():
            _selected_forward[(triton.cdiv(C, 32), triton.cdiv(rows.numel(), 32))](
                x,
                rows,
                out,
                C,
                H,
                W,
                rows.numel(),
                *x.stride(),
                wrap,
                32,
                32,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(rows)
        ctx.shape, ctx.wrap = x.shape, wrap
        return out

    @staticmethod
    def backward(ctx, grad):
        (rows,) = ctx.saved_tensors
        B, C, H, W = ctx.shape
        row_map = torch.full((B * H * W,), -1, device=grad.device, dtype=torch.int32)
        row_map[rows] = torch.arange(rows.numel(), device=grad.device, dtype=torch.int32)
        dx = torch.empty(
            ctx.shape, device=grad.device, dtype=grad.dtype, memory_format=torch.channels_last
        )
        _selected_backward[(triton.cdiv(B * C * H * W, 256),)](
            grad.contiguous(),
            row_map,
            dx,
            B,
            C,
            H,
            W,
            ctx.wrap,
            256,
            enable_fp_fusion=False,
        )
        return dx, None, None


def selected_sobel_cuda(x, rows, padding):
    if padding not in ("wrap", "zero"):
        raise ValueError("selected Sobel supports wrap/zero padding")
    if torch.is_autocast_enabled("cuda"):
        x = x.to(torch.get_autocast_dtype("cuda"))
    return _SelectedSobel.apply(x, rows, padding == "wrap")
