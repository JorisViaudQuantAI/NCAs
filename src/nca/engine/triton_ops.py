"""Optional CUDA kernels for wrap/zero fixed attention; imported only on CUDA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wrap_mix_forward(
    V,
    A,
    Z,
    B: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    BLOCK: tl.constexpr,
    NHWC: tl.constexpr,
    WRAP: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < B * K * D * H * W
    if NHWC:
        d, k = q % D, (q // D) % K
        j, i = (q // (K * D)) % W, (q // (K * D * W)) % H
        b = q // (K * D * H * W)
    else:
        j, i = q % W, (q // W) % H
        d, k, b = (q // (H * W)) % D, (q // (D * H * W)) % K, q // (K * D * H * W)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(9):
        ni, nj = i + n // 3 - 1, j + n % 3 - 1
        inside = valid
        if WRAP:
            ni, nj = (ni + H) % H, (nj + W) % W
        else:
            inside = valid & (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
        v = tl.load(V + b * S0 + k * S1 + d * S2 + ni * S3 + nj * S4, inside, 0).to(tl.float32)
        a = tl.load(A + k * A0 + n * A1 + i * A2 + j * A3, valid, 0).to(tl.float32)
        acc = tl.fma(v, a, acc)
    tl.store(Z + q, acc, valid)


@triton.jit
def _wrap_mix_dvalues(
    G,
    A,
    DV,
    B: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    G4: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    BLOCK: tl.constexpr,
    NHWC: tl.constexpr,
    WRAP: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < B * K * D * H * W
    if NHWC:
        d, k = q % D, (q // D) % K
        j, i = (q // (K * D)) % W, (q // (K * D * W)) % H
        b = q // (K * D * H * W)
    else:
        j, i = q % W, (q // W) % H
        d, k, b = (q // (H * W)) % D, (q // (D * H * W)) % K, q // (K * D * H * W)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(9):
        oi, oj = i + 1 - n // 3, j + 1 - n % 3
        inside = valid
        if WRAP:
            oi, oj = (oi + H) % H, (oj + W) % W
        else:
            inside = valid & (oi >= 0) & (oi < H) & (oj >= 0) & (oj < W)
        g = tl.load(G + b * G0 + k * G1 + d * G2 + oi * G3 + oj * G4, inside, 0).to(tl.float32)
        a = tl.load(A + k * A0 + n * A1 + oi * A2 + oj * A3, inside, 0).to(tl.float32)
        acc = tl.fma(g, a, acc)
    tl.store(DV + q, acc, valid)


@triton.jit
def _wrap_mix_dweights(
    V,
    G,
    DA,
    B: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    G4: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    WRAP: tl.constexpr,
    SPLITS: tl.constexpr = 1,
):
    kn = tl.program_id(0)
    k, n = kn // 9, kn % 9
    pos = tl.program_id(1) * P + tl.arange(0, P)
    i, j = pos // W, pos % W
    ni, nj = i + n // 3 - 1, j + n % 3 - 1
    inside = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
    if WRAP:
        ni, nj = (ni + H) % H, (nj + W) % W
        inside = pos < H * W
    acc = tl.full((R, P), 0, tl.float32)
    split = tl.program_id(2)
    for start in range(split, tl.cdiv(B * D, R), SPLITS):
        row = start * R + tl.arange(0, R)
        b, d = row // D, row % D
        valid = (row[:, None] < B * D) & (pos[None, :] < H * W)
        v = tl.load(
            V + b[:, None] * S0 + k * S1 + d[:, None] * S2 + ni[None, :] * S3 + nj[None, :] * S4,
            valid & inside[None, :],
            0,
        ).to(tl.float32)
        g = tl.load(
            G + b[:, None] * G0 + k * G1 + d[:, None] * G2 + i[None, :] * G3 + j[None, :] * G4,
            valid,
            0,
        ).to(tl.float32)
        acc += v * g
    tl.store(DA + (split * K * 9 + kn) * H * W + pos, tl.sum(acc, axis=0), pos < H * W)


class _WrappedMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, mix, wrap, split_k):
        B, K, D, H, W = v.shape
        nhwc = v.stride(2) == 1
        z = torch.empty(
            (B, K * D, H, W),
            device=v.device,
            dtype=torch.float32,
            memory_format=torch.channels_last if nhwc else torch.contiguous_format,
        ).view(B, K, D, H, W)
        _wrap_mix_forward[(triton.cdiv(v.numel(), 256),)](
            v, mix, z, B, K, D, H, W, *v.stride(), *mix.stride(), 256, nhwc, wrap
        )
        ctx.save_for_backward(v, mix)
        ctx.wrap = wrap
        ctx.split_k = split_k
        return z

    @staticmethod
    def backward(ctx, g):
        v, mix = ctx.saved_tensors
        B, K, D, H, W = v.shape
        nhwc = v.stride(2) == 1
        dv = torch.empty(
            (B, K * D, H, W),
            device=v.device,
            dtype=v.dtype,
            memory_format=torch.channels_last if nhwc else torch.contiguous_format,
        ).view(B, K, D, H, W)
        da = torch.empty((ctx.split_k, *mix.shape), device=mix.device, dtype=mix.dtype)
        _wrap_mix_dvalues[(triton.cdiv(v.numel(), 256),)](
            g, mix, dv, B, K, D, H, W, *g.stride(), *mix.stride(), 256, nhwc, ctx.wrap
        )
        if v.stride(2) == 1 and g.stride(2) == 1:
            reduce_rows, positions = 32, 16
        elif v.stride(-1) == 1 and g.stride(-1) == 1:
            reduce_rows, positions = 64, 32
        else:
            reduce_rows, positions = 64, 16
        _wrap_mix_dweights[(K * 9, triton.cdiv(H * W, positions), ctx.split_k)](
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
            reduce_rows,
            positions,
            ctx.wrap,
            ctx.split_k,
            enable_fp_fusion=False,
        )
        return dv, da[0] if ctx.split_k == 1 else da.sum(0), None, None


def wrapped_attention_mix_cuda(v, mix, padding="wrap", split_k=1):
    if v.dtype not in (torch.float32, torch.bfloat16) or mix.dtype != torch.float32:
        raise ValueError("wrapped Triton attention supports FP32/BF16 values and FP32 weights")
    if padding not in ("wrap", "zero"):
        raise ValueError("Triton attention supports wrap or zero padding")
    if split_k not in (1, 2, 4, 8, 16, 32):
        raise ValueError("split_k must be a power of two in 1..32")
    return _WrappedMix.apply(v, mix, padding == "wrap", split_k)


@triton.jit
def _shared_dinput(
    G,
    WT,
    DX,
    B: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    G4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    Hp: tl.constexpr = H + 2
    Wp: tl.constexpr = W + 2
    valid = q < B * C * Hp * Wp
    j, i = q % Wp, (q // Wp) % Hp
    c, b = (q // (Hp * Wp)) % C, q // (C * Hp * Wp)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(9):
        oi, oj = i - n // 3, j - n % 3
        mask = valid & (oi >= 0) & (oi < H) & (oj >= 0) & (oj < W)
        for k in tl.static_range(K):
            g = tl.load(G + b * G0 + k * G1 + c * G2 + oi * G3 + oj * G4, mask, 0).to(tl.float32)
            w = tl.load(WT + k * 9 + n).to(tl.float32)
            acc = tl.fma(g, w, acc)
    tl.store(DX + q, acc, valid)


@triton.jit
def _shared_dweight_partials(
    G,
    X,
    PART,
    B: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    G4: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    X3: tl.constexpr,
    NPART: tl.constexpr,
    BLOCK: tl.constexpr,
):
    kn, part = tl.program_id(0), tl.program_id(1)
    k, n = kn // 9, kn % 9
    q = part * BLOCK + tl.arange(0, BLOCK)
    valid = q < B * C * H * W
    j, i = q % W, (q // W) % H
    c, b = (q // (H * W)) % C, q // (C * H * W)
    g = tl.load(G + b * G0 + k * G1 + c * G2 + i * G3 + j * G4, valid, 0).to(tl.float32)
    x = tl.load(X + b * X0 + c * X1 + (i + n // 3) * X2 + (j + n % 3) * X3, valid, 0).to(tl.float32)
    tl.store(PART + kn * NPART + part, tl.sum(g * x, axis=0))


@triton.jit
def _shared_dweight_finish(PART, DW, NPART: tl.constexpr, BLOCK: tl.constexpr):
    kn = tl.program_id(0)
    q = tl.arange(0, BLOCK)
    values = tl.load(PART + kn * NPART + q, q < NPART, 0)
    tl.store(DW + kn, tl.sum(values, axis=0))


def shared_kernel_backward_cuda(gout, xp, weight):
    if xp.dtype not in (torch.float32, torch.bfloat16) or weight.dtype != xp.dtype:
        raise ValueError("Triton shared-kernel backward supports matching FP32/BF16 tensors")
    B, C, Hp, Wp = xp.shape
    K, H, W = weight.shape[0], Hp - 2, Wp - 2
    g = gout.reshape(B, K, C, H, W)
    gx = torch.empty(xp.shape, dtype=xp.dtype, device=xp.device)
    gw = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
    parts = triton.cdiv(B * C * H * W, 4096)
    partial = torch.empty((K * 9, parts), device=xp.device, dtype=torch.float32)
    _shared_dinput[(triton.cdiv(xp.numel(), 256),)](g, weight, gx, B, C, K, H, W, *g.stride(), 256)
    _shared_dweight_partials[(K * 9, parts)](
        g,
        xp,
        partial,
        B,
        C,
        K,
        H,
        W,
        *g.stride(),
        *xp.stride(),
        parts,
        4096,
        enable_fp_fusion=False,
    )
    _shared_dweight_finish[(K * 9,)](partial, gw, parts, triton.next_power_of_2(parts))
    return gx, gw
