"""Opt-in BF16 two-layer ReLU MLP with fused forward and input-gradient kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    W1,
    B1,
    W2,
    B2,
    H,
    Y,
    M: tl.constexpr,
    CI: tl.constexpr,
    J: tl.constexpr,
    CO: tl.constexpr,
    BM: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
    BO: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ci, co = tl.arange(0, BI), tl.arange(0, BO)
    x = tl.load(X + rows[:, None] * CI + ci[None, :], (rows[:, None] < M) & (ci[None, :] < CI), 0)
    acc = tl.full((BM, BO), 0, tl.float32)
    for start in range(tl.cdiv(J, BJ)):
        cj = start * BJ + tl.arange(0, BJ)
        w1 = tl.load(W1 + cj[None, :] * CI + ci[:, None], (ci[:, None] < CI) & (cj[None, :] < J), 0)
        b1 = tl.load(B1 + cj, cj < J, 0).to(tl.float32)
        hidden = (tl.dot(x, w1) + b1[None, :]).to(tl.bfloat16)
        hidden = tl.maximum(hidden, 0).to(tl.bfloat16)
        tl.store(
            H + rows[:, None] * J + cj[None, :], hidden, (rows[:, None] < M) & (cj[None, :] < J)
        )
        w2 = tl.load(W2 + co[None, :] * J + cj[:, None], (cj[:, None] < J) & (co[None, :] < CO), 0)
        acc += tl.dot(hidden, w2)
    b2 = tl.load(B2 + co, co < CO, 0).to(tl.float32)
    y = (acc + b2[None, :]).to(tl.bfloat16)
    tl.store(Y + rows[:, None] * CO + co[None, :], y, (rows[:, None] < M) & (co[None, :] < CO))


@triton.jit
def _backward(
    G,
    W1,
    W2,
    H,
    GH,
    GX,
    M: tl.constexpr,
    CI: tl.constexpr,
    J: tl.constexpr,
    CO: tl.constexpr,
    BM: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
    BO: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ci, co = tl.arange(0, BI), tl.arange(0, BO)
    g = tl.load(G + rows[:, None] * CO + co[None, :], (rows[:, None] < M) & (co[None, :] < CO), 0)
    acc = tl.full((BM, BI), 0, tl.float32)
    for start in range(tl.cdiv(J, BJ)):
        cj = start * BJ + tl.arange(0, BJ)
        w2 = tl.load(W2 + co[:, None] * J + cj[None, :], (co[:, None] < CO) & (cj[None, :] < J), 0)
        hidden = tl.load(
            H + rows[:, None] * J + cj[None, :], (rows[:, None] < M) & (cj[None, :] < J), 0
        )
        gh = tl.where(hidden > 0, tl.dot(g, w2).to(tl.bfloat16), 0).to(tl.bfloat16)
        tl.store(GH + rows[:, None] * J + cj[None, :], gh, (rows[:, None] < M) & (cj[None, :] < J))
        w1 = tl.load(W1 + cj[:, None] * CI + ci[None, :], (cj[:, None] < J) & (ci[None, :] < CI), 0)
        acc += tl.dot(gh, w1)
    gx = acc.to(tl.bfloat16)
    tl.store(GX + rows[:, None] * CI + ci[None, :], gx, (rows[:, None] < M) & (ci[None, :] < CI))


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        M, CI = x.shape
        J, CO = w1.shape[0], w2.shape[0]
        h = torch.empty((M, J), device=x.device, dtype=x.dtype)
        y = torch.empty((M, CO), device=x.device, dtype=x.dtype)
        _forward[(triton.cdiv(M, 64),)](
            x,
            w1,
            b1,
            w2,
            b2,
            h,
            y,
            M,
            CI,
            J,
            CO,
            64,
            max(16, triton.next_power_of_2(CI)),
            min(64, max(16, triton.next_power_of_2(J))),
            max(16, triton.next_power_of_2(CO)),
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(x, w1, w2, h)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w1, w2, h = ctx.saved_tensors
        g = grad.contiguous()
        M, CI = x.shape
        J, CO = w1.shape[0], w2.shape[0]
        gh, gx = torch.empty_like(h), torch.empty_like(x)
        _backward[(triton.cdiv(M, 64),)](
            g,
            w1,
            w2,
            h,
            gh,
            gx,
            M,
            CI,
            J,
            CO,
            64,
            max(16, triton.next_power_of_2(CI)),
            min(64, max(16, triton.next_power_of_2(J))),
            max(16, triton.next_power_of_2(CO)),
            num_warps=4,
            num_stages=1,
        )
        return gx, gh.T @ x, gh.sum(0), g.T @ h, g.sum(0)


def fused_mlp_update(z, update):
    if (
        not z.is_cuda
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise ValueError("triton_mlp requires CUDA BF16 autocast")
    B, C, H, W = z.shape
    if max(C, update.fc1.out_channels, update.fc2.out_channels) > 512:
        raise ValueError("triton_mlp supports widths up to 512")
    rows = z.permute(0, 2, 3, 1).reshape(-1, C).to(torch.bfloat16).contiguous()
    y = _MLP.apply(
        rows,
        update.fc1.weight[:, :, 0, 0].to(torch.bfloat16).contiguous(),
        update.fc1.bias.to(torch.bfloat16),
        update.fc2.weight[:, :, 0, 0].to(torch.bfloat16).contiguous(),
        update.fc2.bias.to(torch.bfloat16),
    )
    return y.reshape(B, H, W, -1).permute(0, 3, 1, 2)
