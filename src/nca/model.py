"""The Neural Cellular Automaton (paper Section 3.1 and Appendix B.1).

One NCA step, for every cell (i, j) of a `(B, C, H, W)` state tensor `x`:

1. (ARC only) add the learned step embedding to the hidden slice (paper eq. B.3-(1)); a cell that
   fires keeps it in its state, a cell that does not fire stays frozen.
2. **Perception**: look at the 3x3 Moore neighbourhood and produce a vector `z`:
   * `kernels`     - `K_heads` learned 3x3 kernels, each a 3x3 matrix of scalars shared by all
                     channels (paper B.1, eq. for z^(k)); `z` has `K_heads * C` channels (Maze);
   * `conv`        - `K_heads` dense 3x3 convolutions (C -> K_heads * C), the perception of
                     the first Maze runs (2026-10-10); see docs/DEVIATIONS.md 2026-10-12;
   * `sobel`       - fixed identity + Sobel-x + Sobel-y depthwise filters (Visual Sudoku);
   * `fixed_attn`  - per-position, per-head mixing weights over the 9 neighbours applied to a
                     shared value projection; `z` has `C` channels (Sudoku, ARC);
   * `self_attn`   - standard query-key attention over the 9 neighbours (ablation, Fig. 13).
3. **Update**: a shared MLP (`mlp`, ReLU) or `swiglu` block maps `z` to a residual update `dx`.
4. **Asynchronous firing**: `x <- x + m * dx` with a Bernoulli(p_fire) mask `m` per cell. With
   `project_update` (the authors' public code, Sudoku) the masked update is first made tangent
   to the state, group by group: `dx <- dx - <dx, x> x`.
5. (Sudoku, ARC) **group normalisation**: the state is split into `U` groups, each rescaled to
   unit L2 norm. Training noise is added before it, or after it with `noise_after_norm` (public
   code).
6. **Clamping**: the input channels of given cells are reset to their token embedding, so they
   never change (walls, clues, the ARC input image).

Predictions are read from the output slice with an L2 head (Maze), a cosine head (Sudoku) or a
linear head (ARC); the per-cell confidence is the maximum score.

Shapes: states are channels-first `(B, C, H, W)`; `given` masks are `(B, 1, H, W)` bool; clamp
values are `(B, C_in, H, W)`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import make_embeddings

# 3x3 Moore neighbourhood offsets (dy, dx), row-major, centre included
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


@dataclass
class NCAConfig:
    """Architecture hyper-parameters (paper Tables 3, 4 and 6)."""

    vocab_size: int
    channels: int  # C
    c_in: int  # size of the input (clamped) slice
    c_out: int  # size of the output (readout) slice
    output_tokens: tuple[int, ...]  # tokens the readout can predict
    in_offset: int = 0  # the input slice is channels [in_offset, in_offset + c_in)
    out_offset: int = 0  # the output slice is channels [out_offset, out_offset + c_out)
    c_task: int = 0  # ARC: size of the task-embedding slice (placed at the end)
    n_tasks: int = 0  # ARC: number of learned task embeddings
    embedding: str = "onehot"  # "onehot" | "orthonormal" | "semantic" (maze, public code)
    perception: str = "kernels"  # "kernels" | "conv" | "sobel" | "fixed_attn" | "self_attn"
    heads: int = 4  # K_heads
    grid_size: tuple[int, int] | None = None  # needed by fixed_attn (position-specific weights)
    padding: str = "wall"  # "wall" | "wrap" | "zero"
    wall_token: int | None = 2  # token whose embedding fills the border when padding == "wall"
    update: str = "mlp"  # "mlp" | "swiglu"
    expansion: int = 2  # E: hidden width = E * C (or E * dim(z), see hidden_from_perception)
    hidden_from_perception: bool = False  # MLP hidden = E * dim(z) instead of E * C (Maze)
    norm_groups: int = 0  # U: 0 = no normalisation
    # normalise only the cells that fired (or received noise) this step, so a non-firing cell
    # stays frozen as in the paper. False = the behaviour before 2026-10-13 (every cell
    # renormalised each step), which checkpoints saved without this key keep.
    norm_fired_only: bool = False
    sigma_init: float = 0.15  # std of the Gaussian initial state
    init_uniform: bool = False  # Visual Sudoku: U(-sigma, sigma) instead of N(0, sigma^2)
    fire_rate: float = 0.8  # p_fire
    head: str = "l2"  # "l2" | "cosine" | "mlp"
    time_steps: int = 0  # ARC: size of the learned step-embedding table (0 = none)
    zero_init_update: bool = True  # last layer of the update block starts at zero (do-nothing rule)
    attn_bias: bool = True  # self_attn: learned per-head relative-position bias
    # Switches for the choices where the authors' public code (reasoning-nca, a34414f) differs
    # from this implementation; the defaults keep the behaviour of every earlier run.
    project_update: bool = False  # remove the radial part of dx, per group, before the residual
    noise_after_norm: bool = False  # add the step noise after the normalisation, not before
    norm_init: bool = False  # normalise the initial state (otherwise the first step does it)
    norm_style: str = (
        "sqrt_eps"  # "sqrt_eps": x / sqrt(sum x^2 + 1e-4); "norm_eps": x / (|x| + 1e-6)
    )
    init_style: str = "torch"  # "torch" | "public": the public code's parameter initialisation

    def __post_init__(self) -> None:
        self.output_tokens = tuple(int(t) for t in self.output_tokens)
        if self.grid_size is not None:
            self.grid_size = tuple(int(s) for s in self.grid_size)
        c = self.channels
        assert self.in_offset + self.c_in <= c, "input slice exceeds the state"
        assert self.out_offset + self.c_out <= c, "output slice exceeds the state"
        assert self.c_task <= c, "task slice exceeds the state"
        if self.perception in ("fixed_attn",):
            assert self.grid_size is not None, "fixed_attn needs grid_size"
        if self.perception in ("fixed_attn", "self_attn"):
            assert c % self.heads == 0, "channels must be divisible by heads"
        if self.norm_groups:
            assert c % self.norm_groups == 0, "channels must be divisible by norm_groups"
        assert self.norm_style in ("sqrt_eps", "norm_eps"), self.norm_style
        assert self.init_style in ("torch", "public"), self.init_style
        if self.head in ("l2", "cosine"):
            # the readout compares against token embeddings, so it must sit inside the input slice
            assert self.in_offset <= self.out_offset, "output slice must lie inside the input slice"
            assert self.out_offset + self.c_out <= self.in_offset + self.c_in, (
                "output slice must lie inside the input slice for embedding-based heads"
            )

    @property
    def in_slice(self) -> slice:
        return slice(self.in_offset, self.in_offset + self.c_in)

    @property
    def out_slice(self) -> slice:
        return slice(self.out_offset, self.out_offset + self.c_out)

    @property
    def task_slice(self) -> slice:
        return slice(self.channels - self.c_task, self.channels)

    @property
    def hid_slice(self) -> slice:
        """Channels that are neither input, output nor task (used by the time embedding)."""
        lo = max(self.in_offset + self.c_in, self.out_offset + self.c_out)
        return slice(lo, self.channels - self.c_task)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NCAConfig:
        return cls(**d)


def _pad(x: torch.Tensor, mode: str, pad_vec: torch.Tensor | None) -> torch.Tensor:
    """Pad a (B, C, H, W) tensor by one cell on each side."""
    if mode == "wrap":
        return F.pad(x, (1, 1, 1, 1), mode="circular")
    xp = F.pad(x, (1, 1, 1, 1))
    if mode == "wall":
        assert pad_vec is not None
        v = pad_vec.to(xp.dtype)[None, :, None]
        xp[:, :, 0, :] = v
        xp[:, :, -1, :] = v
        xp[:, :, :, 0] = v
        xp[:, :, :, -1] = v
    return xp


def _shifts(xp: torch.Tensor, H: int, W: int):
    """Yield the 9 neighbour views of a padded tensor, in `OFFSETS` order."""
    for dy, dx in OFFSETS:
        yield xp[:, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]


class KernelPerception(nn.Module):
    """K_heads learned 3x3 kernels of scalars, each applied to every channel (paper B.1,
    "Convolutional Sensing": z^(k) = sum_n W_perc,n^(k) x_n with scalar W_perc,n^(k)). Head k's
    output is C channels, so `z` has K_heads * C channels; the parameter count (9 per head) is
    what makes the paper's Maze models 10K and 41K parameters. Mordvintsev et al.'s Sobel
    filters are the fixed version of this (`SobelPerception`)."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(heads, 1, 3, 3) / 3)
        self.channels = channels
        self.heads = heads
        self.out_dim = heads * channels

    def forward(self, xp: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, C = xp.shape[:2]
        z = F.conv2d(xp.reshape(B * C, 1, H + 2, W + 2), self.weight)  # (B*C, heads, H, W)
        return z.view(B, C, self.heads, H, W).transpose(1, 2).reshape(B, self.heads * C, H, W)


class ConvPerception(nn.Module):
    """K_heads dense 3x3 convolutions over the full state (C -> K_heads * C). This is the
    perception the Maze runs of 2026-10-10 used; `kernels` is the paper's."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, heads * channels, 3, bias=False)
        self.out_dim = heads * channels

    def forward(self, xp: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.conv(xp)


class SobelPerception(nn.Module):
    """Fixed identity + Sobel filters, depthwise (the original NCA of Mordvintsev et al.)."""

    def __init__(self, channels: int):
        super().__init__()
        ident = torch.tensor([[0.0, 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        kernels = torch.stack([ident, sx, sx.T])  # (3, 3, 3)
        # grouped conv: output channels 3c, 3c+1, 3c+2 read input channel c, so the three
        # filters must repeat per channel (ident, sx, sy, ident, sx, sy, ...)
        w = kernels[None].repeat(channels, 1, 1, 1).reshape(3 * channels, 1, 3, 3)
        self.register_buffer("weight", w)
        self.channels = channels
        self.out_dim = 3 * channels

    def forward(self, xp: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return F.conv2d(xp, self.weight, groups=self.channels)


class FixedAttention(nn.Module):
    """Position-specific mixing weights A[k, n, i, j] over the 9 neighbours of a shared value
    projection V^(k) (paper B.1, "Fixed-Attention Sensing")."""

    def __init__(self, channels: int, heads: int, grid_size: tuple[int, int]):
        super().__init__()
        self.heads = heads
        self.channels = channels
        H, W = grid_size
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        # start close to a uniform average over the neighbourhood, plus a little noise
        a = torch.full((heads, 9, H, W), 1.0 / 9) + 0.02 * torch.randn(heads, 9, H, W)
        self.mix = nn.Parameter(a)
        self.out_dim = channels

    def forward(self, xp: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = xp.shape[0]
        # NOTE: the value projection is applied to the padded tensor, so the border cells are
        # projected too (for wrap padding they are copies, for wall/zero they are constants).
        v = self.value(xp).view(B, self.heads, self.channels // self.heads, H + 2, W + 2)
        assert self.mix.shape[-2:] == (H, W), (
            f"fixed attention was built for a {tuple(self.mix.shape[-2:])} grid, got {(H, W)}"
        )
        z = None
        for n, (dy, dx) in enumerate(OFFSETS):
            vn = v[:, :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            term = vn * self.mix[None, :, n, None]  # (B, K, C/K, H, W)
            z = term if z is None else z + term
        return z.reshape(B, self.channels, H, W)


class SelfAttention(nn.Module):
    """Query-key attention over the 3x3 neighbourhood (the ablation of paper Appendix D.1)."""

    def __init__(self, channels: int, heads: int, rel_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads, 9)) if rel_bias else None
        self.out_dim = channels

    def forward(self, xp: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = xp.shape[0]
        d = self.channels // self.heads
        qkv = self.qkv(xp)
        q, k, v = qkv.chunk(3, dim=1)
        q = q[:, :, 1 : 1 + H, 1 : 1 + W].reshape(B, self.heads, d, H, W)
        k = k.view(B, self.heads, d, H + 2, W + 2)
        v = v.view(B, self.heads, d, H + 2, W + 2)
        scores, values = [], []
        for n, (dy, dx) in enumerate(OFFSETS):
            kn = k[:, :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            vn = v[:, :, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            s = (q * kn).sum(2) / d**0.5  # (B, K, H, W)
            if self.bias is not None:
                s = s + self.bias[None, :, n, None, None]
            scores.append(s)
            values.append(vn)
        att = torch.softmax(torch.stack(scores, 2), dim=2)  # (B, K, 9, H, W)
        z = sum(att[:, :, n, None] * values[n] for n in range(9))
        return z.reshape(B, self.channels, H, W)


class MLPUpdate(nn.Module):
    def __init__(self, in_dim: int, channels: int, hidden: int, zero_init: bool):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(z)))


class SwiGLUUpdate(nn.Module):
    def __init__(self, in_dim: int, channels: int, hidden: int, zero_init: bool):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, 2 * hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        g, u = self.fc1(z).chunk(2, dim=1)
        return self.fc2(F.silu(g) * u)


class NCA(nn.Module):
    """The cell update rule, shared by every cell (see module docstring)."""

    def __init__(self, cfg: NCAConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        self.register_buffer(
            "embed", make_embeddings(cfg.vocab_size, cfg.c_in, cfg.embedding, c_out=cfg.c_out)
        )
        # embeddings of the predictable tokens, restricted to the output slice
        if cfg.head in ("l2", "cosine"):
            lo = cfg.out_offset - cfg.in_offset
            self.register_buffer(
                "out_embed", self.embed[list(cfg.output_tokens)][:, lo : lo + cfg.c_out].clone()
            )
        # border value for "wall" padding: the wall token embedding in the input slice, 0 elsewhere
        pad_vec = torch.zeros(C)
        if cfg.padding == "wall":
            assert cfg.wall_token is not None, "wall padding needs wall_token"
            pad_vec[cfg.in_slice] = self.embed[cfg.wall_token]
        self.register_buffer("pad_vec", pad_vec)

        if cfg.perception == "kernels":
            self.perceive = KernelPerception(C, cfg.heads)
        elif cfg.perception == "conv":
            self.perceive = ConvPerception(C, cfg.heads)
        elif cfg.perception == "sobel":
            self.perceive = SobelPerception(C)
        elif cfg.perception == "fixed_attn":
            self.perceive = FixedAttention(C, cfg.heads, cfg.grid_size)
        elif cfg.perception == "self_attn":
            self.perceive = SelfAttention(C, cfg.heads, cfg.attn_bias)
        else:
            raise ValueError(f"unknown perception {cfg.perception!r}")

        Update = {"mlp": MLPUpdate, "swiglu": SwiGLUUpdate}[cfg.update]
        hidden = cfg.expansion * (self.perceive.out_dim if cfg.hidden_from_perception else C)
        self.update = Update(self.perceive.out_dim, C, hidden, cfg.zero_init_update)

        if cfg.head == "mlp":
            self.head = nn.Conv2d(cfg.c_out, len(cfg.output_tokens), 1)
        if cfg.n_tasks:
            self.task_embed = nn.Embedding(cfg.n_tasks, cfg.c_task)
            nn.init.normal_(self.task_embed.weight, std=cfg.sigma_init)
        if cfg.time_steps:
            n_hid = cfg.hid_slice.stop - cfg.hid_slice.start
            self.time_embed = nn.Embedding(cfg.time_steps, n_hid)
            nn.init.normal_(self.time_embed.weight, std=0.02)
        if cfg.init_style == "public":
            self._public_init()

    @torch.no_grad()
    def _public_init(self) -> None:
        """The initialisation of the authors' public code (`reset_params`): both MLP layers
        N(0, 1e-3^2) with zero biases (a near-do-nothing rule that is not exactly zero), 3x3
        kernels and fixed-attention weights N(0, 1/9) centred on zero, value projection
        N(0, 1/C)."""
        for fc in (self.update.fc1, self.update.fc2):
            nn.init.normal_(fc.weight, std=1e-3)
            nn.init.zeros_(fc.bias)
        if isinstance(self.perceive, KernelPerception):
            nn.init.normal_(self.perceive.weight, std=1 / 3)
        elif isinstance(self.perceive, FixedAttention):
            nn.init.normal_(self.perceive.mix, std=1 / 3)
            nn.init.normal_(self.perceive.value.weight, std=self.cfg.channels**-0.5)
        else:
            raise ValueError(f"init_style public is not defined for {self.cfg.perception!r}")

    # ------------------------------------------------------------------ state construction
    @property
    def device(self) -> torch.device:
        return self.embed.device

    def clamp_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Token embeddings of the input grid: `(B, C_in, H, W)` from `(B, H, W)` tokens."""
        if self.cfg.head == "pixels":
            return inputs[:, None].float()
        return self.embed[inputs].permute(0, 3, 1, 2).contiguous()

    def init_state(
        self,
        inputs: torch.Tensor,
        given: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build `(x, given4, clamp)` for a batch of boards (paper B.1, "State Initialization").

        inputs: (B, H, W) long tokens, given: (B, H, W) bool. Returns x (B, C, H, W),
        given4 (B, 1, H, W) and clamp (B, C_in, H, W).
        """
        cfg = self.cfg
        B, H, W = inputs.shape
        shape = (B, cfg.channels, H, W)
        if cfg.init_uniform:
            x = (
                torch.rand(shape, generator=generator, device=self.device) * 2 - 1
            ) * cfg.sigma_init
        else:
            x = torch.randn(shape, generator=generator, device=self.device) * cfg.sigma_init
        given4 = given[:, None].to(self.device)
        clamp = self.clamp_values(inputs.to(self.device))
        if cfg.n_tasks:
            x = self.write_task_embedding(x, task_ids)
        x = self.apply_clamp(x, given4, clamp)
        if cfg.norm_init:
            x = self.apply_clamp(self.normalize(x), given4, clamp)
        return x, given4, clamp

    def write_task_embedding(
        self, x: torch.Tensor, task_ids: torch.Tensor | None, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Overwrite the task slice of `x` (all boards, or the boolean `rows` (B,)) with the
        *current* learned task embedding of `task_ids`, differentiably: the result carries the
        autograd history of `task_embed`, `x` itself may be a detached state (e.g. from the
        replay pool)."""
        cfg = self.cfg
        assert task_ids is not None, "this model needs task ids"
        B, C, H, W = x.shape
        te = self.task_embed(task_ids.to(self.device))[:, :, None, None].expand(B, cfg.c_task, H, W)
        sl = cfg.task_slice
        if rows is not None:
            te = torch.where(rows.to(x.device).view(B, 1, 1, 1), te, x[:, sl])
        return torch.cat([x[:, : sl.start], te.to(x.dtype)], dim=1)

    def apply_clamp(
        self, x: torch.Tensor, given4: torch.Tensor, clamp: torch.Tensor
    ) -> torch.Tensor:
        """Reset the input channels of given cells to their token embedding."""
        sl = self.cfg.in_slice
        x_in = torch.where(given4, clamp, x[:, sl])
        if sl.start == 0 and sl.stop == self.cfg.channels:
            return x_in
        return torch.cat([x[:, : sl.start], x_in, x[:, sl.stop :]], dim=1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell group normalisation: U groups of C/U channels, each to unit L2 norm."""
        U = self.cfg.norm_groups
        if not U:
            return x
        B, C, H, W = x.shape
        xg = x.view(B, U, C // U, H, W)
        # sqrt(sum + eps) rather than .norm(): the gradient of the norm at a zero vector is NaN,
        # and damage perturbations do create all-zero cells. eps = 1e-4 also bounds the Jacobian
        # of the normalisation at a (nearly) zero cell to 100 per step, which keeps BPTT finite
        # while such a cell waits for its first firing (see docs/DEVIATIONS.md).
        if self.cfg.norm_style == "norm_eps":  # the public code's formula
            norm = xg.norm(dim=2, keepdim=True) + 1e-6
        else:
            norm = (xg.pow(2).sum(dim=2, keepdim=True) + 1e-4).sqrt()
        return (xg / norm).view(B, C, H, W)

    def project_tangent(self, dx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """`dx - <dx, x> x` in every channel group (the public code's `proj_emb`): the part of
        the update along the state is removed, so the residual step mostly rotates each unit
        vector. The formula is kept as is (no division by |x|^2), as in the public code."""
        U = self.cfg.norm_groups
        B, C, H, W = x.shape
        xg = x.view(B, U, C // U, H, W)
        dg = dx.view(B, U, C // U, H, W)
        return (dg - (dg * xg).sum(dim=2, keepdim=True) * xg).view(B, C, H, W)

    # ------------------------------------------------------------------ one step
    def step_embedding(self, x: torch.Tensor, clock: torch.Tensor | None) -> torch.Tensor | None:
        """The learned step embedding e_step(t) placed on the hidden slice (zero on the other
        channels), broadcastable to `x`: (1, C, 1, 1) for a global clock (training), (B, C, H, W)
        for per-cell local clocks `(B, 1, H, W)` (inference). None when the model has no time
        embedding."""
        cfg = self.cfg
        if not cfg.time_steps:
            return None
        assert clock is not None, "this model needs a step clock"
        idx = clock.clamp(max=cfg.time_steps - 1)
        if idx.dim() == 0:  # global clock (training)
            # an embedding lookup rather than `weight[idx]`: indexing with a 0-dim tensor calls
            # .item() (one host sync per step, a graph break under torch.compile); same row,
            # same gradient
            te = self.time_embed(idx.reshape(1))[:, :, None, None]
        else:  # per-cell local clocks (B, 1, H, W)
            te = self.time_embed(idx[:, 0]).permute(0, 3, 1, 2)
        sl = cfg.hid_slice
        # zeros before and after the hidden slice (output / input channels, task slice)
        return F.pad(te, (0, 0, 0, 0, sl.start, cfg.channels - sl.stop))

    def sense_update(self, x: torch.Tensor) -> torch.Tensor:
        """Perception + update rule on `x` as is: the proposed update dx of every cell."""
        cfg = self.cfg
        B, C, H, W = x.shape
        xp = _pad(x, cfg.padding, self.pad_vec)
        z = self.perceive(xp, H, W)
        return self.update(z)

    def delta(self, x: torch.Tensor, clock: torch.Tensor | None = None) -> torch.Tensor:
        """Proposed update dx for every cell (before the firing mask), sensed from `x` with the
        step embedding added to the hidden slice (`step_embedding`)."""
        te = self.step_embedding(x, clock)
        return self.sense_update(x if te is None else x + te)

    def step(
        self,
        x: torch.Tensor,
        given4: torch.Tensor,
        clamp: torch.Tensor,
        fire_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        clock: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One asynchronous NCA step. `fire_mask` is (B, 1, H, W) in {0, 1}; `noise` (optional) is
        added to the state after the update (training/test-time perturbation).

        With a time embedding (ARC) the step follows paper eq. B.3-(1): the hidden slice of every
        cell receives e_step(t) *before* sensing, `x_hid <- x_hid + e_step(t)`, and the residual
        update applies to that shifted state, so a firing cell keeps the embedding in its state.
        A cell that does not fire stays frozen (its local clock does not advance either), which
        is why the shift is gated by the firing mask like the update."""
        te = self.step_embedding(x, clock)
        if te is None:
            dx = self.sense_update(x)
        else:
            dx = self.sense_update(x + te) + te
        return self.residual_update(x, dx, fire_mask, noise, given4, clamp)

    def residual_update(
        self,
        x: torch.Tensor,
        dx: torch.Tensor,
        fire_mask: torch.Tensor,
        noise: torch.Tensor | None,
        given4: torch.Tensor,
        clamp: torch.Tensor,
        noise_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Steps 4-6 of the module docstring: masked residual update, noise, normalisation,
        clamp. `noise_mask` (B, 1, H, W), when known, saves recomputing it from `noise`."""
        cfg = self.cfg
        if cfg.project_update and cfg.norm_groups:
            x = x + self.project_tangent(fire_mask * dx, x)
        else:
            x = x + fire_mask * dx
        early_noise = noise is not None and not cfg.noise_after_norm
        if early_noise:
            x = x + noise
        if cfg.norm_groups and cfg.norm_fired_only:
            # a cell that neither fired nor received noise keeps its state, unnormalised
            moved = fire_mask.to(x.dtype)
            if early_noise:
                if noise_mask is None:
                    noise_mask = (noise != 0).any(dim=1, keepdim=True)
                moved = torch.maximum(moved, noise_mask.to(x.dtype))
            x = torch.where(moved > 0, self.normalize(x), x)
        else:
            x = self.normalize(x)
        if noise is not None and cfg.noise_after_norm:
            x = x + noise
        return self.apply_clamp(x, given4, clamp)

    @property
    def public_update(self) -> bool:
        """True when the update uses a public-code switch the fused engine kernels lack."""
        cfg = self.cfg
        return cfg.project_update or cfg.noise_after_norm or cfg.norm_style != "sqrt_eps"

    def sample_fire_mask(
        self,
        x: torch.Tensor,
        p_fire: float | None = None,
        p_low: float | None = None,
        tau: float | None = None,
        generator: torch.Generator | None = None,
        given: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bernoulli firing mask (B, 1, H, W). With `p_low` and `tau`, cells whose confidence
        exceeds `tau` fire with probability `p_low` instead (adaptive firing, paper 4.4).

        `given` (B, 1, H, W bool) marks the clamped cells: when passed, they count as confident
        and fire at `p_low` too. Their output slice is pinned to the embedding of their token,
        so their L2 score is a constant 0.5 and they would otherwise never cross `tau` (see
        docs/DEVIATIONS.md, "Confidence of given cells").
        """
        p = self.cfg.fire_rate if p_fire is None else p_fire
        B, _, H, W = x.shape
        u = torch.rand((B, 1, H, W), generator=generator, device=x.device)
        if p_low is None or tau is None:
            return (u < p).to(x.dtype)
        confident = self.confidence(x)[:, None] > tau
        if given is not None:
            confident = confident | given
        prob = torch.where(confident, torch.full_like(u, p_low), torch.full_like(u, p))
        return (u < prob).to(x.dtype)

    # ------------------------------------------------------------------ readout
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell scores over the predictable tokens: (B, n_out, H, W)."""
        cfg = self.cfg
        out = x[:, cfg.out_slice]
        if cfg.head == "l2":
            d = out[:, None] - self.out_embed[None, :, :, None, None]  # (B, n_out, C_out, H, W)
            return 1.0 / (1.0 + (d.pow(2).sum(dim=2) + 1e-8).sqrt())
        if cfg.head == "cosine":
            o = out / (out.pow(2).sum(dim=1, keepdim=True) + 1e-8).sqrt()
            e = self.out_embed / self.out_embed.norm(dim=1, keepdim=True)
            return torch.einsum("bchw,vc->bvhw", o, e)
        if cfg.head == "mlp":
            return self.head(out)
        raise ValueError(cfg.head)

    def logits_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Scores suitable for a cross-entropy loss (mlp head)."""
        return self.scores(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted token of every cell: (B, H, W) long (ignores the given cells; combine with
        the inputs via `torch.where(given, inputs, decoded)`)."""
        if self.cfg.head == "pixels":
            return x[:, self.cfg.out_offset]
        idx = self.scores(x).argmax(dim=1)
        tokens = torch.tensor(self.cfg.output_tokens, device=x.device)
        return tokens[idx]

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell confidence in [0, 1]: (B, H, W)."""
        s = self.scores(x)
        if self.cfg.head == "mlp":
            s = torch.softmax(s, dim=1)
        return s.max(dim=1).values.clamp(0.0, 1.0)

    def board_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Mean cell confidence per board: (B,)."""
        return self.confidence(x).mean(dim=(1, 2))

    def target_embedding(self, targets: torch.Tensor) -> torch.Tensor:
        """Target vectors in the output slice for an MSE loss: (B, C_out, H, W)."""
        cfg = self.cfg
        if cfg.head == "pixels":
            return targets[:, None].float()
        lo = cfg.out_offset - cfg.in_offset
        e = self.embed[targets][..., lo : lo + cfg.c_out]  # (B, H, W, C_out)
        return e.permute(0, 3, 1, 2)

    def output_index(self, targets: torch.Tensor) -> torch.Tensor:
        """Map target tokens to indices in `output_tokens` (for cross-entropy). Unmapped tokens
        (input-only tokens on given cells) get index 0; mask them out of the loss."""
        lut = torch.zeros(self.cfg.vocab_size, dtype=torch.long, device=targets.device)
        for i, t in enumerate(self.cfg.output_tokens):
            lut[t] = i
        return lut[targets]

    def extra_repr(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.cfg.to_dict().items())


# ---------------------------------------------------------------------- presets (paper tables)
PRESETS: dict[str, dict] = {
    # Table 3
    "maze_ood": dict(
        vocab_size=4,
        channels=16,
        c_in=16,
        c_out=2,
        output_tokens=(0, 1),
        perception="kernels",
        hidden_from_perception=True,
        heads=4,
        padding="wall",
        wall_token=2,
        update="mlp",
        expansion=2,
        norm_groups=0,
        sigma_init=0.15,
        fire_rate=0.8,
        head="l2",
        embedding="onehot",
    ),
    "maze_hard": dict(
        vocab_size=5,
        channels=32,
        c_in=32,
        c_out=3,
        output_tokens=(0, 1),
        perception="kernels",
        hidden_from_perception=True,
        heads=4,
        padding="wall",
        wall_token=2,
        update="mlp",
        expansion=2,
        norm_groups=0,
        sigma_init=0.25,
        fire_rate=0.8,
        head="l2",
        embedding="onehot",
    ),
    "sudoku_ood": dict(
        vocab_size=10,
        channels=128,
        c_in=32,
        c_out=32,
        output_tokens=tuple(range(1, 10)),
        perception="fixed_attn",
        heads=16,
        grid_size=(9, 9),
        padding="wrap",
        update="mlp",
        expansion=4,
        norm_groups=4,
        # historical behaviour: the published Sudoku-OOD runs were trained with it
        norm_fired_only=False,
        sigma_init=0.15,
        fire_rate=0.8,
        head="cosine",
        embedding="orthonormal",
        zero_init_update=False,
    ),
    "sudoku_extreme": dict(
        vocab_size=10,
        channels=256,
        c_in=32,
        c_out=32,
        output_tokens=tuple(range(1, 10)),
        perception="fixed_attn",
        heads=16,
        grid_size=(9, 9),
        padding="wrap",
        update="mlp",
        expansion=4,
        norm_groups=8,
        norm_fired_only=True,
        sigma_init=0.15,
        fire_rate=0.8,
        head="cosine",
        embedding="orthonormal",
        zero_init_update=False,
    ),
    # Table 4
    "arc": dict(
        vocab_size=12,
        channels=512,
        c_in=32,
        c_out=32,
        in_offset=0,
        out_offset=32,
        c_task=128,
        output_tokens=tuple(range(12)),
        perception="fixed_attn",
        heads=32,
        grid_size=(32, 32),
        padding="zero",
        update="swiglu",
        expansion=4,
        norm_groups=32,
        norm_fired_only=True,
        sigma_init=0.10,
        fire_rate=0.9,
        head="mlp",
        embedding="orthonormal",
        time_steps=64,
        zero_init_update=False,
    ),
}


def build_model(preset: str | None = None, **overrides) -> NCA:
    """`build_model("sudoku_ood", channels=64)` builds a preset with modifications."""
    d = dict(PRESETS[preset]) if preset else {}
    d.update(overrides)
    d["output_tokens"] = tuple(d["output_tokens"])
    return NCA(NCAConfig(**d))


@dataclass
class ModelSpec:
    """What a config file says about the model: a preset name plus overrides."""

    preset: str | None = None
    overrides: dict = field(default_factory=dict)

    def build(self) -> NCA:
        return build_model(self.preset, **self.overrides)
