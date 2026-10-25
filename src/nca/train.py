"""Training with sample replay and truncated BPTT (paper Section 3.2, Appendix B.2, Table 3).

One iteration:

1. draw a batch from the replay pool, replacing a fraction `seed_ratio` by fresh boards;
2. apply the pre-rollout perturbations: target swap (`swap.p`) and damage (`damage.p`);
3. unroll the NCA for `rollout` steps with state noise (`noise`); a random window of `chunk`
   consecutive steps is run with gradients (truncated BPTT), the rest without;
4. average the loss over the supervised steps of the chunk (`loss_window`: "all", "last" or an
   integer `w` = the last w steps), clip gradients, AdamW step, EMA update;
5. write the final states back into the pool.

Opt-in switches reproduce the recipe of the authors' public JAX code for A/B tests (all default
to the behaviour above; see the config-key table of docs/PAPER_MAP.md): `betas`,
`init_lr_ratio`, `loss_reduce`, `ema_warmup`, `damage.fill` / `damage.geometry`,
`batch_noise`, `swap_source`, and, in this trainer only, `chunk_start` / `writeback`.

The EMA weights are used for evaluation and are what `final.pt` stores (`final_raw.pt`: the
trained weights; `ema_extra`: more EMAs of the same run, in `final_<name>.pt`). `checkpoint.pt` holds
everything needed to resume an interrupted run (model, EMA, optimizer, pool, RNG, step).
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .model import NCA, build_model
from .perturb import apply_batch_noise, apply_damage, sample_noise, swap_indices
from .pool import ReplayPool
from .rollout import RolloutConfig, decode_boards, rollout, score_candidates
from .tasks import get_task
from .tasks.base import Boards, overlap_counts
from .utils import count_parameters, get_device, human_time, save_json, seed_everything

DEFAULTS = dict(
    steps=5000,
    batch_size=64,
    pool_mult=4,
    seed_ratio=0.25,
    rollout=100,
    chunk=100,
    loss_window="all",
    loss="mse",
    loss_reduce="sum",  # MSE over the output channels: "sum" (ours) or "mean" (public code)
    lr=4e-4,
    schedule="constant",
    warmup_ratio=0.0,
    init_lr_ratio=None,  # None: warmup from 0; r: from r x lr to lr (optax warmup_cosine_decay)
    weight_decay=0.0,
    betas=(0.9, 0.999),  # AdamW betas (PyTorch's default; the public code uses (0.9, 0.95))
    grad_clip=1.0,
    ema=0.999,
    ema_warmup=True,  # decay_t = min(ema, (1 + t) / (10 + t)); false: constant decay from step 0
    # extra EMAs of the same trajectory, {name: {ema_warmup: bool}}: they never feed back into the
    # training, so comparing them to the main EMA isolates the averaging; saved as final_<name>.pt
    ema_extra={},
    noise=dict(p_t=0.1, p_s=0.2, sigma=0.15),
    # `fill`: value written inside the discs (public code: 1.0 for normalised models); `geometry`:
    # "ours" (radius x min(H, W), centres anywhere) or "official" (the public make_circle_masks:
    # coordinates in [-1, 1], centres in [-0.5, 0.5], radius in those units)
    damage=dict(p=0.1, n=1, r_min=0.1, r_max=0.4, fill=0.0, geometry="ours"),
    swap=dict(p=0.1),
    swap_source="fresh",  # swapped boards come from the dataset ("fresh") or the pool ("pool")
    batch_noise=None,  # public code: {p, ps, sigma} noise on the prepared batch (BATCH_NOISE)
    chunk_start="shared",  # one BPTT chunk start per batch, or "per_sample" (reference trainer)
    writeback="rollout_end",  # pool gets the state at the rollout end, or at "chunk_end"
    augment=True,
    asynchronous=True,
    replay=True,
    perturb=True,
    log_every=50,
    eval_every=500,
    checkpoint_every=500,
    stop_at=None,  # probe runs: stop after this step, without final.pt; resuming continues the run
    amp=False,
    grad_checkpoint="auto",
    eval=dict(n=256, depth=None, k=1),
)


# `grad_checkpoint: auto` turns checkpointing on above this many state elements kept alive by
# the BPTT chunk (chunk x batch x channels x H x W). One step saves several state-sized fp32
# tensors (about 25 x for the ARC fixed-attention / SwiGLU cell, 5-10 x for the maze and Sudoku
# cells), so 2**27 elements (512 MiB per state-sized tensor) is where a chunk stops fitting an
# 8 GB GPU. Maze-OOD (2**23) and Maze-Hard (2**27 - 12%) stay below, Sudoku-Extreme (2**28)
# and arc_small (2**28) above; Sudoku-OOD (2**27 - 5%) sets the flag explicitly in its config.
GRAD_CHECKPOINT_CELLS = 2**27

# `train.batch_noise` (public code): after the swap and the damage, each board with probability
# `p` gets N(0, sigma^2) noise on a random `ps` fraction of its cells (every channel but the
# input channels of the given cells), then the state is renormalised and clamped. A dict in the
# config is merged into these defaults; absent / null = off.
BATCH_NOISE = dict(p=0.1, ps=0.2, sigma=0.15)

_CHOICES = dict(
    loss_reduce=("sum", "mean"),
    swap_source=("fresh", "pool"),
    chunk_start=("shared", "per_sample"),
    writeback=("rollout_end", "chunk_end"),
)


def resolve_train_cfg(cfg: dict) -> dict:
    """The `train` section of `cfg` merged into `DEFAULTS` (the sub-dicts key by key), with the
    choices of the public-code switches checked. `loss_reduce: mean` is folded into the loss
    kind (`loss: mse_mean`) because that string is what reaches `step_loss` in both trainers
    (the fast engine passes `train.loss` to it unchanged)."""
    given = cfg.get("train") or {}
    tr = {**DEFAULTS, **given}
    for k in ("noise", "damage", "swap", "eval"):
        tr[k] = {**DEFAULTS[k], **(given.get(k) or {})}
    if tr["batch_noise"] is not None:
        tr["batch_noise"] = {**BATCH_NOISE, **tr["batch_noise"]}
    for k, allowed in _CHOICES.items():
        if tr[k] not in allowed:
            raise ValueError(f"train.{k} must be one of {allowed}, got {tr[k]!r}")
    if tr["damage"]["geometry"] not in ("ours", "official"):
        raise ValueError(f"train.damage.geometry must be ours or official: {tr['damage']}")
    if tr["loss_reduce"] == "mean":
        if tr["loss"] not in ("mse", "mse_mean"):
            raise ValueError("train.loss_reduce: mean applies to the MSE loss only")
        tr["loss"] = "mse_mean"
    return tr


def auto_grad_checkpoint(
    chunk: int, batch_size: int, channels: int, shape: tuple[int, int]
) -> bool:
    """The `grad_checkpoint: auto` rule: True when the chunk's activations are too big."""
    H, W = shape
    return chunk * batch_size * channels * H * W > GRAD_CHECKPOINT_CELLS


def graph_span(tr: dict) -> int:
    """Steps whose graph is alive at the backward: the chunk, or with `chunk_start: per_sample`
    up to the whole rollout (from the earliest chunk start of the batch to the latest chunk end).
    Maze-Hard with per-sample chunks spans ~2 x its chunk, past the `auto` threshold: counting the
    chunk alone left checkpointing off and spilled the 8 GB GPU into shared memory (22 s/step)."""
    return tr["rollout"] if tr["chunk_start"] == "per_sample" else tr["chunk"]


def lr_at(step: int, total: int, cfg: dict) -> float:
    """Learning-rate multiplier: warmup then constant or cosine decay (Table 3).

    The warmup goes linearly from 1 / warm to 1 or, with `init_lr_ratio: r`, from r to 1 as
    `r + (1 - r) * step / warm` (the public code's optax `warmup_cosine_decay_schedule` with
    `init_value = r * lr`); its constant end value is `schedule: constant` or `cosine` with
    `cosine_floor: 1.0`."""
    warm = int(cfg["warmup_ratio"] * total)
    if step < warm:
        r = cfg.get("init_lr_ratio")
        if r is None:
            return (step + 1) / warm
        return r + (1 - r) * step / warm
    if cfg["schedule"] == "constant":
        return 1.0
    if cfg["schedule"] == "cosine":
        p = (step - warm) / max(1, total - warm)
        alpha = cfg.get("cosine_floor", 0.0)
        return alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * p))
    raise ValueError(cfg["schedule"])


def build_task_and_data(tcfg: dict, root: str):
    """Instantiate the task and load the train and validation boards from a config section."""
    task = get_task(tcfg["name"], **tcfg.get("args", {}))
    train = task.load("train", root, **tcfg.get("train", {}))
    val_cfg = dict(tcfg.get("val", {}))
    split = val_cfg.pop("split", "test")
    val = task.load(split, root, **val_cfg) if split else None
    return task, train, val


def cell_loss(model: NCA, x: torch.Tensor, targets: torch.Tensor, loss_kind: str) -> torch.Tensor:
    """(B, H, W) loss of every cell: MSE to the target embedding summed over the output channels
    (`mse`) or averaged over them (`mse_mean`, the public code; `train.loss_reduce: mean`), or
    cross-entropy (`ce`)."""
    if loss_kind in ("mse", "mse_mean"):
        d = (x[:, model.cfg.out_slice] - model.target_embedding(targets)) ** 2
        return d.sum(dim=1) if loss_kind == "mse" else d.mean(dim=1)
    if loss_kind == "ce":
        logits = model.logits_for_loss(x)
        return F.cross_entropy(logits, model.output_index(targets), reduction="none")
    raise ValueError(loss_kind)


def step_loss(
    model: NCA,
    x: torch.Tensor,
    targets: torch.Tensor,
    loss_kind: str,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Loss of one step: `cell_loss` averaged over the batch's (masked) cells."""
    per_cell = cell_loss(model, x, targets, loss_kind)
    if loss_mask is not None:
        return (per_cell * loss_mask).sum() / loss_mask.sum().clamp_min(1)
    return per_cell.mean()


def ema_decay(tr: dict, step: int, warmup: bool | None = None) -> float:
    """EMA decay of iteration `step`: with `ema_warmup` (default) decay_t = min(decay,
    (1 + t) / (10 + t)) so that short runs are not dominated by the random initial weights;
    without, the constant decay of the public code. `warmup` overrides `ema_warmup` (extra EMAs)."""
    if tr["ema_warmup"] if warmup is None else warmup:
        return min(tr["ema"], (1 + step) / (10 + step))
    return tr["ema"]


def perturb_batch(
    model: NCA,
    pool: ReplayPool,
    batch: dict,
    tr: dict,
    rng: np.random.Generator,
    gen_dev: torch.Generator,
) -> None:
    """The pre-rollout perturbations of a training batch, in place (B.2; shared by both
    trainers): target swap (a fresh board, or with `swap_source: pool` another pool entry's),
    damage, clamp, and the public code's optional batch noise (then renormalised and clamped)."""
    which = swap_indices(tr["batch_size"], tr["swap"]["p"], rng)
    if tr["swap_source"] == "pool":
        pool.swap_from_pool(batch, which)
    else:
        pool.replace_boards(batch, which)
    d = tr["damage"]
    batch["x"], _ = apply_damage(
        batch["x"],
        d["p"],
        d["n"],
        d["r_min"],
        d["r_max"],
        rng,
        fill=d["fill"],
        geometry=d["geometry"],
    )
    batch["x"] = model.apply_clamp(batch["x"], batch["given4"], batch["clamp"])
    bn = tr["batch_noise"]
    if bn is not None:
        x = apply_batch_noise(
            batch["x"], batch["given4"], model.cfg.in_slice, bn["p"], bn["ps"], bn["sigma"], gen_dev
        )
        batch["x"] = model.apply_clamp(model.normalize(x), batch["given4"], batch["clamp"])


def build_run_model(cfg: dict) -> NCA:
    """The model a training run of `cfg` uses, with the training options that change the
    model's own config applied (`train.asynchronous: false` sets every cell to fire). Used by
    `train` and by `scripts/archive_stale_runs.py`, so both see the same model config."""
    mcfg = cfg.get("model", {})
    model = build_model(mcfg.get("preset"), **mcfg.get("overrides", {}))
    if not {**DEFAULTS, **cfg.get("train", {})}["asynchronous"]:
        model.cfg.fire_rate = 1.0
    return model


def restore_model_config(model: NCA, ema: NCA, run_json: Path) -> None:
    """Before resuming a run, make the model behave as it did when the run started.

    Options added to `NCAConfig` after the run began (absent from its `run.json`) are set back
    to their dataclass default, which is the historical behaviour (e.g. `norm_fired_only`), so
    a resumed run never changes dynamics silently. Any other difference between the stored and
    the current model config means the preset changed under the run: that is an error (archive
    the run with scripts/archive_stale_runs.py or restart it with --no-resume)."""
    if not run_json.exists():
        return
    stored = json.loads(run_json.read_text()).get("model_config")
    if not stored:
        return
    from dataclasses import MISSING, fields

    from .model import NCAConfig

    current = json.loads(json.dumps(model.cfg.to_dict()))
    for f in fields(NCAConfig):
        if f.default is MISSING:
            continue
        if f.name not in stored and current.get(f.name) != f.default:
            for m in (model, ema):
                setattr(m.cfg, f.name, f.default)
    current = json.loads(json.dumps(model.cfg.to_dict()))
    diff = sorted(k for k in stored if k in current and stored[k] != current[k])
    if diff:
        raise RuntimeError(
            f"{run_json.parent}: the model config differs from the one this run started with "
            f"({', '.join(diff)}); archive the run (scripts/archive_stale_runs.py) or use --no-resume"
        )


def _step_with_grad(model, x, given4, clamp, fm, noise, clock, amp: bool, use_ckpt: bool):
    """One differentiable step (bf16 autocast with `amp`, checkpointed with `use_ckpt`)."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
        if use_ckpt:
            x = grad_checkpoint(model.step, x, given4, clamp, fm, noise, clock, use_reentrant=False)
        else:
            x = model.step(x, given4, clamp, fm, noise, clock)
    return x.float()


def _rollout_shared(
    model: NCA,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    given4: torch.Tensor,
    clamp: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    c0: int,
    tr: dict,
    gen_dev: torch.Generator,
    use_ckpt: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rollout of one iteration with one BPTT chunk `[c0, c0 + chunk)` for the whole batch
    (`chunk_start: shared`, `writeback: rollout_end`). Leaves the clipped gradients in `.grad`
    and returns the final state, the loss and the gradient norm."""
    N, Nc, window = tr["rollout"], tr["chunk"], tr["loss_window"]
    dev = x.device
    if window == "all":
        sup = set(range(c0, c0 + Nc))
    elif window == "last":
        sup = {c0 + Nc - 1}
    else:
        sup = set(range(c0 + Nc - int(window), c0 + Nc))
    loss_sum, n_sup = 0.0, 0
    for t in range(N):
        in_chunk = c0 <= t < c0 + Nc
        fm = model.sample_fire_mask(x, generator=gen_dev)
        nz = tr["noise"]
        noise = (
            sample_noise(x, nz["p_t"], nz["p_s"], nz["sigma"], gen_dev) if tr["perturb"] else None
        )
        clock = torch.tensor(t, device=dev)
        if in_chunk:
            x = _step_with_grad(model, x, given4, clamp, fm, noise, clock, tr["amp"], use_ckpt)
            if t in sup:
                loss_sum = loss_sum + step_loss(model, x, targets, tr["loss"], loss_mask)
                n_sup += 1
            if t == c0 + Nc - 1:
                # the backward runs as soon as the chunk ends, to free its activations; the
                # gradients wait in .grad until the rollout is over
                loss = loss_sum / n_sup
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                x = x.detach()
        else:
            with torch.no_grad():
                x = model.step(x, given4, clamp, fm, noise, clock)
    return x, loss, gn


def _rollout_per_sample(
    model: NCA,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    given4: torch.Tensor,
    clamp: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    c0s: np.ndarray,
    tr: dict,
    gen_dev: torch.Generator,
    use_ckpt: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rollout with the public code's TBPTT options (reference trainer only):

    * `chunk_start: per_sample`: board b has its own chunk `[c0s[b], c0s[b] + chunk)`. The steps
      between the first chunk start and the last chunk end build the graph for the whole batch
      (under `amp` autocast if set), and each board's history is cut at its own start
      (`x.detach()` on its rows), so its gradient covers its own chunk only; its loss is taken
      over the `loss_window` steps of its own chunk;
    * `writeback: chunk_end`: the state returned (and written back to the pool) is each board's
      state at the last step of its own chunk, and the rollout stops at the last chunk end.

    The loss is the mean over the supervised steps and the (masked) cells of the batch, the
    normalisation of `_rollout_shared` (equal to it when every c0s[b] is the same)."""
    N, Nc, window = tr["rollout"], tr["chunk"], tr["loss_window"]
    dev = x.device
    B = x.shape[0]
    c0t = torch.as_tensor(c0s, device=dev)
    lo, hi = int(c0s.min()), int(c0s.max()) + Nc
    end = hi if tr["writeback"] == "chunk_end" else N
    n_sup = Nc if window == "all" else 1 if window == "last" else min(int(window), Nc)
    sup_lo, sup_hi = c0t + Nc - n_sup, c0t + Nc
    if loss_mask is not None:
        denom = n_sup * loss_mask.sum().clamp_min(1)
    else:
        denom = n_sup * B * x.shape[-2] * x.shape[-1]
    loss_sum = 0.0
    out = x.detach()
    for t in range(end):
        fm = model.sample_fire_mask(x, generator=gen_dev)
        nz = tr["noise"]
        noise = (
            sample_noise(x, nz["p_t"], nz["p_s"], nz["sigma"], gen_dev) if tr["perturb"] else None
        )
        clock = torch.tensor(t, device=dev)
        if lo <= t < hi:
            if t > 0:  # at t = 0 the batch keeps its graph (the ARC task embedding), as shared
                x = torch.where((c0t == t).view(B, 1, 1, 1), x.detach(), x)
            x = _step_with_grad(model, x, given4, clamp, fm, noise, clock, tr["amp"], use_ckpt)
            rows = (sup_lo <= t) & (t < sup_hi)
            if bool(rows.any()):
                w = rows.view(B, 1, 1).to(x.dtype)
                if loss_mask is not None:
                    w = w * loss_mask
                loss_sum = loss_sum + (cell_loss(model, x, targets, tr["loss"]) * w).sum()
            if t == hi - 1:
                loss = loss_sum / denom
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                x = x.detach()
        else:
            with torch.no_grad():
                x = model.step(x, given4, clamp, fm, noise, clock)
        if tr["writeback"] == "chunk_end":
            out = torch.where((c0t + Nc - 1 == t).view(B, 1, 1, 1), x.detach(), out)
    if tr["writeback"] == "rollout_end":
        out = x.detach()
    return out, loss, gn


def train(
    cfg: dict,
    out_dir: str | Path,
    seed: int = 0,
    device: str | None = None,
    resume: bool = True,
    quiet: bool = False,
) -> dict:
    """Train one model. Returns the final evaluation metrics. See `scripts/train.py`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["task"]
    mcfg = cfg.get("model", {})
    tr = resolve_train_cfg(cfg)
    dev = get_device(device)
    seed_everything(seed)
    gen_dev = torch.Generator(device=dev)
    gen_dev.manual_seed(seed)
    rng = np.random.default_rng(seed)

    task, train_boards, val_boards = build_task_and_data(tcfg, cfg.get("data_root", "data"))
    model = build_run_model(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_extra = {name: copy.deepcopy(ema) for name in tr["ema_extra"]}
    opt = torch.optim.AdamW(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"], betas=tuple(tr["betas"])
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, tr["steps"], tr))

    pool_mult = tr["pool_mult"] if tr["replay"] else 1
    seed_ratio = tr["seed_ratio"] if tr["replay"] else 1.0
    task_ids = train_boards.meta.get("task_id")
    pool = ReplayPool(
        model,
        task,
        train_boards,
        pool_mult * tr["batch_size"],
        rng,
        gen_dev,
        augment=tr["augment"],
        task_ids=task_ids,
    )
    use_ckpt = tr["grad_checkpoint"]
    if use_ckpt == "auto":
        use_ckpt = dev.type == "cuda" and auto_grad_checkpoint(
            graph_span(tr), tr["batch_size"], model.cfg.channels, train_boards.shape
        )
    perturb = tr["perturb"]

    step0 = 0
    log_path = out / "log.jsonl"
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        restore_model_config(model, ema, out / "run.json")
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        for name, m in ema_extra.items():
            m.load_state_dict(ck["ema_extra"][name])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        pool.load_state_dict(ck["pool"])
        rng.bit_generator.state = ck["rng"]
        # map_location moved the generator state to the GPU; set_state wants a CPU ByteTensor
        gen_dev.set_state(ck["gen_dev"].cpu())
        step0 = ck["step"]
        if not quiet:
            print(f"resumed from step {step0}")
        if step0 >= tr["steps"] and (out / "summary.json").exists():
            # already finished: keep the summary (and its final eval) instead of overwriting it
            with open(out / "summary.json") as f:
                return json.load(f).get("final_eval", {})
    elif not quiet:
        print(
            f"{task.name}: {len(train_boards)} train boards ({train_boards.meta.get('source')}), "
            f"model {mcfg.get('preset')} with {count_parameters(model):,} parameters on {dev}"
        )
    data_info = {
        "train_source": train_boards.meta.get("source"),
        "train_n": len(train_boards),
        "train_shape": list(train_boards.shape),
        "val_source": None if val_boards is None else val_boards.meta.get("source"),
        "val_n": None if val_boards is None else len(val_boards),
    }
    if val_boards is not None:
        # a generated validation split must not share puzzles or solutions with the train split
        data_info["train_val_overlap"] = overlap_counts(train_boards, val_boards)
        if max(data_info["train_val_overlap"].values()) > 0 and not quiet:
            print(f"WARNING: train/val overlap {data_info['train_val_overlap']}")
    save_json(
        {
            "config": cfg,
            "seed": seed,
            "params": count_parameters(model),
            "model_config": model.cfg.to_dict(),
            "data": data_info,
        },
        out / "run.json",
    )

    N, Nc = tr["rollout"], tr["chunk"]
    assert Nc <= N
    per_sample_tbptt = tr["chunk_start"] != "shared" or tr["writeback"] != "rollout_end"
    t_start = time.time()
    last_eval: dict = {}
    n_skipped = 0
    model.train()
    for step in range(step0, tr["steps"]):
        idx, batch = pool.sample(tr["batch_size"], seed_ratio)
        if perturb:
            perturb_batch(model, pool, batch, tr, rng, gen_dev)
        x, given4, clamp, targets = batch["x"], batch["given4"], batch["clamp"], batch["targets"]
        loss_mask = batch.get("loss_mask")
        if not per_sample_tbptt:
            c0 = int(rng.integers(0, N - Nc + 1))
            x, loss, gn = _rollout_shared(
                model, opt, x, given4, clamp, targets, loss_mask, c0, tr, gen_dev, use_ckpt
            )
        else:
            if tr["chunk_start"] == "per_sample":
                c0s = rng.integers(0, N - Nc + 1, size=x.shape[0])
            else:
                c0s = np.full(x.shape[0], int(rng.integers(0, N - Nc + 1)))
            x, loss, gn = _rollout_per_sample(
                model, opt, x, given4, clamp, targets, loss_mask, c0s, tr, gen_dev, use_ckpt
            )
        # B.2: the whole rollout, the steps after the chunk included, runs with the parameters of
        # the start of the iteration; they are updated once the last step is done, so that the
        # states written back to the replay buffer come from a single set of weights
        if torch.isfinite(gn):
            opt.step()
        else:  # a non-finite gradient: skip the update rather than poison the weights
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
        sched.step()
        with torch.no_grad():
            d = ema_decay(tr, step)  # warm-up unless `ema_warmup: false`
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for name, m in ema_extra.items():
                d = ema_decay(tr, step, tr["ema_extra"][name].get("ema_warmup", True))
                for pe, pm in zip(m.parameters(), model.parameters(), strict=True):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        batch["x"] = x.detach()
        pool.write_back(idx, batch)

        if (step + 1) % tr["log_every"] == 0 or step == step0:
            with torch.no_grad():
                dec = decode_boards(model, task, x, given4[:, 0], batch["inputs"])
                # scored cells: the loss mask when the task has one (ARC: every cell with a
                # target), otherwise the mutable cells; not the clamp mask `given`
                mut = ~given4[:, 0] if loss_mask is None else loss_mask > 0
                correct = ((dec == targets) & mut).sum(dim=(1, 2)).float()
                cell_acc = (correct / mut.sum(dim=(1, 2)).clamp_min(1)).mean().item()
                exact = (correct == mut.sum(dim=(1, 2))).float().mean().item()
            rec = dict(
                step=step + 1,
                loss=float(loss.item()),
                grad_norm=float(gn),
                lr=sched.get_last_lr()[0],
                batch_cell_acc=cell_acc,
                batch_exact=exact,
                mean_age=float(batch["age"].float().mean().item()),
                skipped_steps=n_skipped,
                elapsed=time.time() - t_start,
            )
            with open(log_path, "a") as f:
                f.write(_json_line(rec))
            if not quiet:
                eta = (tr["steps"] - step - 1) * rec["elapsed"] / max(1, step + 1 - step0)
                print(
                    f"step {step + 1:>7d}/{tr['steps']}  loss {rec['loss']:.4f}  "
                    f"cell {cell_acc:.3f}  exact {exact:.3f}  age {rec['mean_age']:.1f}  "
                    f"lr {rec['lr']:.2e}  {human_time(rec['elapsed'])} (eta {human_time(eta)})"
                )

        if val_boards is not None and (
            (step + 1) % tr["eval_every"] == 0 or step + 1 == tr["steps"]
        ):
            last_eval = evaluate_quick(ema, task, val_boards, tr["eval"], N, seed)
            last_eval["step"] = step + 1
            with open(log_path, "a") as f:
                f.write(_json_line({"eval": last_eval}))
            if not quiet:
                print(
                    f"  eval@{step + 1}: "
                    + "  ".join(f"{k} {v:.3f}" for k, v in last_eval.items() if k != "step")
                )
            model.train()

        stop = bool(tr["stop_at"]) and step + 1 >= tr["stop_at"]
        if (step + 1) % tr["checkpoint_every"] == 0 or step + 1 == tr["steps"] or stop:
            torch.save(
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    ema_extra={name: m.state_dict() for name, m in ema_extra.items()},
                    opt=opt.state_dict(),
                    sched=sched.state_dict(),
                    pool=pool.state_dict(),
                    rng=rng.bit_generator.state,
                    gen_dev=gen_dev.get_state(),
                    step=step + 1,
                ),
                ckpt_path,
            )
        if stop:
            # the LR schedule still follows `steps`, so the probe matches the start of a full run
            if not quiet:
                print(f"stopped at step {step + 1} (stop_at); resume to continue the run")
            return last_eval

    # final.pt: the main EMA; final_raw.pt: the trained weights; final_<name>.pt: the extra EMAs
    for fname, m in [("final", ema), ("final_raw", model)] + [
        (f"final_{name}", m) for name, m in ema_extra.items()
    ]:
        torch.save(
            dict(
                model=m.state_dict(),
                model_config=model.cfg.to_dict(),
                config=cfg,
                seed=seed,
                step=tr["steps"],
            ),
            out / f"{fname}.pt",
        )
    save_json(
        {
            "final_eval": last_eval,
            "train_seconds": time.time() - t_start,
            "steps": tr["steps"],
            "params": count_parameters(model),
            "skipped_steps": n_skipped,
        },
        out / "summary.json",
    )
    return last_eval


def evaluate_quick(
    model: NCA, task, boards: Boards, ecfg: dict, rollout_len: int, seed: int
) -> dict:
    """Small validation during training: K=1, depth = the training rollout unless set."""
    n = min(ecfg.get("n", 256) or len(boards), len(boards))
    sub = boards.subset(np.arange(n))
    rc = RolloutConfig(
        depth=ecfg.get("depth") or rollout_len,
        k=ecfg.get("k", 1),
        amp=ecfg.get("amp", False),
        engine=ecfg.get("engine"),
    )
    res = rollout(model, task, sub, rc, seed=seed, task_ids=sub.meta.get("task_id"))
    m = score_candidates(task, res, sub)
    out = {f"val_{k}": float(v.mean()) for k, v in m.items()}
    out["val_cell_acc"] = float(task.cell_accuracy(res.pred[:, 0], sub).mean())
    out["val_conf"] = float(res.conf.mean())
    return out


def load_model(path: str | Path, device: str | None = None) -> tuple[NCA, dict]:
    """Load `final.pt` (EMA weights) and return (model, run config)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from .model import NCAConfig

    model = NCA(NCAConfig.from_dict(ck["model_config"]))
    model.load_state_dict(ck["model"])
    return model.to(get_device(device)).eval(), ck["config"]


def _json_line(d: dict) -> str:
    import json

    return json.dumps(d) + "\n"
