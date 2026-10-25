"""Inference: roll the NCA out for `D` steps on `K` parallel stochastic trials (paper B.4).

`rollout()` is the single engine behind every evaluation of the repository:

* depth scaling (`depth`), width scaling (`k` parallel trials, selected by board confidence),
* test-time noise injection (`noise`), adaptive firing (`p_low`, `tau`), damage events
  (`damage_at`), Niche-Capped Diversity Pruning (`pruning`, paper B.6),
* traces of accuracy / confidence / cell updates every `record_every` steps, and the decoded
  boards of a few trials for animations.

Everything is batched: boards x trials are flattened into one batch of states and processed in
chunks of at most `max_cells` cells so that a 201x201 maze and a 9x9 Sudoku with K=256 both fit
on an 8 GB GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import NCA
from .perturb import circular_masks, sample_noise
from .pruning import niche_capped_pruning
from .tasks.base import Boards, Task


@dataclass
class RolloutConfig:
    depth: int
    k: int = 1
    fire_rate: float | None = None  # None: the model's training value
    p_low: float | None = None  # adaptive firing: low rate for confident cells
    tau: float | None = None  # adaptive firing: confidence threshold
    given_confident: bool = True  # adaptive firing: clamped cells count as confident
    noise: dict | None = None  # {"r": 0.25, "p_t": 0.1, "p_s": 0.2, "sigma": 0.15}
    damage_at: int | None = None  # test-time damage event (step index)
    damage: dict = field(default_factory=lambda: {"n": 1, "r_min": 0.1, "r_max": 0.4})
    pruning: dict | None = None  # {"m": 4, "n_cap": 3}
    record_every: int = 0  # 0: only the final state is scored
    record_boards: int = 0  # keep decoded boards of the first trial of this many boards
    max_cells: int = 2_000_000  # chunking: boards per pass = max_cells // (k * H * W)
    amp: bool = False
    local_clock: bool = True  # ARC: per-cell clocks at inference
    engine: dict | None = None  # opt-in compiled uniform-firing blocks, same RNG stream


@dataclass
class RolloutResult:
    pred: np.ndarray  # (N, K, H, W) final decoded boards (see `decode_boards`)
    conf: np.ndarray  # (N, K) final board confidence
    alive: np.ndarray  # (N, K) bool, False for trajectories pruned away
    cell_updates: np.ndarray  # (N, K) total number of cell updates (fired cells) per trajectory
    steps: np.ndarray  # recorded step indices (T,)
    trace_exact: np.ndarray | None  # (N, K, T) bool: board exactly solved at recorded steps
    trace_cell_acc: np.ndarray | None  # (N, K, T)
    trace_conf: np.ndarray | None  # (N, K, T)
    trace_updates: np.ndarray | None  # (N, K, T) cumulative cell updates
    boards_trace: (
        np.ndarray | None
    )  # (n_rec, T, H, W) decoded boards of trial 0 of the first boards
    damage_masks: np.ndarray | None  # (N, H, W) when a damage event happened

    @property
    def n_boards(self) -> int:
        return self.pred.shape[0]


def _noise_active(step: int, depth: int, noise: dict | None) -> bool:
    return noise is not None and step < noise.get("r", 1.0) * depth


def decode_boards(
    model: NCA, task: Task, x: torch.Tensor, given: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    """Predicted tokens `(B, H, W)` of a batch of states. When the task's given cells are not
    decodable from the readout (`task.fill_given`: maze, sudoku), they show their input token;
    otherwise (ARC: the readout is a separate slice) every cell's prediction comes from the NCA.
    `given` is `(B, H, W)` bool."""
    dec = model.decode(x)
    return torch.where(given, inputs, dec) if task.fill_given else dec


@torch.no_grad()
def rollout(
    model: NCA,
    task: Task,
    boards: Boards,
    rc: RolloutConfig,
    seed: int = 0,
    task_ids: np.ndarray | None = None,
    progress: bool = False,
) -> RolloutResult:
    """Run the NCA on every board of `boards`, `rc.k` times each."""
    model.eval()
    device = model.device
    N = len(boards)
    H, W = boards.shape
    K = rc.k
    per_pass = max(1, rc.max_cells // (K * H * W))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fast_blocks = None
    if rc.engine is not None:
        if model.cfg.time_steps or rc.p_low is not None or rc.tau is not None:
            raise ValueError(
                "compiled evaluation blocks currently require uniform firing and no step embedding"
            )
        from .engine.engine import EngineConfig
        from .engine.inference import UniformRolloutBlocks

        fast_blocks = UniformRolloutBlocks(model, EngineConfig.from_dict(rc.engine), rc.amp)

    record = rc.record_every > 0
    steps = list(range(rc.record_every, rc.depth + 1, rc.record_every)) if record else []
    if record and (not steps or steps[-1] != rc.depth):  # depth < record_every: record the end
        steps.append(rc.depth)
    T = len(steps)
    out_pred = np.zeros((N, K, H, W), dtype=np.int64)
    out_conf = np.zeros((N, K), dtype=np.float32)
    out_alive = np.ones((N, K), dtype=bool)
    out_updates = np.zeros((N, K), dtype=np.float64)
    tr_exact = np.zeros((N, K, T), dtype=bool) if record else None
    tr_cell = np.zeros((N, K, T), dtype=np.float32) if record else None
    tr_conf = np.zeros((N, K, T), dtype=np.float32) if record else None
    tr_upd = np.zeros((N, K, T), dtype=np.float64) if record else None
    boards_trace = (
        np.zeros((rc.record_boards, T, H, W), dtype=np.int64) if rc.record_boards else None
    )
    damage_masks = np.zeros((N, H, W), dtype=bool) if rc.damage_at is not None else None

    iterator = range(0, N, per_pass)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"rollout D={rc.depth} K={K}", leave=False)

    for start in iterator:
        sub = boards.subset(np.arange(start, min(N, start + per_pass)))
        n = len(sub)
        # flatten boards x trials
        inputs = torch.from_numpy(np.repeat(sub.inputs, K, axis=0)).to(device)
        targets = torch.from_numpy(np.repeat(sub.targets, K, axis=0)).to(device)
        given = torch.from_numpy(np.repeat(sub.given, K, axis=0)).to(device)
        tids = None
        if task_ids is not None:
            tids = torch.from_numpy(np.repeat(task_ids[start : start + n], K)).to(device)
        x, given4, clamp = model.init_state(inputs, given, tids, gen)
        B = n * K
        # Pruned trajectories are removed from the batch ("compaction", paper B.6), so the GPU
        # work really shrinks. `idx` maps each row of the compact batch to its (board, trial)
        # index in [0, B); the output arrays of this pass are addressed through flat views.
        idx = torch.arange(B, device=device)
        updates = torch.zeros(B, dtype=torch.float64, device=device)
        clock = torch.zeros((B, 1, H, W), dtype=torch.long, device=device)
        # the cells that are scored (maze, sudoku: the mutable cells; ARC: every cell with a
        # target, see Task.eval_mask)
        mutable = torch.from_numpy(np.repeat(task.eval_mask(sub), K, axis=0)).to(device).float()
        n_mut = mutable.sum(dim=(1, 2)).clamp_min(1)
        t_rec = 0
        prune_steps = set()
        if rc.pruning:
            m = int(rc.pruning["m"])
            dt = rc.depth // m
            prune_steps = {i * dt for i in range(1, m)}
        # flat (board x trial) views of this pass's slice of the output arrays
        o = dict(
            pred=out_pred[start : start + n].reshape(B, H, W),
            conf=out_conf[start : start + n].reshape(B),
            alive=out_alive[start : start + n].reshape(B),
            upd=out_updates[start : start + n].reshape(B),
        )
        if record:
            o["tr_exact"] = tr_exact[start : start + n].reshape(B, T)
            o["tr_cell"] = tr_cell[start : start + n].reshape(B, T)
            o["tr_conf"] = tr_conf[start : start + n].reshape(B, T)
            o["tr_upd"] = tr_upd[start : start + n].reshape(B, T)

        def score(x, inputs, targets, given, mutable, n_mut):
            dec = decode_boards(model, task, x, given, inputs)
            correct = ((dec == targets).float() * mutable).sum(dim=(1, 2))
            return dec, correct / n_mut, correct == n_mut, model.board_confidence(x)

        def store(rows, dec, cell, exact, conf, upd, t_from, o=o):
            """Final values of the rows `rows` (numpy flat indices); a trajectory that leaves the
            batch keeps its last state for the remaining recorded steps."""
            o["pred"][rows] = dec.cpu().numpy()
            o["conf"][rows] = conf.cpu().numpy()
            o["upd"][rows] = upd.cpu().numpy()
            if record and t_from < T:
                o["tr_exact"][rows, t_from:] = exact.cpu().numpy()[:, None]
                o["tr_cell"][rows, t_from:] = cell.cpu().numpy()[:, None]
                o["tr_conf"][rows, t_from:] = conf.cpu().numpy()[:, None]
                o["tr_upd"][rows, t_from:] = upd.cpu().numpy()[:, None]

        skip_until = 0
        boundaries = sorted(
            {rc.depth, *steps, *prune_steps}
            | ({rc.damage_at} if rc.damage_at is not None else set())
        )
        for t in range(rc.depth):
            if t < skip_until:
                continue
            if rc.damage_at is not None and t == rc.damage_at:
                masks = circular_masks(
                    n, H, W, rc.damage["n"], rc.damage["r_min"], rc.damage["r_max"], rng
                )
                damage_masks[start : start + n] = masks
                mt = torch.from_numpy(np.repeat(masks, K, axis=0)).to(device)[idx][:, None]
                x = model.apply_clamp(x.masked_fill(mt, 0.0), given4, clamp)
            stop = (
                t + 1
                if fast_blocks is None
                else min(t + fast_blocks.config.block, next(b for b in boundaries if b > t))
            )
            fires, noises = [], []
            for draw_t in range(t, stop):
                fm = model.sample_fire_mask(
                    x, rc.fire_rate, rc.p_low, rc.tau, gen, given4 if rc.given_confident else None
                )
                noise = None
                if _noise_active(draw_t, rc.depth, rc.noise):
                    nz = rc.noise
                    noise = sample_noise(x, nz["p_t"], nz["p_s"], nz["sigma"], gen)
                if fast_blocks is not None and rc.noise is not None and noise is None:
                    # Keep the compiled signature stable even when all boards
                    # miss the noise draw. Allocating zeros consumes no RNG.
                    noise = torch.zeros_like(x)
                fires.append(fm)
                noises.append(noise)
                updates += fm.sum(dim=(1, 2, 3)).double()
            if fast_blocks is None:
                clk = (
                    clock
                    if (model.cfg.time_steps and rc.local_clock)
                    else torch.tensor(t, device=device)
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=rc.amp):
                    x = model.step(x, given4, clamp, fm, noise, clk)
                x = x.float()
                clock = clock + fm.long()
            else:
                x = fast_blocks(x, given4, clamp, fires, noises)
            skip_until = stop
            t = stop - 1

            if (t + 1) in prune_steps:
                dec, cell, exact, conf = score(x, inputs, targets, given, mutable, n_mut)
                # the pruning rule sees the full (board, trial) layout: dead rows are not alive
                dec_full = torch.zeros((B, H, W), dtype=dec.dtype, device=device)
                conf_full = torch.full((B,), -float("inf"), device=device)
                alive_full = torch.zeros(B, dtype=torch.bool, device=device)
                dec_full[idx], conf_full[idx], alive_full[idx] = dec, conf, True
                keep = niche_capped_pruning(
                    dec_full.view(n, K, H, W),
                    conf_full.view(n, K),
                    alive_full.view(n, K),
                    rc.pruning.get("n_cap", 3),
                    rng,
                ).view(B)[idx]
                drop = ~keep
                if drop.any():
                    rows = idx[drop].cpu().numpy()
                    store(
                        rows, dec[drop], cell[drop], exact[drop], conf[drop], updates[drop], t_rec
                    )
                    o["alive"][rows] = False
                    sel = keep.nonzero().squeeze(1)
                    x, given4, clamp = x[sel], given4[sel], clamp[sel]
                    inputs, targets, given = inputs[sel], targets[sel], given[sel]
                    mutable, n_mut = mutable[sel], n_mut[sel]
                    clock, updates, idx = clock[sel], updates[sel], idx[sel]
                    tids = None if tids is None else tids[sel]

            if record and (t + 1) == steps[t_rec]:
                dec, cell, exact, conf = score(x, inputs, targets, given, mutable, n_mut)
                rows = idx.cpu().numpy()
                o["tr_exact"][rows, t_rec] = exact.cpu().numpy()
                o["tr_cell"][rows, t_rec] = cell.cpu().numpy()
                o["tr_conf"][rows, t_rec] = conf.cpu().numpy()
                o["tr_upd"][rows, t_rec] = updates.cpu().numpy()
                if boards_trace is not None and start < rc.record_boards:
                    nb = min(n, rc.record_boards - start)
                    o["pred"][rows] = dec.cpu().numpy()  # first trial of the first boards
                    boards_trace[start : start + nb, t_rec] = out_pred[start : start + nb, 0]
                t_rec += 1

        dec, cell, exact, conf = score(x, inputs, targets, given, mutable, n_mut)
        store(idx.cpu().numpy(), dec, cell, exact, conf, updates, T)

    return RolloutResult(
        out_pred,
        out_conf,
        out_alive,
        out_updates,
        np.array(steps),
        tr_exact,
        tr_cell,
        tr_conf,
        tr_upd,
        boards_trace,
        damage_masks,
    )


# ---------------------------------------------------------------------- scoring helpers
def score_candidates(task: Task, res: RolloutResult, boards: Boards) -> dict[str, np.ndarray]:
    """Task metrics for every (board, trial): dict of (N, K) arrays."""
    N, K, H, W = res.pred.shape
    flat = res.pred.reshape(N * K, H, W)
    m = task.metrics(flat, boards.repeat(K))  # per-board meta (ARC) repeated along K too
    return {k: np.asarray(v).reshape(N, K) for k, v in m.items()}


def conf_at_k(
    metric: np.ndarray,
    conf: np.ndarray,
    alive: np.ndarray,
    ks: list[int],
    weights: np.ndarray | None = None,
) -> dict[int, float]:
    """Conf@k for several k from one run with K trials: for each k, use the first k trials,
    pick the alive one with the highest board confidence and report the mean metric (paper
    B.4, "Final Selection"). `weights` (one per board) gives a weighted mean instead."""
    N, K = metric.shape
    out = {}
    for k in ks:
        k = min(k, K)
        c = np.where(alive[:, :k], conf[:, :k], -np.inf)
        best = c.argmax(axis=1)
        out[k] = float(np.average(metric[np.arange(N), best], weights=weights))
    return out


def pass_at_k(
    metric: np.ndarray, alive: np.ndarray, ks: list[int], weights: np.ndarray | None = None
) -> dict[int, float]:
    """Pass@k (oracle selection): any of the first k trials correct; `weights` (one per board)
    gives a weighted mean."""
    N, K = metric.shape
    return {
        min(k, K): float(
            np.average(
                (metric[:, : min(k, K)] & alive[:, : min(k, K)]).any(axis=1), weights=weights
            )
        )
        for k in ks
    }


def majority_vote(pred: np.ndarray, alive: np.ndarray, top: int = 2) -> list[list[np.ndarray]]:
    """Per board, the `top` most frequent distinct predicted boards (ARC pass@M)."""
    N, K = pred.shape[:2]
    out = []
    for n in range(N):
        counts: dict[bytes, int] = {}
        keep: dict[bytes, np.ndarray] = {}
        for k in range(K):
            if not alive[n, k]:
                continue
            key = pred[n, k].tobytes()
            counts[key] = counts.get(key, 0) + 1
            keep[key] = pred[n, k]
        ranked = sorted(counts, key=lambda b: -counts[b])[:top]
        out.append([keep[b] for b in ranked])
    return out
