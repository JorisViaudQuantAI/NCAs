"""The experiments of the paper, as functions that return plain dictionaries.

Each function takes a trained model and a `Boards` set, runs `rollout()` with the right options
and returns JSON-serialisable results. `scripts/eval_*.py` call them and write
`results/<experiment>/<task>_seed<k>.json`; `scripts/make_figures.py` draws the figures from
those files only. Nothing here depends on matplotlib.

Experiment  -> paper figure / table
----------------------------------
tts_curves          Fig. 4A, 16, 17: accuracy vs depth D and vs parallel trials K (conf@K)
noise_sweep         Fig. 7B: test-time noise magnitude sweep
maze_size_scaling   Fig. 4B and Table 1 (Maze-OOD): accuracy vs maze size with D scaled
adaptive_compute    Fig. 8A/C: adaptive firing, steps and cell updates to solve, damage recovery
pruning_pareto      Fig. 6A: accuracy vs compute with Niche-Capped Diversity Pruning
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .model import NCA
from .pruning import pruned_compute_fraction
from .rollout import (
    RolloutConfig,
    RolloutResult,
    conf_at_k,
    majority_vote,
    pass_at_k,
    rollout,
    score_candidates,
)
from .tasks.base import Boards, Task


def _group_means(values: np.ndarray, groups: np.ndarray | None) -> dict:
    """Mean of `values` per group label (e.g. per difficulty level)."""
    if groups is None:
        return {}
    return {str(g): float(values[groups == g].mean()) for g in np.unique(groups)}


def _ks(k_max: int) -> list[int]:
    ks = [1]
    while ks[-1] * 2 <= k_max:
        ks.append(ks[-1] * 2)
    return ks


def majority_pass_at_m(
    res: RolloutResult, boards: Boards, ks: list[int], top: int = 2
) -> dict[int, dict[int, float]]:
    """ARC selection of the paper (B.4): every trial's canvas is mapped back to the task's
    canonical frame (`nca.tasks.arc.map_back` with the board's scale / offset / out_shape /
    dihedral_k / color_perm meta) and the `top` most-voted grids are the attempts. For every k in
    `ks` (the first k trials): {k: {1: pass@1, 2: pass@2, ...}}, the fraction of boards whose
    canonical target is among the first m attempts. Boards without a target never pass."""
    from .tasks.arc import map_back

    N, K = res.pred.shape[:2]
    meta = boards.meta
    canon = np.empty((N, K), dtype=object)
    target = []
    for n in range(N):
        s, o, shape = meta["scale"][n], meta["offset"][n], meta["out_shape"][n]
        kd, perm = meta["dihedral_k"][n], meta["color_perm"][n]
        for j in range(K):
            canon[n, j] = map_back(res.pred[n, j], s, o, shape, kd, perm)
        has = bool(meta["has_target"][n])
        target.append(map_back(boards.targets[n], s, o, shape, kd, perm) if has else None)
    out: dict[int, dict[int, float]] = {}
    for k in ks:
        k = min(k, K)
        voted = majority_vote(canon[:, :k], res.alive[:, :k], top=top)
        hits = np.zeros((N, top), dtype=bool)
        for n in range(N):
            if target[n] is None:
                continue
            for m, g in enumerate(voted[n]):
                hits[n, m] = g.shape == target[n].shape and bool(np.array_equal(g, target[n]))
        out[k] = {m + 1: float(hits[:, : m + 1].any(axis=1).mean()) for m in range(top)}
    return out


def tts_curves(
    model: NCA,
    task: Task,
    boards: Boards,
    depth: int,
    k: int,
    record_every: int,
    noise: dict | None = None,
    seed: int = 0,
    group_key: str | None = None,
    max_cells: int = 2_000_000,
    progress: bool = False,
    amp: bool = False,
    select: str = "conf",
    engine: dict | None = None,
) -> dict:
    """One rollout with K trials and traces: gives the depth curve (K=1 and conf@K at every
    recorded step is not stored; we store the K=1 trace and the final conf@k curve).

    `select`: "conf" (board confidence, conf@K; always reported) or "majority" (ARC only: the
    paper's majority vote over the trials mapped back to the canonical frame, reported as
    pass@1 / pass@2 of the most-voted grids under `majority_pass_at_k`)."""
    if select not in ("conf", "majority"):
        raise ValueError(f"select must be 'conf' or 'majority', got {select!r}")
    if select == "majority" and "out_shape" not in boards.meta:
        raise ValueError("majority selection needs ARC boards (map_back meta)")
    rc = RolloutConfig(
        depth=depth,
        k=k,
        noise=noise,
        record_every=record_every,
        max_cells=max_cells,
        amp=amp,
        engine=engine,
    )
    res = rollout(
        model, task, boards, rc, seed=seed, task_ids=boards.meta.get("task_id"), progress=progress
    )
    m = score_candidates(task, res, boards)
    main = m[task.main_metric]
    groups = boards.meta.get(group_key) if group_key else None
    ks = _ks(k)
    out = dict(
        depth=depth,
        k=k,
        noise=noise,
        n_boards=len(boards),
        metric=task.main_metric,
        engine=engine,
        steps=res.steps.tolist(),
        # depth curve: fraction of boards solved at each recorded step, K=1 (first trial)
        depth_curve=res.trace_exact[:, 0, :].mean(axis=0).tolist(),
        depth_curve_mean_over_trials=res.trace_exact.mean(axis=(0, 1)).tolist(),
        cell_acc_curve=res.trace_cell_acc[:, 0, :].mean(axis=0).tolist(),
        conf_curve=res.trace_conf[:, 0, :].mean(axis=0).tolist(),
        # width curve
        conf_at_k=conf_at_k(main, res.conf, res.alive, ks),
        pass_at_k=pass_at_k(main, res.alive, ks),
        mean_trial_acc=float(main.mean()),
        metrics={name: float(v.mean()) for name, v in m.items()},
        # per-difficulty breakdown of conf@K (max K) and K=1
        by_group_k1=_group_means(main[:, 0], groups),
        by_group_confk=_group_means(
            main[np.arange(len(boards)), np.where(res.alive, res.conf, -np.inf).argmax(1)], groups
        ),
        by_group_passk=_group_means((main & res.alive).any(axis=1), groups),
        cell_updates_per_trial=float(res.cell_updates.mean()),
        source=boards.meta.get("source"),
        max_cells=max_cells,
        amp=amp,
        select=select,
        # True: stratified sample, the global scores to compare are the *_weighted ones;
        # False: uniform sample or full set, conf_at_k is already the benchmark's mean
        weighted="weight" in boards.meta,
    )
    if "weight" in boards.meta:
        # stratified sample (Sudoku-Extreme): the global score re-weighted by the benchmark's
        # share of each difficulty bin, comparable to the paper's score on the full test set
        w = np.asarray(boards.meta["weight"], dtype=np.float64)
        out["conf_at_k_weighted"] = conf_at_k(main, res.conf, res.alive, ks, weights=w)
        out["pass_at_k_weighted"] = pass_at_k(main, res.alive, ks, weights=w)
        out["depth_curve_weighted"] = np.average(
            res.trace_exact[:, 0, :], axis=0, weights=w
        ).tolist()
    if select == "majority":
        out["majority_pass_at_k"] = majority_pass_at_m(res, boards, ks, top=2)
    # depth curve per group (K=1)
    if groups is not None:
        out["depth_curve_by_group"] = {
            str(g): res.trace_exact[groups == g, 0, :].mean(axis=0).tolist()
            for g in np.unique(groups)
        }
    return out


def noise_sweep(
    model: NCA,
    task: Task,
    boards: Boards,
    depth: int,
    k: int,
    sigmas: list[float],
    p_t: float,
    p_s: float,
    r: float = 0.25,
    seed: int = 0,
    max_cells: int = 2_000_000,
    progress: bool = False,
    amp: bool = False,
) -> dict:
    """Fig. 7B: conf@K and K=1 accuracy as a function of the test-time noise magnitude
    (benchmark-weighted means on a stratified Sudoku-Extreme sample, `weighted: true`)."""
    w = boards.meta.get("weight")
    w = None if w is None else np.asarray(w, dtype=np.float64)
    rows = []
    for sigma in sigmas:
        noise = None if sigma == 0 else dict(r=r, p_t=p_t, p_s=p_s, sigma=sigma)
        rc = RolloutConfig(depth=depth, k=k, noise=noise, max_cells=max_cells, amp=amp)
        res = rollout(
            model,
            task,
            boards,
            rc,
            seed=seed,
            task_ids=boards.meta.get("task_id"),
            progress=progress,
        )
        m = score_candidates(task, res, boards)[task.main_metric]
        rows.append(
            dict(
                sigma=sigma,
                k1=float(np.average(m[:, 0], weights=w)),
                conf_k=conf_at_k(m, res.conf, res.alive, [k], weights=w)[min(k, m.shape[1])],
                pass_k=pass_at_k(m, res.alive, [k], weights=w)[min(k, m.shape[1])],
                mean_trial=float(np.average(m.mean(axis=1), weights=w)),
            )
        )
    return dict(
        depth=depth,
        k=k,
        p_t=p_t,
        p_s=p_s,
        r=r,
        n_boards=len(boards),
        source=boards.meta.get("source"),
        max_cells=max_cells,
        amp=amp,
        weighted=w is not None,
        rows=rows,
    )


def maze_size_scaling(
    model: NCA,
    task: Task,
    load_size,
    sizes: list[int],
    depth_for,
    n_boards: int,
    record_every_frac: float = 0.05,
    seed: int = 0,
    max_cells: int = 2_000_000,
    progress: bool = False,
    amp: bool = False,
) -> dict:
    """Fig. 4B / Table 1: run the 9x9-trained NCA on larger mazes. `load_size(size, n)` returns the
    boards, `depth_for(size)` the rollout depth."""
    rows = []
    for size in sizes:
        boards = load_size(size, n_boards)
        depth = int(depth_for(size))
        rc = RolloutConfig(
            depth=depth,
            k=1,
            record_every=max(1, int(depth * record_every_frac)),
            max_cells=max_cells,
            amp=amp,
        )
        res = rollout(model, task, boards, rc, seed=seed, progress=progress)
        m = score_candidates(task, res, boards)
        rows.append(
            dict(
                size=size,
                depth=depth,
                n_boards=len(boards),
                source=boards.meta.get("source"),
                **{k: float(v.mean()) for k, v in m.items()},
                steps=res.steps.tolist(),
                exact_curve=res.trace_exact[:, 0, :].mean(0).tolist(),
                cell_updates=float(res.cell_updates.mean()),
            )
        )
    return dict(sizes=sizes, max_cells=max_cells, amp=amp, rows=rows)


ADAPTIVE_VARIANTS = ("uniform", "adaptive", "adaptive_given")
"""Firing rules compared by `adaptive_compute`:

uniform         every cell fires with probability p_fire (the training rule)
adaptive        paper 4.4 as written: a cell whose confidence exceeds tau fires at p_low; the
                clamped (given) cells never do, since their output slice is pinned
adaptive_given  the same, with the clamped cells counted as confident (they fire at p_low);
                see docs/DEVIATIONS.md, "Confidence of given cells"
"""


def adaptive_compute(
    model: NCA,
    task: Task,
    boards: Boards,
    depth: int,
    p_fire: float,
    p_low: float,
    tau: float,
    record_every: int,
    damage_at: int | None = None,
    damage: dict | None = None,
    seed: int = 0,
    max_cells: int = 2_000_000,
    progress: bool = False,
    amp: bool = False,
    variants: tuple[str, ...] = ADAPTIVE_VARIANTS,
    previous: dict | None = None,
) -> dict:
    """Fig. 8: uniform vs adaptive firing; steps and cumulative cell updates to reach 95% of boards
    solved; optional damage event at `damage_at`.

    `previous` is the result of an earlier call: its variants are kept (not recomputed) when
    they already reached 95% or were run at least as deep as `depth`, and only if `previous`
    carries the same `fingerprint` (model weights, boards, seed, firing and damage parameters,
    chunking, precision); anything else is recomputed. Each variant stores its own `depth`;
    `ratios` compares each adaptive rule to `uniform` as the paper does (steps and cell updates
    to 95%, adaptive / uniform).
    """
    damage = damage or {"n": 1, "r_min": 0.1, "r_max": 0.4}
    fingerprint = experiment_fingerprint(
        model,
        boards,
        seed=seed,
        p_fire=p_fire,
        p_low=p_low,
        tau=tau,
        damage_at=damage_at,
        damage=damage,
        max_cells=max_cells,
        amp=amp,
        record_every=record_every,
    )
    if previous is not None and previous.get("fingerprint") != fingerprint:
        previous = None  # a different experiment (or a file written before fingerprints)
    out = dict(
        depth=depth,
        p_fire=p_fire,
        p_low=p_low,
        tau=tau,
        damage_at=damage_at,
        n_boards=len(boards),
        source=boards.meta.get("source"),
        max_cells=max_cells,
        amp=amp,
        fingerprint=fingerprint,
        variants={},
    )
    rules = {
        "uniform": dict(p_low=None, tau=None, given_confident=False),
        "adaptive": dict(p_low=p_low, tau=tau, given_confident=False),
        "adaptive_given": dict(p_low=p_low, tau=tau, given_confident=True),
    }
    for name in variants:
        if name not in rules:
            raise ValueError(f"unknown variant {name!r}; choose from {list(rules)}")
        old = (previous or {}).get("variants", {}).get(name)
        old_depth = old.get("depth", previous.get("depth", 0)) if old is not None else 0
        if old is not None and damage_at is not None:
            # a curve that stops before the damage event, or whose 95% crossing predates it
            # (files written before the crossing was searched after the event), says nothing
            # about the recovery: recompute it
            reached = old.get("steps_to_95")
            if old_depth <= damage_at or (reached is not None and reached <= damage_at):
                old = None
        if old is not None and (old.get("steps_to_95") is not None or old_depth >= depth):
            old.setdefault("depth", old_depth)  # files written before the per-variant key
            out["variants"][name] = old
            continue
        rc = RolloutConfig(
            depth=depth,
            k=1,
            fire_rate=p_fire,
            record_every=record_every,
            damage_at=damage_at,
            damage=damage,
            max_cells=max_cells,
            amp=amp,
            **rules[name],
        )
        res = rollout(model, task, boards, rc, seed=seed, progress=progress)
        solved = res.trace_exact[:, 0, :].mean(axis=0)
        upd = res.trace_updates[:, 0, :].mean(axis=0)
        out["variants"][name] = dict(
            depth=depth,
            steps=res.steps.tolist(),
            solved_curve=solved.tolist(),
            cell_updates_curve=upd.tolist(),
            cell_acc_curve=res.trace_cell_acc[:, 0, :].mean(axis=0).tolist(),
            final_exact=float(solved[-1]),
            final_updates=float(upd[-1]),
            **recovery_point(res.steps, solved, upd, damage_at),
        )
    out["ratios"] = adaptive_ratios(out["variants"])
    return out


def experiment_fingerprint(model: NCA, boards: Boards, **params) -> str:
    """A short hash of what determines a rollout's result: the model weights and its config
    (initial-state scale, normalisation, ...), the boards (inputs, targets, given mask, source)
    and the given parameters (which must include anything that changes the recorded curves,
    such as `record_every`). Used to decide whether a stored result can be reused."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().float().cpu().numpy().tobytes())
    h.update(json.dumps(model.cfg.to_dict(), sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(boards.inputs).tobytes())
    h.update(np.ascontiguousarray(boards.targets).tobytes())
    h.update(np.ascontiguousarray(boards.given).tobytes())
    meta = {"source": boards.meta.get("source"), **params}
    h.update(json.dumps(meta, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


RECOVERY_KEYS = ("steps_to_95", "updates_to_95")
RECOVERY_SINCE_DAMAGE_KEYS = ("steps_since_damage_to_95", "updates_since_damage_to_95")


def recovery_point(steps, solved, upd, damage_at: int | None = None) -> dict:
    """The first recorded step at which at least 95% of the boards are solved, as the paper's
    Fig. 8 reads it: `steps_to_95` (absolute step index, i.e. the total solve time) and
    `updates_to_95` (cumulative cell updates per board since step 0).

    With a damage event at `damage_at` (applied before step `damage_at` runs, so the record at
    `steps == damage_at` is the state just before it), only records strictly after the event
    count: a board solved before the damage and never repaired is not a recovery. The file
    then also carries the time elapsed since the event, `steps_since_damage_to_95` and
    `updates_since_damage_to_95` (updates counted from the last record before the damage)."""
    steps = np.asarray(steps)
    solved = np.asarray(solved, dtype=np.float64)
    upd = np.asarray(upd, dtype=np.float64)
    after = np.ones(len(steps), dtype=bool) if damage_at is None else steps > damage_at
    hits = np.flatnonzero((solved >= 0.95) & after)
    reach = None if len(hits) == 0 else int(hits[0])
    out = {
        "steps_to_95": None if reach is None else int(steps[reach]),
        "updates_to_95": None if reach is None else float(upd[reach]),
    }
    if damage_at is not None:
        before = np.flatnonzero(steps <= damage_at)
        upd_at_damage = float(upd[before[-1]]) if len(before) else 0.0
        out["steps_since_damage_to_95"] = None if reach is None else int(steps[reach]) - damage_at
        out["updates_since_damage_to_95"] = (
            None if reach is None else float(upd[reach]) - upd_at_damage
        )
    return out


def adaptive_ratios(variants: dict) -> dict:
    """Paper Fig. 8 numbers: for each adaptive rule, steps to 95% and cell updates to 95% divided
    by the uniform rule's; None when either side did not reach 95%. After a damage event the
    same ratios are also given for the time elapsed since the event (`*_since_damage_to_95`).
    `updates_per_step` is the mean firing rate relative to uniform, defined whatever the
    accuracy reached."""
    base = variants.get("uniform")
    ratios = {}
    for name, v in variants.items():
        if name == "uniform" or base is None:
            continue
        r = {"steps_to_95": None, "updates_to_95": None, "updates_per_step": None}
        if v["steps_to_95"] is not None and base["steps_to_95"]:
            r["steps_to_95"] = v["steps_to_95"] / base["steps_to_95"]
            r["updates_to_95"] = v["updates_to_95"] / base["updates_to_95"]
        if "steps_since_damage_to_95" in v and "steps_since_damage_to_95" in base:
            for key in RECOVERY_SINCE_DAMAGE_KEYS:
                r[key] = None
                if v[key] is not None and base[key]:
                    r[key] = v[key] / base[key]
        if v["steps"] and base["steps"]:
            r["updates_per_step"] = (v["final_updates"] / v["steps"][-1]) / (
                base["final_updates"] / base["steps"][-1]
            )
        ratios[name] = r
    return ratios


def pruning_pareto(
    model: NCA,
    task: Task,
    boards: Boards,
    depths: list[int],
    k: int,
    ms: list[int],
    n_cap: int = 3,
    noise: dict | None = None,
    seed: int = 0,
    max_cells: int = 2_000_000,
    progress: bool = False,
    amp: bool = False,
) -> dict:
    """Fig. 6A: accuracy (conf@K) against total compute (in cell-step units relative to K x D)
    for unpruned rollouts (m = 1) and pruning divisors `ms`. On a stratified sample with
    benchmark weights (Sudoku-Extreme) accuracies are the weighted means (`weighted: true`)."""
    w = boards.meta.get("weight")
    w = None if w is None else np.asarray(w, dtype=np.float64)
    rows = []
    for depth in depths:
        for m in [1] + list(ms):
            pruning = None if m == 1 else dict(m=m, n_cap=n_cap)
            rc = RolloutConfig(
                depth=depth, k=k, noise=noise, pruning=pruning, max_cells=max_cells, amp=amp
            )
            res = rollout(model, task, boards, rc, seed=seed, progress=progress)
            met = score_candidates(task, res, boards)[task.main_metric]
            acc = conf_at_k(met, res.conf, res.alive, [k], weights=w)[min(k, met.shape[1])]
            rows.append(
                dict(
                    depth=depth,
                    m=m,
                    k=k,
                    conf_k=acc,
                    pass_k=pass_at_k(met, res.alive, [k], weights=w)[min(k, met.shape[1])],
                    compute_units=float(
                        np.average(res.cell_updates.sum(axis=1), weights=w)
                    ),  # fired cells, per board
                    compute_rel=depth * k * pruned_compute_fraction(m) if m > 1 else depth * k,
                    theoretical_fraction=pruned_compute_fraction(m) if m > 1 else 1.0,
                )
            )
    return dict(
        k=k,
        depths=depths,
        ms=ms,
        n_cap=n_cap,
        noise=noise,
        n_boards=len(boards),
        source=boards.meta.get("source"),
        max_cells=max_cells,
        amp=amp,
        weighted=w is not None,
        rows=rows,
    )


def aggregate_seeds(runs: list[dict], keys: list[str]) -> dict:
    """mean/std over seeds of scalar or list-valued entries `keys` found in each run dict."""
    out = {}
    for key in keys:
        vals = [np.asarray(_dig(r, key), dtype=float) for r in runs if _dig(r, key) is not None]
        if not vals:
            continue
        arr = np.stack(vals)
        out[key] = dict(mean=arr.mean(axis=0).tolist(), std=arr.std(axis=0).tolist(), n=len(vals))
    return out


def _dig(d: dict, dotted: str):
    for p in dotted.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d
