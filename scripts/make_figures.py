#!/usr/bin/env python
"""Draw every figure from the JSON files in results/ (and the training logs in runs/).

    python scripts/make_figures.py                 # all figures that have data, into figures/
    python scripts/make_figures.py --only tts noise

Each figure averages over the seeds found (`*_seed<k>.json`) and shades +-1 std. Figures that
have no data yet are skipped with a message. Paper correspondence:

    figures/F1_maze_size_scaling      Fig. 4B, Table 1 (Maze-OOD)
    figures/F2_tts_<task>             Fig. 4A, 16, 17 (depth and parallel-trial curves)
    figures/F3_noise_<task>           Fig. 7B
    figures/F4_ablations_<task>       Fig. 7A and Fig. 13
    figures/F5_adaptive_compute       Fig. 8A/C
    figures/F6_pruning                Fig. 6A
    figures/F7_training_<task>        training curves (loss, validation accuracy) over 3 seeds
    figures/F8_recipe_search          paper recipe vs public-code recipe (docs/RESULTS.md)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.tasks import TASKS  # noqa: E402
from nca.viz import CATEGORICAL, band, finish, seq_colors, set_style  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("figures")


def load_group(folder: str, prefix: str) -> list[dict]:
    files = (
        sorted((RESULTS / folder).glob(f"{prefix}_seed*.json"))
        if (RESULTS / folder).exists()
        else []
    )
    return [json.load(open(f)) for f in files]


def groups_in(folder: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    if not (RESULTS / folder).exists():
        return out
    for f in sorted((RESULTS / folder).glob("*_seed*.json")):
        m = re.fullmatch(r"(.+)_seed\d+\.json", f.name)
        if m:
            out[m.group(1)].append(json.load(open(f)))
    return out


def mean_std(runs: list[dict], key, cast=float):
    vals = np.array(
        [cast(key(r)) if not isinstance(key(r), list) else key(r) for r in runs], dtype=float
    )
    return vals.mean(axis=0), vals.std(axis=0)


# ---------------------------------------------------------------------------- F1
def fig_maze_scaling():
    import matplotlib.pyplot as plt

    runs = load_group("maze_scaling", "maze_ood")
    if not runs:
        return print("F1: no results/maze_scaling/maze_ood_seed*.json yet")
    sizes = runs[0]["sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for metric, color in zip(["exact", "valid", "optimal"], CATEGORICAL, strict=False):
        m, s = mean_std(runs, lambda r, k=metric: [row[k] for row in r["rows"]])
        band(axes[0], sizes, 100 * m, 100 * s, label=metric, color=color, marker="o")
    axes[0].set_xlabel("maze size (trained on 9x9)")
    axes[0].set_ylabel("% of test mazes")
    axes[0].set_ylim(0, 102)
    axes[0].set_xscale("log")
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels([str(s) for s in sizes], rotation=45)
    axes[0].minorticks_off()  # the log scale's minor labels (2x10^1, ...) overlap the sizes
    axes[0].legend()
    axes[0].set_title(f"Spatial substrate scaling ({len(runs)} seeds)")
    # convergence curves per size
    cols = seq_colors(len(sizes))
    for row, c in zip(runs[0]["rows"], cols, strict=True):
        steps = np.array(row["steps"]) / row["depth"]
        axes[1].plot(
            steps, 100 * np.array(row["exact_curve"]), color=c, label=f"{row['size']}x{row['size']}"
        )
    axes[1].set_xlabel("fraction of the rollout depth D(size)")
    axes[1].set_ylabel("% solved (seed 0)")
    axes[1].legend(ncol=2, fontsize=7)
    axes[1].set_title("Solving as the rollout proceeds")
    finish(fig, FIGURES / "F1_maze_size_scaling")
    print("F1 drawn")


# ---------------------------------------------------------------------------- F2
def overall(r: dict, key: str):
    """A global (all-boards) curve or score: the benchmark-weighted version when the sample is
    stratified (Sudoku-Extreme), so that global numbers follow the paper's distribution; the
    per-difficulty curves stay on the balanced sample."""
    return r.get(key + "_weighted", r[key])


def fig_tts():
    import matplotlib.pyplot as plt

    groups = groups_in("tts")
    bases = sorted({k.replace("_noise", "") for k in groups})
    for base in bases:
        runs = groups.get(base, [])
        runs_noise = groups.get(base + "_noise", [])
        if not runs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        # depth curves per group (K = 1)
        by = runs[0].get("depth_curve_by_group")
        if by:
            keys = sorted(by, key=lambda k: float(k))
            cols = seq_colors(len(keys))
            for k, c in zip(keys, cols, strict=True):
                m, s = mean_std(runs, lambda r, k=k: r["depth_curve_by_group"][k])
                band(axes[0], runs[0]["steps"], 100 * m, 100 * s, label=k, color=c)
            axes[0].legend(title=runs[0].get("group", "level"), ncol=2, fontsize=6)
        m, s = mean_std(runs, lambda r: overall(r, "depth_curve"))
        all_label = "all (benchmark-weighted)" if "depth_curve_weighted" in runs[0] else "all"
        band(
            axes[0],
            runs[0]["steps"],
            100 * m,
            100 * s,
            label=all_label,
            color="black",
            linewidth=1.2,
        )
        axes[0].set_xlabel("rollout steps D")
        axes[0].set_ylabel("% boards solved (K = 1)")
        axes[0].set_title(f"{base}: depth scaling ({len(runs)} seeds)")
        # width curves
        ks = sorted(int(k) for k in runs[0]["conf_at_k"])
        for label, rs, ls in [("conf@K", runs, "-"), ("conf@K + test noise", runs_noise, "--")]:
            if not rs:
                continue
            m, s = mean_std(rs, lambda r, ks=ks: [overall(r, "conf_at_k")[str(k)] for k in ks])
            band(
                axes[1],
                ks,
                100 * m,
                100 * s,
                label=label,
                color=CATEGORICAL[0],
                linestyle=ls,
                marker="o",
            )
            m, s = mean_std(rs, lambda r, ks=ks: [overall(r, "pass_at_k")[str(k)] for k in ks])
            band(
                axes[1],
                ks,
                100 * m,
                100 * s,
                label=label.replace("conf", "pass"),
                color=CATEGORICAL[1],
                linestyle=ls,
                marker="s",
            )
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("parallel trials K")
        axes[1].set_ylabel(f"% solved at D = {runs[0]['depth']}")
        axes[1].set_title("Parallel trials scaling")
        axes[1].legend()
        finish(fig, FIGURES / f"F2_tts_{base}")
        print(f"F2 drawn for {base}")


# ---------------------------------------------------------------------------- F3
def fig_noise():
    import matplotlib.pyplot as plt

    groups = groups_in("noise")
    if not groups:
        return print("F3: no results/noise yet")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (name, runs), color in zip(sorted(groups.items()), CATEGORICAL, strict=False):
        sig = [r["sigma"] for r in runs[0]["rows"]]
        m, s = mean_std(runs, lambda r: [row["conf_k"] for row in r["rows"]])
        band(
            ax, sig, 100 * m, 100 * s, label=f"{name} conf@{runs[0]['k']}", color=color, marker="o"
        )
        m, s = mean_std(runs, lambda r: [row["k1"] for row in r["rows"]])
        band(
            ax, sig, 100 * m, 100 * s, label=f"{name} K=1", color=color, linestyle="--", marker="s"
        )
    ax.set_xlabel("test-time noise magnitude sigma")
    ax.set_ylabel("% solved")
    ax.set_title("Noise injection at test time")
    ax.legend(fontsize=7)
    finish(fig, FIGURES / "F3_noise")
    print("F3 drawn")


# ---------------------------------------------------------------------------- F4
def fig_ablations():
    import matplotlib.pyplot as plt

    groups = groups_in("ablations")
    if not groups:
        return print("F4: no results/ablations yet")
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for name, runs in groups.items():
        task = next((t for t in TASKS if name.startswith(t + "_")), None)
        if task is None:
            continue
        by_task[task][name[len(task) + 1 :]] = runs
    for task, variants in by_task.items():
        order = [
            v for v in ["full", "no_async", "no_perturb", "no_replay", "self_attn"] if v in variants
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        for v, color in zip(order, CATEGORICAL, strict=False):
            runs = variants[v]
            m, s = mean_std(runs, lambda r: overall(r, "depth_curve"))
            band(axes[0], runs[0]["steps"], 100 * m, 100 * s, label=v, color=color)
        axes[0].set_xlabel("rollout steps D")
        axes[0].set_ylabel("% test boards solved (K = 1)")
        axes[0].set_title(f"{task}: training ingredients")
        axes[0].legend()
        xs = np.arange(len(order))
        kmax = str(max(int(k) for k in variants[order[0]][0]["conf_at_k"]))
        for i, (v, color) in enumerate(zip(order, CATEGORICAL, strict=False)):
            runs = variants[v]
            m1, s1 = mean_std(runs, lambda r: overall(r, "conf_at_k")["1"])
            mk, sk = mean_std(runs, lambda r, kmax=kmax: overall(r, "conf_at_k")[kmax])
            axes[1].bar(i - 0.2, 100 * m1, 0.4, yerr=100 * s1, color=color, alpha=0.6, capsize=2)
            axes[1].bar(i + 0.2, 100 * mk, 0.4, yerr=100 * sk, color=color, capsize=2)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(order, rotation=20)
        axes[1].set_ylabel("% solved (light: K=1, dark: conf@K)")
        axes[1].set_title("Final accuracy, mean ± std over seeds")
        finish(fig, FIGURES / f"F4_ablations_{task}")
        print(f"F4 drawn for {task}")


# ---------------------------------------------------------------------------- F5
def fig_adaptive():
    import matplotlib.pyplot as plt

    groups = groups_in("adaptive")
    if not groups:
        return print("F5: no results/adaptive yet")
    fig, axes = plt.subplots(
        1, 2 * len(groups), figsize=(4.2 * 2 * len(groups), 3.2), squeeze=False
    )
    axes = axes[0]
    labels = {
        "uniform": "uniform (p = 0.8)",
        "adaptive": "adaptive, given cells at 0.8",
        "adaptive_given": "adaptive, given cells at 0.4",
    }
    for gi, (name, runs) in enumerate(sorted(groups.items())):
        for variant, color in zip(labels, CATEGORICAL[:3], strict=True):
            # seeds may have been run at different depths: cut every curve to the shortest
            rs = [r for r in runs if variant in r["variants"]]
            if not rs:
                continue
            T = min(len(r["variants"][variant]["steps"]) for r in rs)
            steps = rs[0]["variants"][variant]["steps"][:T]
            m, s = mean_std(rs, lambda r, v=variant, T=T: r["variants"][v]["solved_curve"][:T])
            band(axes[2 * gi], steps, 100 * m, 100 * s, label=labels[variant], color=color)
            upd, _ = mean_std(
                rs, lambda r, v=variant, T=T: r["variants"][v]["cell_updates_curve"][:T]
            )
            band(axes[2 * gi + 1], upd, 100 * m, 100 * s, label=labels[variant], color=color)
        dmg = runs[0].get("damage_at")
        if dmg:
            for ax in axes[2 * gi : 2 * gi + 2]:
                ax.axvline(
                    dmg if ax is axes[2 * gi] else np.nan, color="grey", linestyle=":", linewidth=1
                )
        axes[2 * gi].set_xlabel("steps")
        axes[2 * gi].set_ylabel("% mazes solved")
        axes[2 * gi].set_title(f"{name}: vs steps" + (f" (damage at {dmg})" if dmg else ""))
        axes[2 * gi + 1].set_xlabel("cumulative cell updates (per maze)")
        axes[2 * gi + 1].set_title("vs cell updates")
        axes[2 * gi].legend()
    finish(fig, FIGURES / "F5_adaptive_compute")
    print("F5 drawn")


# ---------------------------------------------------------------------------- F6
def fig_pruning():
    import matplotlib.pyplot as plt

    groups = groups_in("pruning")
    if not groups:
        return print("F6: no results/pruning yet")
    for name, runs in groups.items():
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ms = [1] + list(runs[0]["ms"])
        for m_, color in zip(ms, CATEGORICAL, strict=False):
            rows = [[row for row in r["rows"] if row["m"] == m_] for r in runs]
            x = np.array([[row["compute_units"] for row in rr] for rr in rows]).mean(0)
            y = np.array([[row["conf_k"] for row in rr] for rr in rows])
            band(
                ax,
                x,
                100 * y.mean(0),
                100 * y.std(0),
                label="unpruned" if m_ == 1 else f"pruning m={m_}",
                color=color,
                marker="o",
            )
        ax.set_xscale("log")
        ax.set_xlabel("compute per board (cell updates)")
        ax.set_ylabel(f"% solved, conf@{runs[0]['k']}")
        ax.set_title(f"{name}: accuracy vs compute, depths {runs[0]['depths']}")
        ax.legend()
        finish(fig, FIGURES / f"F6_pruning_{name}")
        print(f"F6 drawn for {name}")


# ---------------------------------------------------------------------------- F7
def fig_training():
    import matplotlib.pyplot as plt

    runs_root = Path("runs")
    if not runs_root.exists():
        return print("F7: no runs/ folder")
    for cfg_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        seeds = sorted(cfg_dir.glob("seed*/log.jsonl"))
        if not seeds:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for log, color in zip(seeds, CATEGORICAL, strict=False):
            recs = [json.loads(line) for line in open(log)]
            tr = [r for r in recs if "loss" in r]
            ev = [r["eval"] for r in recs if "eval" in r]
            axes[0].plot(
                [r["step"] for r in tr],
                [r["loss"] for r in tr],
                color=color,
                linewidth=1,
                label=log.parent.name,
            )
            if ev:
                key = next((k for k in ("val_exact",) if k in ev[0]), None)
                if key:
                    axes[1].plot(
                        [e["step"] for e in ev],
                        [100 * e[key] for e in ev],
                        color=color,
                        marker="o",
                        label=log.parent.name,
                    )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("training step")
        axes[0].set_ylabel("training loss")
        axes[0].set_title(cfg_dir.name)
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("% validation boards solved (EMA weights)")
        finish(fig, FIGURES / f"F7_training_{cfg_dir.name}")
        print(f"F7 drawn for {cfg_dir.name}")


# ---------------------------------------------------------------------------- F8
def _load(path: str) -> dict | None:
    f = RESULTS / path
    return json.load(open(f)) if f.exists() else None


def fig_recipe_search():
    """Left: Maze-OOD 201x201 on fresh mazes, one dot per seed, per recipe. Right: Sudoku-OOD
    conf@K on the RRN test per recipe and budget, with the paper's two points and the authors'
    weights (on validation boards, so drawn apart as a dotted line)."""
    import matplotlib.pyplot as plt

    from nca.viz import TEXT, TEXT_2

    mazes = [
        ("paper recipe", CATEGORICAL[0], "confirm/maze_ood_ours_seed{}.json", (3, 4, 5)),
        (
            "public recipe,\nwall padding",
            CATEGORICAL[2],
            "confirm/maze_ood_codepub_wall_seed{}.json",
            (3, 4, 5),
        ),
        (
            "public recipe\non the ring",
            CATEGORICAL[1],
            "confirm/maze_ood_codepub_ring_seed{}.json",
            (1, 2),
        ),
    ]
    maze_vals = []
    for _, _, pattern, seeds in mazes:
        vals = []
        for sd in seeds:
            r = _load(pattern.format(sd))
            if r is not None:
                vals += [100 * row["exact"] for row in r["rows"] if row["size"] == 201]
        maze_vals.append(vals)
    sudoku = [
        (
            "paper recipe, 50K steps",
            CATEGORICAL[0],
            "-",
            [f"tts/sudoku_ood_seed{k}.json" for k in (0, 1, 2)],
        ),
        (
            "paper recipe, 20K steps",
            CATEGORICAL[0],
            "--",
            [f"ab/sudoku_ood_20k_seed{k}.json" for k in (0, 1)],
        ),
        (
            "public recipe, 20K steps",
            CATEGORICAL[1],
            "-",
            [f"ab/sudoku_ood_codepub_seed{k}.json" for k in (0, 1)],
        ),
    ]
    sudoku_runs = [[r for r in map(_load, files) if r is not None] for *_, files in sudoku]
    if not any(maze_vals) and not any(sudoku_runs):
        return print("F8: no results/confirm or results/ab recipe files yet")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for i, ((_, color, _, _), vals) in enumerate(zip(mazes, maze_vals, strict=True)):
        if not vals:
            continue
        x = i + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([i])
        ax.scatter(x, vals, s=36, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        m = float(np.mean(vals))
        ax.hlines(m, i - 0.25, i + 0.25, color=TEXT, linewidth=1.2, zorder=2)
        ax.annotate(f"{m:.1f}", (i + 0.28, m), va="center", fontsize=8, color=TEXT_2)
    ax.axhline(100, color=TEXT_2, linewidth=0.8, linestyle=":")
    ax.annotate("paper, and their weights in our code", (-0.4, 103), fontsize=7, color=TEXT_2)
    ax.set_xticks(range(len(mazes)))
    ax.set_xticklabels(
        [f"{label}\n(seeds {'/'.join(map(str, seeds))})" for label, _, _, seeds in mazes],
        fontsize=7.5,
    )
    ax.set_xlim(-0.5, len(mazes) - 0.3)
    ax.set_ylim(-3, 110)
    ax.set_ylabel("% of 201x201 mazes solved (K=1)")
    ax.set_title("Maze-OOD, 100 fresh mazes, one dot per seed")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for (label, color, ls, _), runs in zip(sudoku, sudoku_runs, strict=True):
        if not runs:
            continue
        ks = sorted(int(k) for k in runs[0]["conf_at_k"])
        m, sd = mean_std(runs, lambda r, ks=ks: [r["conf_at_k"][str(k)] for k in ks])
        band(
            ax,
            ks,
            100 * m,
            100 * sd,
            label=f"{label} ({len(runs)} seeds)",
            color=color,
            linestyle=ls,
            marker="o",
            markersize=3.5,
        )
    pub = _load("public/sudoku_ood_dev.json")
    if pub is not None:
        ks = sorted(int(k) for k in pub["conf_at_k"])
        ax.plot(
            ks,
            [100 * pub["conf_at_k"][str(k)] for k in ks],
            color=TEXT_2,
            linestyle=":",
            marker="o",
            markersize=3,
            label="their weights in our code (validation boards)",
        )
    ax.scatter(
        [1, 2048], [66.1, 98.5], marker="D", s=30, color=TEXT, zorder=4, label="paper (K=1, K=2048)"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 2048])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "2048"])
    ax.minorticks_off()
    ax.set_ylim(0, 102)
    ax.set_xlabel("parallel trials K (conf@K)")
    ax.set_ylabel("% of RRN test boards solved")
    ax.set_title("Sudoku-OOD, D=256")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    finish(fig, FIGURES / "F8_recipe_search")
    print("F8 drawn")


ALL = {
    "maze_scaling": fig_maze_scaling,
    "tts": fig_tts,
    "noise": fig_noise,
    "ablations": fig_ablations,
    "adaptive": fig_adaptive,
    "pruning": fig_pruning,
    "training": fig_training,
    "recipe": fig_recipe_search,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="*", default=None, choices=list(ALL))
    args = ap.parse_args()
    set_style()
    FIGURES.mkdir(exist_ok=True)
    for name, fn in ALL.items():
        if args.only and name not in args.only:
            continue
        fn()


if __name__ == "__main__":
    main()
