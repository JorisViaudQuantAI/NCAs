#!/usr/bin/env python
"""Aggregate every results/*/*_seed*.json over seeds (mean, std, n) into results/summary.json and
print the headline numbers next to the paper's, as a Markdown table (pasted into docs/RESULTS.md).

    python scripts/aggregate.py            # writes results/summary.json and prints the table
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.utils import save_json  # noqa: E402

# (experiment folder, file prefix, dotted key in the JSON, label, paper value, paper setting)
HEADLINES = [
    ("maze_scaling", "table1", "rows[13].exact", "Maze-OOD 13x13, D=300, exact", 100.0, "100.0"),
    ("maze_scaling", "table1", "rows[59].exact", "Maze-OOD 59x59, D=2000, exact", 100.0, "100.0"),
    (
        "maze_scaling",
        "table1",
        "rows[201].exact",
        "Maze-OOD 201x201, D=13000, exact",
        100.0,
        "100.0",
    ),
    ("tts", "maze_hard_astar", "conf_at_k.1", "Maze-Hard (A*), K=1, exact", 86.6, "86.6"),
    (
        "tts",
        "maze_hard_astar",
        "conf_at_k.max",
        "Maze-Hard (A*), conf@K, exact",
        89.2,
        "89.2 (K=128)",
    ),
    ("tts", "maze_hard_bfs", "conf_at_k.1", "Maze-Hard (BFS), K=1, exact", 94.7, "94.7"),
    (
        "tts",
        "maze_hard_bfs",
        "conf_at_k.max",
        "Maze-Hard (BFS), conf@K, exact",
        96.2,
        "96.2 (K=128)",
    ),
    ("tts", "sudoku_ood", "conf_at_k.1", "Sudoku-OOD, K=1", 66.1, "66.1 (D=256)"),
    ("tts", "sudoku_ood", "conf_at_k.max", "Sudoku-OOD, conf@K", 98.5, "98.5 (K=2048)"),
    # Sudoku-Extreme: the test sample is stratified over the difficulty bins, so the global score
    # is the mean re-weighted by each bin's share of the benchmark (the paper's distribution)
    (
        "tts",
        "sudoku_extreme",
        "conf_at_k_weighted.1|conf_at_k.1@weighted=false",
        "Sudoku-Extreme, K=1 (benchmark-weighted)",
        75.6,
        "75.6 (D=2048)",
    ),
    (
        "tts",
        "sudoku_extreme",
        "conf_at_k_weighted.max|conf_at_k.max@weighted=false",
        "Sudoku-Extreme, conf@K (benchmark-weighted)",
        92.7,
        "92.7 (K=512)",
    ),
    # ARC: proof of concept on 30 training tasks (not the benchmark score); the paper selects by
    # majority vote (pass@2), see docs/PAPER_MAP.md
    (
        "tts",
        "arc",
        "majority_pass_at_k.max.2",
        "ARC (30 tasks, proof of concept), majority pass@2",
        48.8,
        "48.8 (full ARC-AGI-1, K=3264)",
    ),
    ("tts", "arc", "conf_at_k.max", "ARC (30 tasks, proof of concept), conf@K", None, "-"),
]


def dig(d, key: str):
    """Value at a dotted `key`; `a|b` tries `a` first, then `b`. An alternative written
    `b@field=value` is only used when the file's top-level `field` equals `value` (JSON
    literal), e.g. `conf_at_k.1@weighted=false`: the unweighted score is accepted only from a
    file that says its sample is not stratified (old stratified files without the flag are not
    mislabelled as benchmark-weighted)."""
    if "|" in key:
        for alt in key.split("|"):
            v = dig(d, alt)
            if v is not None:
                return v
        return None
    if "@" in key:
        key, cond = key.split("@", 1)
        field, value = cond.split("=", 1)
        if not isinstance(d, dict) or field not in d or d[field] != json.loads(value):
            return None
    for part in key.split("."):
        m = re.fullmatch(r"(\w+)\[(\w+)\]", part)
        if m:  # rows[13] -> the row whose "size" (or "depth"/"m") equals 13
            d = d[m.group(1)]
            d = next(
                (r for r in d if str(r.get("size", r.get("depth", r.get("m")))) == m.group(2)), None
            )
            if d is None:
                return None
            continue
        if isinstance(d, dict):
            if part == "max" and d:
                part = str(max(int(k) for k in d))
            if part not in d:
                return None
            d = d[part]
        else:
            return None
    return d


def main() -> None:
    root = Path("results")
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in sorted(root.glob("*/*_seed*.json")):
        m = re.fullmatch(r"(.+)_seed(\d+)\.json", p.name)
        if m:
            groups[(p.parent.name, m.group(1))].append(p)
    summary = {}
    for (exp, prefix), paths in groups.items():
        runs = [json.load(open(p)) for p in paths]
        summary[f"{exp}/{prefix}"] = {
            "n_seeds": len(runs),
            "files": [str(p) for p in paths],
            "scalars": _aggregate_scalars(runs),
        }
    save_json(summary, root / "summary.json")
    lines = ["| Experiment | Paper | Ours (mean ± std over seeds) | n |", "|---|---|---|---|"]
    for exp, prefix, key, label, _paper, paper_txt in HEADLINES:
        g = groups.get((exp, prefix))
        if not g:
            lines.append(f"| {label} | {paper_txt} | not run yet | 0 |")
            continue
        vals = [dig(json.load(open(p)), key) for p in g]
        vals = [100 * v for v in vals if v is not None]
        if not vals:
            lines.append(f"| {label} | {paper_txt} | key {key} missing | {len(g)} |")
            continue
        lines.append(
            f"| {label} | {paper_txt} | {np.mean(vals):.1f} ± {np.std(vals):.1f} | {len(vals)} |"
        )
    lines += _adaptive_lines(groups)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(groups)} experiment groups -> results/summary.json, results/summary.md\n")
    print("\n".join(lines))


PAPER_ADAPTIVE = (("steps_to_95", "1.29x"), ("updates_to_95", "0.71x"))
# Fig. 8C quotes the "total repair-and-solve time" (steps and updates since step 0, with the
# crossing searched after the damage); the elapsed time since the event is reported as well
PAPER_ADAPTIVE_DAMAGE = (
    ("steps_to_95", "0.94x"),
    ("updates_to_95", "0.52x"),
    ("steps_since_damage_to_95", "- (0.94x if read as elapsed)"),
    ("updates_since_damage_to_95", "- (0.52x if read as elapsed)"),
)


def _adaptive_lines(groups: dict) -> list[str]:
    """Paper Fig. 8 ratios (adaptive / uniform, steps and cell updates to 95%) per group."""
    lines = []
    for (exp, prefix), paths in sorted(groups.items()):
        if exp != "adaptive":
            continue
        runs = [json.load(open(p)) for p in paths]
        for variant in ("adaptive", "adaptive_given"):
            have = [r for r in runs if variant in r.get("variants", {})]
            # paper Fig. 8: 1.29x steps / 0.71x updates without damage, 0.94x / 0.52x when
            # recovering from damage (Fig. 8C)
            ref = PAPER_ADAPTIVE_DAMAGE if "damage" in prefix else PAPER_ADAPTIVE
            for key, paper in ref:
                vals = [
                    r["ratios"][variant][key]
                    for r in have
                    if r.get("ratios", {}).get(variant, {}).get(key) is not None
                ]
                label = f"{prefix} {variant}: {key.replace('_', ' ')} vs uniform"
                if not have:
                    lines.append(f"| {label} | {paper} | not run yet | 0 |")
                elif not vals:
                    lines.append(f"| {label} | {paper} | 95% not reached | {len(have)} |")
                else:
                    lines.append(
                        f"| {label} | {paper} | {np.mean(vals):.2f}x ± {np.std(vals):.2f} "
                        f"| {len(vals)} |"
                    )
    return lines


def _aggregate_scalars(runs: list[dict]) -> dict:
    out = {}
    keys = set()
    for r in runs:
        keys |= {
            k
            for k, v in _flatten(r).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    for k in sorted(keys):
        vals = [v for v in (_flatten(r).get(k) for r in runs) if isinstance(v, (int, float))]
        if len(vals) == len(runs):
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
