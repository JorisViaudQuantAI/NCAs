#!/usr/bin/env python
"""Test-time scaling curves (paper Fig. 4A, 16, 17): accuracy vs rollout depth D and vs the
number of parallel trials K, with confidence-based selection (conf@K) and oracle pass@K.

    python scripts/eval_tts.py --run runs/sudoku_ood/seed0 --n 1000 --depth 256 --k 128 \
        --record-every 8 --group level --out results/tts/sudoku_ood_seed0.json
    python scripts/eval_tts.py ... --noise-sigma 0.15 --out results/tts/sudoku_ood_noise_seed0.json
    python scripts/eval_tts.py --run runs/arc_small/seed0 --depth 64 --k 32 --select majority \
        --out results/tts/arc_small_seed0.json      # ARC: majority vote, pass@1 / pass@2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.cli import add_common, load_boards, load_run, parse_kv  # noqa: E402
from nca.experiments import tts_curves  # noqa: E402
from nca.utils import Timer, load_config, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(ap)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--record-every", type=int, default=8)
    ap.add_argument(
        "--group", default=None, help="meta key for per-difficulty breakdown (level, bin, size)"
    )
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="> 0 enables test-time noise")
    ap.add_argument("--noise-pt", type=float, default=0.1)
    ap.add_argument("--noise-ps", type=float, default=0.2)
    ap.add_argument("--noise-r", type=float, default=0.25)
    ap.add_argument(
        "--engine-config", help="YAML containing an engine section for compiled evaluation"
    )
    ap.add_argument(
        "--select",
        choices=("conf", "majority"),
        default="conf",
        help="candidate selection: board confidence (conf@K, all tasks) or, for ARC, the "
        "paper's majority vote over the trials mapped back to the canonical frame (pass@2)",
    )
    args = ap.parse_args()
    model, cfg, seed, task = load_run(args.run, args.device)
    boards = load_boards(task, cfg, args.split, args.n, parse_kv(args.load))
    noise = None
    if args.noise_sigma > 0:
        noise = dict(r=args.noise_r, p_t=args.noise_pt, p_s=args.noise_ps, sigma=args.noise_sigma)
    with Timer() as t:
        res = tts_curves(
            model,
            task,
            boards,
            args.depth,
            args.k,
            args.record_every,
            noise=noise,
            seed=args.seed if args.seed is not None else seed,
            group_key=args.group,
            max_cells=args.max_cells,
            progress=args.progress,
            amp=args.amp,
            select=args.select,
            engine=None
            if args.engine_config is None
            else load_config(args.engine_config)["engine"],
        )
    res.update(
        run=args.run,
        seed=args.seed if args.seed is not None else seed,
        split=args.split,
        source=boards.meta.get("source"),
        seconds=t.seconds,
        load=parse_kv(args.load),  # the board sampling options (seed, exclude_seed, ...)
    )
    save_json(res, args.out)
    ks = sorted(res["conf_at_k"])
    extra = ""
    if args.select == "majority":
        mv = res["majority_pass_at_k"][ks[-1]]
        extra = f"majority K={ks[-1]}: pass@1 {mv[1]:.3f} pass@2 {mv[2]:.3f}  "
    print(
        f"{task.name}: {len(boards)} boards, D={args.depth}  K=1 {res['conf_at_k'][1]:.3f}  "
        f"conf@{ks[-1]} {res['conf_at_k'][ks[-1]]:.3f}  pass@{ks[-1]} {res['pass_at_k'][ks[-1]]:.3f}  "
        f"{extra}({t.seconds:.0f}s) -> {args.out}"
    )


if __name__ == "__main__":
    main()
