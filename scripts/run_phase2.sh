#!/usr/bin/env bash
# Phase 2: Maze-Hard (A* and BFS targets) and Sudoku-Extreme (with pruning), 3 seeds each.
# Sudoku-Extreme evaluations (C=256, K=64) are chunked with --max-cells 200000 (38 boards per pass):
# the default 2M cells would need about 10 GB of state tensors on top of the model (DEVIATIONS 2026-10-12).
#   bash scripts/run_phase2.sh [all|maze_hard|sudoku_extreme]      (about 25 h on an 8 GB GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
PART="${1:-all}"
SEEDS="0 1 2"
run() { if [ -f "$1" ]; then echo "skip (exists): $1"; else shift; "$@"; fi; }

if [ "$PART" = all ] || [ "$PART" = maze_hard ]; then
  for cfg in maze_hard maze_hard_bfs_dlru; do  # BFS: down/left/right/up expansion (DEVIATIONS 2026-10-19)
    for s in $SEEDS; do python scripts/train.py --config configs/$cfg.yaml --seed $s; done
    for s in $SEEDS; do
      R=runs/$cfg/seed$s; name=$([ $cfg = maze_hard ] && echo maze_hard_astar || echo maze_hard_bfs)
      run results/tts/${name}_seed$s.json       python scripts/eval_tts.py --run $R --n 1000 --depth 200 --k 64 --record-every 8 --out results/tts/${name}_seed$s.json
      run results/tts/${name}_noise_seed$s.json python scripts/eval_tts.py --run $R --n 1000 --depth 200 --k 64 --record-every 8 --noise-sigma 0.15 --noise-ps 0.3 --out results/tts/${name}_noise_seed$s.json
    done
  done
  python scripts/animate.py --run runs/maze_hard/seed0 --depth 200 --every 2 --out figures/anim/maze_hard
fi

if [ "$PART" = all ] || [ "$PART" = sudoku_extreme ]; then
  for s in $SEEDS; do python scripts/train.py --config configs/sudoku_extreme.yaml --seed $s; done
  for s in $SEEDS; do
    R=runs/sudoku_extreme/seed$s
    run results/tts/sudoku_extreme_seed$s.json       python scripts/eval_tts.py --run $R --n 1000 --depth 512 --k 64 --record-every 16 --group bin --max-cells 200000 --out results/tts/sudoku_extreme_seed$s.json
    run results/tts/sudoku_extreme_noise_seed$s.json python scripts/eval_tts.py --run $R --n 1000 --depth 512 --k 64 --record-every 16 --group bin --noise-sigma 0.15 --max-cells 200000 --out results/tts/sudoku_extreme_noise_seed$s.json
    run results/pruning/sudoku_extreme_seed$s.json   python scripts/eval_pruning.py --run $R --n 500 --k 64 --depths 64 128 256 512 --ms 2 4 8 --max-cells 200000 --out results/pruning/sudoku_extreme_seed$s.json
  done
  python scripts/animate.py --run runs/sudoku_extreme/seed0 --depth 512 --every 4 --out figures/anim/sudoku_extreme --board 5
fi

python scripts/aggregate.py
python scripts/make_figures.py
