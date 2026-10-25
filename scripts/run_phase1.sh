#!/usr/bin/env bash
# Phase 1 of the replication (the "cheminement"): Maze-OOD and Sudoku-OOD, 3 seeds each, plus
# every evaluation that uses them. Rerunnable: finished steps are skipped (training resumes from
# checkpoints; evaluations are skipped when their JSON exists). Windows: run_phase1.ps1.
#
#   bash scripts/run_phase1.sh            # everything (about 10 h on an 8 GB GPU)
#   bash scripts/run_phase1.sh maze       # only the Maze-OOD part (about 1 h)
#   bash scripts/run_phase1.sh sudoku     # only the Sudoku-OOD part
set -euo pipefail
cd "$(dirname "$0")/.."
PART="${1:-all}"
SEEDS="0 1 2"
run() { if [ -f "$1" ]; then echo "skip (exists): $1"; else shift; "$@"; fi; }

if [ "$PART" = all ] || [ "$PART" = maze ]; then
  # runs trained with an older architecture of the preset are archived (never resumed), with their results
  python scripts/archive_stale_runs.py --config configs/maze_ood.yaml configs/ablations_maze_ood.yaml \
    --results "results/maze_scaling/*.json" "results/adaptive/*.json" "results/noise/maze_ood_*.json" "results/ablations/maze_ood_*.json" "figures/anim/maze37*"
  for s in $SEEDS; do python scripts/train.py --config configs/maze_ood.yaml --seed $s; done
  for s in $SEEDS; do
    R=runs/maze_ood/seed$s
    run results/maze_scaling/maze_ood_seed$s.json python scripts/eval_maze_scaling.py --run $R --n 200 --out results/maze_scaling/maze_ood_seed$s.json
    run results/maze_scaling/table1_seed$s.json   python scripts/eval_maze_scaling.py --run $R --table1 --n 100 --max-cells 500000 --out results/maze_scaling/table1_seed$s.json
    python scripts/eval_adaptive.py --run $R --size 201 --n 20 --depth 20000 --max-cells 500000 --record-every 100 --out results/adaptive/maze201_seed$s.json
    python scripts/eval_adaptive.py --run $R --size 201 --n 20 --depth 20000 --max-cells 500000 --damage-at 6000 --record-every 100 --out results/adaptive/maze201_damage_seed$s.json
    run results/noise/maze_ood_seed$s.json        python scripts/eval_noise.py --run $R --n 500 --depth 300 --k 8 --load size=13 --out results/noise/maze_ood_seed$s.json
  done
  # evaluated on 13x13 mazes at D=300 (the protocol in the docs); finished trainings are resumed, not redone
  python scripts/run_ablations.py --config configs/ablations_maze_ood.yaml --seeds $SEEDS --eval-n 500 --eval-depth 300 --load size=13
  python scripts/animate.py --run runs/maze_ood/seed0 --size 37 --depth 1200 --every 10 --out figures/anim/maze37
fi

if [ "$PART" = all ] || [ "$PART" = sudoku ]; then
  for s in $SEEDS; do python scripts/train.py --config configs/sudoku_ood.yaml --seed $s; done
  for s in $SEEDS; do
    R=runs/sudoku_ood/seed$s
    run results/tts/sudoku_ood_seed$s.json       python scripts/eval_tts.py --run $R --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --out results/tts/sudoku_ood_seed$s.json
    run results/tts/sudoku_ood_noise_seed$s.json python scripts/eval_tts.py --run $R --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --noise-sigma 0.15 --out results/tts/sudoku_ood_noise_seed$s.json
    run results/noise/sudoku_ood_seed$s.json     python scripts/eval_noise.py --run $R --n 500 --depth 256 --k 16 --max-cells 500000 --out results/noise/sudoku_ood_seed$s.json
  done
  python scripts/run_ablations.py --config configs/ablations_sudoku_ood.yaml --seeds $SEEDS --eval-n 500 --eval-depth 256
  python scripts/animate.py --run runs/sudoku_ood/seed0 --depth 256 --every 2 --out figures/anim/sudoku --board 3
fi

python scripts/aggregate.py
python scripts/make_figures.py
