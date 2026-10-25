#!/usr/bin/env bash
# Phase 3: ARC-AGI-1 proof of concept (one NCA on 30 training tasks with RE-ARC extras), 3 seeds.
#   bash scripts/run_phase3.sh      (about 6 h on an 8 GB GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="0 1 2"
run() { if [ -f "$1" ]; then echo "skip (exists): $1"; else shift; "$@"; fi; }
for s in $SEEDS; do python scripts/train.py --config configs/arc_small.yaml --seed $s; done
for s in $SEEDS; do
  run results/tts/arc_seed$s.json python scripts/eval_tts.py --run runs/arc_small/seed$s --depth 64 --k 32 --record-every 4 --select majority --max-cells 100000 --out results/tts/arc_seed$s.json
done
python scripts/aggregate.py
python scripts/make_figures.py
