# Phase 1 of the replication (Windows twin of run_phase1.sh): Maze-OOD and Sudoku-OOD, 3 seeds
# each, plus every evaluation that uses them. Rerunnable: training resumes from checkpoints and
# evaluations whose JSON exists are skipped.
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1          # everything (~10 h, 8 GB GPU)
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 maze     # Maze-OOD only (~1 h)
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 sudoku   # Sudoku-OOD only
param([string]$Part = "all")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Seeds = 0, 1, 2
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
function Run { param($target, [Parameter(ValueFromRemainingArguments)]$a); if (Test-Path $target) { Write-Host "skip (exists): $target" } else { Py @a } }

if ($Part -in "all", "maze") {
  # runs trained with an older architecture of the preset are archived (never resumed), with their results
  Py scripts/archive_stale_runs.py --config configs/maze_ood.yaml configs/ablations_maze_ood.yaml --results "results/maze_scaling/*.json" "results/adaptive/*.json" "results/noise/maze_ood_*.json" "results/ablations/maze_ood_*.json" "figures/anim/maze37*"
  foreach ($s in $Seeds) { Py scripts/train.py --config configs/maze_ood.yaml --seed $s }
  foreach ($s in $Seeds) {
    $R = "runs/maze_ood/seed$s"
    Run results/maze_scaling/maze_ood_seed$s.json scripts/eval_maze_scaling.py --run $R --n 200 --out results/maze_scaling/maze_ood_seed$s.json
    Run results/maze_scaling/table1_seed$s.json scripts/eval_maze_scaling.py --run $R --table1 --n 100 --max-cells 500000 --out results/maze_scaling/table1_seed$s.json
    Py scripts/eval_adaptive.py --run $R --size 201 --n 20 --depth 20000 --max-cells 500000 --record-every 100 --out results/adaptive/maze201_seed$s.json
    Py scripts/eval_adaptive.py --run $R --size 201 --n 20 --depth 20000 --max-cells 500000 --damage-at 6000 --record-every 100 --out results/adaptive/maze201_damage_seed$s.json
    Run results/noise/maze_ood_seed$s.json scripts/eval_noise.py --run $R --n 500 --depth 300 --k 8 --load size=13 --out results/noise/maze_ood_seed$s.json
  }
  # evaluated on 13x13 mazes at D=300 (the protocol in the docs); finished trainings are resumed, not redone
  Py scripts/run_ablations.py --config configs/ablations_maze_ood.yaml --seeds 0 1 2 --eval-n 500 --eval-depth 300 --load size=13
  Py scripts/animate.py --run runs/maze_ood/seed0 --size 37 --depth 1200 --every 10 --out figures/anim/maze37
}

if ($Part -in "all", "sudoku") {
  foreach ($s in $Seeds) { Py scripts/train.py --config configs/sudoku_ood.yaml --seed $s }
  foreach ($s in $Seeds) {
    $R = "runs/sudoku_ood/seed$s"
    Run results/tts/sudoku_ood_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --out results/tts/sudoku_ood_seed$s.json
    Run results/tts/sudoku_ood_noise_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --noise-sigma 0.15 --out results/tts/sudoku_ood_noise_seed$s.json
    Run results/noise/sudoku_ood_seed$s.json scripts/eval_noise.py --run $R --n 500 --depth 256 --k 16 --max-cells 500000 --out results/noise/sudoku_ood_seed$s.json
  }
  Py scripts/run_ablations.py --config configs/ablations_sudoku_ood.yaml --seeds 0 1 2 --eval-n 500 --eval-depth 256
  Py scripts/animate.py --run runs/sudoku_ood/seed0 --depth 256 --every 2 --out figures/anim/sudoku --board 3
}

Py scripts/aggregate.py
Py scripts/make_figures.py
