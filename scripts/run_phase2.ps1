# Phase 2 (Windows twin of run_phase2.sh): Maze-Hard (A* and BFS targets) and Sudoku-Extreme
# (with pruning), 3 seeds each. About 25 h on an 8 GB GPU.
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase2.ps1 [all|maze_hard|sudoku_extreme]
param([string]$Part = "all")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Seeds = 0, 1, 2
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
function Run { param($target, [Parameter(ValueFromRemainingArguments)]$a); if (Test-Path $target) { Write-Host "skip (exists): $target" } else { Py @a } }

if ($Part -in "all", "maze_hard") {
  foreach ($cfg in "maze_hard", "maze_hard_bfs_dlru") {  # BFS: down/left/right/up expansion (DEVIATIONS 2026-10-19)
    foreach ($s in $Seeds) { Py scripts/train.py --config configs/$cfg.yaml --seed $s }
    $name = if ($cfg -eq "maze_hard") { "maze_hard_astar" } else { "maze_hard_bfs" }
    foreach ($s in $Seeds) {
      $R = "runs/$cfg/seed$s"
      Run results/tts/${name}_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 200 --k 64 --record-every 8 --out results/tts/${name}_seed$s.json
      Run results/tts/${name}_noise_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 200 --k 64 --record-every 8 --noise-sigma 0.15 --noise-ps 0.3 --out results/tts/${name}_noise_seed$s.json
    }
  }
  Py scripts/animate.py --run runs/maze_hard/seed0 --depth 200 --every 2 --out figures/anim/maze_hard
}

if ($Part -in "all", "sudoku_extreme") {
  foreach ($s in $Seeds) { Py scripts/train.py --config configs/sudoku_extreme.yaml --seed $s }
  foreach ($s in $Seeds) {
    $R = "runs/sudoku_extreme/seed$s"
    Run results/tts/sudoku_extreme_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 512 --k 64 --record-every 16 --group bin --max-cells 200000 --out results/tts/sudoku_extreme_seed$s.json
    Run results/tts/sudoku_extreme_noise_seed$s.json scripts/eval_tts.py --run $R --n 1000 --depth 512 --k 64 --record-every 16 --group bin --noise-sigma 0.15 --max-cells 200000 --out results/tts/sudoku_extreme_noise_seed$s.json
    Run results/pruning/sudoku_extreme_seed$s.json scripts/eval_pruning.py --run $R --n 500 --k 64 --depths 64 128 256 512 --ms 2 4 8 --max-cells 200000 --out results/pruning/sudoku_extreme_seed$s.json
  }
  Py scripts/animate.py --run runs/sudoku_extreme/seed0 --depth 512 --every 4 --out figures/anim/sudoku_extreme --board 5
}

Py scripts/aggregate.py
Py scripts/make_figures.py
