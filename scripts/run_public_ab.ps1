# Our recipe against the authors' public code (DEVIATIONS 2026-10-20). Rerunnable: training resumes
# from checkpoints and evaluations whose JSON exists are skipped. Arm A is our recipe, arm B the
# public-code switches (configs/*_codepub.yaml); same seeds, data and evaluation for both.
#   powershell -ExecutionPolicy Bypass -File scripts\run_public_ab.ps1 public -Repo ..\reasoning-nca  # their weights, our evaluation (~2 h)
#   powershell -ExecutionPolicy Bypass -File scripts\run_public_ab.ps1 maze_ood    # arm B, 3 seeds (arm A = the published runs) (~1.5 h)
#   powershell -ExecutionPolicy Bypass -File scripts\run_public_ab.ps1 maze_hard   # A and B at 20K steps, seed 0 (~9 h)
#   powershell -ExecutionPolicy Bypass -File scripts\run_public_ab.ps1 sudoku      # A and B pilots at 20K steps, seeds 0 and 1 (~5 h)
param([string]$Part = "public", [string]$Repo = "..\reasoning-nca")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
function Run { param($target, [Parameter(ValueFromRemainingArguments)]$a); if (Test-Path $target) { Write-Host "skip (exists): $target" } else { Py @a } }

if ($Part -eq "public") {
  Py scripts/import_public.py --repo $Repo --out runs/public --tasks Maze-OOD Maze-Hard Sudoku-OOD
  Run results/public/maze_ood_table1.json scripts/eval_maze_scaling.py --run runs/public/maze_ood --table1 --n 100 --max-cells 500000 --seed 0 --out results/public/maze_ood_table1.json
  Run results/public/maze_hard_astar.json scripts/eval_tts.py --run runs/public/maze_hard --n 1000 --depth 200 --k 64 --record-every 8 --seed 0 --out results/public/maze_hard_astar.json
  Run results/public/sudoku_ood.json scripts/eval_tts.py --run runs/public/sudoku_ood --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --seed 0 --out results/public/sudoku_ood.json
}

if ($Part -eq "maze_ood") {
  foreach ($s in 0, 1, 2) {
    Py scripts/train.py --config configs/maze_ood_codepub.yaml --seed $s
    $R = "runs/maze_ood_codepub/seed$s"
    Run results/ab/maze_ood_codepub_table1_seed$s.json scripts/eval_maze_scaling.py --run $R --table1 --n 100 --max-cells 500000 --out results/ab/maze_ood_codepub_table1_seed$s.json
  }
}

if ($Part -eq "maze_hard") {
  # arm A: our recipe with the paper's 20K steps (the cosine horizon is 20K from the start)
  Py scripts/train.py --config configs/maze_hard.yaml --seed 0 --set train.steps=20000 --out runs/maze_hard_20k/seed0
  Py scripts/train.py --config configs/maze_hard_codepub.yaml --seed 0
  Run results/ab/maze_hard_20k_seed0.json scripts/eval_tts.py --run runs/maze_hard_20k/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --out results/ab/maze_hard_20k_seed0.json
  Run results/ab/maze_hard_codepub_seed0.json scripts/eval_tts.py --run runs/maze_hard_codepub/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --out results/ab/maze_hard_codepub_seed0.json
}

if ($Part -eq "sudoku") {
  foreach ($s in 0, 1) {
    Py scripts/train_fast.py --config configs/sudoku_ood.yaml --seed $s --set train.steps=20000 --set task.val.split=val --out runs/sudoku_ood_20k/seed$s
    Py scripts/train_fast.py --config configs/sudoku_ood_codepub.yaml --seed $s --out runs/sudoku_ood_codepub/seed$s
    foreach ($arm in "sudoku_ood_20k", "sudoku_ood_codepub") {
      Run results/ab/${arm}_seed$s.json scripts/eval_tts.py --run runs/$arm/seed$s --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --out results/ab/${arm}_seed$s.json
    }
  }
}
