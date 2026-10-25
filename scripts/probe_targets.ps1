# Maze-Hard target checks (DEVIATIONS 2026-10-19): which rule produced the HRM paths (CPU, about
# two minutes), then a seed-0 probe on our own A* paths stopped at step 3000 (about 40 min on an
# 8 GB GPU; the HRM paths reach 27 % of the validation mazes there, our BFS paths 0 %).
#   powershell -ExecutionPolicy Bypass -File scripts\probe_targets.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
Py scripts/compare_hard_targets.py --split train | Tee-Object -FilePath results/compare_hard_targets.txt
Py scripts/train.py --config configs/maze_hard_astar_own.yaml --seed 0 --set train.stop_at=3000 --out runs/probe_bfs/astar_own_seed0
