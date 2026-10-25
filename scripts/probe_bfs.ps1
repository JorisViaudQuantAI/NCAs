# Short Maze-Hard BFS probes (seed 0, stopped at step 3000, where A* already solves 27 % of the
# validation mazes). About 40 min each on an 8 GB GPU. See docs/DEVIATIONS.md (2026-10-19).
#   local : BFS targets with the local tie-break (maze.bfs_local_path), augmentation on
#   noaug : our first BFS targets (global tie-break), augmentation off, as for A*
# A probe that learns can be finished as a full run: train.py --config <same> --seed 0
# --out <same folder> resumes from step 3000 up to 10 000.
#   powershell -ExecutionPolicy Bypass -File scripts\probe_bfs.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
Py scripts/train.py --config configs/maze_hard_bfs_local.yaml --seed 0 --set train.stop_at=3000 --out runs/probe_bfs/local_seed0
Py scripts/train.py --config configs/maze_hard_bfs.yaml --seed 0 --set train.augment=false --set train.stop_at=3000 --out runs/probe_bfs/noaug_seed0
