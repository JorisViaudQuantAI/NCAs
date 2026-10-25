# Phase 3 (Windows twin of run_phase3.sh): ARC-AGI-1 proof of concept, 3 seeds. About 6 h on an 8 GB GPU.
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase3.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Seeds = 0, 1, 2
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
function Run { param($target, [Parameter(ValueFromRemainingArguments)]$a); if (Test-Path $target) { Write-Host "skip (exists): $target" } else { Py @a } }

foreach ($s in $Seeds) { Py scripts/train.py --config configs/arc_small.yaml --seed $s }
foreach ($s in $Seeds) {
  Run results/tts/arc_seed$s.json scripts/eval_tts.py --run runs/arc_small/seed$s --depth 64 --k 32 --record-every 4 --select majority --max-cells 100000 --out results/tts/arc_seed$s.json
}
Py scripts/aggregate.py
Py scripts/make_figures.py
