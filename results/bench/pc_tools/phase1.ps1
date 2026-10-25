$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..\..")
$env:TORCHINDUCTOR_CACHE_DIR = "C:\tic"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$w = "results\bench\pc_tools\bench_wrap.py"
$d = "data_root=data"
New-Item -ItemType Directory -Force results\bench\pc_logs | Out-Null

function Wait-Gpu {
  while ($true) {
    # every Python process but this script's own children: a training or benchmark from any
    # checkout, NCAs-fast included, must finish before the GPU is used again
    $others = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.ParentProcessId -ne $PID }
    if (-not $others) { return }
    "$(Get-Date -Format s) WAIT: other python on the machine: $($others.CommandLine -join ' | ')"
    Start-Sleep 300
  }
}

$jobs = @(
  @{n="sudoku_ood";     a=@("--config","configs/sudoku_ood.yaml","--variants","reference,fused,refcompiled,compiled4,preset","--preset-config","configs/l4_round3/sudoku_ood.yaml")},
  @{n="maze_ood";       a=@("--config","configs/maze_ood.yaml","--variants","reference,fused,compiled4,graphs4,graphs16")},
  @{n="maze_hard";      a=@("--config","configs/maze_hard.yaml","--variants","reference,fused,refcompiled,compiled4,preset","--preset-config","configs/budget_optimized/maze_hard.yaml","--segments","0,16")},
  @{n="sudoku_extreme"; a=@("--config","configs/sudoku_extreme.yaml","--variants","reference,fused,compiled4")}
)
foreach ($j in $jobs) {
  Wait-Gpu
  $t = Get-Date
  "$(Get-Date -Format s) START $($j.n)"
  # retried once on failure (a transient compiler or GPU error); a second failure stops the phase
  for ($try = 1; $try -le 2; $try++) {
    if ($try -gt 1) { Wait-Gpu }
    & $py $w @($j.a) --iters 10 --warmup 3 --seed 0 --set $d --out "results/bench/pc_$($j.n).json" *> "results/bench/pc_logs/pc_$($j.n).log"
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) END $($j.n) try=$try exit=$code ($([int]((Get-Date)-$t).TotalMinutes) min)"
    if ($code -eq 0) { break }
    "$(Get-Date -Format s) FAILED $($j.n) (try $try), log: results/bench/pc_logs/pc_$($j.n).log"
  }
  if ($code -ne 0) { "PHASE1 STOPPED at $($j.n)"; exit 1 }
  Select-String -Path "results/bench/pc_logs/pc_$($j.n).log" -Pattern "summary:" -Context 0,40 | ForEach-Object { $_.Context.PostContext } | Where-Object { $_ -match "ms/iter|FAILED" }
}
"PHASE1 DONE"
