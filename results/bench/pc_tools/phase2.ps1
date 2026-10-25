param([string]$Part = "maze")   # maze (Maze-OOD then Maze-Hard A*) | sudoku
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..\..")
$env:TORCHINDUCTOR_CACHE_DIR = "C:\tic"
$env:PYTHONUNBUFFERED = "1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$d = "data_root=data"
$Seeds = 0, 1, 2
New-Item -ItemType Directory -Force results\bench\pc_logs | Out-Null
$driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim()
$vers = & $py -c "import torch,triton,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),triton=triton.__version__)))"

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

# one python call, retried once on failure; returns $true on success
function Invoke-Py([string]$log, [string[]]$a) {
  for ($try = 1; $try -le 2; $try++) {
    Wait-Gpu
    "$(Get-Date -Format s) RUN (try $try): python $($a -join ' ')"
    & $py @a *>> $log
    if ($LASTEXITCODE -eq 0) { return $true }
    "$(Get-Date -Format s) FAILED exit=$LASTEXITCODE (try $try), log: $log"
  }
  Get-Content $log -Tail 40
  return $false
}

function Train([string]$task, [string]$cfg, [string[]]$sets, [int]$s, [double]$estH) {
  $out = "runs/engine/${task}_opt/seed$s"
  $log = "results/bench/pc_logs/train_${task}_opt_seed$s.log"
  if (Test-Path "$out/summary.json") { "skip (done): $out"; return $true }
  $a = @("scripts/train_fast.py", "--config", $cfg, "--seed", "$s", "--out", $out, "--disable-tf32", "--set", $d)
  foreach ($x in $sets) { $a += @("--set", $x) }
  $t0 = Get-Date
  $ok = $false
  for ($try = 1; $try -le 2 -and -not $ok; $try++) {
    Wait-Gpu
    "$(Get-Date -Format s) TRAIN (try $try): python $($a -join ' ')"
    $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
    $rateDone = ($s -ne 0) -or ($try -gt 1)
    $checked = $false
    while (-not $p.HasExited) {
      Start-Sleep 20
      if (-not $checked -and (Test-Path "$out/run.json")) {
        $checked = $true
        $mc = (Get-Content "$out/run.json" -Raw | ConvertFrom-Json).model_config
        "$(Get-Date -Format s) RUNJSON $task seed$s norm_fired_only=$($mc.norm_fired_only)"
        if ($task -eq "sudoku_ood" -and $mc.norm_fired_only -ne $false) {
          "$(Get-Date -Format s) ABORT: norm_fired_only is not false"; Stop-Process -Id $p.Id; return $false
        }
      }
      if (-not $rateDone -and (Test-Path "$out/log.jsonl")) {
        $r = Get-Content "$out/log.jsonl" | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.step -ge 200 } | Select-Object -First 1
        if ($r) {
          $rateDone = $true
          $steps = ($r.step / ($r.elapsed / [Math]::Max(1, $r.step)))
          $projH = 3 * $r.elapsed / $r.step * $(if ($task -eq "maze_ood") { 5000 } elseif ($task -eq "maze_hard") { 10000 } else { 50000 }) / 3600
          "$(Get-Date -Format s) RATE $task seed0 step$($r.step): $([Math]::Round($r.elapsed / $r.step, 3)) s/step (elapsed avg), last window $([Math]::Round($r.iter_seconds, 3)) s/it, 3 seeds training ~ $([Math]::Round($projH, 2)) h vs estimate $estH h" + $(if ($projH -gt 1.5 * $estH) { "  OVER 1.5x" } else { "" })
        }
      }
    }
    $p.WaitForExit()
    Get-Content "$log.out", "$log.err" -ErrorAction SilentlyContinue | Add-Content $log
    $ok = Test-Path "$out/summary.json"
    if (-not $ok) { "$(Get-Date -Format s) TRAIN FAILED (try $try) exit=$($p.ExitCode)"; Get-Content "$log.err" -Tail 40 }
  }
  $wall = ((Get-Date) - $t0).TotalSeconds
  if ($ok) {
    $sum = Get-Content "$out/summary.json" -Raw | ConvertFrom-Json
    @{ task = $task; seed = $s; config = $cfg; overrides = $sets + @($d); disable_tf32 = $true; wall_seconds_incl_compile = $wall;
       driver = $driver; versions = ($vers | ConvertFrom-Json); gpu = (nvidia-smi --query-gpu=name --format=csv,noheader).Trim();
       command = "python " + ($a -join " ") } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$out/pc_provenance.json"
    "$(Get-Date -Format s) TRAINED $task seed$s wall $([Math]::Round($wall/60,1)) min, train_seconds $([Math]::Round($sum.train_seconds/60,1)) min, peak $([Math]::Round($sum.peak_mem_gb,2)) GiB, final_eval $($sum.final_eval | ConvertTo-Json -Compress)"
  }
  return $ok
}

if ($Part -eq "maze") {
  # ---- Maze-OOD: budget_optimized preset (5000 steps, bf16), evals of the published phase 1
  $q = "results/bench/quality/maze_ood_opt"; New-Item -ItemType Directory -Force $q | Out-Null
  foreach ($s in $Seeds) { Train "maze_ood" "configs/budget_optimized/maze_ood.yaml" @() $s 0.25 }
  foreach ($s in $Seeds) {
    $R = "runs/engine/maze_ood_opt/seed$s"; $L = "results/bench/pc_logs/eval_maze_ood_opt_seed$s.log"
    if (-not (Test-Path "$R/final.pt")) { continue }
    if (-not (Test-Path "$q/maze_scaling_seed$s.json")) { Invoke-Py $L @("scripts/eval_maze_scaling.py", "--run", $R, "--n", "200", "--out", "$q/maze_scaling_seed$s.json") }
    if (-not (Test-Path "$q/table1_seed$s.json")) { Invoke-Py $L @("scripts/eval_maze_scaling.py", "--run", $R, "--table1", "--n", "100", "--max-cells", "500000", "--out", "$q/table1_seed$s.json") }
    if (-not (Test-Path "$q/noise_seed$s.json")) { Invoke-Py $L @("scripts/eval_noise.py", "--run", $R, "--n", "500", "--depth", "300", "--k", "8", "--load", "size=13", "--out", "$q/noise_seed$s.json") }
    Get-Content $L | Select-String -Pattern "->|table|acc" | Select-Object -Last 12
  }
  "MAZE_OOD DONE"
  # ---- Maze-Hard A*: budget_optimized preset at the published 10000 steps, fp32
  $q = "results/bench/quality/maze_hard_opt"; New-Item -ItemType Directory -Force $q | Out-Null
  foreach ($s in $Seeds) { Train "maze_hard" "configs/budget_optimized/maze_hard.yaml" @("train.steps=10000") $s 4.0 }
  foreach ($s in $Seeds) {
    $R = "runs/engine/maze_hard_opt/seed$s"; $L = "results/bench/pc_logs/eval_maze_hard_opt_seed$s.log"
    if (-not (Test-Path "$R/final.pt")) { continue }
    if (-not (Test-Path "$q/maze_hard_astar_seed$s.json")) { Invoke-Py $L @("scripts/eval_tts.py", "--run", $R, "--n", "1000", "--depth", "200", "--k", "64", "--record-every", "8", "--out", "$q/maze_hard_astar_seed$s.json") }
    if (-not (Test-Path "$q/maze_hard_astar_noise_seed$s.json")) { Invoke-Py $L @("scripts/eval_tts.py", "--run", $R, "--n", "1000", "--depth", "200", "--k", "64", "--record-every", "8", "--noise-sigma", "0.15", "--noise-ps", "0.3", "--out", "$q/maze_hard_astar_noise_seed$s.json") }
    Get-Content $L | Select-String -Pattern "K=1" | Select-Object -Last 4
  }
  "MAZE_HARD DONE"
}

if ($Part -eq "sudoku") {
  $q = "results/bench/quality/sudoku_ood_opt"; New-Item -ItemType Directory -Force $q | Out-Null
  foreach ($s in $Seeds) { Train "sudoku_ood" "configs/budget_optimized/sudoku_ood.yaml" @("train.steps=50000", "model.overrides.norm_fired_only=false", "engine.normalized_backward=false") $s 9.0 }
  foreach ($s in $Seeds) {
    $R = "runs/engine/sudoku_ood_opt/seed$s"; $L = "results/bench/pc_logs/eval_sudoku_ood_opt_seed$s.log"
    if (-not (Test-Path "$R/final.pt")) { continue }
    if (-not (Test-Path "$q/sudoku_ood_seed$s.json")) { Invoke-Py $L @("scripts/eval_tts.py", "--run", $R, "--n", "1000", "--depth", "256", "--k", "128", "--record-every", "8", "--group", "level", "--max-cells", "500000", "--out", "$q/sudoku_ood_seed$s.json") }
    Get-Content $L | Select-String -Pattern "K=1" | Select-Object -Last 2
  }
  "SUDOKU DONE"
}



