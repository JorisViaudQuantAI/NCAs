# Recipe search (2026-10-21): which ingredients of the paper, the authors' public code and our own
# implementation give the best results at our budget. Short probes, same data, trainer, precision
# and evaluation for every arm; the arms are compared on held-out boards, never on the test sets,
# and the winner is then confirmed with 3 seeds and evaluated once on the test set. Rerunnable:
# finished trainings and evaluations are skipped.
#   powershell -ExecutionPolicy Bypass -File scripts\run_recipe_search.ps1 [all|sudoku|maze_ood|maze_hard|confirm]
#
# Sudoku-OOD: the public-code recipe (configs/sudoku_ood_codepub.yaml) beat ours by ~20 points at
#   20K steps. Leave-one-out from it, 10K steps, seed 0: each arm turns one ingredient back to our
#   recipe; compared on 500 boards of RRN's validation file (--split dev), D=256, K up to 16.
#   One seed only: effects under ~10 points (K=1) are not interpreted; the arms that move are
#   re-run on a second seed before any decision. Short probes measure how fast an ingredient
#   lets the model take off, not the final accuracy.
# Maze-OOD: ours beat the public-code recipe on 201x201 (85.7 vs 20.0%). Leave-one-in on ours,
#   seeds 1 and 2: each arm adds one ingredient of the public code; Table 1 sizes on 100 mazes
#   per size drawn with seed 7 (the published Table 1 sample uses seed 0; 10/100 shared at 201).
# Maze-HARD: an A/B on the test set, not a selection: our recipe at the paper's 20K steps against
#   the public-code recipe with a BPTT window shared by the batch.
# Confirmation (2026-10-23/24, part "confirm"): the Maze-OOD arms on seeds the search never saw,
#   evaluated on 100 fresh official mazes per size (drawn with seed 11 after removing the seed-0
#   Table 1 sample and the seed-7 selection sample), and the Sudoku betas on a second seed at 20K.
#   The PC ran these by hand; the commands below rebuild them from the run folders and result
#   files (docs/RESULTS.md, "Recipe search"). The verdict: configs/maze_ood_codepub_ring.yaml and
#   configs/sudoku_ood_codepub.yaml are the default recipes.
param([string]$Part = "all")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Py { param([Parameter(ValueFromRemainingArguments)]$a); Write-Host "python $a"; & python @a; if ($LASTEXITCODE) { throw "failed: python $a" } }
function Run { param($target, [Parameter(ValueFromRemainingArguments)]$a); if (Test-Path $target) { Write-Host "skip (exists): $target" } else { Py @a } }

if ($Part -in "all", "maze_ood") {
  $arms = [ordered]@{
    "wrap"      = @("model.overrides.padding=wrap")
    "semantic"  = @("model.overrides.embedding=semantic", "model.overrides.output_tokens=[0,1,2,3]")
    "init"      = @("model.overrides.init_style=public")
    # schedule stays constant (ours already is); the public code adds a 2% warmup, weight decay,
    # its AdamW betas and the mean MSE
    "optimizer" = @("train.warmup_ratio=0.02", "train.init_lr_ratio=1.0", "train.weight_decay=1.0e-4", "train.betas=[0.9,0.95]", "train.loss_reduce=mean")
    "ema"       = @("train.ema_warmup=false")
    "perturb"   = @("train.damage={p: 0.1, n: 4, r_min: 0.1, r_max: 0.4, fill: 0.0, geometry: official}", "train.swap_source=pool", "train.batch_noise={p: 0.1, ps: 0.2, sigma: 0.15}")
  }
  $dev = @("--table1", "--n", "100", "--max-cells", "500000", "--load", "seed=7")
  foreach ($s in 1, 2) {
    Run results/search/maze_ood_ours_seed$s.json scripts/eval_maze_scaling.py --run runs/maze_ood/seed$s @dev --out results/search/maze_ood_ours_seed$s.json
    Run results/search/maze_ood_codepub_seed$s.json scripts/eval_maze_scaling.py --run runs/maze_ood_codepub/seed$s @dev --out results/search/maze_ood_codepub_seed$s.json
  }
  foreach ($name in $arms.Keys) {
    foreach ($s in 1, 2) {
      $R = "runs/search/maze_ood_$name/seed$s"
      $sets = @()
      foreach ($x in $arms[$name]) { $sets += @("--set", $x) }
      Py scripts/train.py --config configs/maze_ood.yaml --seed $s --out $R @sets
      Run results/search/maze_ood_${name}_seed$s.json scripts/eval_maze_scaling.py --run $R @dev --out results/search/maze_ood_${name}_seed$s.json
    }
  }
}

if ($Part -in "all", "sudoku") {
  # anchor: our recipe at the same 10K steps (validation on SAT-Net val, not the RRN test)
  $R = "runs/search/sudoku_ours/seed0"
  Py scripts/train_fast.py --config configs/sudoku_ood.yaml --seed 0 --out $R --set train.steps=10000 --set task.val.split=val
  Run results/search/sudoku_ours.json scripts/eval_tts.py --run $R --split dev --n 500 --depth 256 --k 16 --record-every 32 --group level --max-cells 500000 --out results/search/sudoku_ours.json
  $arms = [ordered]@{
    "full"          = @()
    "no_projection" = @("model.overrides.project_update=false")
    "our_lr"        = @("train.cosine_floor=0.0", "train.init_lr_ratio=null")
    "our_betas"     = @("train.betas=[0.9,0.999]")
    "our_loss_sum"  = @("train.loss_reduce=sum")
    "our_init"      = @("model.overrides.init_style=torch")
    "our_readout"   = @("model.overrides.output_tokens=[1,2,3,4,5,6,7,8,9]")
    "our_ema"       = @("train.ema_warmup=true")
    # the damage fill goes with the normalisation (cells reset to 1 for a normalised model, to 0
    # otherwise), so it moves with the norm arm and stays at 1 in the perturbation arm
    "our_norm"      = @("model.overrides.noise_after_norm=false", "model.overrides.norm_init=false", "model.overrides.norm_style=sqrt_eps", "train.damage={p: 0.1, n: 4, r_min: 0.1, r_max: 0.4, fill: 0.0, geometry: official}")
    "our_perturb"   = @("train.damage={p: 0.1, n: 1, r_min: 0.1, r_max: 0.4, fill: 1.0, geometry: ours}", "train.swap_source=fresh", "train.batch_noise=null")
  }
  foreach ($name in $arms.Keys) {
    $R = "runs/search/sudoku_$name/seed0"
    $sets = @("--set", "train.steps=10000")
    foreach ($x in $arms[$name]) { $sets += @("--set", $x) }
    Py scripts/train_fast.py --config configs/sudoku_ood_codepub.yaml --seed 0 --out $R @sets
    Run results/search/sudoku_$name.json scripts/eval_tts.py --run $R --split dev --n 500 --depth 256 --k 16 --record-every 32 --group level --max-cells 500000 --out results/search/sudoku_$name.json
  }
}

if ($Part -in "all", "maze_hard") {
  Run results/ab/maze_hard_20k_seed0.json scripts/eval_tts.py --run runs/maze_hard_20k/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --out results/ab/maze_hard_20k_seed0.json
  Py scripts/train.py --config configs/maze_hard_codepub.yaml --seed 0 --set train.chunk_start=shared --out runs/maze_hard_codepub_shared/seed0
  Run results/ab/maze_hard_codepub_shared_seed0.json scripts/eval_tts.py --run runs/maze_hard_codepub_shared/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --out results/ab/maze_hard_codepub_shared_seed0.json
}

if ($Part -in "all", "confirm") {
  $fresh = @("--table1", "--n", "100", "--max-cells", "500000", "--load", "seed=11", "--load", "exclude_seed=[0,7]", "--load", "exclude_n=100")
  # ours (configs/maze_ood.yaml) with both EMAs of one trajectory and the trained weights; the
  # main EMA (final.pt) has the warm-up, final_const.pt a constant decay, final_raw.pt no EMA
  foreach ($init in "torch", "public") {
    foreach ($s in 3, 4, 5) {
      $R = "runs/search/maze_fact_$init/seed$s"
      Py scripts/train.py --config configs/maze_ood.yaml --seed $s --out $R --set model.overrides.init_style=$init --set "train.ema_extra={const: {ema_warmup: false}}"
      foreach ($w in @(@("emawarm", "final.pt"), @("emaconst", "final_const.pt"), @("raw", "final_raw.pt"))) {
        Run results/search/maze_fact_${init}_seed${s}_$($w[0]).json scripts/eval_maze_scaling.py --run "$R/$($w[1])" @fresh --out results/search/maze_fact_${init}_seed${s}_$($w[0]).json
      }
    }
  }
  foreach ($s in 3, 4, 5) {
    Run results/confirm/maze_ood_ours_seed$s.json scripts/eval_maze_scaling.py --run runs/search/maze_fact_torch/seed$s @fresh --out results/confirm/maze_ood_ours_seed$s.json
    # the public-code recipe with given walls around the board instead of the wrapped border
    $R = "runs/search/maze_ood_codepub_wall/seed$s"
    Py scripts/train.py --config configs/maze_ood_codepub.yaml --seed $s --out $R --set model.overrides.padding=wall
    Run results/confirm/maze_ood_codepub_wall_seed$s.json scripts/eval_maze_scaling.py --run $R @fresh --out results/confirm/maze_ood_codepub_wall_seed$s.json
  }
  # the authors' exact setup: their recipe on mazes with the ring of walls
  foreach ($s in 1, 2) {
    $R = "runs/search/maze_ood_codepub_ring/seed$s"
    Py scripts/train.py --config configs/maze_ood_codepub_ring.yaml --seed $s --out $R
    Run results/confirm/maze_ood_codepub_ring_seed$s.json scripts/eval_maze_scaling.py --run $R @fresh --out results/confirm/maze_ood_codepub_ring_seed$s.json
  }
  # Sudoku: the public-code recipe at its 20K steps, with its betas and with ours, seeds 0 and 1
  $dev = @("--split", "dev", "--n", "500", "--depth", "256", "--k", "128", "--record-every", "32", "--group", "level", "--max-cells", "500000")
  foreach ($s in 0, 1) {
    Py scripts/train_fast.py --config configs/sudoku_ood_codepub.yaml --seed $s --out runs/sudoku_ood_codepub/seed$s
    Run results/confirm/sudoku_codepub_seed$s.json scripts/eval_tts.py --run runs/sudoku_ood_codepub/seed$s @dev --out results/confirm/sudoku_codepub_seed$s.json
    $R = "runs/search/sudoku_codepub_betas999/seed$s"
    Py scripts/train_fast.py --config configs/sudoku_ood_codepub.yaml --seed $s --out $R --set "train.betas=[0.9,0.999]"
    Run results/confirm/sudoku_codepub_betas999_seed$s.json scripts/eval_tts.py --run $R @dev --out results/confirm/sudoku_codepub_betas999_seed$s.json
  }
}
