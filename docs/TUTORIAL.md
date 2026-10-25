# Tutorial: reproducing the paper from scratch

This is the long version of the README's cheminement. Each level is self-contained, ordered from
one minute on CPU to a day of GPU time, and every command is the one the phase scripts run. All
commands are given from the repository root. On Windows, replace `python scripts/...` by
`python scripts\...` (or keep forward slashes, Python accepts both) and use the `.ps1` phase
scripts.

Timings are for one 8 GB GPU: phase 1 about 2 days in total (Maze-OOD alone about 6.5 h with the adaptive-compute evaluations at D=20 000, Sudoku-OOD about 45 h; measured on an RTX 4060 Laptop), phase 2
about 60 h (Maze-Hard about 10 h per target type, Sudoku-Extreme about 45 h), phase 3 about 6 h.
See the compute budget table of [DEVIATIONS.md](DEVIATIONS.md) for the paper budget vs the local
budget of every task.

## Before you start

```bash
pip install -e ".[dev]"
pytest -q
```

The tests run on CPU, offline, in a few minutes. If they pass, the package, the data generators
and the training loop work on your machine.

The data folder is `data/` by default (`data_root` in the configs). Official datasets are
downloaded on first use when the hosts are reachable and otherwise replaced by local generators;
each loader prints which source it used, and `meta["source"]` is written into the result JSONs.
See [../data/README.md](../data/README.md).

## Level 0: smoke test on CPU (about 1 minute)

```bash
python scripts/train.py --config configs/smoke_maze.yaml --device cpu
```

`configs/smoke_maze.yaml` inherits `maze_ood.yaml` (through `_base`) and shrinks it: 2,000
generated 9x9 mazes, 300 steps of batch 16, rollouts of 48 steps. The first run generates and
caches the mazes (`data/maze/maze_ood_train_9_n2000_seed0.npz`), which takes a few seconds.

What you should see on the console: one line per 25 steps, `step 25/300 loss 0.xxxx cell 0.xxx
exact 0.xxx age x.x lr 4.00e-04 ...`, and an `eval@100`, `eval@200`, `eval@300` line. The loss
should drop clearly over the run (the test suite asserts a 30% drop after only 60 steps) and
`cell` (fraction of mutable cells correct) should climb well above where it started. `exact`
(whole boards solved) may stay low after 300 steps; that is expected, the config comment says
"starts solving mazes after a few hundred steps", and the full 5,000-step Maze-OOD run is what
reaches 100%.

Files written to `runs/smoke_maze/seed0/`:

| file | content |
|---|---|
| `run.json` | the resolved config, seed, parameter count, model config |
| `log.jsonl` | one JSON line per 25 steps (`step`, `loss`, `grad_norm`, `lr`, `batch_cell_acc`, `batch_exact`, `mean_age`, `skipped_steps`, `elapsed`) and one `{"eval": {...}}` line per evaluation |
| `checkpoint.pt` | everything needed to resume (model, EMA, optimizer, scheduler, pool, RNG states, step) |
| `final.pt` | the EMA weights plus the config, what every eval script loads |
| `summary.json` | final evaluation, training seconds, steps, parameters, skipped steps |

Try an interruption: press Ctrl+C in the middle, rerun the same command, and it prints
`resumed from step N`.

## Level 1: Maze-OOD (about 6.5 hours)

This is the one task whose training budget is identical to the paper's (5,000 steps, batch 64,
50K training mazes), so it is the best first real experiment. The shortcut for the whole level is
`bash scripts/run_phase1.sh maze` (or `powershell -ExecutionPolicy Bypass -File
scripts\run_phase1.ps1 maze`). The steps it runs, in order:

### 1. Train three seeds (10 to 15 minutes each)

```bash
python scripts/train.py --config configs/maze_ood.yaml --seed 0
python scripts/train.py --config configs/maze_ood.yaml --seed 1
python scripts/train.py --config configs/maze_ood.yaml --seed 2
```

Output: `runs/maze_ood/seed{0,1,2}/` with the five files listed above. The validation line
every 500 steps is a K=1 rollout of depth 100 on 256 test mazes of size 9; `val_exact` should
approach 1.0 by the end.

### 2. Spatial substrate scaling (Fig. 4B and the Maze-OOD rows of Table 1)

```bash
python scripts/eval_maze_scaling.py --run runs/maze_ood/seed0 --n 200 --out results/maze_scaling/maze_ood_seed0.json
python scripts/eval_maze_scaling.py --run runs/maze_ood/seed0 --table1 --n 100 --out results/maze_scaling/table1_seed0.json
```

(and the same for seeds 1 and 2). The first command runs the 9x9-trained model unchanged on
mazes of sizes 9, 11, 13, ..., 37 and 59, with a depth that grows with the size (the rule is in
the script; `--sizes` and `--depth-scale` change it). The second uses the paper's exact depths for
13x13 (D=300), 59x59 (D=2000) and 201x201 (D=13000). The JSON holds one row per size with
`exact`, `valid`, `optimal`, `cell_acc`, the solving curve along the rollout and the number of
cell updates.

### 3. Adaptive compute and damage recovery (Fig. 8)

```bash
python scripts/eval_adaptive.py --run runs/maze_ood/seed0 --size 201 --n 20 --depth 13000 --record-every 100 --out results/adaptive/maze201_seed0.json
python scripts/eval_adaptive.py --run runs/maze_ood/seed0 --size 201 --n 20 --depth 13000 --damage-at 6000 --record-every 100 --out results/adaptive/maze201_damage_seed0.json
```

Uniform firing (p_fire = 0.8) against adaptive firing (p_low = 0.4 once a cell's confidence
exceeds tau = 0.95), on 20 mazes of size 201, with and without a damage event at step 6000. The
JSON has a `variants` dict with, for `uniform` and `adaptive`, the solved curve, the cumulative
cell-update curve, and `steps_to_95` / `updates_to_95`.

### 4. Test-time noise sweep (Fig. 7B)

```bash
python scripts/eval_noise.py --run runs/maze_ood/seed0 --n 500 --depth 300 --k 8 --load size=13 --out results/noise/maze_ood_seed0.json
```

Accuracy at K=1 and conf@8 for noise magnitudes 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 injected during
the first quarter of a 300-step rollout on 500 mazes of size 13.

### 5. Training-ingredient ablations (Fig. 7A)

```bash
python scripts/run_ablations.py --config configs/ablations_maze_ood.yaml --seeds 0 1 2 --eval-n 500
```

Trains `full`, `no_async`, `no_perturb` and `no_replay` with three seeds each (12 short trainings
in `runs/ablations_maze_ood/<variant>/seed<k>/`, resumable), then evaluates each at K=1 and
conf@8 on 500 test mazes with a depth of twice the training rollout, writing
`results/ablations/maze_ood_<variant>_seed<k>.json`.

### 6. Animation (Fig. 2A style)

```bash
python scripts/animate.py --run runs/maze_ood/seed0 --size 37 --depth 1200 --every 10 --out figures/anim/maze37
```

Writes `figures/anim/maze37.gif` and `figures/anim/maze37_strip.png` (six snapshots) and prints
the step at which the maze was solved.

### 7. Aggregate and draw

```bash
python scripts/aggregate.py
python scripts/make_figures.py
```

`aggregate.py` reads every `results/<experiment>/<name>_seed<k>.json`, writes
`results/summary.json` (mean, std, n over seeds of every scalar) and prints the Markdown table of
headline numbers next to the paper's. `make_figures.py` writes, for the data available so far,
`figures/F1_maze_size_scaling`, `figures/F3_noise`, `figures/F4_ablations_maze_ood`,
`figures/F5_adaptive_compute` and `figures/F7_training_maze_ood` (each as `.png` and `.pdf`),
and says which figures are skipped for lack of data.

## Level 2: Sudoku-OOD (the rest of phase 1, about 45 hours)

```bash
bash scripts/run_phase1.sh sudoku
```

runs, for seeds 0, 1 and 2:

```bash
python scripts/train.py --config configs/sudoku_ood.yaml --seed 0
python scripts/eval_tts.py --run runs/sudoku_ood/seed0 --n 1000 --depth 256 --k 128 --record-every 8 --group level --out results/tts/sudoku_ood_seed0.json
python scripts/eval_tts.py --run runs/sudoku_ood/seed0 --n 1000 --depth 256 --k 128 --record-every 8 --group level --noise-sigma 0.15 --out results/tts/sudoku_ood_noise_seed0.json
python scripts/eval_noise.py --run runs/sudoku_ood/seed0 --n 500 --depth 256 --k 16 --out results/noise/sudoku_ood_seed0.json
```

then

```bash
python scripts/run_ablations.py --config configs/ablations_sudoku_ood.yaml --seeds 0 1 2 --eval-n 500 --eval-depth 256
python scripts/animate.py --run runs/sudoku_ood/seed0 --depth 256 --every 2 --out figures/anim/sudoku --board 3
python scripts/aggregate.py
python scripts/make_figures.py
```

Notes:

* Training is 50,000 steps of batch 256 (the paper: 150,000), rollouts of 48 steps, the 160K
  parameter fixed-attention model with group normalisation. Validation every 2,500 steps on 180
  test boards (10 per difficulty level).
* `eval_tts.py` runs 128 parallel trials on 1,000 test boards spread over the 18 difficulty
  levels (47 to 64 empty cells). The JSON holds the K=1 depth curve (`depth_curve`, every 8
  steps, overall and per level in `depth_curve_by_group`), the width curve (`conf_at_k` and
  `pass_at_k` for K = 1, 2, 4, ..., 128) and per-level accuracies. The `_noise` variant injects
  sigma = 0.15 during the first quarter of the rollout (Fig. 16 dashed lines).
* The ablations config adds the `self_attn` variant (Fig. 13) and trains every variant for 20,000
  steps.
* If the official RRN test set could not be downloaded, the generated test set does not reach the
  hardest levels (17 to 21 clues); the achieved level distribution is printed and stored in
  `meta["level"]`. See data/README.md.
* New figures: `F2_tts_sudoku_ood`, `F4_ablations_sudoku_ood`, `F7_training_sudoku_ood`, and
  `F3_noise` now has both tasks.

## Level 3: phase 2, Maze-Hard and Sudoku-Extreme (about 60 hours; only Maze-Hard was run)

```bash
bash scripts/run_phase2.sh                  # or: maze_hard | sudoku_extreme
```

Maze-Hard, for `maze_hard` (A* targets) and `maze_hard_bfs_dlru` (BFS targets in the order
down, left, right, up, recomputed on each dihedral augmentation; see `docs/DEVIATIONS.md`,
2026-10-19), seeds 0, 1, 2:

```bash
python scripts/train.py --config configs/maze_hard.yaml --seed 0
python scripts/eval_tts.py --run runs/maze_hard/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --out results/tts/maze_hard_astar_seed0.json
python scripts/eval_tts.py --run runs/maze_hard/seed0 --n 1000 --depth 200 --k 64 --record-every 8 --noise-sigma 0.15 --noise-ps 0.3 --out results/tts/maze_hard_astar_noise_seed0.json
python scripts/animate.py --run runs/maze_hard/seed0 --depth 200 --every 2 --out figures/anim/maze_hard
```

(the BFS runs write `results/tts/maze_hard_bfs_seed<k>.json`). Training is 10,000 steps instead
of the paper's 20,000, rollouts of 256 steps with a 128-step chunk, cosine schedule with 25%
warmup. The maze metrics in the JSON are `exact` (matches the reference path), `valid` (one
simple path between start and goal), `optimal` (valid and as short as the reference) and
`cell_acc`.

Sudoku-Extreme, seeds 0, 1, 2:

```bash
python scripts/train.py --config configs/sudoku_extreme.yaml --seed 0
python scripts/eval_tts.py --run runs/sudoku_extreme/seed0 --n 1000 --depth 512 --k 64 --record-every 16 --group bin --out results/tts/sudoku_extreme_seed0.json
python scripts/eval_tts.py --run runs/sudoku_extreme/seed0 --n 1000 --depth 512 --k 64 --record-every 16 --group bin --noise-sigma 0.15 --out results/tts/sudoku_extreme_noise_seed0.json
python scripts/eval_pruning.py --run runs/sudoku_extreme/seed0 --n 500 --k 64 --depths 64 128 256 512 --ms 2 4 8 --out results/pruning/sudoku_extreme_seed0.json
python scripts/animate.py --run runs/sudoku_extreme/seed0 --depth 512 --every 4 --out figures/anim/sudoku_extreme --board 5
```

Training is 30,000 steps of batch 128 (the paper: 200,000 of batch 800), 96-step rollouts with
last-step supervision, the 603K-parameter model. The test boards are stratified over the 7
difficulty bins of the paper's Fig. 12 (`--group bin`). `eval_pruning.py` runs, for each depth,
the unpruned K=64 ensemble and the pruned ones with m = 2, 4, 8 halvings, with the paper's
exploration noise (sigma 0.1, p_t 0.2, p_s 0.8, first quarter); each row stores conf@K, pass@K
and the measured compute in cell updates per board. New figures: `F2_tts_maze_hard_astar`,
`F2_tts_maze_hard_bfs`, `F2_tts_sudoku_extreme`, `F6_pruning_sudoku_extreme`.

## Level 4: phase 3, ARC-AGI-1 proof of concept (about 6 hours)

```bash
bash scripts/run_phase3.sh
```

runs, for seeds 0, 1, 2:

```bash
python scripts/train.py --config configs/arc_small.yaml --seed 0
python scripts/eval_tts.py --run runs/arc_small/seed0 --depth 64 --k 32 --record-every 4 --out results/tts/arc_seed0.json
```

One NCA is trained on the demonstration pairs of 30 ARC-AGI-1 training tasks (chosen by
`task_seed`), each task with its own learned task embedding, plus 20 RE-ARC generated examples
per task when RE-ARC is available. The model is the paper's ARC architecture at C=128 instead of
512 (fixed attention, SwiGLU update, group norm, learned step embedding with per-cell local
clocks at inference), trained for 20,000 steps with cross-entropy on the last 16 steps of a
64-step rollout, with online augmentation (random scale and offset on the 32x32 canvas, random
dihedral symmetry, random colour permutation) and no replay pool. The test split is the test
pairs of the same 30 tasks, i.e. unseen inputs of seen rules (the setting of the paper's Fig. 3).
The first run downloads the ARC-AGI repository and the four RE-ARC source files into `data/arc/`.

Test-time training on the evaluation tasks (paper B.3) is implemented (`ARCTask.ttt_variants`)
but not part of the phase script.

## Reading the outputs

### `runs/<config>/seed<k>/log.jsonl`

Training lines, every `log_every` steps:

| key | meaning |
|---|---|
| `step` | optimizer steps done so far |
| `loss` | the loss of this step's chunk (MSE to the target embeddings, or cross-entropy for ARC) |
| `grad_norm` | gradient norm before clipping (clipped at `grad_clip`, 1.0 everywhere) |
| `lr` | current learning rate (constant for Maze-OOD, warmup then cosine elsewhere) |
| `batch_cell_acc` | fraction of mutable cells correct at the end of the rollout, on this training batch |
| `batch_exact` | fraction of boards of this batch fully solved at the end of the rollout |
| `mean_age` | mean number of rollouts the batch's states have gone through in the replay pool (0 = fresh) |
| `skipped_steps` | cumulative number of steps skipped because the gradient was not finite (should stay 0) |
| `elapsed` | seconds since the start (or the resume) |

Evaluation lines (`{"eval": {...}}`), every `eval_every` steps, with the EMA weights, K=1, on the
validation boards of the config: `val_exact` (boards solved), `val_cell_acc`, `val_conf` (mean
board confidence), `step`, plus the task's extra metrics: `val_valid` and `val_optimal` for mazes,
`val_valid` for Sudoku, `val_exact_with_border` for ARC.

`summary.json` repeats the last evaluation with `train_seconds`, `steps`, `params` and
`skipped_steps`.

### `results/<experiment>/<name>_seed<k>.json`

Every result file carries `run`, `seed`, `n_boards`, `seconds` and, where it applies, `source`
(official or generated data). The main keys:

* `tts/`: `steps`, `depth_curve` (K=1 solved fraction at each recorded step),
  `depth_curve_mean_over_trials`, `cell_acc_curve`, `conf_curve`, `conf_at_k` and `pass_at_k`
  (dicts keyed by K), `mean_trial_acc`, `metrics` (every task metric averaged over all trials),
  `by_group_k1` / `by_group_confk` / `by_group_passk` and `depth_curve_by_group` when `--group`
  was given, `cell_updates_per_trial`.
* `maze_scaling/`: `sizes` and `rows` (one per size: `size`, `depth`, `exact`, `valid`,
  `optimal`, `cell_acc`, `steps`, `exact_curve`, `cell_updates`).
* `adaptive/`: `variants.uniform` and `variants.adaptive`, each with `steps`, `solved_curve`,
  `cell_updates_curve`, `cell_acc_curve`, `final_exact`, `final_updates`, `steps_to_95`,
  `updates_to_95`.
* `noise/`: `rows` with `sigma`, `k1`, `conf_k`, `pass_k`, `mean_trial`.
* `pruning/`: `rows` with `depth`, `m` (1 = unpruned), `conf_k`, `pass_k`, `compute_units`
  (measured cell updates per board), `compute_rel` and `theoretical_fraction`.
* `ablations/`: the `tts` keys plus `variant` and `overrides`.

`results/summary.json` (from `aggregate.py`) has one entry per `<experiment>/<name>` with
`n_seeds`, `files` and `scalars` (mean, std, n of every numeric key present in all seeds).

### Figures

`figures/F1_maze_size_scaling`, `F2_tts_<task>`, `F3_noise`, `F4_ablations_<task>`,
`F5_adaptive_compute`, `F6_pruning_<task>`, `F7_training_<config>`, each as `.png` and `.pdf`;
curves are means over the seeds found with a ±1 std band. `figures/anim/<name>.gif` and
`<name>_strip.png` come from `animate.py`.

## Resuming and interrupting

* Training checkpoints every `checkpoint_every` steps (100 for the smoke run, 500 for the maze
  and ARC configs, 1,000 for Sudoku). Interrupt with Ctrl+C at any time; rerunning the very same
  `train.py` command continues from `checkpoint.pt` (it prints `resumed from step N`). The
  checkpoint contains the replay pool and the RNG states, so a resumed run follows the same
  trajectory as an uninterrupted one would have, up to GPU non-determinism.
* `--no-resume` starts over in the same folder; `--out some/other/folder` keeps the old run.
* Changing a config value after a checkpoint exists is not detected: the checkpoint is loaded as
  is (and the new `train.steps` is honoured, which is how the tests extend a run). Delete the run
  folder or use `--out` when you change anything else.
* The phase scripts are rerunnable: a training that is complete is a no-op (the loop starts at
  `train.steps`), and an evaluation is skipped when its output JSON exists. To redo an
  evaluation, delete its JSON. Evaluations themselves are not checkpointed; the long ones are the
  201x201 adaptive runs (depth 13,000) and the pruning sweep.
* Evaluation seeds default to the training run's seed (`--seed` overrides).

## Memory tips for 8 GB

The training memory is dominated by backprop through the chunk: roughly
`chunk x batch_size x channels x H x W` activations, times the perception width. The configs are
sized for 8 GB, but if you hit `CUDA out of memory`:

* `--set train.batch_size=128` (halves memory; this changes the experiment, so note it in your
  own deviations log) or, better for faithfulness, `--set train.chunk=48` with the same
  `train.rollout`: the rollout length stays, only the window that gets gradients shrinks (the paper
  itself uses a 128-step chunk inside 256-step rollouts for Maze-Hard).
* `--set train.amp=true` runs the chunk in bfloat16 autocast (off by default; the loss and the
  state are kept in float32 between steps).
* Gradient checkpointing (`train.grad_checkpoint`) is `auto`: on CUDA it switches itself on when
  `chunk x batch_size x channels` exceeds about a million, i.e. for the Sudoku and ARC configs,
  trading some extra compute (the chunk is recomputed during the backward pass) for much less
  activation memory. Set it to `true` or `false` to
  force it.
* The replay pool lives on the GPU: `pool_mult x batch_size` states of `channels x H x W`
  floats. It is small for every config here (a few hundred MB at most for Sudoku-Extreme).
* For evaluations, memory is `K x H x W x channels` per board in a pass. `--max-cells`
  (default 2,000,000 cells per pass) bounds it: the engine processes `max_cells // (K x H x W)`
  boards at a time, so lowering it only makes the evaluation slower. `--amp` is available on every
  eval script. A 201x201 maze with K=1 is 40K cells, so 20 of them fit in one pass; Sudoku with
  K=128 is 10K cells per board.
* `--device cpu` forces the CPU (slow but useful for the smoke config and for debugging).

## Troubleshooting

* **`torch.cuda.is_available()` is False** on a machine with an NVIDIA GPU: you installed the
  CPU wheel. Reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (Windows and Linux) and update the NVIDIA driver if the CUDA version is not supported.
* **`[maze] official Maze-OOD ... unavailable ... using the local generator`** (and the same
  notices for Sudoku and ARC): the download failed or the host was unreachable; the local
  generator is used and `meta["source"]` says `generated`. The Maze-OOD generator produces the
  same kind of mazes (perfect mazes, unique path). The Sudoku-OOD generator cannot produce the
  hardest test levels (17 to 21 clues); only the official RRN data can, so compare per-level
  results with care. ARC has no generator: without the data, `load` raises (the tests use
  `source=synthetic`). To force a source: `--set task.train.source=official` on `train.py`, or
  `--load source=generated` on the eval scripts.
* **Generating data is slow the first time**: 50,000 9x9 mazes take a minute or two, 1,000
  201x201 mazes a few minutes, 9,000 Sudoku boards a few minutes with multiprocessing; everything
  is cached under `data/`.
* **`fixed attention was built for a (9, 9) grid, got (H, W)`**: fixed attention has
  position-specific weights, so Sudoku and ARC models only run at their training grid size.
  Spatial substrate scaling is a Maze-OOD experiment (convolutional perception).
* **`skipped_steps` grows** in `log.jsonl`: non-finite gradients were skipped. Occasional ones at
  the very start of a run are harmless; a steady increase means the run is unstable (try a lower
  learning rate or check `grad_norm`).
* **The loss is flat at the start**: the update block starts at zero (a do-nothing rule) for the
  maze presets, so the first few dozen steps look still; this is expected.
* **Validation `val_exact` is 0 while `val_cell_acc` is high**: normal early on, whole-board
  exact match is strict. For Sudoku-OOD at 50K local steps, expect K=1 accuracy below the paper's
  66% on the hardest levels; the conf@K curve is where the gap closes.
* **Evaluation is very slow**: check it is on the GPU (`--device cuda`), reduce `--n`, and
  remember that `eval_adaptive.py` at depth 13,000 and the pruning sweep are the long ones.
* **PowerShell refuses to run the `.ps1` scripts**: use
  `powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1`, as in the script headers.
* **`pip install -e ".[dev]"` fails on Windows with a quoting error**: PowerShell is fine with
  the double quotes; in `cmd.exe` use `pip install -e .[dev]` without quotes.
* **The PC went to sleep during a phase**: the run resumes from the last checkpoint when you rerun
  the phase script; see [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for keeping the machine awake.
