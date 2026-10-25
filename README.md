# Reasoning with Neural Cellular Automata, replicated on one consumer GPU

A from-scratch PyTorch replication of **"Reasoning with Neural Cellular Automata"**
(Etcheverry et al., 2026, [arXiv:2609.36126](https://arxiv.org/abs/2609.36126)) that runs on a
single Windows PC with an 8 GB GPU. The paper was trained on TPUs; this repo rebuilds the model,
the training recipe and the evaluations from the paper, shrinks only what has to be shrunk
(training steps, batch sizes, test-set sizes), runs the authors' published weights through its
own code as a check, and compares the recipe written in the paper with the one in the authors'
public code. Where they differ, both are kept and the measured winner is the default (see
[Paper, public code, our choice](#paper-public-code-our-choice)). Every departure from the paper
is written down in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

## TL;DR of the paper

A Neural Cellular Automaton (NCA) is a grid of identical cells. Each cell holds a small vector,
looks only at its 3x3 neighbours, and applies one shared update rule, over and over. Cells fire
asynchronously (each one updates with probability p_fire per step), so the dynamics are stochastic.
The paper shows that this strictly local substrate can solve real reasoning tasks: finding paths in
mazes, completing Sudoku boards, and ARC-AGI-1 puzzles, with models of 10K to a few million
parameters.

The interesting part is how compute can be scaled at test time without touching the weights:

* **depth D**: run more steps, so information travels further (a 9x9-trained maze NCA solves
  201x201 mazes when given a bigger grid and 13,000 steps);
* **parallel trials K**: run K stochastic rollouts of the same board and keep the one with the
  highest self-reported **confidence** (conf@K), which turns ~66% into ~98% on hard Sudoku;
* **pruning**: Niche-Capped Diversity Pruning periodically discards duplicate trajectories so that
  K parallel trials cost 2 to 4 times less for the same accuracy;
* **noise** at test time is not only tolerated but slightly helpful;
* **adaptive compute**: cells that are confident fire less often, which saves cell updates and
  speeds up recovery from damage.

Training uses backprop through time over a short chunk of a rollout, a replay pool of evolving
states, and perturbations (noise, damage, target swap). The ablations show these are what make the
out-of-distribution generalisation possible.

## Results at a glance

The Paper column below reproduces Table 1. Local quality results are documented in
[docs/RESULTS.md](docs/RESULTS.md); pending entries remain unmeasured. The GPU speed benchmarks
linked below are separate measurements and do not establish benchmark accuracy.
`scripts/aggregate.py` aggregates quality results over 3 seeds. Local test configurations are
smaller than the paper's (fewer boards, smaller K or D); the budgets are listed in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md) and each command in [docs/PAPER_MAP.md](docs/PAPER_MAP.md).

| Benchmark (test setting) | Metric | Paper | This repo (3 seeds) |
|---|---|---|---|
| Maze-OOD 13x13 (D=300, K=1) | exact | 100.0 | 100.0 ± 0.0 |
| Maze-OOD 59x59 (D=2000, K=1) | exact | 100.0 | 98.3 ± 1.7 |
| Maze-OOD 201x201 (D=13000, K=1) | exact | 100.0 | 85.7 ± 10.8 (seeds 100 / 83 / 74; see docs/RESULTS.md) |
| Maze-Hard, A* targets (D=200, K=1) | exact | 86.6 | 65.4 ± 4.1 (10K steps instead of 20K) |
| Maze-Hard, A* targets (D=200, conf@K) | exact | 89.2 (K=128) | 70.3 ± 2.9 (K=64) |
| Maze-Hard, BFS targets (D=200, K=1) | exact | 94.7 | 77.0 ± 3.9 (10K steps, `bfs_dlru` targets, see DEVIATIONS) |
| Maze-Hard, BFS targets (D=200, conf@K) | exact | 96.2 (K=128) | 80.4 ± 4.3 (K=64) |
| Sudoku-OOD (D=256, K=1) | exact | 66.1 | 26.0 ± 6.8 (50K steps instead of 150K) |
| Sudoku-OOD (D=256, conf@K) | exact | 98.5 (K=2048) | 67.0 ± 8.4 (K=128) |
| Sudoku-Extreme (K=1) | exact | 75.6 (D=2048) | not run (postponed: about 30 h of training, 45 h with evaluations) |
| Sudoku-Extreme (conf@K) | exact | 92.7 (K=512, D=2048) | not run (postponed) |
| ARC-AGI-1 (pass@2, K=3264) | exact | 48.8 | left out of this replication (too long on one 8 GB GPU); a 30-task proof-of-concept setup exists but was not run |
| Visual Sudoku | exact | 87.8 / 18.9 | not replicated |

The table above is the recipe read from the paper (`configs/maze_ood.yaml`, `maze_hard*.yaml`,
`sudoku_ood.yaml`), seeds 0 / 1 / 2. The recipe search of 21-24 October added the authors'
published weights run through our code and the recipe of their public code, which is now the
default for Maze-OOD and Sudoku-OOD:

| Benchmark (test setting) | Paper | Their weights in our code | Paper recipe here | Public-code recipe here (default) |
|---|---|---|---|---|
| Maze-OOD 201x201, 100 fresh mazes (D=13000, K=1) | 100.0 | 100 (Table 1 sample) | 83.3 ± 23.6 (seeds 3 / 4 / 5: 50 / 100 / 100) | 97.5 ± 0.5 (seeds 1 / 2: 98 / 97) |
| Maze-OOD 59x59, 100 fresh mazes (D=2000, K=1) | 100.0 | 100 (Table 1 sample) | 96.7 ± 4.7 | 100.0 ± 0.0 |
| Sudoku-OOD, RRN test (D=256, K=1) | 66.1 | 62.4 (500 validation boards) | 26.0 ± 6.8 at 50K steps; 8.0 ± 6.0 at 20K (2 seeds) | 34.8 ± 2.0 at 20K steps (2 seeds) |
| Sudoku-OOD, RRN test (D=256, conf@128) | 98.5 (K=2048) | 91.2 (500 validation boards) | 67.0 ± 8.4 at 50K; 24.5 ± 19.6 at 20K | 64.6 ± 0.2 at 20K |
| Maze-Hard, A* targets (D=200, K=1) | 86.6 | 79.1 (1 000 test mazes) | 65.4 ± 4.1 (10K steps) | not compared at the same budget |

"Fresh mazes" are 100 official test mazes per size that share no maze with the Table 1 sample or
the sample used to select recipes; the 3-seed Table 1 numbers above use the seed-0 sample. Their
weights in our code reproduce Maze-OOD exactly (100 / 100 / 100%) and come close on the other two
tasks: Sudoku 62.4% at K=1 on validation boards (paper 66.1% on test, a different set) and
Maze-Hard 79.1% against 86.6% on the same 1 000 test mazes, a 7.5-point gap we have not
explained. So the large gaps of the paper recipe (26.0% on Sudoku, 65.4% on Maze-Hard) come
mostly from training; a smaller part may sit in our model or evaluation. Details, the arms of the search
and the false leads on the way: [docs/RESULTS.md](docs/RESULTS.md#recipe-search-21-24-october).

The optimised training engine (`docs/FAST_ENGINE.md`) was run on the same RTX 4060 with the same
recipes and evaluation commands; its quality and speed are in
[results/bench/pc_rtx4060/README.md](results/bench/pc_rtx4060/README.md). Its Maze-Hard A* models
(3 seeds, 10K steps) give K=1 62.7 ± 3.0, conf@64 68.0 ± 2.0, pass@64 75.6 ± 0.4; its Sudoku-OOD
models give K=1 28.5 ± 3.4 and conf@128 71.0 ± 5.8, within the seed spread of the table above.
Standard deviations are population standard deviations over the three seeds, as everywhere here.

Parameter counts of the presets: maze_ood 10.4K (paper 10K), maze_hard 41.3K (41K), sudoku_ood
159.8K (160K), sudoku_extreme 602.8K (603K), arc 3.7M with the full width (3.5M).

## What was reproduced, and what was left aside

Each claim or experiment of the paper, with its status here. **Reproduced**: same result within
the seed spread. **Trend reproduced**: the paper's finding holds at a lower level, with a reduced
training budget. **Not reproduced**: run as described, without the paper's result. **Not run**:
implemented, not trained (over the time budget). **Left out**: deliberately not replicated.

| Paper claim / experiment | Status | Note |
|---|---|---|
| Maze-OOD: trained on 9x9, solves 13x13 and 59x59 (Table 1) | Reproduced | 100.0 and 98.3%, paper's exact budget |
| Maze-OOD: solves 201x201 (Table 1, Fig. 4B) | Reproduced with the authors' setup | 97.5 ± 0.5% with their public recipe and data format (2 seeds); the paper recipe leaves one seed in three below 75% (85.7 ± 10.8%) |
| Our code runs the authors' published weights (positive control) | Reproduced on Maze-OOD, close on the others | Maze-OOD 100 / 100 / 100%, Sudoku-OOD 62.4% at K=1 (validation boards), Maze-Hard 79.1% at K=1 on the paper's test mazes (paper 86.6%, gap not explained) |
| Adaptive firing saves cell updates (Fig. 8A) | Reproduced under an interpretation | 0.72x (paper 0.71x) only if the clamped input cells count as confident; read literally, the rule saves nothing (1.07x). Counted updates, not GPU time |
| Adaptive firing repairs damage faster (Fig. 8C) | Not reproduced | 1.30x the steps of uniform firing (paper 0.94x) |
| Training ingredients: replay, perturbations, asynchrony (Fig. 7A) | Trend partly reproduced | Maze-OOD only, evaluated on 13x13: replay matters with one trial, perturbations keep the confidence honest, asynchrony changes nothing at this size. The paper's no-replay failure on larger mazes was not tested |
| Test-time noise helps slightly (Fig. 7B) | Trend reproduced | Sudoku-OOD +0.7 to +1 point at the paper's noise level; no effect on mazes |
| Maze-Hard, A* and BFS targets, BFS better (Table 1) | Trend reproduced | 65.4 and 77.0% at K=1 (paper 86.6 and 94.7), with 10K training steps instead of 20K |
| Sudoku-OOD: depth and parallel trials raise accuracy (Fig. 4A) | Trend reproduced | public-code recipe: 34.8% at K=1, 64.6% at conf@128 in 20K steps (paper 66.1, 98.5 at K=2048, 150K steps); paper recipe 26.0 / 67.0% in 50K steps |
| Sudoku ablations, fixed vs self attention (Table 2, Fig. 13) | Not run | stopped at 5K steps, where every variant is at 0%; at least 20K steps per variant needed (about 50 h) |
| Sudoku-Extreme (Table 1) | Not run | about 45 h on the 8 GB GPU, postponed |
| Niche-Capped Diversity Pruning (Fig. 6A) | Not run | implemented; it is evaluated on Sudoku-Extreme |
| ARC-AGI-1 with task embeddings | Left out | about 100 A100-hours at the paper's size; a 30-task proof of concept is implemented, not run |
| Visual Sudoku | Left out | three days on 64 TPUs in the paper |
| Comparisons with TRM, HRM, PTRM | Left out | the baselines are not re-run |

Two things the paper does not specify had to be reconstructed: how its BFS breaks ties between
equally short paths (only the order down/left/right/up, the one of the HRM data, is learnable;
see `docs/DEVIATIONS.md`, 2026-10-19), and whether clamped cells count as confident for adaptive
firing. Several ingredients of the authors' public code are not in the paper, and on Sudoku three
of them decide whether the model learns at all (see below). With them, the gap that remains on
Sudoku-OOD is the training length (20K steps against 150K) and the number of trials (128 against
2048); that is a hypothesis, not a result, since the 150K-step run (about 19.5 h here) was not
made. Maze-Hard was not re-run with the public recipe nor at the paper's 20K steps.

## Paper, public code, our choice

The paper and the authors' public JAX code do not train the same way. Each difference is a switch
in our code, off by default in the paper-recipe configs (`docs/PAPER_MAP.md` lists them all); the
recipe search measured them and the defaults follow the measurements.

| Point | Paper | Public code | Default here | Measured |
|---|---|---|---|---|
| Sudoku-OOD recipe | cosine LR decay; no projection of the update; init not stated | update made tangent to the normalised cell (radial part removed), LR from 10x the peak down to the peak then constant, their init, a 10-token readout, mean MSE, their perturbations | public code: `configs/sudoku_ood_codepub.yaml` | 20K steps, RRN test: 34.8 vs 8.0% at K=1. Leaving out the projection, the LR schedule or the init gives 0% at 10K steps |
| AdamW betas | not stated (0.999 here, the PyTorch default) | 0.95 | public code | 0.999 takes off faster at 10K steps, ends 6-9 points lower at conf@128 at 20K (2 seeds, RRN validation boards) |
| Maze-OOD data format | 9x9 cells | 11x11: a ring of given walls around each maze | the ring, with their recipe: `configs/maze_ood_codepub_ring.yaml` | 201x201: 97.5 ± 0.5% (2 seeds) vs 83.3 ± 23.6% for the paper recipe (3 seeds), on fresh mazes |
| Maze-OOD border | wall padding (Table 3) | wrapped (torus) | wrapped, on the ring | without the ring, wrapping joins corridors on opposite edges: 85 / 53 / 20% at 13 / 59 / 201 |
| EMA decay | not stated | constant from step 0 | public code in its configs | no effect beyond run-to-run variance when both EMAs of one run are compared |
| Initialisation | not stated | their init | public code in its configs | Maze-OOD: no effect beyond seed variance (3 seeds); Sudoku: needed |
| Maze-Hard recipe | as written | public code | paper recipe: `configs/maze_hard.yaml` | not compared at the same budget |

The paper-recipe configs (`maze_ood.yaml`, `sudoku_ood.yaml`) are kept unchanged: the 3-seed
table above was produced with them.

## Quickstart

Python 3.11 or newer is required. Install PyTorch first so that you get the right build for your
machine, then the package.

Linux / macOS:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                       # CUDA build on Linux with an NVIDIA driver, CPU otherwise
pip install -e ".[dev]"
pytest -q                               # a few minutes on CPU, no download needed
```

Windows (PowerShell, from the repo root; see [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md) for
the long version):

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())"   # must print True for the GPU runs
pytest -q
```

Level 0, a one-minute smoke run on CPU (a tiny maze NCA that starts solving 9x9 mazes):

```bash
python scripts/train.py --config configs/smoke_maze.yaml --device cpu
# -> runs/smoke_maze/seed0/{run.json, log.jsonl, checkpoint.pt, final.pt, summary.json}
```

Phase 1, Maze-OOD part only (about 6.5 hours on an 8 GB laptop GPU; trains 3 seeds, runs every
Maze-OOD evaluation, the ablations, an animation, then aggregates and draws the figures):

```bash
bash scripts/run_phase1.sh maze                                     # Linux
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 maze   # Windows
```

Every script resumes: training continues from `checkpoint.pt`, and evaluations whose JSON already
exists are skipped, so an interrupted phase is simply rerun with the same command.

## The cheminement

The replication is ordered so that the cheap experiments come first and each phase is useful on
its own. [docs/TUTORIAL.md](docs/TUTORIAL.md) walks through each level with the exact commands
and the files you should see.

1. **Level 0, smoke test** (CPU, about 1 minute): `configs/smoke_maze.yaml`. Checks that
   training runs, logs, checkpoints and resumes.
2. **Level 1, Maze-OOD** (about 6.5 h, 14 min per training run): `scripts/run_phase1.sh maze`. Trains the 10K-parameter maze
   NCA with the paper's exact budget (5K steps, batch 64), then: spatial substrate scaling up to
   59x59 and the Table 1 sizes 13 / 59 / 201 (`eval_maze_scaling.py`), adaptive compute and damage
   recovery on 201x201 mazes (`eval_adaptive.py`), the test-time noise sweep (`eval_noise.py`),
   the training-ingredient ablations (`run_ablations.py`) and a GIF of a 37x37 solve
   (`animate.py`).
3. **Level 2, Sudoku-OOD** (about 45 h on an 8 GB laptop GPU): `scripts/run_phase1.sh sudoku`.
   50K steps instead of the paper's 150K; depth and parallel-trial curves with and without noise
   (`eval_tts.py`), noise sweep, ablations including fixed vs self attention (stopped: see
   above), animation.
4. **Level 3, phase 2** (measured on the RTX 4060 Laptop: about 2 h of training and 1 h 30 of
   evaluation per Maze-Hard seed and target, about 10 h of training per Sudoku-Extreme seed,
   about 60 h in all): `scripts/run_phase2.sh`. Maze-Hard with A* and with BFS
   targets (10K steps instead of 20K), Sudoku-Extreme (30K steps, batch 128 instead of 200K,
   batch 800) with the Niche-Capped Diversity Pruning Pareto curves (`eval_pruning.py`). Only
   the Maze-Hard part has been run.
5. **Level 4, phase 3** (about 6 h, not run: ARC is left out of this replication): `scripts/run_phase3.sh`. ARC-AGI-1 proof of concept: one NCA
   conditioned by learned task embeddings on 30 training tasks plus RE-ARC extras, C=128,
   20K steps.
6. After any phase: `python scripts/aggregate.py` (writes `results/summary.json` and prints the
   Markdown table pasted into docs/RESULTS.md) and `python scripts/make_figures.py` (writes
   `figures/F1..F7*.png` and `.pdf`).

Each phase script has a PowerShell twin (`run_phase1.ps1`, `run_phase2.ps1`, `run_phase3.ps1`).

## What is in the repo

```
.
├── README.md                     this file
├── pyproject.toml                package metadata, dependencies, ruff and pytest settings
├── configs/                      one YAML per experiment; paper values vs local reductions are commented
│   ├── smoke_maze.yaml           level 0: tiny CPU run (inherits maze_ood.yaml through _base)
│   ├── maze_ood.yaml             Maze-OOD, identical to paper Table 3
│   ├── maze_ood_codepub_ring.yaml Maze-OOD default: the authors' public recipe on their 11x11 format
│   ├── maze_ood_codepub.yaml     the same recipe without the ring of walls (search arm)
│   ├── maze_hard.yaml            Maze-Hard with A* targets (10K steps locally)
│   ├── maze_hard_bfs_dlru.yaml   same with BFS targets (down/left/right/up, augmentation on)
│   ├── sudoku_ood.yaml           Sudoku-OOD, paper recipe (50K steps locally)
│   ├── sudoku_ood_codepub.yaml   Sudoku-OOD default: the authors' public recipe (20K steps locally)
│   ├── sudoku_extreme.yaml       Sudoku-Extreme (30K steps, batch 128 locally)
│   ├── arc_small.yaml            ARC-AGI-1 proof of concept (30 tasks, C=128)
│   ├── ablations_maze_ood.yaml   Fig. 7A variants: full / no_async / no_perturb / no_replay
│   ├── ablations_sudoku_ood.yaml same plus self_attn (Fig. 13), 5K steps (stopped, too short)
│   ├── sudoku_ood_fast.yaml      Sudoku-OOD with the fast engine (extension, docs/FAST_ENGINE.md)
│   └── maze_hard_fast.yaml       Maze-Hard with the fast engine
├── src/nca/                      the package (`import nca`)
│   ├── embeddings.py             fixed orthonormal token embeddings (paper B.1)
│   ├── model.py                  NCAConfig, the NCA cell, PRESETS from Tables 3-4, build_model
│   ├── pool.py                   sample replay buffer (B.2)
│   ├── perturb.py                state noise, damage, target swap (B.2)
│   ├── train.py                  BPTT training loop: DEFAULTS, EMA, AdamW, schedules, checkpoints
│   ├── rollout.py                inference engine: depth D, K trials, noise, adaptive firing, damage, pruning
│   ├── pruning.py                Niche-Capped Diversity Pruning (B.6)
│   ├── experiments.py            the paper's experiments as functions returning plain dicts
│   ├── tasks/                    benchmarks: base.py (Boards, Task), maze.py, sudoku.py, arc.py
│   ├── engine/                   fast training engine: fused operators, compiled blocks, segmented checkpointing (extension)
│   ├── cli.py                    shared command-line arguments of the eval scripts
│   ├── utils.py                  seeding, device choice, YAML configs with overrides, JSON output
│   └── viz.py                    figure style, board images, GIFs
├── scripts/
│   ├── train.py                  train one model from a config (resumable)
│   ├── train_fast.py             same with the fast engine; bench_engine.py and check_engine.py measure it
│   ├── eval_tts.py               depth and parallel-trial curves, conf@K / pass@K (Fig. 4A, 16, 17)
│   ├── eval_maze_scaling.py      spatial substrate scaling on Maze-OOD (Fig. 4B, Table 1)
│   ├── eval_adaptive.py          adaptive firing and damage recovery on 201x201 mazes (Fig. 8)
│   ├── eval_noise.py             test-time noise sweep (Fig. 7B)
│   ├── eval_pruning.py           pruning Pareto curves (Fig. 6A)
│   ├── run_ablations.py          training-ingredient ablations (Fig. 7A, 13)
│   ├── animate.py                GIF and snapshot strip of one rollout (Fig. 2)
│   ├── aggregate.py              results/*/*_seed*.json -> results/summary.json + Markdown table
│   ├── make_figures.py           results/ -> figures/F1..F7 (png + pdf)
│   ├── run_phase{1,2,3}.{sh,ps1} the ordered cheminement, Linux and Windows
│   ├── import_public.py          the authors' published weights as run folders (positive control)
│   └── run_recipe_search.ps1     paper vs public-code recipe: search arms and confirmation runs
├── tests/                        pytest suite (CPU, offline, a few minutes)
├── docs/                         tutorial, paper map, deviations, experiments, results, Windows setup
├── data/                         datasets (downloaded or generated on first use; gitignored except data/README.md)
├── runs/<config>/seed<k>/        created by training: run.json, log.jsonl, checkpoint.pt, final.pt, summary.json
├── results/<experiment>/         created by the eval scripts: <task>_seed<k>.json, summary.json
├── results/bench/pc_rtx4060/     speed and quality of the fast engine on the RTX 4060 (logs and drivers next to it)
└── figures/                      created by make_figures.py and animate.py (figures/anim/)
```

The `commit` field recorded in run and result files points to the development history, which
is not part of this public repository.

## Extension: a faster training engine for 8 GB

`src/nca/engine/` executes the same training recipe with fused operators (fixed-attention
mixing and the residual / noise / normalisation / clamp update, both with explicit backward
passes), `torch.compile` over blocks of steps, segmented gradient checkpointing tuned to the
memory budget, and reduced CPU synchronisation inside the rollout. The model, its parameters, the
data pipeline, the losses and the checkpoints are the replication's; every operator is
checked against the reference implementation (outputs and gradients) and the whole rollout is
compared backend against backend. `scripts/bench_engine.py` measures time, memory and FLOPs
per variant; `scripts/train_fast.py` trains with it. Design, validation and what to run on the
GPU: [docs/FAST_ENGINE.md](docs/FAST_ENGINE.md). The published numbers above were produced by
`scripts/train.py`, not by the engine.

Three later campaigns on a cloud L4 GPU capped at 8 GiB added opt-in options, a compiled
inference runner, ARC test-time training and a Visual Sudoku compute fixture; they are summarised
in [docs/FAST_ENGINE.md](docs/FAST_ENGINE.md#further-opt-in-options-and-apis-l4-campaigns-14-16-october-2026). Those were compute benchmarks on generated data: they do not establish benchmark accuracy
or laptop performance.

## How we checked

* **Tests** (`pytest -q`, CPU only, no network): `tests/test_model.py` checks the invariants of
  the cell (given cells stay clamped while free cells move, a zero fire mask is the identity,
  group normalisation gives unit-norm groups, wall padding blocks information at the border, wrap
  padding keeps convolution translation-equivariant, fixed attention refuses a grid of the wrong
  size, adaptive firing lowers the rate of confident cells, pruning halves the ensemble and keeps
  distinct niches, the ARC time embedding follows per-cell clocks). `tests/test_rollout.py` runs
  the rollout engine and the training loop end to end on tiny problems: reproducibility for a
  fixed seed, traces, selection helpers, pruning / adaptive / damage paths, checkpoint reload and
  resume, and that a few dozen steps on 9x9 mazes already cut the loss by more than 30%.
  `tests/test_maze.py`, `tests/test_sudoku.py` and `tests/test_arc.py` check the generators
  (perfect mazes are trees with unique paths, Sudoku puzzles have unique solutions, ARC rendering
  round-trips), the metrics (corrupted paths are rejected, invalid Sudoku grids are flagged), the
  augmentations (bijective, validity-preserving), the caches and the parsers of the official
  formats on hand-made files.
* **Parameter counts**: every preset is built and counted against Table 1 of the paper. The Sudoku
  presets match to 0.1% and the maze presets round to the paper's 10K and 41K; the ARC preset is
  within 20% (the paper does not say whether biases are counted), see docs/DEVIATIONS.md.
* **Training invariants in the logs**: `grad_norm` is clipped at 1.0 and a non-finite gradient
  skips the step instead of poisoning the weights (`skipped_steps` in `log.jsonl` stays at 0 in
  a healthy run); EMA weights are what every evaluation uses; runs are resumable bit-for-bit
  (model, EMA, optimizer, pool, RNG states are all in `checkpoint.pt`).
* **Protocol**: every experiment of the main table runs with seeds 0, 1 and 2; `aggregate.py`
  reports mean ± std next to the paper's value; figures shade ±1 std. The recipe search compares
  arms on held-out boards, then confirms on seeds it did not use (3, 4, 5 for mazes) and on mazes
  that neither the selection nor Table 1 used.
* **Positive control**: the authors' published weights, loaded into our model
  (`scripts/import_public.py`), match their JAX forward pass to 2e-7 (mazes) and 2e-4 (Sudoku)
  over 60 steps; with our evaluation scripts they reach the paper's accuracy on Maze-OOD and come
  within 4 to 8 points of it on Sudoku-OOD and Maze-Hard.

## Documentation

* [docs/TUTORIAL.md](docs/TUTORIAL.md): reproduce from scratch, level by level, with the exact
  commands, expected files, how to read the logs, memory tips for 8 GB, troubleshooting.
* [docs/PAPER_MAP.md](docs/PAPER_MAP.md): each claim, section and figure of the paper mapped to
  the code (file:function), the command, the result file and the figure; plus a glossary of config
  names vs paper symbols.
* [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md): step-by-step setup on Windows with CUDA.
* [data/README.md](data/README.md): what each dataset is, where it comes from, the local fallback,
  cache file names and caveats.
* [docs/REPLICATION_WORKFLOW.md](docs/REPLICATION_WORKFLOW.md): the reusable process we followed.
* [docs/DEVIATIONS.md](docs/DEVIATIONS.md): journal of every departure from the paper.
* [docs/RESULTS.md](docs/RESULTS.md): the numbers, filled by `scripts/aggregate.py`.
* [docs/FAST_ENGINE.md](docs/FAST_ENGINE.md): the optional faster training engine, its validation and
  its options.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{etcheverry2026reasoning,
  title   = {Reasoning with Neural Cellular Automata},
  author  = {Etcheverry, Mayalen and Miotti, Pietro and Sirbu, Aidan and Sch{\"u}rholt, Konstantin
             and Drozdova, Mariia and Ghosh, Arna and Ag{\"u}era y Arcas, Blaise and Manyika, James
             and Richards, Blake and Niklasson, Eyvind},
  journal = {arXiv preprint arXiv:2609.36126},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.36126}
}
```

## Acknowledgements and license

The ideas and the experimental protocol are the paper authors'. The datasets belong to their
respective authors: easy-to-hard mazes (Bansal et al., 2022), Maze-Hard and Sudoku-Extreme
(Wang et al., 2025, HRM), SAT-Net Sudoku (Wang et al., 2019), RRN Sudoku (Palm et al., 2017),
ARC-AGI-1 (Chollet, 2019) and RE-ARC (Hodel, 2024). The replication code is released under the
MIT License.
