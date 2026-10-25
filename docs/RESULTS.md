# Results

Filled by `python scripts/aggregate.py` (headline table) and `python scripts/make_figures.py`
(figures). The main table uses the recipe read from the paper; the default recipes chosen by the
recipe search (21-24 October) are in the section of that name.

## Headline table

Copied from `results/summary.md` (written by `scripts/aggregate.py`) after phase 1 on the target
PC (Maze-OOD 2026-10-12/13, with the paper's architecture; Sudoku-OOD 2026-10-12). Exact-match
accuracy in %, mean ± std over seeds 0, 1, 2.

| Experiment | Paper | Ours (mean ± std over seeds) | n |
|---|---|---|---|
| Maze-OOD 13x13, D=300, exact | 100.0 | 100.0 ± 0.0 | 3 |
| Maze-OOD 59x59, D=2000, exact | 100.0 | 98.3 ± 1.7 | 3 |
| Maze-OOD 201x201, D=13000, exact | 100.0 | 85.7 ± 10.8 | 3 |
| Maze-Hard (A*), K=1, exact | 86.6 | 65.4 ± 4.1 (10K steps) | 3 |
| Maze-Hard (A*), conf@K, exact | 89.2 (K=128) | 70.3 ± 2.9 (K=64) | 3 |
| Maze-Hard (BFS), K=1, exact | 94.7 | 77.0 ± 3.9 (10K steps, `bfs_dlru` targets) | 3 |
| Maze-Hard (BFS), conf@K, exact | 96.2 (K=128) | 80.4 ± 4.3 (K=64) | 3 |
| Sudoku-OOD, K=1 | 66.1 (D=256) | 26.0 ± 6.8 (D=256, 50K steps) | 3 |
| Sudoku-OOD, conf@K | 98.5 (K=2048) | 67.0 ± 8.4 (K=128) | 3 |
| Sudoku-Extreme, K=1 | 75.6 (D=2048) | not run yet | 0 |
| Sudoku-Extreme, conf@K | 92.7 (K=512) | not run yet | 0 |
| maze201 adaptive (rule as written): steps / updates to 95% vs uniform | 1.29x / 0.71x | 1.41x ± 0.14 / 1.07x ± 0.10 | 2 |
| maze201 adaptive_given: steps / updates to 95% vs uniform | 1.29x / 0.71x | 1.41x ± 0.14 / 0.72x ± 0.06 (counted activations) | 2 |
| maze201 after damage, adaptive_given: steps / updates to 95% vs uniform, total since step 0 | 0.94x / 0.52x (Fig. 8C, "total repair-and-solve time") | 1.30x / 0.66x (not reproduced) | 1 |
| maze201 after damage, adaptive_given: steps / updates to 95% vs uniform, elapsed since the damage | (0.94x / 0.52x if Fig. 8C is read as elapsed time) | 1.54x / 0.77x (not reproduced) | 1 |

Maze-Hard A* (3 seeds, 10K steps instead of 20K, 1 h 58 of training per seed, 1 000 official test
mazes, D=200, K=64, about 1 h per evaluation on the 8 GB GPU): K=1 64.9 / 60.7 / 70.7 per seed,
conf@64 70.3 ± 2.9, pass@64 77.6 ± 1.7, valid paths 86.5%, A*-optimal paths 72.7%. More trials help
little (+5 points from K=1 to K=64) and test-time noise changes nothing (K=1 64.5, conf@64 70.2).
The optimised engine's models trained on the same recipe give K=1 62.7 ± 3.0 and conf@64 68.0 ± 2.0
([results/bench/pc_rtx4060/README.md](../results/bench/pc_rtx4060/README.md)), within the seed
spread. Maze-Hard with BFS targets: every BFS target we first computed (neighbours expanded
up/down/left/right) stayed at 0% solved. The paths shipped with the HRM data turned out to be a BFS
expanding down/left/right/up, and the same BFS recomputed on each augmented maze (`bfs_dlru`)
learns like A* (27.3% at step 3 000, seed 0; `docs/DEVIATIONS.md`, 2026-10-19). With the same
recipe (3 seeds, 10 000 steps, 1 000 official test mazes, D=200), BFS gives K=1 77.0 ± 3.9 (seeds
79.1 / 71.5 / 80.3; paper 94.7), conf@64 80.4 ± 4.3 (paper 96.2 at K=128), pass@64 84.4 ± 2.0,
valid paths 87.8%, optimal paths 81.7%; test-time noise changes nothing (K=1 76.8, conf@64 80.4).
As in the paper, BFS targets beat A* targets, here by 11.6 points at K=1 and 10.1 at conf@64
(paper: 8.1 and 7.0); both stay 16-21 points under the paper with half its training steps. Sudoku-Extreme is postponed, ARC-AGI and
visual Sudoku are left out of this replication.

### Maze-OOD details (3 seeds, trained 5 000 steps on 9x9 mazes, 10 420 parameters)

Re-run of 2026-10-12/13 with the paper's architecture (scalar 3x3 kernels per head, MLP width
E x K_heads x C; `docs/DEVIATIONS.md`, 2026-10-12, fidelity), official easy-to-hard mazes for
training, validation and every evaluation (`source: official` in the three `run.json` and the
27 result files), 6 h 30 of GPU in all. The first run of 2026-10-10 (dense-conv models,
11 824 parameters) is superseded and not kept in the repository; its numbers were
100.0 / 99.3 / 95.0% at 13 / 59 / 201.

- **Training** (`figures/F7_training_maze_ood.png`): validation exact match reaches 100% for
  the three seeds by step 2 000-2 500 of 5 000 (0.97 / 1.00 / 1.00 at step 5 000 for the old
  architecture, which also converged more slowly) and stays there; cell accuracy 1.000, no
  skipped steps; 13.5-14 min per seed on the RTX 4060 (0.16-0.17 s/step).
- **Size generalisation** (`figures/F1_maze_size_scaling.png`, 200 boards per size from 9 to
  59, 100 for the Table 1 sizes, K=1): 100% at every size up to 37x37 for the three seeds,
  100 / 98.5 / 97.5% at 59x59 (200 boards, D=2 300), 100 / 99 / 96% at 59x59 (Table 1 setting,
  D=2 000) and 100 / 83 / 74% at 201x201 (D=13 000). Mean ± std 100.0 ± 0.0 / 98.3 ± 1.7 /
  85.7 ± 10.8 against the paper's 100 at all three sizes. The failures are not near-misses:
  cell accuracy on 201x201 is 100 / 97.5 / 98.7%. So the paper's architecture trains faster
  and better in distribution, while its extrapolation to 15x the training width depends
  strongly on the seed (one seed matches the paper, two do not), whereas the dense-conv models
  were more uniform (92-97% each). With 3 seeds and 100 boards this spread is the finding;
  more seeds or more boards would be needed to say whether 100% is the typical outcome.
- **Training ablations** (`figures/F4_ablations_maze_ood.png`, 500 mazes of 13x13, D=300,
  8 trials per maze, the setting of the compute budget table in `docs/DEVIATIONS.md`; 12 extra trainings
  of 12-14 min). Exact accuracy, mean ± std over seeds (per seed in brackets), recomputed from
  `results/ablations/maze_ood_*.json`:

  | Variant | K=1 | conf@8 |
  |---|---|---|
  | full | 100.0 ± 0.0 (100 / 100 / 100) | 100.0 ± 0.0 |
  | no asynchronous firing | 99.9 ± 0.1 (100 / 100 / 99.8) | 100.0 ± 0.0 |
  | no perturbations | 99.3 ± 0.7 (100 / 99.6 / 98.4) | 98.3 ± 2.4 (100 / 100 / 95.0) |
  | no replay pool | 96.1 ± 4.3 (100 / 90.2 / 98.2) | 99.5 ± 0.8 (100 / 100 / 98.4) |

  The two columns tell different stories. With one trial (K=1) the replay pool is the
  ingredient that matters most. With 8 trials and confidence selection the no-replay models
  recover to 99.5% (their failures are trial-dependent and the confidence picks a good trial),
  while the no-perturbation models get *worse* than at K=1 on seed 2 (95.0% against 98.4%):
  without perturbations in training, wrong trials can be confidently wrong, which is the
  calibration role the paper gives to perturbations. Asynchronous firing has no visible effect
  at this scale. The paper's Table 2 measures these on the harder tasks.
- **Test-time noise** (`figures/F3_noise.png`, 13x13, D=300, K=8, 500 boards, sigma from 0
  to 1): flat at 100 / 99.8 / 100% for every sigma and every selection rule; the maze model is
  insensitive to the noise injected at test time at this size.
- **Adaptive compute** (`figures/F5_adaptive_compute.png`, 201x201, 20 boards, D=20 000,
  19 min per file): both firing rules are reported (`docs/DEVIATIONS.md`, confidence of given
  cells). Per step, the rule as written fires 0.75x the cells of the uniform rule and the
  `adaptive_given` rule 0.50x. Seeds 0 and 2 reach 95% solved under every rule (uniform at
  8 700 and 13 000 steps); seed 1 plateaus at 90% (18 of 20 boards) under all three, so the
  ratios are over 2 seeds. To 95%: `adaptive_given` needs 1.55x / 1.28x the steps and
  0.78x / 0.65x the cell updates of uniform (mean 1.41x and 0.72x; paper 1.29x and 0.71x);
  the rule as written needs the same steps and 1.16x / 0.97x the updates (mean 1.07x), i.e. no
  saving, because the given cells (64% of a maze) never become confident and keep firing at
  0.8. At D=20 000 all three rules end at the same accuracy per seed (100 / 90 / 90%), unlike
  the D=13 000 run of 2026-10-10 where the adaptive rules had not converged. With damage at
  step 6 000 (`maze201_damage`), only seed 0 reaches 95% again (uniform 13 800 steps; adaptive
  1.30x steps, 0.98x updates as written and 0.66x with `adaptive_given`); seeds 1 and 2 end
  at 90% under every rule. The paper's reference after damage is different (Fig. 8C: 0.94x
  steps and 0.52x updates, i.e. faster recovery than uniform): our single usable seed needs
  1.30x the steps, so the accelerated recovery is **not** reproduced. The 95% crossing is
  searched after the damage event only (a board solved before the damage and never repaired
  does not count; `experiments.recovery_point`, fourth review of 2026-10-13), which changes
  nothing in the published files (uniform crosses at 13 800 > 6 000). Those ratios are
  "total repair-and-solve" times since step 0, the wording of Fig. 8C; counted from the damage
  itself (uniform 7 800 steps, `*_since_damage_to_95` in the files) `adaptive_given` needs
  1.54x the steps and 0.77x the updates, so the conclusion holds under either reading.

  Caveats on the 0.72x (no damage): it is a ratio of *counted* cell activations, conditional
  on the two seeds that reach 95% (seed 2 reaches it and later drops back to 90%), and it is
  not a measured GPU saving. In these models the given cells (walls, endpoints) are clamped,
  so `adaptive` and `adaptive_given` produce exactly the same predictions step by step;
  changing the firing probability of a clamped cell changes the operation count, not the
  state. What is reproduced is the count the paper reports, under the given-cell rule; whether
  it translates into wall-clock time depends on skipping those cells, which the rollout engine
  does not do.
- **Animation** (`figures/anim/maze37.gif`, `maze37_strip.png`): a 37x37 maze solved by the
  seed-0 model, one frame every few steps.

### Sudoku-OOD details (3 seeds, 50 000 steps instead of the paper's 150 000, 159 760 parameters)

- **Training** (`figures/F7_training_sudoku_ood.png`): bf16, no gradient checkpointing, 0.59 s/step,
  8 h 10 per seed on the RTX 4060. Validation exact match (K=1, D=96) at 50K steps: 34.4 / 15.6 /
  22.2 % for seeds 0 / 1 / 2, cell accuracy 75 / 67 / 70 %. The spread between seeds is large and
  every seed is still improving slowly at 50K: seed 0 is at 0 / 6.7 / 16.1 / 22.8 / 30.0 % at
  10 / 15 / 20 / 25 / 30K steps, seed 1 at 0 / 0.6 / 2.8 / 7.8 / 11.1 %. The first exact solutions
  appear only after 10-15K steps, which is why our 50K-step models sit well below the paper's
  150K-step ones (the paper's Fig. 4A curves are also still rising late in training).
- **Depth and width scaling** (`figures/F2_tts_sudoku_ood.png`, 1 000 official RRN test boards,
  D=256, K up to 128): K=1 26.0 ± 6.8 % (paper 66.1 %), conf@K 33.6 / 40.2 / 46.7 / 52.4 / 57.2 /
  62.5 / 67.0 % at K=2 / 4 / 8 / 16 / 32 / 64 / 128 (paper 98.5 % at K=2048). The width-scaling
  trend of Fig. 4A is reproduced: every doubling of K adds 4-8 points (7.6 / 6.5 / 6.6 / 5.7 / 4.7 / 5.4 / 4.4) and the curve has not
  flattened at K=128. Confidence selection is nearly as good as an oracle here: on average conf@K is within one
  point of pass@K at every K (the largest single-seed gap is 2.8 points, seed 2 at K=128: 67.8
  against 70.6), so the confidence head almost always picks a correct trial when one exists. Depth: K=1 accuracy is 0 at D=16, reaches 90 % of its final value at D=120 and 95 % at
  D=160, then plateaus; D=256 is enough for these models.
- **By difficulty level** (number of empty cells, the 18 RRN levels from 47 to 64 empty cells,
  i.e. 34 down to 17 givens; K=128, mean over seeds): 96-99 % solved at 47-51 empty cells,
  89-92 % at 52-54, 79 % at 55-56, 59-64 % at 57-58, 42-48 % at 59-60, 10-32 % at 61-64. At K=1
  the same curve goes from 73 % (47 empty cells) to 0 % (63-64 empty cells). The gap to the
  paper is entirely on the hard levels: the easy half of the test set is solved.
- **Test-time noise** (`figures/F3_noise.png`, 500 boards, D=256, K=16, sigma from 0 to 1): K=1
  26.7 % at sigma 0 and 29-31 % for every sigma > 0; conf@16 55-58 % throughout. Noise at test
  time gives a small, flat gain of 2-4 points at K=1 and nothing at K=16, in line with the
  paper's claim that the models are robust to it. With the paper's default noise
  (sigma 0.15, `results/tts/sudoku_ood_noise_seed*.json`) the full scaling curve is unchanged
  (K=1 26.9 %, conf@128 68.0 %).
- **Training ablations** (Table 2): run at 5 000 steps per variant to fit the budget, see
  `docs/DEVIATIONS.md` (2026-10-12). At that budget every variant, the full recipe included, is at
  0 % exact and 22-24 % cell accuracy (chance level), so the runs carry no signal and no F4
  figure is drawn for Sudoku. The first exact solutions appear after 10-15K steps, so a
  meaningful ablation needs at least 20K steps per variant (3.3 h each on the RTX 4060).
  Budget decision (2026-10-12): the Sudoku ablations stop here; Table 2 is not reproduced for
  Sudoku in this replication.
- **Animation**: pending; needs a checkpoint copied from the PC (matplotlib is blocked there).

## Recipe search (21-24 October)

Goal: the best replication at our budget, combining the paper, the authors' public code and our
own choices, each difference measured rather than assumed. Protocol: arms compared on held-out
boards (RRN validation for Sudoku, a seed-7 maze sample for Maze-OOD), the winner confirmed on
seeds the search did not use and, for mazes, on 100 fresh official mazes per size that share no
maze with the Table 1 sample or the selection sample (`--load seed=11 --load exclude_seed=[0,7]
--load exclude_n=100`). The result files of these evaluations do not record that sampling (their
`table1: true` flag only says the Table 1 sizes and depths were used); the protocol is recorded
here and in `scripts/run_recipe_search.ps1`, from the PC's commands. Commands: `scripts/run_recipe_search.ps1`; result files under
`results/public/`, `results/search/`, `results/confirm/` and `results/ab/`. Figure:
`figures/F8_recipe_search.png`. Means ± population std over the seeds listed.

**Positive control first** (`scripts/import_public.py`, `results/public/`). The authors' published
weights in our model and evaluation scripts:

| Task | Their weights here | Paper |
|---|---|---|
| Maze-OOD 13 / 59 / 201, Table 1 sample, with the ring of walls | 100 / 100 / 100 | 100 / 100 / 100 |
| Maze-OOD, same sample, 9x9-cell format (no ring) | 0 / 0 / 0 | |
| Sudoku-OOD, 500 RRN validation boards, K=1 / conf@16 / conf@128 | 62.4 / 85.0 / 91.2 | 66.1 (test, K=1), 98.5 (K=2048) |
| Maze-Hard A*, 1 000 test mazes, K=1 / conf@16 | 79.1 / 81.8 | 86.6 / 89.2 (K=128) |

Our model, data loading and evaluation reproduce their Maze-OOD results exactly once their data
format is used. On Sudoku-OOD the 62.4% is on validation boards, not the paper's test set, so it
is close but not a like-for-like match; on Maze-Hard their weights give 79.1% against the paper's
86.6% on the same 1 000 test mazes, a 7.5-point gap we have not explained (data split, selection
or a remaining difference in our rollout). Most of what separates our trainings from the paper is
therefore the training recipe and its budget, but not necessarily all of it.

**Sudoku-OOD: the public-code recipe wins.** `configs/sudoku_ood_codepub.yaml` against
`configs/sudoku_ood.yaml`, same 20K steps, 1 000 RRN test boards, D=256 (`results/ab/`, seeds 0
and 1):

| Recipe | Steps | K=1 | conf@128 |
|---|---|---|---|
| Paper recipe | 20K | 8.0 ± 6.0 (14.0 / 2.0) | 24.5 ± 19.6 (44.1 / 4.9) |
| Paper recipe (3 seeds, main table) | 50K | 26.0 ± 6.8 | 67.0 ± 8.4 |
| Public-code recipe | 20K | 34.8 ± 2.0 (32.8 / 36.8) | 64.6 ± 0.2 (64.4 / 64.8) |

Leave-one-out from the public-code recipe at 10K steps, seed 0, 500 validation boards
(`results/search/sudoku_*.json`, K=1 / conf@16): full recipe 9.2 / 20.2; the update projection,
the LR schedule or the initialisation turned back to ours each give 0.0 (0.0 / 0.0, 0.0 / 0.0,
0.0 / 0.6), so these three are what the paper leaves out and the model needs. Our normalisation
6.2 / 17.8, sum loss 9.6 / 22.4, our EMA 9.4 / 20.6, our readout 9.2 / 20.2, our perturbations
11.8 / 28.2 and our betas (0.999) 20.2 / 34.8 move less or help at 10K. The betas were checked at
20K on seeds 0 and 1 (`results/confirm/`, 500 validation boards): K=1 36.7 ± 2.1 with the public
code's 0.95 and 35.8 ± 1.4 with 0.999, conf@128 66.8 ± 1.0 and 59.4 ± 0.6. 0.999 only speeds up
the take-off; the public recipe stays as is. The remaining gap to the paper (66.1 / 98.5) is the
150K-step training and K=2048, not run (about 19.5 h of training here; dropped on 2026-10-24 for budget).

**Maze-OOD: the authors' exact setup is the default.** The public code samples the maze images
with `[1:-1:2]`, which keeps a ring of given walls around each maze (9x9 cells become an 11x11
grid), and wraps the border. On the 9x9-cell format the wrap joins corridors of opposite edges,
which is what the first A/B measured (public-code recipe without the ring, Table 1 sample, seeds
0-2: 85.3 / 52.7 / 20.0 at 13 / 59 / 201, `results/ab/`). Confirmation on fresh mazes
(`results/confirm/`):

| Recipe | Seeds | 13x13 | 59x59 | 201x201 |
|---|---|---|---|---|
| Paper recipe (`maze_ood.yaml`) | 3 / 4 / 5 | 100.0 ± 0.0 | 96.7 ± 4.7 (90 / 100 / 100) | 83.3 ± 23.6 (50 / 100 / 100) |
| Public recipe, wall padding instead of the wrap (no ring) | 3 / 4 / 5 | 100.0 ± 0.0 | 82.7 ± 24.5 (100 / 48 / 100) | 47.0 ± 30.4 (69 / 4 / 68) |
| Public recipe on the ring (`maze_ood_codepub_ring.yaml`) | 1 / 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 97.5 ± 0.5 (98 / 97) |

On the seed-7 selection sample the wall-padding arm had given 100 / 100 / 100 on seeds 1 and 2;
on new seeds and new mazes it fails one seed in three, like the paper recipe: at 201x201 the
spread between seeds dominates every recipe difference except the ring. The ring setup is the
default because it is the authors' own and the best measured, with the caveat that it has two
seeds, not three. The paper-recipe numbers of the main table stay as published.

**Two false leads, corrected on the way** (`docs/DEVIATIONS.md`, 2026-10-23). (1) "The torus
destroys generalisation": it was the missing ring. (2) "A constant EMA lifts 201x201 from 71 to
96.5%": two runs of the same recipe already diverge on CUDA. With both EMAs and the raw weights of
the same trajectory (`train.ema_extra`, seeds 3-5, `results/search/maze_fact_*`), neither EMA
dominates (201x201, torch init: warm-up 42 / 100 / 100, constant 19 / 100 / 100, raw 2 / 95 / 93)
and the initialisation does not either (public init: 100 / 7 / 82 with the warm-up EMA). An EMA
can matter a lot on one seed (42 against 2% for the raw weights) and the raw weights can win on
another (93 against 82%): seed luck, not a rule.

**Maze-Hard: not settled.** Our runs have 10K steps against the paper's 20K; the public-code
recipe was not trained at that budget. `configs/maze_hard.yaml` stays the default.

## Figures

| Figure | Paper | File | Status |
|---|---|---|---|
| F1 maze size scaling | Fig. 4B, Table 1 | `figures/F1_maze_size_scaling.png` | done (Maze-OOD) |
| F2 depth / width scaling per task | Fig. 4A, 16, 17 | `figures/F2_tts_<task>.png` | done for sudoku_ood |
| F3 test-time noise sweep | Fig. 7B | `figures/F3_noise.png` | done (Maze-OOD and Sudoku-OOD) |
| F4 training ablations | Table 2 | `figures/F4_ablations_<task>.png` | done for maze_ood (13x13, D=300); sudoku_ood drawn but every variant is at 0% after 5K steps |
| F5 adaptive compute and damage | Fig. 8 | `figures/F5_adaptive_compute.png` | done at D=20 000 with both firing rules |
| F6 pruning Pareto front | Fig. 6A | `figures/F6_pruning.png` | pending |
| F7 training curves | - | `figures/F7_training_<task>.png` | done for maze_ood and sudoku_ood |
| F8 recipe search | - | `figures/F8_recipe_search.png` | done: Maze-OOD 201x201 per seed on fresh mazes, Sudoku-OOD conf@K per recipe |
| Animations | Fig. 2 | `figures/anim/*.gif` | done (maze37); sudoku pending (needs a checkpoint from the PC) |

## Journal

- 2026-10-10: pipeline validated on CPU (tests, a 300-step smoke run, the figure script on
  empty results). No GPU run yet.
- 2026-10-10: the 5 000-step Maze-OOD convergence check passed on a GitHub runner (CPU) and on
  the target GPU (val exact 0.97 at 5K steps, seed 0).
- 2026-10-10: Maze-OOD half of phase 1 on the target PC (RTX 4060 Laptop, 8 GB, CUDA): training
  0.17-0.18 s/step, three seeds, all evaluations, ablations, figures F1 / F3 / F4 / F5 / F7 and
  the animation; about 4.5 h of GPU time in all (ablations 2.5-3 h of it). Two fixes on the
  way: resuming a CUDA checkpoint (RNG state had to be restored on CPU) and the 201x201
  evaluations chunked with `--max-cells 500000` after the GPU dropped once. Adaptive compute
  falls short of the paper's ratios with the rule as written: see `docs/DEVIATIONS.md`,
  "Confidence of given cells"; the re-run with both rules at D=20 000 is queued behind Sudoku.
- 2026-10-11 to 2026-10-12: Sudoku-OOD half of phase 1 on the target PC. The first measured
  step time (1.2 s/step in fp32 with gradient checkpointing) made the plan 15x longer than
  estimated; bf16 autocast without checkpointing brought it to 0.59-0.65 s/step and the
  ablations were cut to 5K steps (`docs/DEVIATIONS.md`, 2026-10-11). Three seeds trained in
  24 h 30 of GPU time, then the nine evaluations (11 min each; the first K=128 evaluation ran
  out of VRAM and the three Sudoku evaluations now use `--max-cells 500000`). Results: K=1
  26.0 ± 6.8 %, conf@128 67.0 ± 8.4 %, conf@K within one point of pass@K on average. The 5K-step ablations came out at 0 %
  for every variant, so they are reported as such and not drawn. Windows Application Control
  started blocking matplotlib on the PC, so figures F2 / F3 / F7 for Sudoku were drawn from
  the pushed result files on another machine.
- 2026-10-12: external review of the branch (12 findings) answered. Fixed in code: the shared
  random stream of the generated Sudoku splits (train/test leak), the data provenance in
  `run.json` and in every result file, the memory bound of the phase-2 Sudoku-Extreme
  evaluations, the evaluation protocol of the maze ablations, the ARC training and evaluation
  path (second backward, output mask, metadata, checkpointing, resume), the pruning rollout
  (pruned trajectories no longer computed). Found on the way: the maze presets did not have the
  paper's perception and MLP width (10 420 / 41 284 parameters now, see `docs/DEVIATIONS.md`).
  The data source of the Sudoku-OOD trainings was confirmed from the PC's console logs
  (`official:satnet` for training, `official:rrn` for the test set).
- 2026-10-12 to 2026-10-13: Maze-OOD half of phase 1 re-run on the target PC with the paper's
  architecture (19:27 to about 02:00 Paris time, 6 h 30 of GPU): the 15 old runs and their
  result and figure files were archived automatically (`scripts/archive_stale_runs.py`), then
  3 trainings (13.5-14 min each), the size-scaling and Table 1 evaluations (21 min per seed),
  adaptive compute and damage recovery at D=20 000 with both firing rules (2 x 19 min per
  seed), the noise sweep, 12 ablation trainings evaluated at 13x13 / D=300, the animation and
  the figures (matplotlib worked again on the PC). Every `run.json` and result file says
  `source: official`. Results: 100.0 ± 0.0 / 98.3 ± 1.7 / 85.7 ± 10.8% at 13 / 59 / 201
  (seeds 100 / 83 / 74 at 201x201), ablations at K=1 full 100.0 / no async 99.9 / no perturb
  99.3 / no replay 96.1 (conf@8: 100.0 / 100.0 / 98.3 / 99.5), adaptive_given 0.72x the
  counted updates of uniform to reach 95% (paper 0.71x; 2 seeds, not a measured GPU saving)
  and 1.41x the steps (paper 1.29x), the rule as written 1.07x the updates; after damage
  1.30x / 0.66x on one seed against the paper's 0.94x / 0.52x (not reproduced). Numbers
  corrected after the third code review (2026-10-13); since-damage recovery times added to
  the three damage files on 2026-10-13 from their stored curves (fourth review, no re-run). The
  architecture fix
  made training faster and the in-distribution ablations cleaner, and the 201x201
  extrapolation more seed-dependent than with the dense-conv models (95.0 ± 2.2 before).
- 2026-10-14 to 2026-10-16: compute campaigns of the optimised engine on cloud L4 and H100 GPUs
  (timings on generated data, short exploratory recipes); they changed no published number and
  their raw files are not kept in the repository. The engine's RTX 4060 results are in
  `results/bench/pc_rtx4060/README.md`.
- 2026-10-21 to 2026-10-24: recipe search on the target PC (`scripts/run_recipe_search.ps1`,
  section "Recipe search" above): the authors' published weights run through our code (positive
  control), the Sudoku-OOD leave-one-out at 10K steps, the Maze-OOD leave-one-in, the wall ring
  found in their data format, the EMA and init factorial on one trajectory per seed, and the
  confirmations on new seeds and fresh mazes. Verdict: the public-code recipe is the default
  for Sudoku-OOD, the authors' exact setup (public recipe on the ring of walls) for Maze-OOD;
  the Sudoku 150K-step run was dropped for budget (2026-10-24).
