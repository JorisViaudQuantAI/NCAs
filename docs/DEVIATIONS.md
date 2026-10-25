# Deviations from the paper

A journal of every place where this repository departs from *Reasoning with Neural Cellular
Automata* (Etcheverry et al., 2026), with the reason and the expected effect. Dates are the
day the decision was made. Entries are never deleted; a reverted decision gets a new entry.

Legend: **budget** = forced by the 8 GB GPU / time budget, **data** = dataset not available
offline, **numerics** = needed to train stably, **unspecified** = the paper does not say and
we had to choose, **scope** = deliberately not replicated.

## Compute budget (2026-10-10, budget)

| Experiment | Paper | Here | Expected effect |
|---|---|---|---|
| Maze-OOD 9x9 | 5K steps, batch 64, N=100 | identical | none |
| Maze-Hard 30x30 | 20K steps, batch 32, N=256 (chunk 128) | 10K steps, same batch and N | lower absolute accuracy at D=200, same depth-scaling trend |
| Sudoku-OOD | 150K steps, batch 256, N=48 | 50K steps, batch 256 | lower K=1 accuracy on the hardest levels; conf@K trend expected to hold |
| Sudoku-Extreme | 200K steps, batch 800, N=96 | 30K steps, batch 128 | clearly lower accuracy; pruning comparison is relative so it still applies |
| ARC-AGI-1 | 400 tasks, C=512, 3.5M params, 650K steps, batch 128 | 30 tasks, C=128, about 0.3M params, 20K steps, batch 32, with RE-ARC extra demonstrations | proof of concept of task conditioning only; no comparable accuracy number |
| Test sets | full official test sets, K up to 2048 | subsets (100 to 1000 boards), K up to 128 | wider confidence intervals; three seeds keep the dispersion honest |
| Maze size scaling | sizes up to 201 with D up to 13K | same sizes, 20 to 200 boards per size | same |
| Training ablations | full budget per variant | Maze-OOD: 5K steps (the paper's), evaluated on 500 mazes of 13x13 at D=300, K=8; Sudoku-OOD: 5K steps, stopped (2026-10-12) | the maze ablations test generalisation to 13x13 only |
| Adaptive compute and damage | 201x201, D=13K | 201x201, D=20K, 20 mazes, both readings of the firing rule | ratios over the seeds that reach 95% |

The reduced step counts were chosen so that each phase script finishes in a night or a
week-end on the target PC (measured times in `docs/TUTORIAL.md`). Where the paper reports the model
still improving late in training, our numbers are expected to sit below the paper's.

## Datasets (2026-10-10, data)

- The official files are downloaded when the machine can reach them: easy-to-hard mazes
  (9x9 to 201x201), the HRM Maze-Hard csv, the SAT-Net / RRN Sudoku sets, the
  `sapientinc/sudoku-extreme` csv, ARC-AGI-1 and RE-ARC. When a download fails the loader
  falls back to a **local generator** with the same token format and prints which source it
  used; the source is recorded in the boards' metadata and in `run.json`.
- Generated mazes are perfect mazes from an iterative depth-first search with a random
  start/goal pair; the official Maze-OOD mazes are also perfect mazes but their generator and
  the endpoint sampling may differ slightly. Generated Maze-Hard mazes imitate the HRM set
  (30x30, extra openings, minimum path length) rather than copying it.
- The generated Sudoku-OOD test set is stratified by number of empty cells but **cannot
  reach the hardest RRN levels** (17 to 21 clues) in reasonable time; results on those levels
  exist only when the official RRN data was downloaded. The results tables say which.
- Sudoku-Extreme difficulty is the backtrack count of our own solver, not the rating of the
  original dataset; the bins are therefore only comparable in ordering.
- ARC synthetic tasks (three simple grid transformations) are used only when the official
  archive is unreachable, so that the pipeline can be tested; no ARC number is reported from
  synthetic data.

## Model and numerics

- **Normalisation epsilon (2026-10-10, numerics).** The per-cell group normalisation divides
  by `sqrt(sum of squares + 1e-4)` instead of the exact norm. With damage perturbations some
  cells are exactly zero, and the gradient of the exact norm at zero is unbounded; the loss
  turned NaN on the second training step. The output is unchanged for non-zero cells to
  within 1e-4 relative.
- **Non-zero initial update for normalised models (2026-10-10, numerics).** Presets with group
  normalisation (Sudoku, ARC) do not zero-initialise the last layer of the update MLP, so that
  a damaged (zero) cell leaves the singular point of the normalisation on its first update.
  Maze presets (no normalisation) keep the zero initialisation of the paper.
- **EMA warm-up (2026-10-10, unspecified).** The paper evaluates EMA weights with decay 0.999.
  With that decay the EMA is dominated by the random initial weights for the first few
  thousand steps, which makes short runs look worse than they are. We use the usual warm-up
  `decay = min(0.999, (1 + step) / (10 + step))`; after about 10K steps this is identical to
  the paper.
- **Parameter counts (2026-10-10, unspecified; resolved 2026-10-12).** Our Sudoku presets
  match Table 1 to 0.1%. The maze presets were first built with a dense 3x3 convolution and
  an MLP of width E x C (11 824 and 47 200 parameters against the paper's 10K and 41K); the
  2026-10-12 entry below explains the gap and the fix (10 420 and 41 284 now). The ARC preset is
  within 20%; the paper does not say whether biases are counted. Checked by
  `tests/test_model.py`.
- **Skipped steps (2026-10-10, numerics).** A training step whose gradient norm is not finite
  is skipped and counted (`skipped_steps` in `log.jsonl`). It never happened after the two
  fixes above; the counter is there so that it cannot happen silently.
- **Gradient checkpointing and bf16 (2026-10-10, budget).** On CUDA the BPTT chunk is
  checkpointed per step so that Sudoku-OOD at batch 256 fits in 8 GB; autocast to bf16 is
  optional (`train.amp`). Neither changes the computation in exact arithmetic.

## Evaluation protocol

- **Pruning as accounting (2026-10-10, budget; resolved 2026-10-12).** Until 2026-10-12
  Niche-Capped Diversity Pruning marked pruned trials as dead and stopped updating them, but
  the batch was not compacted on the GPU, so the measured wall-clock saving was nil (the
  external review measured a constant batch). Since 2026-10-12 pruned trajectories leave the
  batch (`rollout.py: rollout`, the `idx` map and the `store` helper): the model is applied to
  the survivors only, as in the paper's B.6, and `tests/test_rollout.py` checks the batch
  shrinks 16 -> 8 -> 4 -> 2. A pruned trajectory keeps its last state in `pred`, `conf` and the
  traces. `results/pruning/` still reports both the theoretical fraction of cell updates
  (`theoretical_fraction`, equation in `src/nca/pruning.py`, what Fig. 6 plots) and the measured
  fired cells per board (`compute_units`).
- **Confidence of given cells (2026-10-10, fidelity).** The input slice of a clamped (given)
  cell is pinned to its token embedding, which includes the output slice, so its L2 score is
  the constant 1/(1+1) = 0.5 for every output token. Board confidence keeps these cells (a
  constant offset that does not change conf@K rankings). For adaptive firing (paper 4.4) the
  paper's rule as written never lets a given cell cross tau = 0.95, so 64% of a maze keeps
  firing at 0.8 and the per-step saving is bounded at 0.82x; with the paper's reported 1.29x
  steps that could never give its 0.71x total updates. `eval_adaptive.py` therefore reports
  both rules: `adaptive` (the rule as written) and `adaptive_given` (given cells count as
  confident and fire at p_low, `RolloutConfig.given_confident`, the default in `rollout`).
  First GPU run with the rule as written (201x201, D = 13 000, 3 seeds): adaptive reaches 90%
  solved with 0.76x the updates of uniform but never 95% within D; with damage at step 6 000,
  45-85% against 95-100% for uniform. The phase-1 scripts now evaluate at D = 20 000 with both
  rules. Re-run of 2026-10-13 (paper's architecture, D = 20 000): `adaptive_given` reaches 95%
  with 0.72x the cell updates of uniform (paper 0.71x) and 1.41x the steps (paper 1.29x); the
  rule as written fires 0.75x the cells per step but needs 1.07x the updates in total, so it
  saves nothing. Both rules end at the same accuracy as uniform at D = 20 000. Ratios over 2
  seeds, the third plateaus at 90% under every rule; see `docs/RESULTS.md`. The ratios count
  cell activations: the given cells are clamped, so both rules give identical predictions and
  the saving is not a measured GPU time. After damage the paper's reference is 0.94x / 0.52x
  (Fig. 8C), not 1.29x / 0.71x; `scripts/aggregate.py` now uses it, and our one seed that
  recovers (1.30x / 0.66x) does not reproduce the faster recovery.
- **Chunked 201x201 evaluation (2026-10-10, hardware).** The Table 1 and adaptive evaluations on
  201x201 mazes pass `--max-cells 500000` (12 boards per pass instead of 49): a full-batch pass
  at D = 13 000-16 200 made the laptop GPU drop off the bus once. Chunking does not change the
  distribution of the results, but it does change the random draws: one generator serves a
  whole `rollout` call, so the firing masks and noise a given board receives depend on the
  pass boundaries (checked 2026-10-12: predictions differ, confidences by up to 0.02). The
  `max_cells` used is therefore stored in every result file since 2026-10-12, and a result is
  reproduced bit for bit only with the same value.
- **Depth for maze size scaling.** For sizes the paper does not list we use
  `D = 0.3 * size^2 + 20 * size`, which matches the paper's 300 / 2000 / 13000 at 13 / 59 /
  201 within 10%; the `--table1` option uses the paper's exact depths.

## Scope (2026-10-10, scope)

- **Visual Sudoku** (MNIST digits as input) is not replicated: it needs the SAT-Net image
  pipeline and adds nothing to the reasoning claims we care about.
- **ARC** is a proof of concept on 30 training tasks; the paper's full ARC run (3.5M
  parameters, 650K steps) is about 100 GPU-hours on an A100 and out of budget.
- **C++/CUDA kernels** are not written; the pure PyTorch implementation is the deliverable.
  The fast engine (`docs/FAST_ENGINE.md`) later added Triton kernels for the hottest operators.

## How to add an entry

One bullet or table row: date, tag, what changed, why, expected effect on which result.
Reference the file and the config key when there is one.

## Later entries

- 2026-10-10, **unspecified**: ARC candidates are selected by confidence (conf@K) like the
  other tasks, while the paper reports ARC with a majority vote over trials.
  `nca.rollout.majority_vote` exists; a `--select majority` option for `eval_tts.py` is on the
  list for phase 3.
- 2026-10-11, **budget**: Sudoku-OOD measured at 1.2 s/step on the target PC (RTX 4060 Laptop
  8 GB, batch 256, N=48, GPU at 100% without throttling), which put phase 1 Sudoku at about
  140 h instead of the 9 h planned. Two changes in `configs/sudoku_ood.yaml`:
  `train.grad_checkpoint: false` (the run uses about 1.7 GB of VRAM, so checkpointing only
  cost time: 0.98 s/step) and `train.amp: true` (bf16 autocast: 0.65 s/step with both; the
  loss matched the fp32 run to 4 decimals over the first 60 steps). Expected effect: none
  beyond bf16 rounding. The Sudoku ablations (`configs/ablations_sudoku_ood.yaml`) go from
  20K to 5K steps per run (chosen over 85 h for the full plan): the variants are
  compared with each other at equal budget, so the ranking is what Table 2 is about, but the
  absolute accuracies sit well below the main models'. Measured budget: main models
  3 x 50K steps about 27 h, ablations 15 x 5K steps about 14 h.
- 2026-10-12, **budget**: the 5K-step Sudoku ablations carry no signal. At 5K steps every
  variant, the full recipe included, scores 0 % exact and 22-24 % cell accuracy on 500 RRN
  boards (chance level), because a Sudoku-OOD model produces its first exact solutions only
  after 10-15K steps (validation of the main runs: seed 0 at 0 / 6.7 / 16.1 / 22.8 / 30.0 %
  exact after 10 / 15 / 20 / 25 / 30K steps; seed 1 at 0 / 0.6 / 2.8 / 7.8 / 11.1 %). The 5K
  results are kept in `results/ablations/sudoku_ood_*.json` as a record but no F4 figure is
  drawn for Sudoku and the ranking of Table 2 is not reproduced at this budget. A meaningful
  comparison needs at least 20K steps per variant (3.3 h each at 0.59 s/step, 17 h for one
  seed of the five variants, 50 h for three seeds); which of these to run is a budget decision
  recorded in `docs/RESULTS.md` once taken.
- 2026-10-12, **unspecified**: Sudoku-OOD evaluation at K=128 with D=256 on 1 000 boards
  needed 3.8 GB more VRAM than the 8 GB available in one pass; the three Sudoku evaluations in
  `scripts/run_phase1.*` run with `--max-cells 500000` (trials are processed in chunks, results
  identical, 11 min per evaluation).
- 2026-10-12, **tooling**: Windows Application Control on the target PC blocks matplotlib's
  `ft2font` extension, so `scripts/make_figures.py` and `scripts/animate.py` cannot run there
  any more. Figures are drawn on another machine from the pushed `results/` and `runs/` files
  (no checkpoint needed); animations need a `final.pt` copied from the PC. No effect on the
  numbers.
- 2026-10-12, **fidelity** (found while answering an external review of the branch): the Maze
  presets did not have the paper's architecture. Appendix B.1 writes the convolutional
  sensing as z^(k) = sum_n W_perc,n^(k) x_n with one scalar W per neighbour and head (the
  learned version of the Sobel filters of Mordvintsev et al.), so a head costs 9 parameters,
  and the update MLP "expands the perception vector by E", i.e. its hidden width is
  E x K_heads x C. With those two readings the counts are 36 + (64 x 128 + 128) +
  (128 x 16 + 16) = 10 420 for Maze-OOD and 36 + (128 x 256 + 256) + (256 x 32 + 32) = 41 284
  for Maze-Hard, which round to the paper's 10K and 41K; no other reading of B.1 gives both.
  Our presets used a dense C -> K_heads x C convolution (9 216 and 36 864 parameters in the
  perception alone) and a hidden width of E x C (32 and 64), i.e. a more expressive
  perception and a narrower MLP, 11 824 and 47 200 parameters in all. Fix: `perception:
  kernels` (`model.KernelPerception`) and `hidden_from_perception: true` in both maze presets;
  `conv` stays available and the old checkpoints still load as `conv` models through their
  `run.json`. Every Maze-OOD number published before this date (Table 1 sizes, ablations,
  noise sweep, adaptive compute, animation) comes from the dense-conv models and is labelled as
  such in `docs/RESULTS.md`; the Maze-OOD half of phase 1 (about 4.5 h of GPU) is to be re-run
  with the paper's architecture. Expected effect: unknown a priori; the dense-conv model solved
  95-100% of the Table 1 sizes, the paper's reaches 100%. Outcome (re-run of 2026-10-12/13,
  6 h 30 of GPU, old runs superseded): training converges
  faster (100% validation by step 2 000-2 500 for the three seeds instead of 0.97-1.00 at
  5 000), 13x13 and 59x59 are at 100.0 and 98.3%, but 201x201 drops to 85.7 ± 10.8%
  (100 / 83 / 74% per seed, against 92-97% per seed before). The adaptive-compute ratio of
  Fig. 8 is reproduced with this architecture (above). The seed dependence of the 201x201
  extrapolation is reported as such in `docs/RESULTS.md`, not averaged away.
- 2026-10-12, **data** (external review, P1): the Sudoku generator derived every split from the
  same seed, and board k of any split starts from the k-th child of that seed, so the generated
  test split shared most of its solution grids with the generated train split (10 of 12 in a
  check with the default settings) and the generated Sudoku-Extreme test puzzles were often
  identical to train ones (7 of 12). Fix in `tasks/sudoku.py: split_seed`: one stream per
  (benchmark, split, seed); `tasks/base.py: overlap_counts` counts shared puzzles and solutions
  and `train.py` records the train/val overlap in `run.json` (a warning is printed when it is not
  zero); `tests/test_sudoku.py` checks both. The official loaders (SAT-Net, RRN, HRM) were
  never affected. The Sudoku-OOD runs of 2026-10-11/12 were evaluated on the official RRN test
  set (`source: official:rrn` in every result file); which source their training used is not in
  the files pushed from the PC, since `run.json` did not record it until today: see
  `docs/RESULTS.md` for the check.
- 2026-10-12, **provenance** (external review, P1): `run.json` now stores the data block
  (source, size and shape of the train split, source and size of the validation split, the
  train/val overlap) and every result file written by `nca.experiments` stores `source`,
  `n_boards` and `max_cells`. For the scientific runs the configs should say `source: official`
  rather than rely on the automatic fallback to the generators.
- 2026-10-12, **protocol** (external review, P2): the Maze-OOD ablations were evaluated on the
  config's default test set, i.e. 9x9 mazes at D = 2 x N = 200 with K = 8 on 500 boards, while
  the docs announced 13x13 at D = 300. The published ablation numbers are therefore in
  distribution (`docs/RESULTS.md` says so now). `run_ablations.py` takes `--load size=13`
  and `--eval-depth 300`, records the board shape and the loading options in each result file,
  and `scripts/run_phase1.*` pass them; done in the Maze re-run of 2026-10-13 (ablation
  models re-trained with the paper's architecture and evaluated on 500 13x13 mazes at D = 300:
  full 100.0, no async 99.9, no perturb 99.3, no replay 96.1 ± 4.3%).
- 2026-10-12, **budget** (external review, P1): the Sudoku-Extreme evaluations of phase 2 are
  chunked with `--max-cells 200000` (38 boards per pass at K = 64) in `scripts/run_phase2.*`:
  with the default 2 000 000 cells the state tensors alone (C = 256, 24 640 trajectories) would
  take about 10 GB.
- 2026-10-12, **budget**: the Sudoku-OOD ablations stop at the 5K-step runs already made (a budget
  decision of 2026-10-12, after the finding above that they carry no signal); Table 2 for Sudoku
  is not reproduced in this replication.
- 2026-10-12, **budget**: ARC activation memory. `configs/arc_small.yaml` (C=128, batch 32,
  32x32 canvas, chunk = rollout = 64 steps under BPTT) keeps about 24 state-sized fp32 tensors
  per step alive for the backward pass (the SwiGLU block with E=4 alone holds 16: its 2EC-channel
  pre-activation, gate and product; the padded state, value projection, mixed perception, dx,
  pre- and post-normalisation states make up the rest). One state is 32 x 128 x 1024 x 4 B =
  16 MiB, so about 380 MiB per step and about 24 GiB for the chunk, three times the 8 GB GPU.
  The old `grad_checkpoint: auto` rule ignored H x W (it counted chunk x batch x channels, which
  put arc_small below the Maze-OOD level); it now counts cells (`nca.train.auto_grad_checkpoint`,
  threshold 2**27 state elements: Maze-OOD and Maze-Hard stay unchecked, Sudoku-Extreme and
  arc_small are checkpointed, Sudoku-OOD keeps its explicit `false`), and `configs/arc_small.yaml`
  sets `train.grad_checkpoint: true` explicitly. With per-step checkpointing the chunk holds the
  64 step inputs (1 GiB) plus one recomputed step (380 MiB), so batch 32 and chunk 64 are kept;
  the measured peak will be recorded in `docs/RESULTS.md` with the first ARC run. Expected
  effect: none on the numbers (checkpointing recomputes the same arithmetic), about 20% more
  time per step.
- 2026-10-12, **unspecified**: ARC step embedding and candidate selection. `NCA.step` now
  follows paper eq. B.3-(1) literally: the hidden slice of the state receives the learned step
  embedding before sensing and keeps it (`x_hid <- x_hid + e_step(t)`, then the residual
  update); until now the embedding was added to a copy used for sensing only, so the state never
  carried it. The paper does not say what happens to a cell that does not fire; since at
  inference "its state and counter stay frozen", the shift is gated by the firing mask like the
  update. Only models with `time_steps > 0` (ARC) are affected; no ARC run had been trained
  before this change (ARC training crashed at the second step, see the pool fix of the same
  day), so no result changes. The majority vote of the paper (B.4) is available as
  `scripts/eval_tts.py --select majority` (`nca.experiments.majority_pass_at_m`: trials mapped
  back to the canonical frame, pass@1 / pass@2 of the most-voted grids); `--select conf` stays
  the default and the only option for Maze and Sudoku.
- 2026-10-13, **fidelity** (second review of 2026-10-13, point 3): with group normalisation
  (Sudoku, ARC) every cell was renormalised at every step, so a cell that did not fire still
  changed (up to 0.59 on the reduced ARC model, from the Gaussian initial state), against the
  paper's "a non-firing cell stays frozen". New model option `norm_fired_only`: only cells that
  fired or received noise are normalised. On for the presets not trained yet (`sudoku_extreme`,
  `arc`); **off for `sudoku_ood`**, whose published runs were trained with the old behaviour,
  and for every checkpoint saved without the key (so their evaluations are unchanged). Size of
  the old effect, measured on the `sudoku_ood` preset: the normalisation is idempotent, so a
  non-firing cell moved only when its state was off the unit sphere, i.e. at the first step
  (the Gaussian initial state, max change 0.21 per channel with sigma_init = 0.15) and right
  after a noise injection; from the second step on a non-firing cell changed by less than
  1e-4 (the eps of the norm). The old dynamics therefore amount to "every cell is projected
  onto the sphere at step 1, and noisy cells are re-projected", not to a drift of frozen cells
  throughout the rollout. A Sudoku-OOD retrain with the option on (about 26 h of GPU for 3
  seeds, 8 h for one) is still the only way to measure its share of the gap to the paper
  before attributing the whole of it to the 50K-step budget.
- 2026-10-13, **fidelity** (point 1): `configs/arc_small.yaml` no longer uses online colour
  permutations and dihedral symmetries. They kept the task id while some ARC rules depend on
  specific colours, which gave the same input two different targets (6 of the 30 selected
  tasks, e.g. `08ed6ac7`). The paper gives such variants their own id; online scale / offset
  is kept.
- 2026-10-13, **protocol** (point 4): the Sudoku-Extreme test sample takes about the same number
  of boards in each of the 7 difficulty bins, which over-weights the rare hard bins in a plain
  mean. Each sampled board now carries a weight (its bin's share of the benchmark / boards
  sampled in that bin); `tts_curves` reports `conf_at_k_weighted` and the headline table uses
  it, while the per-bin curves keep the balanced sample. `stratify` is now part of the cache
  name (`..._official_strat_v2.npz` / `..._uniform.npz`).
- 2026-10-13, **hardware** (point 2): the phase-3 ARC evaluation (31 test grids x K=32) passes
  `--max-cells 100000`; peak VRAM to be measured on the first run.
- 2026-10-13, **fidelity** (fourth review of 2026-10-13, point 1): the optimiser step, the
  schedule and the EMA ran at the end of the differentiable chunk, so with `chunk < rollout`
  the steps after the chunk used the updated weights and the states written back to the replay
  buffer mixed two sets of parameters. B.2 unrolls the whole rollout with the current
  parameters and updates them afterwards. `train.py` now keeps the backward at the end of the
  chunk (to free its activations) and applies the update after the last step. Only
  `maze_hard` (N=256, chunk 128) is affected; the published Maze-OOD (100/100), Sudoku-OOD
  (48/48) and ablation runs had `chunk = rollout` and are unchanged. The three Maze-Hard A*
  models (trained 2026-10-13/14) already use the corrected order.
- 2026-10-13, **protocol** (point 2): the recovery after damage looked for the first 95%
  crossing over the whole curve, so a set of boards solved before the damage and never
  repaired would have reported the pre-damage crossing. `experiments.recovery_point` searches
  after the event only, reports both the total time since step 0 (`steps_to_95`,
  `updates_to_95`, the "total repair-and-solve time" of Fig. 8C) and the time elapsed since
  the damage (`steps_since_damage_to_95`, `updates_since_damage_to_95`), and a stored curve
  that stops before the event, or whose crossing predates it, is recomputed instead of reused.
  The three published damage files did not contain a premature crossing (uniform crosses at
  13 800 > 6 000); their since-damage fields were filled from the stored curves.
- 2026-10-13, **fidelity, measured**: effect of `norm_fired_only` on Sudoku-OOD, one seed
  (seed 0, same recipe, `--set model.overrides.norm_fired_only=true`,
  `runs/sudoku_ood_normfired/seed0`, `results/tts_normfired/sudoku_ood_seed0.json`; 1 000
  official RRN boards, D=256). K=1 32.7% against 33.6% for the published seed 0, conf@128
  78.6% against 76.8%, cell accuracy 74.6% against 74.5%; per-level differences of a few
  boards both ways. Both are well inside the seed-to-seed spread (K=1 26.0 ± 6.8 over three
  seeds). The old normalisation is not what separates us from the paper's 66.1% / 98.5%; the
  50K-step budget (a third of the paper's) remains the main suspect.
- 2026-10-13, **budget**: Sudoku-Extreme measured on the RTX 4060 Laptop at 2.27 s/step (fp32,
  automatic gradient checkpointing), i.e. 19 h per seed for 30K steps. As for Sudoku-OOD,
  `configs/sudoku_extreme.yaml` now sets `train.amp: true` and `train.grad_checkpoint: false`:
  1.22 s/step, 5.1 GiB peak, same loss as fp32 over the first 30 steps (10 h per seed).
  Maze-Hard stays in fp32: bf16 was slower there (0.79 vs 0.72 s/step) and its loss exploded
  in the first steps; fp32 peaks at 5.9 GiB, which fits.
- 2026-10-13, **extension** (fast engine, `docs/FAST_ENGINE.md`): a second execution path for
  training, `scripts/train_fast.py` / `nca.engine`, built to measure how much faster the paper's
  recipe can run on the 8 GB GPU at equal model and protocol. It changes nothing in the
  published runs (which keep `scripts/train.py`); its runs go to `runs/engine/`. Two
  departures from `nca.train.train`, which apply to the engine's reference backend as much as
  to its optimised one: (1) the firing masks and the noise of a step are regenerated from a
  per-step seed (`engine.rng`), with no CPU branch; the distribution is the reference's but
  the random stream is not, so an engine run and a `train.py` run of the same seed are not
  bitwise comparable; (2) `train.grad_checkpoint` is replaced by `engine.checkpoint_segment` (segments of q steps
  with the random draws regenerated during the recomputation, `auto` picks the fastest that
  fits the VRAM budget). The update order is the corrected one of the entry above (the
  optimiser and EMA update after the last step of the rollout); `engine.update_after_rollout:
  false` reproduces the order of the runs published before this date, for A/B tests only.
  Expected effect on the numbers: none beyond fp32 / bf16 rounding,
  to be verified on the GPU by `scripts/check_engine.py` and by the three-seed comparison
  described in `docs/FAST_ENGINE.md`.
- 2026-10-14, **H100 performance estimates**: a cloud campaign timed short, isolated windows at
  the paper's main training dimensions, plus inference and ARC TTT, with the engine's
  `reference` backend as comparator (not `train.py` or the authors' JAX code). Repeated
  processes on seed 0, not three trained seeds; no RTX speed is inferred from these timings,
  and the raw files are not kept in the repository.
- 2026-10-14, **ARC/Visual compute fixtures**: the H100 ARC fixture uses C512, B128,
  650,000 projected iterations and 400 synthetic task IDs, without official ARC/RE-ARC
  pretraining. It has 3,779,468 parameters versus the paper's reported 3.5M; this fidelity
  difference remains unresolved. Visual uses 256x256/C256/N1024 and a hypothetical B1,
  generated font glyphs, fresh states, no replay/damage and no MNIST classifier. Its
  93,000-iteration projection is not the complete official experiment. TTT timing uses
  51 augmented synthetic demonstrations and initialized shared weights, not a pretrained
  checkpoint. Official-data scores and long-run convergence remain untested by this campaign.
- 2026-10-14, **execution restriction**: an H100 ARC TTT numerical gate exposed overwritten
  CUDA Graph gradient buffers when accumulating microbatches. That candidate was not timed
  or recommended. `EngineConfig` now rejects microbatching with the CUDA Graph compilation
  modes and directs callers to ordinary compilation (`compile_mode: null`), which passed
  the full-horizon GPU gate. This guard changes no computation in the retained presets;
  its regression coverage is CPU-only.
- 2026-10-19, **bug found, BFS runs invalid**: the three Maze-Hard BFS trainings never solved a
  maze (validation exact 0% from step 1 000 to 10 000, valid paths ~0%, cell accuracy 93.5%,
  about what "no path" gives). Cause: the BFS target is *one* shortest path chosen by a fixed
  neighbour order (up/down/left/right, `maze.bfs_path`), which depends on the orientation of
  the maze exactly like the A* tie-break, while `MazeHard.augment` applies the 8 dihedral
  symmetries to BFS boards. Every official Maze-Hard maze has several shortest paths (200/200
  checked) and after a 90-degree rotation the rotated target equals the BFS path of the
  rotated maze for only 89/200 mazes, so the training targets contradict each other. These runs
  and their 6 evaluations are discarded and not reported. Fixed: Appendix A.1 says the
  8 dihedral transformations of the BFS boards are "computed on the fly" and scores `exact`
  against "the provided BFS solution", so `MazeHard.augment` now recomputes the BFS target on
  each transformed maze (`maze.bfs_target`) instead of rotating the stored one; the test boards
  are not augmented and keep their targets. `tests/test_maze.py::test_hard_bfs_augment` checks
  every augmented target against the BFS path of its own board (29 of 64 rotated targets were
  wrong under the old code). About 10 ms per 32 fresh boards on the CPU.
- 2026-10-19, **BFS still not learned after the augmentation fix**: the three BFS seeds re-run
  with targets recomputed on each transformed maze end exactly like the first attempt
  (validation exact 0% from step 1 000 to 10 000, cell accuracy 93.5-93.7%, the loss curve
  step for step on top of the buggy run's), while A* solves 27% at step 3 000 and 68% at
  10 000. The augmentation bug was real but is not what blocks learning. Our BFS target
  follows the "first discoverer" parent pointers of a BFS from the start in up/down/left/right
  order: which of the several shortest paths it picks depends on the global discovery order,
  a non-local rule. The paper does not say how its BFS breaks ties (A.1: "BFS-generated
  targets", 8 dihedral transformations on the fly) and reports BFS above A* (94.7 vs 86.6),
  which suggests a local tie-break. These runs are discarded; the evaluations were stopped (they could only give
  0%). Next: a local deterministic BFS target, checked on a short training before three seeds.
- 2026-10-19, **local BFS target (`targets: bfs_local`) and short probes**: `maze.bfs_local_path`
  computes BFS distances from the start, then walks back from the goal, each step going to the
  first neighbour (up/down/left/right) one step closer to the start. Each cell's choice depends
  only on its four neighbours, unlike the first-discoverer pointers of `bfs_path`. The two rules
  pick different shortest paths on 60 of 100 generated hard mazes. Config
  `configs/maze_hard_bfs_local.yaml`; the targets are recomputed on every augmented maze. The
  final eval of the stopped BFS seed 0 also gives 0% *valid* paths, so the model does not draw
  another shortest path, it draws no path at all. `scripts/probe_bfs.ps1` runs two seed-0
  probes stopped at step 3 000 by the new `train.stop_at` key (the LR schedule still follows
  the 10 000 steps, and resuming without it finishes the same run): the local target with
  augmentation, and our first BFS target without augmentation (as for A*). If only the second
  learns, the blocker is the augmentation rather than the tie-break.
- 2026-10-19, **BFS probes (seed 0, 3 000 steps)**: neither probe learns. `bfs_local` with
  augmentation: exact 0 / 0 / 0% at steps 1 000 / 2 000 / 3 000, cell accuracy 79.9 / 90.1 /
  92.4%; the first BFS target without augmentation: 0 / 0 / 0%, 80.0 / 90.2 / 91.9%. A* at the
  same steps: 0 / 0.8 / 27.3%, 80.0 / 94.0 / 96.8%. So neither the augmentation nor the
  global tie-break explains the failure: the three target sets have the same inputs, the same
  path lengths (113.1 cells), no given cell overwritten, and differ from A* on 35 path cells per
  maze on average. Remaining hypotheses: the BFS paths need a longer chain of local decisions
  (distance field, then a walk back from the goal) than the 256-step rollout / 128-step BPTT
  window allows; or the paper's BFS target is something else (it is not specified). Maze-Hard
  BFS is reported as not reproduced until the authors' definition or code is found.
- 2026-10-19, **what the A\* and BFS targets are**: on the official data, the "A\*" targets are
  the solution paths shipped in the HRM CSV (`parse_hrm_maze`), not our `astar_path`; only the BFS
  targets are computed by us (`convert_official_hard`). The paper scores BFS against "the provided
  BFS solution" but neither ships it nor says how it breaks ties between shortest paths. So the
  learnable A\* targets follow whatever rule produced the HRM paths, and none of our BFS rules
  matches what the paper trained on. BFS rows in README and RESULTS read "not reproduced".
- 2026-10-19, **cause found, BFS fixed**: the Maze-Hard paths shipped with the HRM data (our
  "A*" targets, the only ones the model learned) are exactly a BFS from the start that expands
  neighbours down/left/right/up (100% identical on the first 300 training mazes; every other
  order gives 6% or less). Every target we computed used up/down/left/right and none was
  learned by step 3 000: our BFS with and without augmentation, `bfs_local` with and without
  augmentation, and our own A* without augmentation (0% solved, cell accuracy 92-93%), while
  the HRM paths reach 27%. New target `bfs_dlru` (`maze.bfs_path(order="DLRU")`): equal to the
  HRM paths on the original mazes, recomputed on each of the 8 dihedral transformations (our
  reading of the paper's "BFS-generated targets" with on-the-fly augmentation, A.1). Probe with
  augmentation, seed 0: 0 / 0 / 27.3% solved at steps 1 000 / 2 000 / 3 000 (cell accuracy
  79.9 / 93.6 / 97.3%), the same curve as A*. Why the expansion order matters this much is not
  established; the HRM maze generator probably leaves a structure aligned with it. The phase-2
  scripts now train `configs/maze_hard_bfs_dlru.yaml` for the BFS rows (the probe is reused as
  seed 0 and resumed from step 3 000).
- 2026-10-20, **the authors' public code**: the authors published their JAX code and trained
  weights (github.com/paradigms-of-intelligence/reasoning-nca, commit a34414f: notebooks plus
  `checkpoints/<Task>/{config.json, ema_params.pickle}` for Maze-OOD, Maze-Hard, Sudoku-OOD,
  Sudoku-Extreme and ARC). Their code differs from the PDF in several places, and from this
  repository in more. We do not take either as the reference: each difference is a switch,
  off by default (earlier runs and checkpoints are unchanged), to be compared on matched runs.
  Model side (`NCAConfig`): `embedding: semantic` (maze tokens on fixed points of the readout
  space, every other entry -1; the PDF describes orthogonal embeddings) with a readout over the
  whole vocabulary (`output_tokens`); `project_update` (Sudoku: `dx <- dx - <dx, x> x` per
  group before the residual; not in the PDF); `noise_after_norm`; `norm_init`;
  `norm_style: norm_eps` (`x / (|x| + 1e-6)`); `init_style: public` (both MLP layers
  std 1e-3, fixed attention std 1/3, value std 1/sqrt(C)). Their Maze-OOD model also wraps the
  border (`padding: wrap`) where ours pads with the wall embedding. Training side: see the
  `train` keys `betas`, `init_lr_ratio`, `loss_reduce`, `ema_warmup`, `damage.fill` /
  `damage.geometry`, `batch_noise`, `swap_source`, `chunk_start`, `writeback` in
  docs/PAPER_MAP.md. `nca.public_ckpt.load_public` builds their four single-layer models with
  these switches and their weights, and `scripts/import_public.py` writes them as run folders
  for our evaluation scripts. Parity with their JAX `nca_step` / `nca_pred` on their weights,
  same states and firing masks, 60 steps: state error 2e-7 (mazes) to 2e-4 (Sudoku-OOD,
  float32 JAX against float64 here), identical argmax everywhere. With their embeddings a
  given maze cell has confidence 1, which settles "Confidence of given cells" without a
  special rule.
- 2026-10-20, **corrections**: (1) the normalisation epsilon entry above holds for cells of
  norm about 1, not near zero: a cell of norm 0.01 comes out with norm 0.707 here, 0.9999 with
  the public code's formula. (2) "every other order gives 6% or less" (2026-10-19) is wrong
  for DRLU, which an external review measured at 931 / 1 000 identical training paths (905 on
  test); DLRU is still the only order that matches every path. (3) `train.stop_at` was ignored
  by the fast engine; fixed.
- 2026-10-20, **positive control on generated mazes (CPU)**: the public Maze-OOD weights, run
  through our rollout, solve 15 / 15 generated 9x9 mazes and 13 / 13 generated 13x13 mazes whose
  endpoints sit on lattice cells (odd coordinates), but only 40% and 30% of those whose
  endpoints sit on a corridor between two lattice cells, which our generator allows. Whether
  the official mazes ever place an endpoint there is to be checked on the PC (the data was not
  reachable from the development machine). The official test sets are evaluated on the PC
  (`scripts/run_public_ab.ps1 public`); published numbers never use generated mazes.
- 2026-10-23, **the authors' Maze-OOD data has a ring of walls** (positive control of their
  weights, PC): their loader samples the maze images with `[1:-1:2]`, which keeps one border
  pixel per side, so a size-9 maze is an 11 x 11 grid surrounded by given walls; ours keeps the
  9 x 9 cells. With that ring their `padding: wrap` only joins wall to wall; without it,
  opposite corridors become neighbours. Their Maze-OOD weights in our code: 0 / 32 exact on
  13 x 13 without the ring, 32 / 32 with it (`task.load(..., wall_ring=True)`, `--load
  wall_ring=true`). So the Maze-OOD arms of our search that use wrap padding on ring-less grids
  (the public-code recipe, 85 / 53 / 20% on 13 / 59 / 201, and the "wrap" arm) measured this
  format mismatch, not their recipe. Their Sudoku-OOD weights give 62.4% at K=1, 85.0% conf@16,
  91.2% conf@128 on 500 RRN validation boards (paper 66.1% K=1 on test; 98.5% is K=2048), their
  Maze-Hard weights 79.1% at K=1 on the 1 000 test mazes (`results/public/`). Scope of the
  parity entry of 2026-10-20: the forward pass on the checked states and horizons (60 steps,
  2e-7 mazes, up to 2e-4 Sudoku); data loading, selection, metrics and the training step were
  not part of it.
- 2026-10-23, **recipe search, protocol corrections** (external review): (1) two CUDA runs of the
  same recipe and seed already end with different weights (cuDNN is not set deterministic), so a
  comparison of two runs that differ only by the EMA does not measure the EMA; `train.ema_extra`
  keeps several EMAs of one trajectory and `final_raw.pt` the trained weights. On seeds 3 / 4 /
  5 with both EMAs of the same run, the constant EMA does not beat the warm-up EMA (201 x 201:
  19 vs 42%, then equal); the 71 -> 96.5% of the first search was run-to-run variance. (2) The
  selection mazes (`--load seed=7`) are a resample of the same official test files: 1, 1 and 10
  of the 100 mazes at 13, 59 and 201 are also in the seed-0 sample used for Table 1. (3) The
  Sudoku probes set `train.steps=10000`, which also rescales the warm-up and the cosine: they
  compare recipes made for 10K steps, not prefixes of longer runs (`train.stop_at` gives
  prefixes).
- 2026-10-24, **recipe search, verdict and defaults** (`docs/RESULTS.md`, "Recipe search").
  Paper vs public code vs our choice, as measured. Sudoku-OOD: the public code's update
  projection, LR schedule (10x the peak down to the peak, then constant) and initialisation are
  not in the paper and each is needed (0% at 10K steps without any one of them); the public-code
  recipe gives 34.8 ± 2.0% at K=1 and 64.6 ± 0.2% conf@128 on the RRN test in 20K steps (paper
  recipe 8.0 ± 6.0 / 24.5 ± 19.6 at 20K, 26.0 ± 6.8 / 67.0 ± 8.4 at 50K); our AdamW betas 0.999
  only help before 10K steps. Default: `configs/sudoku_ood_codepub.yaml`. Maze-OOD: the paper
  recipe and the public recipe on the 9x9-cell format both leave about one seed in three far
  below 100% at 201 x 201 on fresh mazes; the public recipe on the authors' 11 x 11 format with
  the ring of walls gives 98 and 97% (seeds 1, 2). Default: `configs/maze_ood_codepub_ring.yaml`,
  two seeds only. Maze-Hard: not compared at the same budget, `configs/maze_hard.yaml` stays.
  EMA policy and initialisation on mazes: no effect beyond seed variance. The paper-recipe
  configs are kept unchanged and the published 3-seed table still comes from them. The 150K-step
  Sudoku run (about 19.5 h) was dropped for budget on 2026-10-24, so the remaining Sudoku gap is
  attributed to training length and K as a hypothesis. The confirmation runs were launched by
  hand on the PC; `scripts/run_recipe_search.ps1 confirm` rebuilds them from the run folders and
  result files.
