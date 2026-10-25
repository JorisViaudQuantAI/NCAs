# Fast engine: training Reasoning NCAs under an 8 GB VRAM budget

An experimental extension of the replication: the same NCA, the same training recipe and the
same evaluation protocol, executed by an engine that exploits the local, recurrent structure
of the cell update to cut the real cost of training on one consumer GPU. The scientific
question is *how much faster can the paper's training run at equal model and protocol*, and
the answer is measured, not assumed: every optimisation is validated against the reference
implementation at the operator, trajectory and training levels, and benchmarked separately
before being combined.

Status: the engine was developed and checked for equivalence on a cloud L4 GPU, then measured
on the target RTX 4060 Laptop against the published `scripts/train.py` runs (same GPU, same
update counts, three seeds): see [results/bench/pc_rtx4060/README.md](../results/bench/pc_rtx4060/README.md).
The published runs, results and figures of the replication were produced by `scripts/train.py`.

## The contract between the two versions

The reference is `nca.model.NCA.step` run by the engine's loop; the optimised version is the
`fused` backend, optionally compiled. Both share, by construction:

| Shared | How |
|---|---|
| parameters, initialisation, checkpoint format | the engine only re-implements the *execution* of `NCA.step`; `nn.Module` classes, parameter names, `final.pt` and `checkpoint.pt` are those of `nca.train` |
| data, augmentations, batches, replay states | `ReplayPool`, `apply_damage`, `swap_indices` of the replication, unchanged |
| horizons, supervision window, optimiser, EMA, schedule | `resolve_train_cfg` reads the same `train:` section as `nca.train.train` |
| distribution of the firing masks and of the noise | `engine.rng.draw_step`: one seed per step, Bernoulli(p_fire) masks and `randn * sigma * active * cells` noise, identical tensors for every backend |
| numerical precision | `train.amp` (bf16 autocast) applies to every backend the same way and, as in `nca.train.train`, to the BPTT chunk only: the rollout steps before and after the chunk run in fp32; `scripts/bench_engine.py` measures fp32 and bf16 for each variant |

Two scientific corrections apply to *both* versions:

* **Weights are updated after the rollout.** Until 2026-10-13 `nca.train.train`
  ran `opt.step()` at the end of the BPTT chunk and the remaining rollout steps used the new
  weights; with Maze-Hard's `rollout: 256, chunk: 128` half of the states written back to the
  replay pool were produced by a different parameter vector than the one the loss was computed
  with. Both the reference loop (since 2026-10-13, DEVIATIONS) and the engine keep
  the backward pass at the end of the chunk (that is what frees the chunk's memory) and apply
  the optimiser and EMA update after the last step of the rollout (`engine.update_after_rollout:
  true`, the default; `false` reproduces the earlier order for A/B tests). Sudoku-OOD
  (`chunk == rollout`) is unaffected either way.
* **The normalisation is chosen explicitly.** `norm_fired_only` is a model option
  (DEVIATIONS 2026-10-13); the engine's fused update implements both values and the tests
  check both. A speed comparison must use the same value for every variant.

## What the engine does

`src/nca/engine/`:

| File | Contents |
|---|---|
| `rng.py` | per-step seeds; `draw_step` regenerates a step's firing mask and noise on the device with no CPU branch |
| `ops.py` | `project_values` (project before padding), `fixed_attention_mix` (9-neighbour mixing with an explicit backward), `fused_cell_update` (residual + noise + group norm + clamp with an explicit backward) |
| `step.py` | the step function of each backend (`StepOptions`: the three pieces above can be toggled one by one) and `run_block`, a block of `n` steps with the loss accumulated inside |
| `engine.py` | `Engine`: blocks, `torch.compile`, segmented checkpointing, autotuning, one training iteration |
| `train.py` | `train_fast`: the training loop (same files as `nca.train`, plus `iter_seconds` and `peak_mem_gb` in `log.jsonl`) |
| `bench.py` | timing, peak memory, FLOPs, hardware info, `compare_backends` |

### 1. No synchronisation inside the rollout

The reference `sample_noise` asks the GPU whether any board is active this step
(`active.sum() == 0`), which synchronises the CPU with the GPU at every step for a branch
taken with probability 0.9^256 ~ 2e-12 (48 synchronisations per Sudoku-OOD iteration). The
engine draws `fire`, `active`, `cells` and the Gaussian from the step's seed and multiplies:
a step with no active board gets a zero noise tensor. The noise mask `(B, 1, H, W)` travels
with the noise, so the fused update does not recompute it from `(noise != 0).any(dim=1)`.
`torch.tensor(t, device=dev)` is created only for models with a time embedding (ARC). The
gradient norm is read once per iteration (the finite check before `opt.step`), as in the
reference.

On ARC `NCA.step_embedding` used to index the time-embedding table with the 0-dim global
clock (`weight[idx]`), which calls `.item()` once per step; since 2026-10-13 it looks the same row up with an embedding lookup on a length-1 index and never
leaves the device, for both backends. `time_forward` and `count_step_flops` pass the clock
an ARC model needs.

### 2. Project before padding

The fixed-attention value projection is linear without bias, so projecting the H x W grid
and padding the projected values equals projecting the padded grid: 81 cells instead of
121 for a 9 x 9 Sudoku (wrap and zero padding; wall padding projects the border vector once).

### 3. Fused mixing and fused update, with explicit backward passes

`fixed_attention_mix` accumulates the 9 shifted products in place and computes

    dL/dv_j = sum_delta A_{j-delta,delta} dL/dz_{j-delta},   dL/dA_{i,delta} = sum_{b,c} dL/dz_{b,i,c} v_{b,i+delta,c}

by 9 in-place scatters and 9 reductions, saving only `v` (padded) and `A`.
`fused_cell_update` computes `u = x + m dx + eta`, the per-group normalisation
`y = u / sqrt(||u||^2 + eps)` on every cell or only on the cells that fired or received noise,
and the clamping, in one autograd node; its backward is

    g_u = g_y / r - u (u . g_y) / r^3,   r = sqrt(||u||^2 + eps),

on the normalised cells, the identity elsewhere, and zero on the clamped channels of given
cells. When normalisation is active, it saves `u`, `r` and the masks instead of the chain of state-sized intermediates
autograd keeps (`x + m dx`, `+ eta`, `xg / norm`, `torch.where`, `torch.cat`).
Without normalisation, the identity backward needs only the masks: the update neither
clones nor retains `u`. This removes a full state per differentiable step on Maze;
the saved-tensor regression checks this as well as the gradients.

### 4. Compilation by blocks of steps

`engine.compile: true` compiles `run_block` with `torch.compile(dynamic=False)`: `block`
consecutive steps and their loss terms become one graph, with no Python between the steps.
The same function serves every block of the rollout, so a run compiles once per (block
length, grad mode) pair: the first iteration pays the compilation (reported as
`compile_seconds` in `run.json`). `compile_mode: reduce-overhead` adds CUDA graphs. The loss
is computed at every chunk step with a per-step weight (0 outside the supervision window) so
that the graph is the same whatever `loss_window` says.

CUDA graph outputs can alias reusable storage. The engine marks iteration boundaries,
copies inference outputs before entering a differentiable chunk, and snapshots the chunk
state before backward. The numerical comparison helper also owns its output snapshots.
GPU regressions exercise several chunk offsets and repeated calls under the benchmark's
explicit TF32-disabled policy.

### Optional GPU operators (2026-10-13 optimization follow-up)

All options below default to `false`; they preserve model parameter names and shapes.

| `engine` option | Scope and behavior |
| --- | --- |
| `triton_mix` | CUDA fixed attention with wrap padding: directly gather the nine wrapped neighbors, then compute value and mixing-weight gradients in Triton. Supports FP32/BF16 values and FP32 mixing weights. Avoids padded intermediates and their backward scatter. |
| `kernel_backward` | Kernel perception: retain the original convolution forward and replace its backward with shared-kernel spatial reductions. `true` uses an Inductor-compiled backward; `triton` uses explicit input-gradient and two-stage weight-gradient kernels. Both work without compiling the forward. |
| `channels_last` | Convert model parameters and iteration input state to channels-last storage. The measured benefit depends on perception and compilation. |
| `kernel_mix` | Experimental replacement of both kernel-perception forward and backward. Changes floating-point reduction order; the unnormalized Maze stress trajectory fails the FP32 and BF16 state tolerances. Do not use it as the validated Maze configuration. |
| `compile_backward_only` | With `compile: true` and no compile mode: execute the AOT forward graph as ATen operations and compile its backward with Inductor. Uses PyTorch 2.8 internals; validate decompositions and gradients for the chosen task. |
| `normalized_backward` | For the fused update with clamp channels aligned to normalization groups, save the returned normalized state and compute `(g - y * sum(g*y)) / r`. This reuses state storage already needed by the next step. Unaligned groups retain the original formula. Because the final state is then a saved activation, `train_iteration` clones it before the backward: without that copy the compiled backward reused its buffer and returned a corrupted state (seen on an RTX 4060 with `triton_mix` and `norm_fired_only=false`; loss and gradients matched). |
| `compile_update` | Compile only the fused residual/noise/clamp/normalization update. Perception and the MLP retain their existing execution. Can be used with `compile: false`. |

These operators target first-order training gradients. Numerical checks and timings must
use the same explicit options, precision, seed and normalization. The bounded tile tuner
in `scripts/tune_attention_mix.py` diagnoses only the attention weight-gradient kernel;
an operator microbenchmark does not establish a full-iteration speedup.
Use `scripts/train_fast.py --disable-tf32` to match the TF32-disabled policy of the L4
measurements. `run.json` records the effective matmul/cuDNN TF32 and cuDNN benchmark settings.

### 5. Segmented checkpointing, tuned for the memory budget

`engine.checkpoint_segment: q` keeps the state at the boundaries of segments of `q` steps and
recomputes the inside of a segment during the backward pass (`torch.utils.checkpoint`,
non-reentrant). The recomputation regenerates the masks and the noise from the step seeds,
so nothing random is stored (storing the 48 dense Sudoku-OOD noise tensors would cost
486 MiB in fp32) and the recomputed segment is bitwise the one of the forward pass
(`tests/test_engine_train.py::test_segmented_checkpointing_is_exact`). `auto` times each
candidate of `checkpoint_candidates` (none, 16, 8, 4, 2 by default) on one iteration and keeps
the fastest whose peak allocated memory fits `vram_budget` x the GPU's memory; "none" is a
candidate and should be measured alongside the checkpointed variants. When
no candidate fits (out of memory, or above the budget) the run stops with an error listing
the trials: the budget is a contract, so the engine never silently starts with an over-budget
configuration (the fix is a smaller batch or chunk, smaller candidates, or a larger budget).
The L4 experiment evaluated an absolute 8 GiB target from both allocated and reserved
peaks (including the cold first iteration), independently of the L4's larger capacity.
It does not establish the CUDA/desktop overhead or performance on an 8 GiB laptop.

### Not implemented (and why)

* **Sparse execution of the update MLP** in the default path. With p_fire = 0.8 the ceiling
  is a 20 % saving on the MLP before the cost of gathering and scattering the active cells,
  and any compaction needs the number of active cells on the CPU (one synchronisation per
  step). It was added later as the opt-in `sparse_mlp` (next section), aimed at Visual
  Sudoku's 50 % firing rate.
* **Temporal blocking** (several steps per GPU tile): exploratory, after the above.

## Further opt-in options and APIs (L4 campaigns, 14-16 October 2026)

Three later campaigns on a cloud L4 capped at 8 GiB extended the engine. Every option below
defaults to off, keeps parameter names and checkpoints compatible, and was checked against the
reference at the same precision before being timed. A valid option is not necessarily faster:
use the measured configuration of the workload (the presets in `configs/budget_optimized/`, which
build on `configs/l4_round3/`)
rather than enabling everything. L4 times are not RTX 4060 Laptop times, and the ARC and Visual
Sudoku fixtures use synthetic data: they measure compute, not accuracy.

| `engine` option | What it does, and its scope |
| --- | --- |
| `checkpoint_recursive` | checkpoints both halves of the trajectory recursively (`checkpoint_segment` sets the leaf length): bounded activation memory for long pixel rollouts, more recomputation, a different loss summation order |
| `microbatch: n` | accumulates the gradient of an unchanged global batch in n pieces (masks and noise drawn for the global shape, losses weighted by supervised cells, one clip, one optimizer and EMA step); needs `update_after_rollout: true` |
| `sparse_mlp` | runs the ReLU update MLP on fired cells only (not SwiGLU); perception still reads the whole grid; the CUDA compaction syncs once per step and stays outside compilation |
| `sparse_sobel` | with `sparse_mlp`, computes the fixed identity/Sobel features at fired cells only (Visual Sudoku perception), Triton forward and backward |
| `checkpoint_mlp` | with `sparse_mlp`, recomputes the hidden MLP activation during backward |
| `cache_fire` | reuses identical noiseless firing draws within one iteration (recomputation, microbatches); 64 MiB cap |
| `skip_zero_loss` | skips the readout and loss at steps whose loss weight is zero, keeping every transition and gradient path |
| `linear_mlp` | runs the same 1x1 MLP or SwiGLU weights as a matrix multiplication with channels-last output |
| `linear_value` | linear-layout value projection for fixed attention |
| `attention_split_k` | deterministic split reduction for the attention-weight gradients |
| `iteration_graph` | captures the whole differentiable rollout, clipping and backward in one CUDA Graph (AdamW and EMA outside); unsegmented full-rollout chunks only, no task or step embeddings, no microbatching |
| `triton_mlp` | bf16-autocast-only fused two-layer ReLU forward and input backward, widths up to 512; leave it off for fp32 evaluation |
| `clone_microbatch_grads` | owns the accumulated gradient buffers between CUDA Graph replays; required to combine CUDA Graphs with microbatching |

`triton_mix` also gained zero borders, which makes it usable for ARC.

Known limits. Under PyTorch 2.8, CUDA Graphs combined with checkpoint recomputation can pass a
first equivalence check and fail on later updates, so the training engine rejects that
combination and its autotuner only tries CUDA Graphs without checkpointing. TF32 on Maze-Hard
fails the fp32 state tolerance and is never selected; Maze-Hard training stays in fp32
(bf16 diverged on the RTX 4060). The public-code switches of the model (`project_update`,
`noise_after_norm`, `norm_style: norm_eps`; DEVIATIONS 2026-10-20) run the cell update unfused
(`NCA.residual_update`), perception stays fused; the training engine refuses
`chunk_start: per_sample` and `writeback: chunk_end`, which only `scripts/train.py` implements.

APIs added by the same campaigns:

* **Inference**: `nca.engine.inference.InferenceEngine(model, engine_config, amp=False,
  noise=None)` runs a compiled forward rollout from an initialised batch and per-step seeds and
  returns the state (and the per-cell ARC clocks, which advance when a cell fires). Keep one
  runner per shape and precision so that compilation or CUDA Graph capture is paid once. It is
  a forward runner, not a replacement for the tracing, pruning and voting of `eval_tts.py`.
* **Evaluation with the engine**: `RolloutConfig.engine`, exposed as `eval_tts.py
  --engine-config` and `run_ablations.py --eval-engine-config`, compiles uniform evaluation
  blocks while keeping the historical random stream, noise, damage and pruning boundaries,
  traces and readout. Adaptive firing and ARC clocks are not supported there. The measured fp32
  evaluation sections are `configs/budget_optimized/inference_{maze_ood,maze_hard,sudoku_ood}.yaml`.
* **Ablations with the fast trainer**: `run_ablations.py --trainer fast` (a new, empty
  `--out-root` is required).
* **ARC test-time training**: `nca.ttt.prepare_ttt_model`, `augmentation_boards` (the original
  task, five dihedral transforms with ten colour permutations each, 51 task IDs) and
  `TTTSession`, which updates shared weights and embeddings with Adam and a cosine schedule.
  Pass training demonstrations only, never test answers.
* **Visual Sudoku compute fixture**: `configs/visual_sudoku_compute.yaml` (Table 6 dimensions,
  256x256, 256 channels, batch 1 by assumption, font glyphs instead of MNIST digits); a compute
  benchmark, not a training or evaluation pipeline.

The cloud measurements behind these options are not kept in the repository (they measured
compute on an L4, not accuracy or laptop speed). One result is worth keeping: a 2,000-step
Maze-OOD recipe is 2.5 times faster but fails the 59x59 check on one seed in three, so no
shorter recipe replaces the paper's budget.

## Validation

| Level | Test | What it checks |
|---|---|---|
| operator | `tests/test_engine_ops.py` | `fixed_attention_mix` vs the reference loop and `gradcheck` (float64); `project_values` for wrap / zero / wall; `fused_cell_update` vs the reference chain and `gradcheck` for no norm / norm all / norm fired-only, with empty, full and mixed firing masks, a zero cell, noise on a non-firing cell and clamped cells; bf16 inputs with fp32 weights |
| step | same file | the fused step equals `NCA.step` (outputs and every parameter gradient, atol 1e-5) on the Maze (kernels, wall), Sudoku-OOD (fixed attention, wrap, norm all), Sudoku-Extreme (norm fired-only) and ARC-like (zero padding, SwiGLU, task and time embeddings, MLP head) cells, with each fused piece on its own; under bf16 autocast |
| trajectory | `tests/test_engine_train.py` | reference vs fused on 1 and 8 steps with the same draws (state 1e-5, gradients 1e-5 relative); checkpoint segments 1 / 3 / 4 vs none (bitwise state and loss); block 3 vs block 1 (bitwise); compiled blocks vs eager (1e-4) |
| training | same file | `train_fast` writes `final.pt` / `run.json` / `log.jsonl` / `summary.json` / `checkpoint.pt`, `load_model` reads it, a run interrupted after 3 steps and resumed equals the 6-step run bitwise; the update-order option changes the result only when `chunk < rollout` |

`scripts/check_engine.py` runs the trajectory level on real configurations (1, 8 and `chunk`
steps, fp32 and bf16, segments against none) and, with `--train-steps n`, trains each
variant for `n` iterations from the same initialisation and prints the loss curves. Each
check is judged: the state absolutely, the gradients relative to the largest reference
gradient, against `--tol-fp32` (default 1e-4, 1e-4), `--tol-bf16` (2e-2, 2e-2) and
`--tol-segment` (1e-6, 1e-6); a parameter with a gradient on one side only, or any non-finite
value, fails whatever the tolerance (`compare_backends` reports `grad_only_a` /
`grad_only_b` / `nonfinite`), and the script exits with status 1 on any failure, so it can
gate a benchmark session. The model, the batch and the draws are seeded with `--seed`
(the model initialisation included). fp32 differences come from summation order only
(1e-5 after one step, ~1e-4 after 48 on CPU); bf16 differences carry the rounding of a
different operator sequence; 50 000 iterations of either precision are *not* expected to be
bitwise equal, which is why the quality comparison below is over seeds.

The accumulated loss is also judged: its absolute error must be at most
`grad_rel * max(abs(reference_loss), 1e-3)` (`loss_rel` can override the multiplier
when calling `within_tolerance` directly). Non-finite comparison metrics fail too.
The CLI regression tests cover both a loss-only discrepancy and NaN metrics.

GPU validation on a cloud L4 (PyTorch 2.8.0, CUDA 12.8) also exposed a limit of the
absolute state tolerance on **unnormalised Maze trajectories**. A stress probe
replacing the initially zero final-layer weights and biases by N(0, 0.01) values
reached a maximum state of 139,696 after 256 steps. Compiled BF16 differed from
eager BF16 by 257 (0.184% of that maximum), so it **failed** the absolute 0.02
threshold even though loss and gradient checks passed. Both reference-compiled
and fused-compiled failed; enabling Inductor precision casts or disabling layout
optimisation separately did not resolve it. The original preset and a stable
nonzero probe (standard deviation 0.001; maximum state 3.53) passed at the real
batch size. These failures are retained, not removed by relaxing tolerances.
Compiled BF16 timings for Maze therefore do not certify strict equivalence for
unstable trajectories; prefer a validated FP32 configuration when that guarantee
matters. This probe says nothing about accuracy on the official Maze benchmark.

## CPU micro-benchmark (cloud container, 4 threads, no GPU)

`results/bench/cpu_sudoku_ood_b32_n8.json`: Sudoku-OOD cell (C=128, 16 heads, generated boards),
batch 32, rollout = chunk = 8 steps, full iteration (rollout + backward + clipping), 10 timed
iterations after 3 warm-ups.

| Variant | fp32 ms / iteration (mean ± std, min) | bf16 ms / iteration | Speed-up vs reference (fp32 / bf16, means) | Matmul + conv FLOPs per step (fwd) | First iteration (compile) |
|---|---|---|---|---|---|
| `reference` (engine loop, `NCA.step`) | 259 ± 18 (235) | 252 ± 32 (201) | 1.00 / 1.00 | 0.806 G | 0.3 s |
| `nopad` = fused mix + fused update, projection after padding | 252 ± 41 (174) | 204 ± 15 (176) | 1.03 / 1.24 | 0.806 G | |
| `nomix` = projection before padding + fused update | 258 ± 11 (246) | 251 ± 62 (178) | 1.00 / 1.00 | 0.764 G | |
| `noupd` = projection before padding + fused mix | 244 ± 31 (199) | 193 ± 17 (168) | 1.06 / 1.30 | 0.764 G | |
| `fused` = all three | 298 ± 18 (271) | 262 ± 27 (200) | 0.87 / 0.96 | 0.764 G | |
| `refcompiled` = reference step, `torch.compile`, block 1 | 161 ± 8 (154) | 104 ± 3 (99) | 1.61 / 2.43 | | 21.5 s |
| `compiled` = fused, block 1 | 235 ± 41 (188) | 117 ± 7 (98) | 1.10 / 2.15 | | 20.0 s |
| `compiled4` = fused, blocks of 4 | 165 ± 11 (145) | 126 ± 33 (107) | 1.57 / 2.00 | | 61.7 s |
| `compiled8` = fused, blocks of 8 | 197 ± 65 (144) | 100 ± 8 (88) | 1.31 / 2.52 | | 68.2 s |

Numerical distance to the reference on the same iteration (`vs_reference` in the file): fp32
variants within 3e-7 on the state and 6e-8 on the parameter gradients, bf16 variants within
1e-3 on the state and 1.2e-4 on the gradients (the state's largest entry is 1, the gradients'
about 1). The timings carry a lot of noise (shared container; the std and the min say so),
so differences under about 10 % are not resolved. The file's `commit` field predates the
engine's commit because the run started before it.

What the CPU numbers say:

* **Projection before padding** removes 5.2 % of the matmul + convolution FLOPs of a step
  (0.806 -> 0.764 GFLOP), as the audit predicted; alone it is within the noise on CPU.
* **The fused operators as plain PyTorch do not pay on CPU**: the fused update is *slower*
  (`fused` vs `noupd`, both precisions). Its explicit backward runs as many elementwise
  kernels as autograd's, plus a clone and a few casts, and what it saves (state-sized saved
  tensors, kernel launches) costs nothing on a CPU. That is a GPU question: the benchmark
  on the PC keeps the four ablations so that the fused pieces can be dropped if they do not
  pay there either.
* **Compilation is the gain on CPU**: 1.6 x in fp32 and 2.4-2.5 x in bf16, whichever step
  is compiled (`refcompiled` is as fast as the fused variants: once inductor fuses the
  elementwise chain itself, the hand-written operators bring nothing). Blocks of 4-8 steps
  are within the noise of block 1 on CPU and cost 3 x the compile time; on the GPU, where
  the per-step launch overhead is what blocks remove, they are expected to matter more.
  The audit's 1.48 x for a compiled step on one thread sits between our fp32 and bf16 numbers.

The CPU numbers validate the tested transformations and show which of them remove work; the
GPU is a different machine (kernel launch overhead and memory traffic dominate there, which is
what blocks, CUDA graphs and the fused update target). Do not quote a CPU speed-up as a GPU
one, in either direction.

### Full-iteration benchmark audit (2026-10-13)

The full-iteration timing now uses the production replay recipe: `pool_mult` (or one batch
when replay is disabled), `augment`, `perturb`, and the same NumPy generator shared by pool
sampling, perturbations, chunk placement and per-step seeds. The earlier helper incorrectly
read `perturbations`, disabled augmentation, used a one-batch pool and split the NumPy stream.
`tests/test_benchmark_protocol.py` compares three complete iterations against `train_fast`,
including identical batches, random step seeds, final states and updated parameters, with
perturbations both enabled and disabled.

The `full` record also reports peak allocated/reserved CUDA memory, the first iteration and
skipped optimizer updates. The temporary weight snapshot lives on CPU and model weights,
gradients and train/eval mode are restored even if measurement fails. Per-variant replay
pools are released before the next measurement. `equivalence_ok` now checks the declared
FP32/BF16 numerical tolerances as well as finite values and gradient presence; a successful
timing alone does not establish equivalence.

`timed_compiled_graphs` and `timed_cuda_graph_recordings` count compilation and CUDA graph
recording during the measured window (the latter uses PyTorch 2.8's graph-tree counters).
Nonzero values identify a window that still includes preparation overhead; report that
window explicitly rather than labeling it a steady-state measurement.

These fixes do not change the published CPU micro-benchmark above, which measured only the
rollout, backward and clipping. GPU timings must identify their GPU, precision, normalization,
data source, batch, rollout, checkpoint segment and measured code revision. A result on a
cloud GPU must not be presented as a measurement on the 8 GB laptop GPU.

## What to run on the GPU (the PC, nothing else running on it)

All commands from the repo root, in the `.venv` of the replication, after `git pull`.

```powershell
# 0. can torch.compile run here? (Windows needs a Triton build: pip install triton-windows)
python scripts/bench_engine.py --config configs/sudoku_ood.yaml --probe

# 1. speed and memory, Sudoku-OOD (the published recipe: batch 256, 48 steps), fp32 and bf16
python scripts/bench_engine.py --config configs/sudoku_ood.yaml `
  --variants reference,nopad,nomix,noupd,fused,refcompiled,compiled,compiled4,compiled8,graphs4 `
  --iters 20 --warmup 5 --out results/bench/pc_sudoku_ood.json
# 2. same for Maze-Hard (chunk 128 of 256: the checkpoint candidates matter here)
python scripts/bench_engine.py --config configs/maze_hard.yaml `
  --variants reference,fused,compiled,compiled4 --segments 0,16,8,4 --iters 10 --warmup 3 `
  --out results/bench/pc_maze_hard.json
# 1. numerical equivalence FIRST, on the real sizes (exit status 1 = do not benchmark)
python scripts/check_engine.py --config configs/sudoku_ood.yaml --variants fused,refcompiled,compiled4 `
  --batch 32 --out results/bench/pc_check_sudoku_ood.json
python scripts/check_engine.py --config configs/maze_hard.yaml --variants fused,compiled4 `
  --batch 8 --segments 16,8 --out results/bench/pc_check_maze_hard.json

# 2. speed and memory, Sudoku-OOD (the published recipe: batch 256, 48 steps), fp32 and bf16
python scripts/bench_engine.py --config configs/sudoku_ood.yaml `
  --variants reference,nopad,nomix,noupd,fused,refcompiled,compiled,compiled4,compiled8,graphs4 `
  --iters 20 --warmup 5 --out results/bench/pc_sudoku_ood.json
# 3. same for Maze-Hard (chunk 128 of 256: the checkpoint candidates matter here)
python scripts/bench_engine.py --config configs/maze_hard.yaml `
  --variants reference,fused,refcompiled,compiled,compiled4 --segments 0,16,8,4 --iters 10 --warmup 3 `
  --out results/bench/pc_maze_hard.json
```

Equivalence comes before any long measurement: a variant that fails `check_engine.py` is
not benchmarked, it is fixed. If `--probe` fails, drop the `*compiled*` and `graphs*`
variants: the fused backend needs no compiler. Expect the first iteration of every compiled
variant to take tens of seconds. The ablation variants (`nopad`, `nomix`, `noupd`) decide
whether each fused piece stays on by default on the GPU: a piece that does not beat the
reference there is switched off in `configs/*_fast.yaml` (the CPU result already says the
fused update should be checked first), and `refcompiled` stays in the table whatever the
outcome, since on CPU it explains the whole gain. `peak_allocated_gb` is the number to read
for the memory side of the fused update and for the checkpoint segments. Two timings are
reported per variant: `train` (rollout + backward + clipping on a fixed batch: the engine's
own cost, used for the ablations) and `full` (the complete iteration as `train_fast` runs it:
replay sampling and perturbations, rollout, backward, clipping, AdamW, scheduler, EMA, write
back to the pool: the number that predicts a run's duration; the optimiser and EMA are the
same for every variant, so the two speed-ups differ only by that fixed cost).

### The three configurations of the final comparison

The comparison that goes in the write-up has three arms, each on the same recipe, the same
normalisation (`norm_fired_only: false`, the Sudoku-OOD preset's default), the same number of
updates (50 000), three seeds (0, 1, 2), evaluated identically (K=1 and conf@128 on the
1 000 RRN boards):

| Arm | Command | What it measures |
|---|---|---|
| A. corrected `train.py` | `python scripts/train.py --config configs/sudoku_ood.yaml --seed k` (any version after 2026-10-13) | the published loop with the update after the rollout: the scientific reference |
| B. engine, `reference` backend | `python scripts/train_fast.py --config configs/sudoku_ood.yaml --seed k --set engine.backend=reference --set engine.compile=false` | the engine's loop (per-step seeds, no sync, blocks) running the unchanged `NCA.step`: isolates the loop from the operators |
| C. engine, best variant | `python scripts/train_fast.py --config configs/sudoku_ood_fast.yaml --seed k` (the `engine:` section set from step 2's winner) | the optimised version |

A and B differ by their random streams only (the engine draws masks and noise from per-step
seeds), so they are expected to agree statistically, not bitwise; B and C are expected
within the `check_engine.py` tolerances per iteration and statistically over a run. Then:

```powershell
# evaluate exactly like the published runs
python scripts/eval_tts.py --run runs/engine/sudoku_ood_fast_fused_compiled/seed0 --n 1000 --depth 256 --k 128 --record-every 8 --group level --max-cells 500000 --out results/bench/tts_engine_seed0.json
```

Outputs to publish, per arm and seed: the *total* wall time (`train_seconds` in `run.json`
/ `summary.json`, which includes compilation and autotuning; `compile_seconds` separately),
`iter_seconds` and `peak_mem_gb` from `log.jsonl`, the validation curve (`eval` records), K=1
and conf@128 on the 1 000 RRN boards, and the time to reach a given validation accuracy (read
off the `eval` records at equal step counts). The ablation table from step 2 (one line per
variant, `train` and `full` timings, peak memory, equivalence status) goes next to it.

## How to read a benchmark file

`results/bench/*.json` from `scripts/bench_engine.py`: `hardware` (GPU, torch, CUDA, Triton,
commit and `dirty`, true when the tree had uncommitted changes), `seed`, `train` (the
resolved training configuration), then one record per variant x precision x segment with
`train` (per-iteration seconds of rollout + backward + clipping on a fixed batch: mean, std,
min, the list, `first_iteration_seconds` = compilation + warm-up, `peak_allocated_gb`,
`peak_reserved_gb` on CUDA), `full` (the complete training iteration, optimiser, EMA and
replay included, same fields), `forward` (no-grad seconds per step), `flops_per_step`
(matmul and convolution FLOPs of one step, forward and forward + backward, from
`torch.utils.flop_counter`; elementwise work is not counted), `vs_reference` (largest
absolute difference of the final state, the loss and the parameter gradients against the
reference backend with the same checkpoint segment on the same iteration, plus
`grad_only_a` / `grad_only_b` / `nonfinite`), `equivalence_ok` (false when that comparison
failed or found a missing or non-finite gradient; the timing of the variant is kept and
`ok` stays true: the two statuses are independent), `baseline` (the reference record the
speed-ups are relative to: same precision and same segment when it ran, else the first
reference that ran) and `speedup_vs_reference` / `speedup_full_vs_reference`.

`first_peak_allocated_gb` and `first_peak_reserved_gb` include the first iteration;
the other peaks cover the measured window. `timed_compiled_graphs` counts new
Dynamo graphs compiled during that window: a nonzero count invalidates a claim
that these are steady-state timings and calls for more warm-up.
