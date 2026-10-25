# Paper map: from each claim of the paper to the code, the command, the result and the figure

Section and figure numbers refer to arXiv:2609.36126v1. "Command" is the one the phase scripts
run for seed 0; every experiment is run with seeds 0, 1 and 2 and `scripts/aggregate.py`
averages them. Local test configurations are smaller than the paper's (see the compute budget table of DEVIATIONS.md).

## Architecture (Section 3.1, Appendix B.1)

| Paper | Code | Notes |
|---|---|---|
| Cell state X in R^{H x W x C}, slices C_in / C_out / C_hid / C_task | `src/nca/model.py: NCAConfig` (`channels`, `c_in`, `c_out`, `in_offset`, `out_offset`, `c_task`; properties `in_slice`, `out_slice`, `hid_slice`, `task_slice`) | states are channels-first `(B, C, H, W)` |
| Fixed orthonormal token embeddings e_v | `src/nca/embeddings.py: make_embeddings` | `onehot` for mazes, `orthonormal` for Sudoku and ARC |
| State initialisation: given cells get e_v in the input slice, everything else N(0, sigma_init^2); input slice stays frozen | `model.py: NCA.init_state`, `NCA.apply_clamp` | clamping is re-applied after every step |
| Convolutional sensing (Maze): K_heads scalar 3x3 kernels shared over channels (B.1) | `model.py: KernelPerception` (`perception: kernel`) | z has K_heads x C channels and the MLP is E x K_heads x C wide (`hidden_from_perception`), which gives the paper's 10 420 / 41 284 parameters; `ConvPerception` (dense 3x3 convolution) is kept for the archived runs of 2026-10-10 only |
| Fixed-attention sensing (Sudoku, ARC) | `model.py: FixedAttention` | position-specific mixing weights over the 9 neighbours of a shared value projection |
| Self-attention (ablation, Fig. 13, 14) | `model.py: SelfAttention` | `perception: self_attn` |
| Sobel sensing (Visual Sudoku) | `model.py: SobelPerception` | implemented, not used (Visual Sudoku is not replicated) |
| Update module: MLP with ReLU (Maze, Sudoku), SwiGLU (ARC), expansion E | `model.py: MLPUpdate`, `SwiGLUUpdate` | `update`, `expansion` |
| Asynchronous stochastic execution, m ~ Bernoulli(p_fire) | `model.py: NCA.sample_fire_mask`, `NCA.step` | `fire_rate` |
| Adaptive firing: p_low when c_ij > tau_halt | `model.py: NCA.sample_fire_mask(p_low=, tau=)` | used by `rollout()` with `RolloutConfig.p_low`, `.tau` |
| Residual update, per-cell group normalisation (U groups) | `model.py: NCA.step`, `NCA.normalize` | `norm_groups` |
| Boundary padding: wall / wrap / zero | `model.py: _pad`, `NCA.pad_vec` | `padding`, `wall_token` |
| Prediction heads: L2 similarity (Maze), cosine (Sudoku), MLP (ARC) | `model.py: NCA.scores`, `NCA.decode` | `head` |
| Per-cell confidence c_ij = max score, board confidence = mean | `model.py: NCA.confidence`, `NCA.board_confidence` | softmax first for the MLP head |
| Time encoding added to the hidden slice (ARC, B.3), local clocks at inference | `model.py: NCA.delta` (`time_embed`), `rollout.py: rollout` (`clock`, `RolloutConfig.local_clock`) | `time_steps` |
| Task embedding e_task injected in the task slice (ARC) | `model.py: NCA.init_state` (`task_embed`) | `c_task`, `n_tasks` |
| Table 3 / Table 4 hyper-parameters | `model.py: PRESETS` (`maze_ood`, `maze_hard`, `sudoku_ood`, `sudoku_extreme`, `arc`), `build_model` | parameter counts checked in `tests/test_model.py: test_parameter_counts_close_to_paper` |

## Training recipe (Section 3.2, Appendix B.2, Table 3)

| Paper | Code | Command / config |
|---|---|---|
| Sample replay buffer of M = pool_mult x B states, batch seed ratio r_seed, ages | `src/nca/pool.py: ReplayPool` (`sample`, `write_back`) | `train.pool_mult`, `train.seed_ratio` |
| State noise during rollout (p_t, p_s, sigma) | `src/nca/perturb.py: sample_noise` | `train.noise` |
| State damage before rollout (p_damage, N_damage, r in [0.1, 0.4]) | `perturb.py: circular_masks`, `apply_damage` | `train.damage` |
| Target swapping (p_swap) | `perturb.py: swap_indices`, `pool.py: ReplayPool.replace_boards` | `train.swap` |
| Unroll N steps, random sub-window of N_chunk with truncated BPTT | `src/nca/train.py: train` (the `for t in range(N)` loop) | `train.rollout`, `train.chunk` |
| Loss: MSE to target embeddings (Maze, Sudoku), cross-entropy (ARC) | `train.py: step_loss`, `cell_loss` | `train.loss`, `train.loss_reduce` |
| All-step / last-step / window-step supervision | `train.py: train` (`loss_window`) | `all`, `last`, or an integer w |
| AdamW, gradient clipping 1.0, peak LR, constant or cosine schedule with warmup, weight decay | `train.py: train`, `lr_at` | `train.lr`, `schedule`, `warmup_ratio`, `weight_decay`, `grad_clip`, `cosine_floor` |
| EMA of the weights (0.999), used at evaluation | `train.py: train` (EMA update), `final.pt` | `train.ema` |
| Data augmentation: 8 dihedral (Maze-OOD, Maze-Hard BFS), Sudoku symmetries, ARC online | `tasks/maze.py: augment_dihedral`, `tasks/sudoku.py: SudokuOOD.augment`, `tasks/arc.py: ARCTask.augment` | `train.augment` |
| The training recipe of the authors' public JAX code, as opt-in switches for A/B tests (extension, not in the paper) | `train.py: resolve_train_cfg`, `perturb_batch`, `lr_at`, `ema_decay`, `_rollout_per_sample`; `perturb.py: apply_batch_noise`; `pool.py: ReplayPool.swap_from_pool`; `tests/test_codepub_recipe.py` | `configs/maze_ood_codepub.yaml`, `sudoku_ood_codepub.yaml`, `maze_hard_codepub.yaml` (the "public-code recipe" arms of A/Bs against the base configs; keys in the glossary below) |
| Same recipe executed by the fast engine (extension, not in the paper) | `src/nca/engine/` (`Engine`, `train_fast`), `docs/FAST_ENGINE.md` | `python scripts/train_fast.py --config configs/sudoku_ood_fast.yaml --seed 0` -> `runs/engine/<name>/seed0/`; `scripts/bench_engine.py`, `scripts/check_engine.py` -> `results/bench/` |
| Training commands | `scripts/train.py` | `python scripts/train.py --config configs/<task>.yaml --seed 0` -> `runs/<task>/seed0/` |
| Training curves | `scripts/make_figures.py: fig_training` | `figures/F7_training_<config>` |

## Test-time scaling (Section 3.3, Appendix B.4)

| Paper | Code | Command | Result | Figure |
|---|---|---|---|---|
| Depth scaling D | `src/nca/rollout.py: rollout` (`RolloutConfig.depth`, `record_every`), `experiments.py: tts_curves` (`depth_curve`) | `scripts/eval_tts.py --run runs/sudoku_ood/seed0 --n 1000 --depth 256 --k 128 --record-every 8 --group level --out results/tts/sudoku_ood_seed0.json` | `results/tts/<task>_seed<k>.json` | `figures/F2_tts_<task>` left panel (Fig. 4A, 16A, 16B, 17A, 17B left) |
| Width scaling: K parallel trials from independent initial noise | `rollout.py: rollout` (`RolloutConfig.k`), `experiments.py: tts_curves` (`conf_at_k`, `pass_at_k`) | same command | same file | `figures/F2_tts_<task>` right panel (Fig. 4A, 16, 17 right) |
| Final selection conf@K (highest board confidence) vs pass@K (oracle) | `rollout.py: conf_at_k`, `pass_at_k` | same | `conf_at_k`, `pass_at_k` keys | same figure, two curve colours |
| Majority voting pass@M (ARC) | `rollout.py: majority_vote`, `experiments.py: majority_pass_at_m` | `scripts/eval_tts.py ... --select majority` in `run_phase3.*` (trials mapped back to the canonical frame, pass@1 / pass@2) | `majority_pass_at_k` in `results/tts/arc_seed<k>.json`; `aggregate.py` reports majority pass@2 | |
| Test-time noise injection Theta_noise = (r, p_t, p_s, sigma) | `rollout.py: rollout` (`RolloutConfig.noise`, `_noise_active`) | `eval_tts.py ... --noise-sigma 0.15 [--noise-ps 0.3]` | `results/tts/<task>_noise_seed<k>.json` | dashed curves in `F2_tts_<task>` (Fig. 16, 17 dashed) |
| Noise magnitude sweep (Fig. 7B) | `experiments.py: noise_sweep` | `scripts/eval_noise.py --run runs/sudoku_ood/seed0 --n 500 --depth 256 --k 16 --out results/noise/sudoku_ood_seed0.json`; maze: `--run runs/maze_ood/seed0 --n 500 --depth 300 --k 8 --load size=13` | `results/noise/<task>_seed<k>.json` | `figures/F3_noise` (Fig. 7B) |
| Spatial substrate scaling on Maze-OOD, up to 201x201 (Fig. 4B, Table 1) | `experiments.py: maze_size_scaling`, `scripts/eval_maze_scaling.py: default_depth`, `TABLE1` | `scripts/eval_maze_scaling.py --run runs/maze_ood/seed0 --n 200 --out results/maze_scaling/maze_ood_seed0.json` and `... --table1 --n 100 --out results/maze_scaling/table1_seed0.json` | `results/maze_scaling/maze_ood_seed<k>.json`, `table1_seed<k>.json` | `figures/F1_maze_size_scaling` (Fig. 4B); Table 1 rows via `aggregate.py` |
| Headline numbers of Table 1 | `scripts/aggregate.py: HEADLINES` | `python scripts/aggregate.py` | `results/summary.json`, Markdown table for `docs/RESULTS.md` | |

## Niche-Capped Diversity Pruning (Section 3.3, Appendix B.6, Fig. 6A)

| Paper | Code | Command | Result | Figure |
|---|---|---|---|---|
| Decode boards at m - 1 equidistant checkpoints, group into niches, cap each at n_cap by confidence, halve the ensemble round-robin | `src/nca/pruning.py: niche_capped_pruning`; called from `rollout.py: rollout` (`RolloutConfig.pruning = {"m", "n_cap"}`; pruned trajectories leave the batch, `alive` in the result) | `scripts/eval_pruning.py --run runs/sudoku_extreme/seed0 --n 500 --k 64 --depths 64 128 256 512 --ms 2 4 8 --out results/pruning/sudoku_extreme_seed0.json` | `results/pruning/sudoku_extreme_seed<k>.json` | `figures/F6_pruning_sudoku_extreme` (Fig. 6A) |
| Theoretical compute fraction (2 - 2^-(m-1)) / m | `pruning.py: pruned_compute_fraction` | | `theoretical_fraction`, `compute_rel` keys | |
| Exploration noise schedule of B.6 (sigma 0.1, p_t 0.2, p_s 0.8, first quarter) | `scripts/eval_pruning.py` (`--noise-sigma`, default 0.1) | | | |
| Measured compute | `rollout.py: RolloutResult.cell_updates` (fired cells per trajectory) | | `compute_units` | x axis of F6 |

## Adaptive compute and damage recovery (Section 4.4, Fig. 8)

| Paper | Code | Command | Result | Figure |
|---|---|---|---|---|
| Adaptive firing p_fire = 0.4 if c_ij > 0.95 else 0.8 on 201x201 mazes; steps and cell updates to solve | `experiments.py: adaptive_compute`; `rollout.py` (`p_low`, `tau`, `trace_updates`) | `scripts/eval_adaptive.py --run runs/maze_ood/seed0 --size 201 --n 20 --depth 20000 --max-cells 500000 --record-every 100 --out results/adaptive/maze201_seed0.json` | `results/adaptive/maze201_seed<k>.json` | `figures/F5_adaptive_compute` (Fig. 8A) |
| Damage at t = 6000 (random discs zeroed), recovery | `rollout.py: rollout` (`RolloutConfig.damage_at`, `damage`), `perturb.py: circular_masks`; `experiments.py: recovery_point` (95% crossing searched after the event; total and since-damage times) | `... --damage-at 6000 ... --out results/adaptive/maze201_damage_seed0.json` | `results/adaptive/maze201_damage_seed<k>.json` | same figure, second pair of panels (Fig. 8C) |
| Localised activity after damage (Fig. 8B) | `RolloutResult.damage_masks`, `trace_updates` | no dedicated figure | | |

## Training-ingredient ablations (Fig. 7A) and fixed vs self attention (Fig. 13)

| Paper | Code | Command | Result | Figure |
|---|---|---|---|---|
| No asynchronous updates (p_fire = 1) | `train.py: train` (`asynchronous: false` sets `fire_rate = 1.0`) | `scripts/run_ablations.py --config configs/ablations_maze_ood.yaml --seeds 0 1 2 --eval-n 500` | `results/ablations/maze_ood_<variant>_seed<k>.json` | `figures/F4_ablations_maze_ood` |
| No perturbations (noise, damage, swap off) | `train.py` (`perturb: false`) | same | same | same |
| No sample replay (every batch fresh, pool_mult 1, seed ratio 1) | `train.py` (`replay: false`) | same | same | same |
| Sudoku ablations, plus self-attention instead of fixed attention | `configs/ablations_sudoku_ood.yaml` (`self_attn`) | `scripts/run_ablations.py --config configs/ablations_sudoku_ood.yaml --seeds 0 1 2 --eval-n 500 --eval-depth 256` | `results/ablations/sudoku_ood_<variant>_seed<k>.json` | `figures/F4_ablations_sudoku_ood` (Fig. 7A, 13) |

## Emergent dynamics (Fig. 2) and ARC task conditioning (Fig. 3)

| Paper | Code | Command | Result |
|---|---|---|---|
| Spatial chain-of-thought in mazes and Sudoku (Fig. 2A, B, D) | `rollout.py` (`record_boards`, `boards_trace`), `viz.py: maze_image`, `sudoku_image`, `save_gif`, `board_strip` | `scripts/animate.py --run runs/maze_ood/seed0 --size 37 --depth 1200 --every 10 --out figures/anim/maze37`; `--run runs/maze_hard/seed0 --depth 200 --every 2 --out figures/anim/maze_hard`; `--run runs/sudoku_ood/seed0 --depth 256 --every 2 --out figures/anim/sudoku --board 3`; `--run runs/sudoku_extreme/seed0 --depth 512 --every 4 --out figures/anim/sudoku_extreme --board 5` | `figures/anim/<name>.gif`, `<name>_strip.png` |
| One NCA, many ARC tasks through task embeddings (Fig. 2C, Fig. 3) | `tasks/arc.py: ARCTask` (`meta["task_id"]`), `model.py: NCA.init_state` (`task_embed`), `configs/arc_small.yaml` | `python scripts/train.py --config configs/arc_small.yaml --seed 0`; `scripts/eval_tts.py --run runs/arc_small/seed0 --depth 64 --k 32 --record-every 4 --out results/tts/arc_seed0.json` | `results/tts/arc_seed<k>.json` (test pairs of the 30 training tasks) |
| ARC canvas, border token, loss mask, online scale / offset augmentation (B.3) | `tasks/arc.py: render_pair`, `make_boards`, `ARCTask.augment` | | |
| RE-ARC extras (B.3) | `tasks/arc.py: fetch_rearc`, `rearc_examples`, `_rearc_cached` | `task.args.rearc_per_task` in `configs/arc_small.yaml` | `data/arc/rearc/<id>_n20_seed<s>.json` |
| Test-time training with 51 offline variants (B.3, C.3) | `tasks/arc.py: ARCTask.ttt_variants`, `map_back` | not part of the phase scripts | |

## Datasets (Appendix A)

| Paper | Code | Local fallback | Details |
|---|---|---|---|
| Maze-OOD (Bansal et al., 2022), easy-to-hard mazes 9x9 train, 9..37, 59, 201 test | `tasks/maze.py: MazeOOD`, `_fetch_official_ood`, `convert_official_ood` | `generate_perfect_maze` (DFS perfect mazes, unique path) | [../data/README.md](../data/README.md) |
| Maze-Hard (Wang et al., 2025), 30x30, A* targets; BFS variant | `tasks/maze.py: MazeHard(targets=...)`, `_fetch_official_hard`, `parse_hrm_maze`, `astar_path`, `bfs_path` | `generate_hard_maze` (approximation) | metrics `exact`, `valid`, `optimal` in `_MazeTask.metrics` |
| Sudoku-OOD (Miyato et al., 2024): SAT-Net train, RRN 18-level test | `tasks/sudoku.py: SudokuOOD`, `load_satnet_official`, `load_rrn_official` | `generate_puzzle` (bitmask solver; hardest levels unreachable) | `meta["level"]` = empty cells |
| Sudoku-Extreme (Wang et al., 2025), tdoku rating bins (Fig. 12) | `tasks/sudoku.py: SudokuExtreme`, `load_extreme_official`, `EXTREME_BINS`, `bin_index` | generated boards binned by our solver's backtracks | `meta["bin"]`, `meta["rating"]` or `meta["difficulty"]` |
| ARC-AGI-1 (Chollet, 2019) + RE-ARC (Hodel, 2024) | `tasks/arc.py: fetch_arc`, `fetch_rearc`, `ARCTask` | `source="synthetic"` (three rule-based toy tasks, tests only) | |
| Visual Sudoku | not replicated | | |

## Glossary: config names vs paper symbols

Model (`model.overrides.*`, see `NCAConfig` and `PRESETS` in `src/nca/model.py`):

| config key | paper symbol / name | where in the paper |
|---|---|---|
| `channels` | C, number of channels | Table 3, 4 |
| `c_in`, `c_out` | C_in / C_out, I/O embedding dims | Table 3, 4 |
| `in_offset`, `out_offset` | position of the input and output slices (output inside input for Maze and Sudoku, separate for ARC) | B.1 |
| `c_task`, `n_tasks` | C_task, task embedding dim; number of tasks (ARC) | Table 4 |
| `embedding` | fixed orthonormal token embeddings e_v | B.1 |
| `perception` | sensing module: `kernels` = the paper's 2D conv (K_heads scalar 3x3 kernels shared by all channels), `conv` = dense 3x3 conv (first Maze runs), `fixed_attn` = Fixed Attn, `self_attn` = self-attention ablation, `sobel` = Sobel + identity | Table 3, 4, B.1, D.1 |
| `heads` | K_heads, sensing heads | Table 3, 4 |
| `grid_size` | H x W (needed by fixed attention) | Table 3, 4 |
| `padding`, `wall_token` | boundary padding: wall / wrap / zero | Table 3, 4 |
| `update` | update block: `mlp` or `swiglu` | B.1 |
| `expansion`, `hidden_from_perception` | E, update expansion; hidden width E x dim(z) (Maze: z has K_heads x C channels) or E x C | Table 3, B.1 |
| `norm_groups` | U, GroupNorm groups (0 = none) | Table 3, 4 |
| `sigma_init`, `init_uniform` | sigma_init, state initialisation | Table 3, 4, 6 |
| `fire_rate` | p_fire, cell fire rate | Table 3, 4 |
| `head` | prediction head: `l2`, `cosine`, `mlp` | B.1 |
| `time_steps` | size of the learned step-embedding table e_step (ARC) | B.3 |
| `zero_init_update` | not in the paper: last layer of the update block starts at zero (maze presets) | |

Training (`train.*`, see `DEFAULTS` in `src/nca/train.py`):

| config key | paper symbol / name |
|---|---|
| `steps` | training steps |
| `batch_size` | B |
| `pool_mult` | M / B, pool size relative to the batch |
| `seed_ratio` | r_seed, batch seed ratio |
| `rollout` | N, rollout length |
| `chunk` | N_chunk, chunk size |
| `loss_window` | loss application: `all` = all chunk, `last` = last step, integer w = window of the last w steps |
| `loss` | MSE (`mse`) or cross-entropy (`ce`) |
| `lr`, `schedule`, `warmup_ratio`, `cosine_floor` | peak learning rate, LR schedule (constant / cosine), warmup ratio, cosine alpha |
| `weight_decay`, `grad_clip`, `ema` | weight decay, gradient clipping, EMA decay rate |
| `noise: {p_t, p_s, sigma}` | training noise (p_t, p_s, sigma) |
| `damage: {p, n, r_min, r_max}` | p_damage, N_damage, radius range r in [0.1, 0.4] |
| `swap: {p}` | p_swap, target swap |
| `betas` | not in the paper: AdamW betas; default `[0.9, 0.999]` (PyTorch), public code `[0.9, 0.95]` (`train.py: train`, `engine/train.py: train_fast`) |
| `init_lr_ratio` | not in the paper: start of the warmup as a multiple of the peak LR; default `null` = warmup from 1 / warm; `r` = linear from r x lr to lr, the public code's optax `warmup_cosine_decay_schedule` (`train.py: lr_at`; public code: 1.0 Maze-OOD, 2.0 Maze-Hard, 10.0 Sudoku, with `cosine_floor: 1.0` or `schedule: constant` for the end value = peak) |
| `loss_reduce` | not in the paper: MSE per cell `sum` (default) or `mean` (public code) over the output channels; folded into the loss kind `mse_mean` by `train.py: resolve_train_cfg`, so the engine's `step_loss` sees it too (`train.py: cell_loss`); MSE only |
| `ema_warmup` | not in the paper: default `true` = decay_t = min(ema, (1 + t) / (10 + t)); `false` = constant decay from step 0, as the public code (`train.py: ema_decay`) |
| `damage: {fill, geometry}` | not in the paper: value inside the discs, default `0.0` (public code: 1.0 for normalised models, i.e. Sudoku); `geometry: ours` (default: radius x min(H, W), centres anywhere) or `official` (the public `make_circle_masks`: coordinates linspace(-1, 1), centres in [-0.5, 0.5], radius in those units) (`perturb.py: circular_masks`, `apply_damage`); the clamp is re-applied after the damage |
| `batch_noise: {p, ps, sigma}` | not in the paper: default `null` = off; the public code's noise on the prepared batch (after swap and damage): per board with probability p, per cell with probability ps, N(0, sigma^2) on every channel but the input channels of given cells, then renormalise and clamp (`perturb.py: apply_batch_noise`, `train.py: perturb_batch`); a dict is merged into `{p: 0.1, ps: 0.2, sigma: 0.15}` |
| `swap_source` | not in the paper: `fresh` (default) = a swapped sample gets a fresh dataset board (`pool.py: ReplayPool.replace_boards`); `pool` = the public code: the board of a random pool entry (uniform, with replacement) and its input-channel state on the given cells, age reset to 0 (`ReplayPool.swap_from_pool`) |
| `chunk_start` | not in the paper: `shared` (default) = one random BPTT chunk start per batch; `per_sample` = the public code: each board draws its own start in [0, N - N_chunk] and its gradient is cut there (`train.py: _rollout_per_sample`); reference trainer only, the engine raises a ValueError |
| `writeback` | not in the paper: `rollout_end` (default) = the pool gets the state after N steps; `chunk_end` = the public code: each board's state at the last step of its own chunk, the rollout stops at the last chunk end; reference trainer only |
| `stop_at` | not in the paper: pause a probe run after this step (checkpoint only, no `final.pt` / `summary.json`, the LR schedule still follows `steps`); resuming with it cleared continues the run; both trainers |
| `augment` | data augmentation on / off |
| `asynchronous`, `perturb`, `replay` | ablation switches of Fig. 7A |
| `amp`, `grad_checkpoint` | not in the paper: bfloat16 autocast and activation checkpointing |
| `eval: {n, depth, k}` | the quick validation during training |

Evaluation (command-line options of `scripts/eval_*.py`, see `RolloutConfig` in `src/nca/rollout.py`):

| option | paper symbol / name |
|---|---|
| `--depth` | D, rollout iterations |
| `--k` | K, parallel trials |
| `--noise-sigma`, `--noise-pt`, `--noise-ps`, `--noise-r` (eval_tts) / `--sigmas`, `--p-t`, `--p-s`, `--r` (eval_noise) | Theta_noise = (r_noise, p_t, p_s, sigma) |
| `--p-fire`, `--p-low`, `--tau` | p_fire, p_low, tau_halt (adaptive firing) |
| `--damage-at`, `--damage-n`, `--damage-r` | damage event step, N_damage, radius range |
| `--ms`, `--n-cap` | m (number of halvings), n_cap |
| `--size` (maze scripts), `--load size=` | S, grid size (spatial substrate scaling) |
| `--group level|bin|size` | per-difficulty breakdown (Fig. 16, 17 colours) |
| `--max-cells`, `--amp`, `--device` | not in the paper: memory chunking, bfloat16, device |
| conf@K, pass@K in the JSON | Conf@K (B.4) and the oracle pass@K |
