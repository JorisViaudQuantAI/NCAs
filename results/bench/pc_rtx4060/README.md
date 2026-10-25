# Optimised engine on the RTX 4060 Laptop (8 GB): speed and quality, 17–18 October 2026

Measured on a Windows laptop (NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 610.60, torch 2.6.0+cu124,
triton-windows 3.2.0, Python 3.11.9), with `--disable-tf32` everywhere and `TORCHINDUCTOR_CACHE_DIR=C:\tic`. Everything in this directory's campaign
lives next to it: raw speed benchmarks `results/bench/pc_*.json`, equivalence checks `results/bench/pc_check_*.json`,
logs `results/bench/pc_logs/`, the scripts that drove the PC `results/bench/pc_tools/`, the nine optimised
training runs `runs/engine/{maze_ood,maze_hard,sudoku_ood}_opt/seed{0,1,2}/` (model, `run.json`, `summary.json`,
`log.jsonl`, `pc_provenance.json`) and their evaluations `results/bench/quality/`.

Two kinds of speed-up appear below and must not be mixed:

- **vs the engine's `reference` backend** (`bench_engine.py`, same precision, same GPU): an engine ablation.
- **vs the published `train.py` runs** of the replication (`runs/maze_ood`, `runs/maze_hard`, `runs/sudoku_ood`,
  same GPU): the only number that says what the laptop actually saves.

Neither is comparable to the cloud L4 numbers of the engine's development, measured on another host.

## Phase 1: speed (`bench_engine.py`, 10 iterations after 3 warm-ups)

`train` = rollout + backward + clip; `full` = complete iteration (replay, AdamW, EMA). Speed-ups are `full` vs the
`reference` backend of the same precision. Presets are timed through `pc_tools/bench_wrap.py` (variant `preset`).

| config | precision | variant | train ms | full ms | x full | peak GiB | first call s |
|---|---|---|---|---|---|---|---|
| Sudoku-OOD (b256, chunk 48) | bf16 | reference | 556 | 569 | 1.00 | 2.62 | 0.9 |
| | bf16 | fused | 369 | 379 | 1.50 | 2.50 | 0.4 |
| | bf16 | refcompiled | 140 | 147 | 3.87 | 2.67 | 18.5 |
| | bf16 | compiled4 | 159 | 167 | 3.40 | 2.52 | 49.8 |
| | bf16 | preset l4_round3 + normalized_backward=false | 153 | 158 | 3.63 | 2.40 | 2.6 |
| | fp32 | reference | 784 | 783 | 1.00 | 4.57 | 1.1 |
| Maze-OOD (b64, chunk 100) | bf16 | reference | 203 | 225 | 1.00 | 0.22 | 0.5 |
| | bf16 | compiled4 | 92 | 95 | 2.35 | 0.23 | 26 |
| | bf16 | graphs16 (= budget_optimized preset) | 56 | 64 | 3.53 | n/a* | 64 |
| Maze-Hard A* (b32, chunk 128) | fp32 | reference | 672 | 677 | 1.00 | 5.93 | 1.0 |
| | fp32 | compiled4 | 607 | 609 | 1.11 | 6.02 | 27 |
| | fp32 | budget_optimized preset (kernel_backward=triton, linear_mlp) | 604 | 607 | 1.12 | 5.92 | 11 |
| | fp32 | preset, checkpoint_segment=16 | 736 | 733 | 1.17 vs reference seg16 (855) | 0.84 | 0.7 |
| | bf16 | reference | 757 | 749 | 1.00 | 3.01 | |
| | bf16 | preset | 512 | 480 | 1.56 | 3.04 | 6.3 |
| Sudoku-Extreme (b128, chunk 96) | bf16 | reference | 1233 | 1220 | 1.00 | 5.20 | 1.6 |
| | bf16 | fused | 831 | 857 | 1.42 | 4.93 | 0.9 |
| | bf16 | compiled4 | 368 | 383 | 3.18 | 4.87 | 51 |

\* peak not measurable with CUDA graphs (memory is allocated before the counter is reset). `fused` alone brings
nothing in fp32 on Maze-Hard (1.00) and is slower than the reference on Maze-OOD bf16 (0.96). Sudoku-Extreme fp32
does not fit: the reference needs 8.84 GiB, runs at 46 s/iteration out of shared memory, and `compiled4` hits OOM.

**Equivalence failure found.** `triton_mix` with `normalized_backward=true` under the published normalisation
(`norm_fired_only=false`) returns a final state 0.59–0.71 away from the reference (the state's own magnitude) while
the loss and the gradients match; that state is written back to the replay pool, so a training run would drift.
`norm_fired_only=true` (the preset as shipped), `normalized_backward=false` or `triton_mix=false` all pass
(`pc_sudoku_ood.json`, `pc_check_preset_sudoku_ood_*.json`). Costs 3 % to turn off. Not reproducible on CPU
(the kernel is CUDA-only).

**Fixed on the PC (commit "Engine: own the final state before backward under normalized_backward").** Cause: with
`normalized_backward` the fused update saves its output for the backward, so the final state is itself a saved
activation, and the compiled backward reused that buffer (the returned state came back with gradient-sized values,
mean |x| 1.4e-5 instead of 0.14). Disabling AOTAutograd buffer donation did not help. `Engine.train_iteration` now
clones the state before the backward when `normalized_backward` is on, and the interim guard that refused
`triton_mix` + `normalized_backward` under `norm_fired_only=false` is removed. GPU regression test
`test_compiled_normalized_backward_returns_the_final_state` (fails without the fix); both Sudoku presets pass with
`norm_fired_only=false` and `normalized_backward=true` (`pc_check_fix_*.json`). The Phase 2 Sudoku runs below were
trained before the fix, with `normalized_backward=false`, which is numerically equivalent.

## Phase 2: quality of the optimised arm at the published update counts, 3 seeds

Same evaluations as the published runs (the replication's phase scripts), same normalisation
(`norm_fired_only=false`), `normalized_backward=false`. Per-seed values are seeds 0 / 1 / 2.

### Maze-OOD, `configs/budget_optimized/maze_ood.yaml` (compile + CUDA graphs, 16-step blocks, bf16, 5 000 updates)

| | optimised | published train.py |
|---|---|---|
| training time per seed | 5.4 min (322 / 320 / 325 s); 6.0 min wall incl. compile | ≈ 13.75 min (0.165 s/step) |
| val_exact | 1.0 / 1.0 / 1.0 | 1.0 |
| Table 1 exact 13×13 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Table 1 exact 59×59 | 1.00 / 0.90 / 1.00 | 1.00 / 0.99 / 0.96 |
| Table 1 exact 201×201 | 1.00 / 0.36 / 0.99 (mean 0.78) | 1.00 / 0.83 / 0.74 (mean 0.86) |
| scaling 9→59 | ≥ 0.995 (seeds 0, 2), seed 1 down to 0.885 at 59 | ≥ 0.975 |
| noise (K=8, D=300, 13×13) | 0.998–1.0 | same |

At 201×201 the spread between seeds (0.36 to 1.00 here, 0.74 to 1.00 published) is larger than the gap between
the means; three seeds do not separate the two engines at that size. Not run: `eval_adaptive` and the ablations.

### Maze-Hard A*, `configs/budget_optimized/maze_hard.yaml` with `train.steps=10000` (compile in 4-step blocks, Triton kernel backward, GEMM MLP, fp32)

| | optimised | published train.py |
|---|---|---|
| training time per seed | 105 min (6313 / 6326 / 6229 s), 104–106 min wall incl. compile | 117.5 min |
| speed-up | x1.12 | |
| peak memory | 5.94 GiB, no checkpointing | |
| val_exact (128 mazes) | 0.633 / 0.609 / 0.633 (mean 0.625) | 0.680 / 0.547 / 0.680 (mean 0.635) |

`eval_tts` on the 1 000 test mazes, D=200, K=64 (published command, default chunking, 51–65 min per run without
noise and 22–29 min with noise because the 2 M-cell passes overflow into shared memory):

| | K=1 | conf@64 | pass@64 |
|---|---|---|---|
| no noise | 0.627 ± 0.036 (0.650 / 0.585 / 0.646) | 0.680 ± 0.025 | 0.756 ± 0.005 |
| noise σ 0.15, p 0.3 | 0.632 ± 0.035 | 0.681 ± 0.029 | 0.765 ± 0.006 |

There is no published Maze-Hard `eval_tts` yet (the replication thread has not evaluated its three runs), so these
are the first Maze-Hard test numbers of the replication, not a comparison. BFS: not run.

### Sudoku-OOD, `configs/budget_optimized/sudoku_ood.yaml` with `train.steps=50000` (compile in 4-step blocks, Triton mixing, Triton MLP, bf16)

Preset chosen after a 300-step timing of real `train_fast.py` steps once Windows Smart App Control was disabled:
`budget_optimized` with `triton_mlp` 0.142 s/step against 0.152 s/step for `l4_round3` without it
(`bench_engine.py` cannot time `triton_mlp`: its no-grad measurement runs outside autocast and the kernel needs bf16).
`eval_tts` identical to the published runs: 1 000 official RRN test boards, D=256, K=128, `--record-every 8
--group level --max-cells 500000`.

| % | K=1 | conf@128 |
|---|---|---|
| optimised | 28.5 ± 3.4 (28.8 / 32.5 / 24.3) | 71.0 ± 5.8 (73.3 / 76.7 / 63.1) |
| published train.py | 26.0 ± 6.8 (33.6 / 17.2 / 27.3) | 67.0 ± 8.4 (76.8 / 56.3 / 67.8) |

Population standard deviation, as in `docs/RESULTS.md` (sample: 4.1 / 7.1 optimised, 8.3 / 10.3 published).
`K=1` is `conf_at_k["1"]` and `conf@128` is `conf_at_k["128"]` of `results/bench/quality/sudoku_ood_opt/`, the same
fields as `results/tts/`.

| per seed | optimised | published train.py |
|---|---|---|
| training time | 6 842–6 861 s (1.9 h) | 29 145–29 586 s (8.1–8.2 h) |
| wall incl. compile | 114.4–114.7 min (compile 2.8 s, warm inductor cache) | |
| speed-up | **x4.3** | |
| peak memory | 2.46 GiB (0.137 s/step) | |
| val_exact at 50 000 (180 boards) | 0.217 / 0.328 / 0.233 | 0.344 / 0.156 / 0.222 |

## Summary of the optimised arm (same update counts, 3 seeds, same GPU, vs the published train.py runs)

| task | precision | training-time gain | quality |
|---|---|---|---|
| Maze-OOD (5k) | bf16 | ≈ x2.5 | equivalent (one seed of three weaker at 201×201) |
| Maze-Hard A* (10k) | fp32 | x1.12 | val_exact equivalent; eval_tts K=1 0.627, conf@64 0.680, no published reference yet |
| Sudoku-OOD (50k) | bf16 | x4.3 | equal or better (K=1 28.5 vs 26.0, conf@128 71.0 vs 67.0) |

Not run (budget): ARC, Visual Sudoku, Sudoku-Extreme training, Maze-Hard BFS, the engine-reference arm.

## Reproducing on the PC

`pc_tools/phase1.ps1` (speed) and `pc_tools/phase2.ps1 -Part maze|sudoku` (quality). Both wait while another Python
process is on the machine (any checkout) and retry a failed command once. The committed drivers were tidied after
the campaign (phase 1 now calls the committed `bench_wrap.py` and retries like phase 2; the GPU wait no longer
ignores processes from this checkout); the logs in `pc_logs/` record the commands that actually ran. `pc_tools/check_preset.py` runs the equivalence check of a
preset's `engine:` section against the reference backend; `pc_tools/summarize.py` prints a speed JSON as a table.
