# Data

Everything in this folder except this file is created on first use and ignored by git. The
folder is `data_root` in the configs (default `data`). Each loader looks for a cached `.npz` or
JSON here, then tries to download and convert the official files, then falls back to a local
generator, and prints which source it used (`[maze] ...`, `[sudoku] ...`, `[arc] ...`). The
source is also stored in `meta["source"]` of the loaded boards and copied into the result JSONs
(`"source": "official"` or `"generated"`), so you always know what a number was computed on.

Layout once everything has been fetched or generated:

```
data/
├── README.md
├── maze/
│   ├── raw/                         downloaded archives and csv files
│   ├── maze_ood_official_{train,test}_{size}.npz
│   ├── maze_hard_official_{train,test}_30.npz
│   ├── maze_ood_{split}_{size}_n{n}_seed{seed}.npz      generated
│   └── maze_hard_{split}_30_n{n}_seed{seed}.npz         generated
├── sudoku/
│   ├── raw/                         sudoku.zip, sudoku-hard.zip, train.csv / test.csv of Sudoku-Extreme
│   └── sudoku_{ood,extreme}_{split}_n{n}_seed{seed}_{official,generated_v2}.npz   (generated caches are versioned; v1 files are ignored)
└── arc/
    ├── raw/ARC-AGI-master.zip, raw/ARC-AGI-master/data/{training,evaluation}/<id>.json
    ├── raw/re-arc/{dsl,utils,generators,verifiers}.py
    └── rearc/<id>_n{n}_seed{seed}.json                  generated RE-ARC examples per task
```

To force a source: on `scripts/train.py`, `--set task.train.source=official` (or `generated`,
`synthetic` for ARC); on the eval scripts, `--load source=generated`. `download=false` (same
mechanism) disables network access. When a download fails once in a process it is not retried.

## Maze-OOD (paper Appendix A.1; Bansal et al., 2022)

* **What**: path finding on perfect mazes. Train on 50K mazes of size 9x9, test on 10K mazes per
  size from 9x9 to 37x37 (steps of 2), 10K of 59x59 and 1K of 201x201. Tokens: 0 empty, 1 path,
  2 wall, 3 endpoint (both endpoints share the token). The NCA sees walls and endpoints and must
  mark the path.
* **Official source**: the `easy-to-hard-data` archives,
  `https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_{split}_{size}.tar.gz`,
  downloaded to `data/maze/raw/` and converted from their RGB pixel renderings to token grids
  (the pixel layout is inferred and checked; every converted maze must contain a valid path).
  Cache: `data/maze/maze_ood_official_{split}_{size}.npz`, the full set, subsampled at load time
  (`n`, default 1,000 per test size).
* **Local fallback**: `generate_perfect_maze` in `src/nca/tasks/maze.py`, randomized DFS on a
  k x k lattice rendered on the (2k+1) x (2k+1) grid. The corridor graph is a tree, so the path
  between the two random endpoints is unique, as in the official data. Cache:
  `data/maze/maze_ood_{split}_{size}_n{n}_seed{seed}.npz` (keys `inputs`, `targets`,
  `path_len`). Train and test use independent random streams.
* **Caveats**: the generated mazes have the same structure as the official ones but are not the
  same instances, and their endpoint placement and dead-end statistics are not guaranteed to
  match. The official 201x201 test set has only 1,000 mazes; the phase scripts use 100 of them
  for Table 1 and 20 for the adaptive-compute runs.

## Maze-Hard (Appendix A.1; Wang et al., 2025, HRM)

* **What**: 30x30 mazes with loops, shortest paths of at least 110 cells, often several optimal
  paths. 1,000 train, 1,000 test. Tokens: 0 empty, 1 path, 2 wall, 3 start, 4 goal. The shipped
  answers, used as the "A*" targets (`targets: astar`), are in fact exactly a BFS from the start
  that expands neighbours down, left, right, up (checked on all 1,000 training mazes,
  `scripts/compare_hard_targets.py`). The BFS targets of the published runs (`targets: bfs_dlru`)
  are that BFS recomputed on each dihedral transform of the maze, as in the paper; BFS targets in
  the order up, down, left, right (`bfs`, `bfs_local`) are never learned (`docs/DEVIATIONS.md`,
  2026-10-19).
* **Official source**: Hugging Face dataset `sapientinc/Maze-30x30-Hard-1k`,
  `https://huggingface.co/datasets/sapientinc/Maze-30x30-Hard-1k/resolve/main/{train,test}.csv`
  (columns `source,q,a,rating`; `q` and `a` are 900-character strings over `# SGo`). Downloaded
  to `data/maze/raw/maze_hard/`, parsed by `parse_hrm_maze`, BFS targets recomputed with our own
  BFS. Cache: `data/maze/maze_hard_official_{split}_30.npz` (keys `inputs`, `targets_astar`,
  `targets_bfs` and the other target variants, `path_len`).
* **Local fallback**: `generate_hard_maze`: a 31x31 perfect maze with about 7% of its removable
  interior walls knocked out (loops), cropped to 30x30, endpoints chosen so that the shortest path
  has at least 110 cells; targets by A* (Manhattan heuristic, deterministic tie-break) and by BFS.
  Cache: `data/maze/maze_hard_{split}_30_n{n}_seed{seed}.npz`.
* **Caveats**: the generator only approximates the HRM data (whose exact generator is not
  reproduced): the amount of ambiguity (number of optimal paths) and the A* tie-breaking of the
  official answers can differ, which directly affects the `exact` metric. `valid` and `optimal`
  are robust to this. Results on generated Maze-Hard data should not be compared to the paper's
  `exact` numbers without that caveat.

## Sudoku-OOD (Appendix A.2; Miyato et al., 2024, AKOrN)

* **What**: train on 9,000 easy boards (31 to 42 clues) plus 1,000 validation boards from
  SAT-Net, test on the RRN set: 18 difficulty levels defined by the number of empty cells (47 to
  64, i.e. 34 down to 17 clues), 1,000 boards per level. Tokens: 0 empty, 1-9 digits. Metrics:
  `exact`, `valid` (the filled grid satisfies every constraint), `cell_acc`.
* **Official sources**:
  * SAT-Net train/val: `https://powei.tw/sudoku.zip` (`features.pt` / `labels.pt`, one-hot
    tensors; the first 9,000 boards are train, the last 1,000 val). Reading them needs torch.
  * RRN test (DGL mirror): `https://data.dgl.ai/dataset/sudoku-hard.zip` (`train.csv`,
    `valid.csv`, `test.csv`, lines of `puzzle,solution`).
  * Downloads go to `data/sudoku/raw/`. Cache:
    `data/sudoku/sudoku_ood_{split}_n{n|all}_seed{seed}_official.npz` with `inputs`, `targets`,
    `given` and the meta arrays `n_empty`, `difficulty` (backtracks of our solver), `level`.
* **Local fallback**: `generate_puzzle` in `src/nca/tasks/sudoku.py`: a random full grid (bitmask
  backtracking solver with MRV and naked singles, shuffled candidates), clues removed in random
  order while the puzzle keeps a unique solution. Train boards use 31 to 42 clues (the SAT-Net
  range); test boards are requested at each of the 18 levels and spread evenly over them. Cache:
  `data/sudoku/sudoku_ood_{split}_n{n}_seed{seed}_generated.npz` (defaults: 9,000 train, 1,000
  val, 1,800 test).
* **Caveats**: one removal pass yields a minimal puzzle with typically 22 to 26 clues, so the
  generator cannot reach the hardest levels (17 to 21 clues). Generation is best effort: the
  lowest reachable clue count is accepted, the achieved distribution over the 18 levels is printed
  and `meta["level"]` holds the actual number of empty cells. Only the official RRN data covers
  the full 47-64 range, so the per-level curves of Fig. 4A / 16A can only be compared to the paper
  when `source` is `official`. `difficulty` (our solver's backtracks) is not the tdoku rating.

## Sudoku-Extreme (Appendix A.2; Wang et al., 2025, HRM)

* **What**: 1,000 training boards and 422,780 test boards whose difficulty `rating` is the number
  of backtracks of the tdoku solver (0 to 289). The paper's Fig. 12 bins them into 0, 1-5, 6-11,
  12-16, 17-28, 29-53, 54+ (`EXTREME_BINS`); `--group bin` uses these bins.
* **Official source**: Hugging Face dataset `sapientinc/sudoku-extreme`,
  `https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/{train,test}.csv`
  (header `source,question,answer,rating`). `train.csv` has about 3.8M lines, so only its first
  100,000 lines are streamed and the 1,000 training boards are sampled from that head. The test
  split is stratified over the 7 bins (`n` boards in total, default 1,000; `n=None` loads all
  422,780). Cache: `data/sudoku/sudoku_extreme_{split}_n{n|all}_seed{seed}_official.npz` with
  `rating`, `bin`, `n_empty`, `difficulty`.
* **Local fallback**: the same generator with 22 to 30 clues, binned by our solver's backtracks
  (`meta["difficulty_scale"] = "backtracks(ours)"`). Cache:
  `data/sudoku/sudoku_extreme_{split}_n{n}_seed{seed}_generated.npz`.
* **Caveats**: generated boards are much easier than the official ones (tdoku ratings of 50+
  backtracks need boards the generator does not produce), and our backtrack count is a different
  scale from tdoku's, so the per-bin breakdown is only meaningful on official data. The headline
  conf@K number on generated data would be optimistic.

## ARC-AGI-1 (Appendix A.3; Chollet, 2019) and RE-ARC (Hodel, 2024)

* **What**: 400 training and 400 evaluation tasks, each a few input/output grid pairs (up to
  30x30, colours 0-9) plus test pairs. Here: `n_tasks` training tasks (30 in
  `configs/arc_small.yaml`, chosen by `task_seed`), their demonstration pairs for training and
  their test pairs for evaluation, rendered on a 32x32 canvas with 12 tokens (colours 0-9,
  `PAD` = 10 outside the grid and masked out of the loss, `BORDER` = 11 around the grid).
* **Official source**: `https://github.com/fchollet/ARC-AGI` as an archive
  (`data/arc/raw/ARC-AGI-master.zip`, extracted to `raw/ARC-AGI-master/data/{training,evaluation}/<id>.json`);
  if the archive cannot be downloaded, the training task files are fetched one by one from the
  raw GitHub URLs (the training ids are listed from RE-ARC's `generators.py`); the evaluation
  split needs the archive.
* **RE-ARC extras**: the four source files of `https://github.com/michaelhodel/re-arc`
  (`dsl.py`, `utils.py`, `generators.py`, `verifiers.py`) are downloaded to `data/arc/raw/re-arc/`
  and imported at runtime (they are never vendored; RE-ARC's code imports matplotlib).
  `rearc_per_task` examples per task are generated with their protocol (verified, non-degenerate,
  distinct) and cached in `data/arc/rearc/<id>_n{n}_seed{seed}.json`. When RE-ARC is
  unavailable, training proceeds with the demonstration pairs only and a notice is printed.
* **Local fallback**: none for the real tasks. `source="synthetic"` provides three tiny
  rule-based tasks (fill enclosed regions, recolour the largest object, horizontal mirror) for
  the tests and CPU smoke runs. Without the data, `load` raises.
* **Caveats**: the paper pre-trains on all 400 tasks plus RE-ARC and then fine-tunes per
  evaluation task; this repo's phase 3 trains on 30 tasks and evaluates on unseen inputs of those
  same tasks. The numbers are therefore not comparable to Table 1's ARC rows; they test the
  multi-task conditioning (Fig. 3) and the local propagation dynamics (Fig. 2C).

## Visual Sudoku (Appendix A.4)

Not replicated (training took three days on 64 TPUs in the paper). The Sobel perception module
exists in `src/nca/model.py` but no dataset code is provided.

## Licenses

The datasets belong to their authors and come with their own licenses (easy-to-hard-data,
HRM / Sapient datasets on Hugging Face, SAT-Net, RRN, ARC-AGI under Apache 2.0, RE-ARC). Nothing
from them is redistributed in this repository.
