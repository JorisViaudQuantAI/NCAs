| Experiment | Paper | Ours (mean ± std over seeds) | n |
|---|---|---|---|
| Maze-OOD 13x13, D=300, exact | 100.0 | 100.0 ± 0.0 | 3 |
| Maze-OOD 59x59, D=2000, exact | 100.0 | 98.3 ± 1.7 | 3 |
| Maze-OOD 201x201, D=13000, exact | 100.0 | 85.7 ± 10.8 | 3 |
| Maze-Hard (A*), K=1, exact | 86.6 | 65.4 ± 4.1 | 3 |
| Maze-Hard (A*), conf@K, exact | 89.2 (K=128) | 70.3 ± 2.9 | 3 |
| Maze-Hard (BFS), K=1, exact | 94.7 | 77.0 ± 3.9 | 3 |
| Maze-Hard (BFS), conf@K, exact | 96.2 (K=128) | 80.4 ± 4.3 | 3 |
| Sudoku-OOD, K=1 | 66.1 (D=256) | 26.0 ± 6.8 | 3 |
| Sudoku-OOD, conf@K | 98.5 (K=2048) | 67.0 ± 8.4 | 3 |
| Sudoku-Extreme, K=1 (benchmark-weighted) | 75.6 (D=2048) | not run yet | 0 |
| Sudoku-Extreme, conf@K (benchmark-weighted) | 92.7 (K=512) | not run yet | 0 |
| ARC (30 tasks, proof of concept), majority pass@2 | 48.8 (full ARC-AGI-1, K=3264) | not run yet | 0 |
| ARC (30 tasks, proof of concept), conf@K | - | not run yet | 0 |
| maze201 adaptive: steps to 95 vs uniform | 1.29x | 1.41x ± 0.14 | 2 |
| maze201 adaptive: updates to 95 vs uniform | 0.71x | 1.07x ± 0.10 | 2 |
| maze201 adaptive_given: steps to 95 vs uniform | 1.29x | 1.41x ± 0.14 | 2 |
| maze201 adaptive_given: updates to 95 vs uniform | 0.71x | 0.72x ± 0.06 | 2 |
| maze201_damage adaptive: steps to 95 vs uniform | 0.94x | 1.30x ± 0.00 | 1 |
| maze201_damage adaptive: updates to 95 vs uniform | 0.52x | 0.98x ± 0.00 | 1 |
| maze201_damage adaptive: steps since damage to 95 vs uniform | - (0.94x if read as elapsed) | 1.54x ± 0.00 | 1 |
| maze201_damage adaptive: updates since damage to 95 vs uniform | - (0.52x if read as elapsed) | 1.15x ± 0.00 | 1 |
| maze201_damage adaptive_given: steps to 95 vs uniform | 0.94x | 1.30x ± 0.00 | 1 |
| maze201_damage adaptive_given: updates to 95 vs uniform | 0.52x | 0.66x ± 0.00 | 1 |
| maze201_damage adaptive_given: steps since damage to 95 vs uniform | - (0.94x if read as elapsed) | 1.54x ± 0.00 | 1 |
| maze201_damage adaptive_given: updates since damage to 95 vs uniform | - (0.52x if read as elapsed) | 0.77x ± 0.00 | 1 |
