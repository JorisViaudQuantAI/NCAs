"""Benchmarks (paper Appendix A). `get_task(name)` returns a `Task` instance.

maze_ood        Maze-OOD: 9x9 training mazes, test up to 201x201      (tasks/maze.py)
maze_hard       Maze-Hard: 30x30 mazes with several optimal paths     (tasks/maze.py)
sudoku_ood      Sudoku-OOD: easy training boards, 17-34 clue tests    (tasks/sudoku.py)
sudoku_extreme  Sudoku-Extreme: boards that need backtracking         (tasks/sudoku.py)
arc             ARC-AGI-1 subset with RE-ARC samples                  (tasks/arc.py)
"""

from __future__ import annotations

from .base import Boards, Task

__all__ = ["Boards", "Task", "get_task", "TASKS"]

TASKS = ("maze_ood", "maze_hard", "sudoku_ood", "sudoku_extreme", "arc", "visual_sudoku")


def get_task(name: str, **kwargs) -> Task:
    if name == "maze_ood":
        from .maze import MazeOOD

        return MazeOOD(**kwargs)
    if name == "maze_hard":
        from .maze import MazeHard

        return MazeHard(**kwargs)
    if name == "sudoku_ood":
        from .sudoku import SudokuOOD

        return SudokuOOD(**kwargs)
    if name == "sudoku_extreme":
        from .sudoku import SudokuExtreme

        return SudokuExtreme(**kwargs)
    if name == "visual_sudoku":
        from .visual_sudoku import VisualSudoku

        return VisualSudoku(**kwargs)
    if name == "arc":
        from .arc import ARCTask

        return ARCTask(**kwargs)
    raise KeyError(f"unknown task {name!r}; known: {TASKS}")
