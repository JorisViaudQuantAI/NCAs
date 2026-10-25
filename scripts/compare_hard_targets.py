"""Which rule produces the Maze-Hard paths shipped with the HRM data? (CPU, about two minutes)

The "A*" targets we train on are the HRM paths; our BFS targets, which no model learns, are
computed by us (docs/DEVIATIONS.md, 2026-10-19). This compares the HRM paths with rule families,
each over the 24 neighbour orders and both search directions (start -> goal, goal -> start):
  local   walk one shortest path cell by cell, each step to the first neighbour one step closer
          to the other end;
  greedy  the same, but among those neighbours prefer the one closest (Manhattan) to the end;
  astar   A* with the Manhattan heuristic, ties on f broken by smaller h, then FIFO or LIFO.
Our `astar_path` and both BFS targets are listed too. The best rules per family are printed.

    python scripts/compare_hard_targets.py [--root data] [--split train] [--n 300]
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca.tasks import maze as M  # noqa: E402

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def local_path(passable, src, dst, order, from_goal):
    """Shortest path; each step goes to the first neighbour (in `order`) one step closer to the
    other end. `from_goal`: walk back from `dst` over distances from `src`, else forward from
    `src` over distances to `dst`."""
    a, b = (src, dst) if from_goal else (dst, src)
    dist = M.bfs_distances(passable, a)
    H, W = passable.shape
    path, (i, j) = [b], b
    while dist[i, j] > 0:
        for d in order:
            di, dj = DIRS[d]
            u, v = i + di, j + dj
            if 0 <= u < H and 0 <= v < W and dist[u, v] == dist[i, j] - 1:
                i, j = u, v
                break
        path.append((i, j))
    return path[::-1] if from_goal else path


def greedy_path(passable, src, dst, order, from_goal):
    """Like `local_path`, but among the neighbours one step closer to the other end, pick the
    one closest (Manhattan) to it, then the first in `order`."""
    a, b = (src, dst) if from_goal else (dst, src)
    dist = M.bfs_distances(passable, a)
    H, W = passable.shape
    rank = {d: k for k, d in enumerate(order)}
    path, (i, j) = [b], b
    while dist[i, j] > 0:
        cand = []
        for d in order:
            di, dj = DIRS[d]
            u, v = i + di, j + dj
            if 0 <= u < H and 0 <= v < W and dist[u, v] == dist[i, j] - 1:
                cand.append((abs(u - a[0]) + abs(v - a[1]), rank[d], (u, v)))
        i, j = min(cand)[2]
        path.append((i, j))
    return path[::-1] if from_goal else path


def astar_variant(passable, src, dst, order, lifo, from_goal):
    """A* (Manhattan) from `src` to `dst`, or from `dst` back to `src` when `from_goal`; ties on f
    broken by smaller h, then by insertion order (FIFO, or LIFO when `lifo`)."""
    a, b = (dst, src) if from_goal else (src, dst)
    H, W = passable.shape
    g, parent, closed, n = {a: 0}, {a: None}, set(), 0
    heap = [(0, 0, 0, a)]
    while heap:
        _, _, _, c = heapq.heappop(heap)
        if c in closed:
            continue
        if c == b:
            break
        closed.add(c)
        for d in order:
            di, dj = DIRS[d]
            nb = (c[0] + di, c[1] + dj)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or not passable[nb] or nb in closed:
                continue
            if nb not in g or g[c] + 1 < g[nb]:
                g[nb], parent[nb] = g[c] + 1, c
                hn = abs(nb[0] - b[0]) + abs(nb[1] - b[1])
                n += 1
                heapq.heappush(heap, (g[nb] + hn, hn, -n if lifo else n, nb))
    path, c = [], b
    while c is not None:
        path.append(c)
        c = parent[c]
    path = path[::-1]  # a ... b
    return path[::-1] if from_goal else path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    data = M._fetch_official_hard(args.root, args.split, download=True)
    inputs, hrm = data["inputs"][: args.n], data["targets_astar"][: args.n]
    rules = {
        "astar (ours)": lambda x: M.astar_target(x),
        "bfs (first discoverer)": lambda x: M.bfs_target(x),
    }
    families = {
        "local": lambda o, fg: lambda P, a, b: local_path(P, a, b, o, fg),
        "greedy": lambda o, fg: lambda P, a, b: greedy_path(P, a, b, o, fg),
        "astar fifo": lambda o, fg: lambda P, a, b: astar_variant(P, a, b, o, False, fg),
        "astar lifo": lambda o, fg: lambda P, a, b: astar_variant(P, a, b, o, True, fg),
    }
    for fam, make in families.items():
        for from_goal in (True, False):
            for order in itertools.permutations("UDLR"):
                name = f"{fam} {'from goal' if from_goal else 'from start'} {''.join(order)}"
                path_fn = make(order, from_goal)

                def rule(x, path_fn=path_fn):
                    ends = M._endpoints(x, (M.START,)) + M._endpoints(x, (M.GOAL,))
                    p = path_fn(x != M.WALL, ends[0], ends[1])
                    return M._write_path(x, p, (M.START, M.GOAL))

                rules[name] = rule
    rows = []
    for name, rule in rules.items():
        same, diff = 0, 0
        for x, t in zip(inputs, hrm, strict=True):
            y = rule(x)
            same += np.array_equal(y, t)
            diff += int(((y == M.PATH) != (t == M.PATH)).sum()) // 2
        rows.append((same / len(inputs), diff / len(inputs), name))
    rows.sort(reverse=True)
    print(f"{len(inputs)} official Maze-Hard {args.split} mazes, HRM path vs each rule")
    print(f"{'rule':32s} {'identical':>10s} {'path cells differing':>22s}")
    shown = rows[:10]
    for fam in ("local", "greedy", "astar fifo", "astar lifo"):
        shown += [r for r in rows if r[2].startswith(fam) and r not in shown][:2]
    shown += [r for r in rows if r[2] in ("astar (ours)", "bfs (first discoverer)")]
    for frac, diff, name in shown:
        print(f"{name:32s} {100 * frac:9.1f}% {diff:22.1f}")


if __name__ == "__main__":
    main()
