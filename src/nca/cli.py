"""Shared pieces of the `scripts/eval_*.py` command lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .tasks import get_task
from .tasks.base import Boards
from .train import load_model
from .utils import load_json


def add_common(ap: argparse.ArgumentParser, default_split: str = "test") -> None:
    ap.add_argument(
        "--run", required=True, help="training run folder (contains final.pt and run.json)"
    )
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--split", default=default_split)
    ap.add_argument(
        "--n", type=int, default=None, help="number of test boards (default: all loaded)"
    )
    ap.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra keyword for task.load (e.g. size=59)",
    )
    ap.add_argument("--seed", type=int, default=None, help="rollout seed (default: the run's seed)")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--max-cells", type=int, default=2_000_000, help="chunking budget: cells per pass"
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--progress", action="store_true")


def eval_source(cfg: dict, info: dict) -> str | None:
    """The data source an evaluation of this run must use, decided at evaluation time so that
    it also covers checkpoints whose embedded config predates `source: official`.

    `official` unless the config's test split names a source or the run itself was trained on
    generated / auto data (smoke runs and tests), where None keeps the loader's default. A
    `--load source=...` on the command line still wins (it is applied after this)."""
    tcfg = cfg["task"]
    if "source" in (tcfg.get("test") or {}):
        return tcfg["test"]["source"]
    trained_on = (info.get("data") or {}).get("train_source")
    asked = (tcfg.get("train") or {}).get("source")
    # a run trained on non-official data is evaluated on the same kind of data: generated
    # mazes / Sudoku, or ARC's synthetic tasks (which must not be swapped for official tasks)
    for s in (asked, trained_on):
        if s in ("generated", "synthetic"):
            return s
    if asked == "auto":
        return None
    return "official"


def load_run(run_dir: str, device: str = "auto"):
    """Return (model, config dict, seed, task). The returned config's `task.test` carries the
    evaluation data source (see `eval_source`), which every eval script loads its boards with."""
    run = Path(run_dir)
    # a run folder (its final.pt), or one of its weight files (final_raw.pt, final_<ema>.pt)
    weights, run = (run, run.parent) if run.suffix == ".pt" else (run / "final.pt", run)
    model, cfg = load_model(weights, device)
    info = load_json(run / "run.json")
    tcfg = cfg["task"]
    task = get_task(tcfg["name"], **tcfg.get("args", {}))
    source = eval_source(cfg, info)
    if source is not None:
        tcfg["test"] = {**(tcfg.get("test") or {}), "source": source}
    return model, cfg, int(info["seed"]), task


def parse_kv(items: list[str]) -> dict:
    return {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in items)}


def load_boards(task, cfg: dict, split: str, n: int | None, extra: dict) -> Boards:
    kw = dict(cfg["task"].get("test", {}))
    kw.update(extra)
    if n is not None:
        kw["n"] = n
    return task.load(split, cfg.get("data_root", "data"), **kw)
