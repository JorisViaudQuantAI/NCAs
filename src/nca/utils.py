"""Small helpers: seeding, device choice, YAML configs with command-line overrides, JSON output."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch; return a CPU generator for explicit sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(name: str | None = None) -> torch.device:
    """`cuda` when available (and not disabled), else `cpu`. Accepts "auto", "cpu", "cuda", "cuda:1"."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config. `overrides` are `a.b.c=value` strings (YAML-parsed values).

    A config may contain `_base: other.yaml` (relative to its own folder); the base is loaded
    first and updated recursively.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base is not None:
        base_cfg = load_config(path.parent / base)
        cfg = _deep_update(base_cfg, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        key, value = ov.split("=", 1)
        _set_dotted(cfg, key, yaml.safe_load(value))
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Timer:
    """`with Timer() as t: ...; t.seconds`."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"
