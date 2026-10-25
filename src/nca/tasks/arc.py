"""ARC-AGI-1 benchmark (Chollet, 2019) as used in the paper (Appendix A.3, B.1, B.3, C.3).

An ARC task is a handful (2-5) of demonstration input/output grid pairs plus one or more test
pairs. Grids are up to 30 x 30 with colours 0-9, and the output grid may have a different size
from the input grid. The NCA is trained on many tasks at once, each with its own learned
*task embedding* (``meta["task_id"]``), and must, from the input grid rendered on a canvas,
produce the output grid.

Canvas encoding (paper B.3)
---------------------------
Every grid is rendered on a fixed ``CANVAS x CANVAS`` (32 x 32) canvas with a vocabulary of
V = 12 tokens: the ten colours ``0-9``, ``PAD = 10`` (outside the grid, masked out of the loss)
and ``BORDER = 11`` (the ring of cells immediately outside the grid, clipped to the canvas, which
marks the extent of the grid). Input and target of a pair are rendered at the *same* integer
scale (nearest-neighbour upsampling) and the *same* offset (their top-left corners coincide), each
with its own border ring, so that the NCA can read the input's extent and must produce the
target's extent. The loss mask of a board is ``target != PAD`` (target grid + its border ring).
Metrics ("exact", "cell_acc") are computed on the target grid region only; "exact_with_border"
also requires the border ring. ``given`` is all True: the input slice of the NCA state is read-only
on the whole canvas, the prediction lives in the output slice.

Online augmentation (``ARCTask.augment``, applied when fresh boards enter the replay pool)
re-renders every pair from its original grids (kept in ``meta["pairs"]``) at a fresh random
integer scale (among the scales that keep both grids inside the canvas) and a fresh random
offset; optionally (``online_dihedral``, ``online_color``) also with a random symmetry of the
square and a random permutation of the colours 1-9 (0, the ARC background, is never permuted).
The paper applies the latter two *offline* to build the 51 test-time-training variants of a
task (``ttt_variants``); here they are also available online.

Scaled-down scope
-----------------
The paper pre-trains on the 400 ARC-AGI-1 training tasks augmented with RE-ARC samples and then
runs test-time training on every evaluation task. On a single consumer GPU we keep the same
pipeline on a subset: ``n_tasks`` training tasks chosen by ``task_seed``, optionally extended
with ``rearc_per_task`` generated examples per task (RE-ARC, Hodel 2024). The "test" split holds
the *test pairs of the same training tasks*, i.e. unseen inputs of seen rules (paper Fig. 3).
Test-time training on evaluation tasks is served by ``ttt_variants``.

Data sources and layout (``<root>/arc/``)
-----------------------------------------
* ARC-AGI-1: ``https://github.com/fchollet/ARC-AGI`` (JSON files ``data/training/<id>.json`` and
  ``data/evaluation/<id>.json`` with keys ``train`` and ``test``, each a list of
  ``{"input": grid, "output": grid}``; 400 + 400 tasks, ids are 8 hex characters). ``fetch_arc``
  downloads the repository archive to ``raw/ARC-AGI-master.zip`` and extracts the task files to
  ``raw/ARC-AGI-master/data/{training,evaluation}/<id>.json``. When the archive cannot be
  downloaded it falls back to the raw files, one request per task, listing the training ids from
  RE-ARC's ``generators.py`` (one ``generate_<id>`` per training task); the evaluation split
  cannot be listed that way and needs the archive. The ``evaluation`` tasks are only used for
  test-time training. Nothing is generated locally: without the data, ``load`` raises, unless
  ``source="synthetic"``.
* RE-ARC: ``https://github.com/michaelhodel/re-arc`` (``dsl.py``, ``utils.py``, ``generators.py``,
  ``verifiers.py``; ``generate_<id>(diff_lb, diff_ub)`` returns ``{"input", "output"}`` as tuples of
  tuples, ``verify_<id>(input) == output`` checks it). ``fetch_rearc`` downloads these four files
  (archive, else raw) to ``raw/re-arc/``; they are imported at runtime, never vendored.
  ``rearc_examples`` generates examples with their protocol (verified, non-degenerate, distinct);
  ``load`` caches them per task in ``rearc/<id>_n{n}_seed{seed}.json``.
* ``source="synthetic"``: three tiny rule-based tasks (``synthetic_fill_enclosed``,
  ``synthetic_recolor_largest``, ``synthetic_mirror_h``, see ``SYNTHETIC_RULES``) generated with
  numpy, so that tests and CPU smoke runs work offline.

Everything here is pure numpy + stdlib (RE-ARC's own code needs matplotlib, which it imports).
"""

from __future__ import annotations

import importlib
import io
import json
import os
import random as pyrandom
import re
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

from nca.tasks.base import Boards, Task

# ----------------------------------------------------------------------------------------------
# Tokens and constants
# ----------------------------------------------------------------------------------------------

PAD = 10
BORDER = 11
CANVAS = 32
N_COLORS = 10
MAX_GRID = 30

ARC_ARCHIVE_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_RAW_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/{split}/{name}.json"
ARC_FOLDER = "ARC-AGI-master"  # top folder of the archive, also used by the raw fallback
REARC_ARCHIVE_URL = "https://github.com/michaelhodel/re-arc/archive/refs/heads/main.zip"
REARC_RAW_URL = "https://raw.githubusercontent.com/michaelhodel/re-arc/main/{name}"
REARC_FILES = ("dsl.py", "utils.py", "generators.py", "verifiers.py")

_TASK_RE = re.compile(r"^[0-9a-f]{8}$")
_ARCHIVE_MEMBER_RE = re.compile(r"^[^/]+/data/(training|evaluation)/([0-9a-f]{8})\.json$")
_GENERATOR_RE = re.compile(r"^def generate_([0-9a-f]{8})\(", re.M)


def _log(msg: str) -> None:
    print(f"[arc] {msg}", flush=True)


def arc_dir(root: str) -> str:
    d = os.path.join(root, "arc")
    os.makedirs(d, exist_ok=True)
    return d


def raw_dir(root: str) -> str:
    return os.path.join(arc_dir(root), "raw")


def data_dir(root: str) -> str:
    """``<root>/arc/raw/ARC-AGI-master/data`` (holds ``training/`` and ``evaluation/``)."""
    return os.path.join(raw_dir(root), ARC_FOLDER, "data")


def rearc_dir(root: str) -> str:
    return os.path.join(raw_dir(root), "re-arc")


# ----------------------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------------------


def as_grid(grid) -> np.ndarray:
    """Validate and convert an ARC grid (list of lists / tuple of tuples / array) to int64."""
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1:
        raise ValueError(f"an ARC grid must be 2-D and non-empty, got shape {g.shape}")
    if g.shape[0] > MAX_GRID or g.shape[1] > MAX_GRID:
        raise ValueError(f"ARC grids are at most {MAX_GRID} x {MAX_GRID}, got {g.shape}")
    if g.min() < 0 or g.max() >= N_COLORS:
        raise ValueError("ARC grids hold colours 0-9")
    return g


def normalize_task(task: dict) -> dict:
    """``{"train": [...], "test": [...]}`` with every grid converted by `as_grid`. A test pair may
    lack its ``"output"`` (hidden test output); it is then stored as None."""
    out = {}
    for split in ("train", "test"):
        pairs = []
        for p in task.get(split, []):
            inp = as_grid(p["input"])
            tgt = as_grid(p["output"]) if p.get("output") is not None else None
            pairs.append({"input": inp, "output": tgt})
        out[split] = pairs
    if not out["train"]:
        raise ValueError("an ARC task needs at least one demonstration pair")
    return out


def load_task_json(path: str) -> dict:
    """Read one ARC task file (keys ``train`` / ``test``) and normalise its grids to arrays."""
    with open(path, encoding="utf-8") as f:
        return normalize_task(json.load(f))


def task_path(root: str, split: str, name: str) -> str:
    return os.path.join(data_dir(root), split, f"{name}.json")


def list_tasks(root: str, split: str = "training") -> list[str]:
    """Sorted task ids present under ``<root>/arc/raw/ARC-AGI-master/data/<split>/``."""
    d = os.path.join(data_dir(root), split)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and _TASK_RE.match(f[:-5]))


# ----------------------------------------------------------------------------------------------
# Dihedral symmetries and colour permutations
# ----------------------------------------------------------------------------------------------


def dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    """Symmetry ``k`` in 0..7 of the square: rotate by ``90 * (k % 4)`` degrees, then transpose
    when ``k >= 4`` (same convention as the maze module). Works on the last two axes."""
    out = np.rot90(grid, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.swapaxes(out, -2, -1)
    return np.ascontiguousarray(out)


def dihedral_inverse(grid: np.ndarray, k: int) -> np.ndarray:
    out = np.swapaxes(grid, -2, -1) if k >= 4 else grid
    return np.ascontiguousarray(np.rot90(out, -(k % 4), axes=(-2, -1)))


def _dihedral_table() -> np.ndarray:
    """``table[k1, k2]`` = k such that ``dihedral(dihedral(g, k1), k2) == dihedral(g, k)``."""
    probe = np.arange(9).reshape(3, 3)
    keys = {dihedral(probe, k).tobytes(): k for k in range(8)}
    table = np.zeros((8, 8), dtype=np.int64)
    for k1 in range(8):
        for k2 in range(8):
            table[k1, k2] = keys[dihedral(dihedral(probe, k1), k2).tobytes()]
    return table


DIHEDRAL_COMPOSE = _dihedral_table()


def compose_dihedral(k1: int, k2: int) -> int:
    """Index of "apply ``k1`` then ``k2``"."""
    return int(DIHEDRAL_COMPOSE[k1, k2])


IDENTITY_PERM = np.arange(N_COLORS, dtype=np.int64)


def random_color_perm(rng: np.random.Generator) -> np.ndarray:
    """A permutation of the colours that keeps 0 (the background) fixed."""
    perm = IDENTITY_PERM.copy()
    perm[1:] = rng.permutation(np.arange(1, N_COLORS))
    return perm


def permute_colors(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """``perm[grid]`` on colour cells; PAD / BORDER tokens (>= 10) are left unchanged."""
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (N_COLORS,) or perm[0] != 0 or set(perm.tolist()) != set(range(N_COLORS)):
        raise ValueError("perm must be a permutation of 0..9 with perm[0] == 0")
    lut = np.concatenate([perm, np.arange(N_COLORS, max(BORDER + 1, N_COLORS))])
    return lut[np.asarray(grid, dtype=np.int64)]


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(perm, dtype=np.int64))


def transform_grid(grid: np.ndarray, k: int, perm: np.ndarray) -> np.ndarray:
    """Dihedral symmetry ``k`` followed by colour permutation ``perm`` (the order does not matter,
    the two commute)."""
    return permute_colors(dihedral(grid, k), perm)


def inverse_transform_grid(grid: np.ndarray, k: int, perm: np.ndarray) -> np.ndarray:
    return dihedral_inverse(permute_colors(grid, inverse_perm(perm)), k)


# ----------------------------------------------------------------------------------------------
# Canvas rendering
# ----------------------------------------------------------------------------------------------


def valid_scales(shapes, canvas: int = CANVAS) -> list[int]:
    """Integer scales ``s >= 1`` such that every ``(h, w)`` in ``shapes`` fits: ``s * h <= canvas``
    and ``s * w <= canvas``."""
    m = max(max(h, w) for h, w in shapes)
    return list(range(1, canvas // m + 1))


def random_placement(
    shapes, rng: np.random.Generator, canvas: int = CANVAS
) -> tuple[int, tuple[int, int]]:
    """Random valid scale and random offset keeping every grid of ``shapes`` inside the canvas."""
    scales = valid_scales(shapes, canvas)
    s = int(scales[rng.integers(len(scales))])
    hmax = max(h for h, _ in shapes)
    wmax = max(w for _, w in shapes)
    oi = int(rng.integers(canvas - s * hmax + 1))
    oj = int(rng.integers(canvas - s * wmax + 1))
    return s, (oi, oj)


def render_grid(
    grid: np.ndarray, canvas: int = CANVAS, scale: int = 1, offset: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Upscale ``grid`` by ``scale`` (nearest neighbour), place its top-left corner at ``offset``
    on a PAD canvas and draw the BORDER ring (clipped to the canvas) around it."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    oi, oj = offset
    if scale < 1 or oi < 0 or oj < 0 or oi + scale * h > canvas or oj + scale * w > canvas:
        raise ValueError(f"grid {g.shape} at scale {scale}, offset {offset} does not fit {canvas}")
    tokens = np.full((canvas, canvas), PAD, dtype=np.int64)
    i0, i1 = max(oi - 1, 0), min(oi + scale * h + 1, canvas)
    j0, j1 = max(oj - 1, 0), min(oj + scale * w + 1, canvas)
    tokens[i0:i1, j0:j1] = BORDER
    tokens[oi : oi + scale * h, oj : oj + scale * w] = np.repeat(np.repeat(g, scale, 0), scale, 1)
    return tokens


def render_pair(
    inp: np.ndarray,
    out: np.ndarray | None,
    canvas: int = CANVAS,
    scale: int | None = None,
    offset: tuple[int, int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render an input/output pair at a common scale and offset (random valid ones when None).

    Returns ``(input_tokens, target_tokens, loss_mask)``, all ``(canvas, canvas)``; the loss mask
    is ``target != PAD``. When ``out`` is None (hidden test output) the target is all PAD and the
    mask all False.
    """
    inp = np.asarray(inp, dtype=np.int64)
    shapes = [inp.shape] + ([out.shape] if out is not None else [])
    if scale is None or offset is None:
        rng = np.random.default_rng() if rng is None else rng
        s, o = random_placement(shapes, rng, canvas)
        scale = s if scale is None else scale
        offset = o if offset is None else offset
    x = render_grid(inp, canvas, scale, offset)
    if out is None:
        y = np.full((canvas, canvas), PAD, dtype=np.int64)
    else:
        y = render_grid(out, canvas, scale, offset)
    return x, y, y != PAD


def unrender(
    tokens: np.ndarray, scale: int, offset: tuple[int, int], shape: tuple[int, int]
) -> np.ndarray:
    """Inverse of `render_grid`: crop the ``shape`` grid rendered at ``scale`` / ``offset`` out of
    a canvas, keeping the top-left cell of every ``scale x scale`` block."""
    h, w = int(shape[0]), int(shape[1])
    oi, oj = int(offset[0]), int(offset[1])
    s = int(scale)
    return np.ascontiguousarray(
        np.asarray(tokens)[oi : oi + s * h : s, oj : oj + s * w : s], dtype=np.int64
    )


def map_back(
    tokens: np.ndarray,
    scale: int,
    offset: tuple[int, int],
    shape: tuple[int, int],
    dihedral_k: int = 0,
    color_perm: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted canvas -> ARC grid in the original task frame: `unrender`, then undo the colour
    permutation and the dihedral symmetry recorded in the board's meta."""
    g = unrender(tokens, scale, offset, shape)
    perm = IDENTITY_PERM if color_perm is None else np.asarray(color_perm)
    return inverse_transform_grid(g, int(dihedral_k), perm)


def grid_region(scale: int, offset: tuple[int, int], shape: tuple[int, int]) -> tuple[slice, slice]:
    """Canvas slices covered by a grid rendered at ``scale`` / ``offset``."""
    s, (oi, oj), (h, w) = int(scale), offset, shape
    return slice(int(oi), int(oi) + s * int(h)), slice(int(oj), int(oj) + s * int(w))


# ----------------------------------------------------------------------------------------------
# Boards construction
# ----------------------------------------------------------------------------------------------


def _object_array(items) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, it in enumerate(items):
        arr[i] = it
    return arr


def _shape_or_zero(g: np.ndarray | None) -> tuple[int, int]:
    return (0, 0) if g is None else (int(g.shape[0]), int(g.shape[1]))


def make_boards(
    pairs: list[tuple[np.ndarray, np.ndarray | None]],
    task_ids,
    task_names,
    canvas: int = CANVAS,
    dihedral_k=None,
    color_perm=None,
    scale: int | None = 1,
    offset: tuple[int, int] | None = (0, 0),
    rng: np.random.Generator | None = None,
    extra_meta: dict | None = None,
) -> Boards:
    """Render ``pairs`` (original grids) transformed by per-board ``dihedral_k`` / ``color_perm``
    (identity when None) at ``scale`` / ``offset`` (random per board when None).

    meta: ``task_id`` (N,), ``task_name`` (N,) str, ``pairs`` (N,) object of (input, output)
    original grids, ``in_shape`` / ``out_shape`` (N, 2) shapes *as rendered* (after the symmetry),
    ``scale`` (N,), ``offset`` (N, 2), ``dihedral_k`` (N,), ``color_perm`` (N, 10),
    ``loss_mask`` (N, canvas, canvas) bool, ``has_target`` (N,) bool.
    """
    N = len(pairs)
    ks = np.zeros(N, dtype=np.int64) if dihedral_k is None else np.asarray(dihedral_k, np.int64)
    perms = (
        np.tile(IDENTITY_PERM, (N, 1)) if color_perm is None else np.asarray(color_perm, np.int64)
    )
    inputs = np.empty((N, canvas, canvas), dtype=np.int64)
    targets = np.empty_like(inputs)
    loss_mask = np.empty((N, canvas, canvas), dtype=bool)
    in_shape = np.zeros((N, 2), dtype=np.int64)
    out_shape = np.zeros((N, 2), dtype=np.int64)
    scales = np.zeros(N, dtype=np.int64)
    offsets = np.zeros((N, 2), dtype=np.int64)
    has_target = np.zeros(N, dtype=bool)
    for i, (inp, out) in enumerate(pairs):
        x = transform_grid(inp, int(ks[i]), perms[i])
        y = None if out is None else transform_grid(out, int(ks[i]), perms[i])
        shapes = [x.shape] + ([y.shape] if y is not None else [])
        if scale is None or offset is None:
            s, o = random_placement(shapes, rng, canvas)
            s = s if scale is None else scale
            o = o if offset is None else offset
        else:
            s, o = scale, offset
        inputs[i], targets[i], loss_mask[i] = render_pair(x, y, canvas, s, o)
        in_shape[i] = x.shape
        out_shape[i] = _shape_or_zero(y)
        scales[i] = s
        offsets[i] = o
        has_target[i] = y is not None
    meta = {
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "task_name": np.asarray(task_names, dtype=str),
        "pairs": _object_array(pairs),
        "in_shape": in_shape,
        "out_shape": out_shape,
        "scale": scales,
        "offset": offsets,
        "dihedral_k": ks,
        "color_perm": perms,
        "loss_mask": loss_mask,
        "has_target": has_target,
    }
    if extra_meta:
        meta.update(extra_meta)
    return Boards(inputs, targets, np.ones_like(inputs, dtype=bool), meta)


# ----------------------------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------------------------

_UNAVAILABLE: set[str] = set()  # URLs whose download failed in this process (no retry)


def _fetch_bytes(url: str, timeout: float = 120.0, retries: int = 6) -> bytes:
    """GET ``url``; HTTP 429 / 5xx answers are retried with exponential back-off (2, 4, ... s).
    Any other failure marks the URL as unavailable for the rest of the process."""
    if url in _UNAVAILABLE:
        raise ConnectionError(f"{url} already failed to download in this process")
    req = urllib.request.Request(url, headers={"User-Agent": "nca-reasoning/0.1"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                _UNAVAILABLE.add(url)
                raise
            wait = 2.0 ** (attempt + 1)
            _log(f"HTTP {e.code} for {url}; retrying in {wait:.0f}s")
            time.sleep(wait)
        except Exception:
            _UNAVAILABLE.add(url)
            raise
    raise ConnectionError(url)  # unreachable


def _download(url: str, dest: str, timeout: float = 120.0) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _log(f"downloading {url}")
    data = _fetch_bytes(url, timeout)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def _extract_arc_archive(archive: str, root: str) -> int:
    """Extract the task files of an ARC-AGI repository archive to `data_dir`. Returns the count."""
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            m = _ARCHIVE_MEMBER_RE.match(info.filename)
            if m is None or info.is_dir():
                continue
            dest = task_path(root, m.group(1), m.group(2))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def rearc_training_ids(root: str, download: bool = True) -> list[str]:
    """The 400 ARC training task ids, read from RE-ARC's ``generators.py`` (``generate_<id>``)."""
    path = os.path.join(rearc_dir(root), "generators.py")
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False")
        _download(REARC_RAW_URL.format(name="generators.py"), path)
    with open(path, encoding="utf-8") as f:
        ids = sorted(set(_GENERATOR_RE.findall(f.read())))
    if not ids:
        raise ValueError(f"no generate_<id> functions found in {path}")
    return ids


N_TRAINING_TASKS = 400
_COMPLETE_MARKER = ".complete"


def arc_complete(root: str) -> bool:
    """True when a full download finished (marker file) or the 400 training tasks are present."""
    marker = os.path.join(raw_dir(root), ARC_FOLDER, _COMPLETE_MARKER)
    return os.path.exists(marker) or len(list_tasks(root, "training")) >= N_TRAINING_TASKS


def _mark_complete(root: str) -> None:
    with open(os.path.join(raw_dir(root), ARC_FOLDER, _COMPLETE_MARKER), "w") as f:
        f.write(
            f"training={len(list_tasks(root, 'training'))} "
            f"evaluation={len(list_tasks(root, 'evaluation'))}\n"
        )


def fetch_arc(root: str, download: bool = True) -> str:
    """Make sure the ARC-AGI-1 task files are under `data_dir(root)`; return that folder.

    Order: existing complete data (marker file ``raw/ARC-AGI-master/.complete`` or 400 training
    tasks; with ``download=False`` any existing files are accepted); the repository archive (both
    splits); the raw files one by one (training split only, ids listed from RE-ARC; an
    interrupted download resumes). Raises FileNotFoundError when nothing works.
    """
    have = len(list_tasks(root, "training"))
    if have and (arc_complete(root) or not download):
        if not arc_complete(root):
            _log(f"warning: only {have} training tasks under {data_dir(root)} (download=False)")
        return data_dir(root)
    if not download:
        raise FileNotFoundError(f"no ARC tasks under {data_dir(root)} and download=False")
    errors = []
    archive = os.path.join(raw_dir(root), f"{ARC_FOLDER}.zip")
    try:
        if not os.path.exists(archive):
            _download(ARC_ARCHIVE_URL, archive)
        n = _extract_arc_archive(archive, root)
        if n == 0:
            raise ValueError("archive holds no data/<split>/<id>.json member")
        _log(f"extracted {n} ARC task files from the archive to {data_dir(root)}")
        _mark_complete(root)
        return data_dir(root)
    except Exception as e:  # noqa: BLE001 - fall back to the raw files
        errors.append(f"archive: {type(e).__name__}: {e}")
        if os.path.exists(archive) and not zipfile.is_zipfile(archive):
            os.remove(archive)
    try:
        ids = rearc_training_ids(root, download=True)
        _log(f"archive unavailable ({errors[-1]}); fetching {len(ids)} raw training task files")
        for name in ids:
            dest = task_path(root, "training", name)
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                data = _fetch_bytes(ARC_RAW_URL.format(split="training", name=name))
                time.sleep(0.05)  # be gentle with raw.githubusercontent.com (rate limited)
                json.loads(data)  # must be valid JSON before it is written
                with open(dest, "wb") as f:
                    f.write(data)
        _log(f"fetched {len(list_tasks(root, 'training'))} training tasks (raw files)")
        _mark_complete(root)
        return data_dir(root)
    except Exception as e:  # noqa: BLE001
        errors.append(f"raw files: {type(e).__name__}: {e}")
    raise FileNotFoundError(
        "could not obtain the ARC-AGI-1 data: "
        + "; ".join(errors)
        + f". Place the repository's data folder at {data_dir(root)} (training/<id>.json, "
        "evaluation/<id>.json) or use source='synthetic'."
    )


def fetch_rearc(root: str, download: bool = True) -> str:
    """Make sure RE-ARC's ``dsl.py``, ``utils.py``, ``generators.py`` and ``verifiers.py`` are under
    ``<root>/arc/raw/re-arc/``; return that folder. Archive first, raw files otherwise."""
    folder = rearc_dir(root)
    if all(os.path.exists(os.path.join(folder, f)) for f in REARC_FILES):
        return folder
    if not download:
        raise FileNotFoundError(f"RE-ARC files not found under {folder} and download=False")
    try:
        _log(f"downloading {REARC_ARCHIVE_URL}")
        data = _fetch_bytes(REARC_ARCHIVE_URL, timeout=300.0)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = {os.path.basename(n): n for n in zf.namelist() if n.count("/") == 1}
            missing = [f for f in REARC_FILES if f not in members]
            if missing:
                raise FileNotFoundError(f"archive lacks {missing}")
            os.makedirs(folder, exist_ok=True)
            for f in REARC_FILES:
                with open(os.path.join(folder, f), "wb") as dst:
                    dst.write(zf.read(members[f]))
        return folder
    except Exception as e:  # noqa: BLE001
        _log(f"RE-ARC archive unavailable ({type(e).__name__}: {e}); fetching the raw files")
    for f in REARC_FILES:
        dest = os.path.join(folder, f)
        if not os.path.exists(dest):
            _download(REARC_RAW_URL.format(name=f), dest)
    return folder


# ----------------------------------------------------------------------------------------------
# RE-ARC generation (their code is imported at runtime from <root>/arc/raw/re-arc/)
# ----------------------------------------------------------------------------------------------

_REARC_MODULES: dict[str, object] = {}


def rearc_modules(root: str, download: bool = True) -> tuple[object, object]:
    """Import (once) RE-ARC's ``generators`` and ``verifiers`` modules; returns both."""
    if "generators" in _REARC_MODULES:
        return _REARC_MODULES["generators"], _REARC_MODULES["verifiers"]
    folder = os.path.abspath(fetch_rearc(root, download))
    for name in ("dsl", "utils", "generators", "verifiers"):
        mod = sys.modules.get(name)
        if (
            mod is not None
            and os.path.dirname(os.path.abspath(getattr(mod, "__file__", "") or "")) != folder
        ):
            raise ImportError(
                f"a top-level module named {name!r} is already imported from elsewhere; "
                "RE-ARC's modules use these names and cannot be loaded in this process"
            )
    if folder not in sys.path:
        sys.path.insert(0, folder)
    gens = importlib.import_module("generators")  # imports dsl and utils (needs matplotlib)
    vers = importlib.import_module("verifiers")
    _REARC_MODULES["generators"], _REARC_MODULES["verifiers"] = gens, vers
    return gens, vers


class _Timeout(Exception):
    pass


def _alarm_available() -> bool:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def rearc_examples(
    task_id: str,
    n: int,
    rng: np.random.Generator,
    root: str,
    diff: tuple[float, float] = (0.0, 1.0),
    verify: bool = True,
    max_attempts: int | None = None,
    timeout: float = 5.0,
    download: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate up to ``n`` RE-ARC examples for ARC training task ``task_id`` (8 hex chars).

    Follows RE-ARC's ``generate_dataset``: an example is kept when the generator returns valid
    grids, ``verify_<id>(input) == output`` (when ``verify``), input != output, and its input was
    not seen before. Every attempt is wrapped in try/except (and a ``timeout`` in seconds when
    ``signal.SIGALRM`` is usable, i.e. on Unix in the main thread) and ``max_attempts``
    (default ``10 * n + 20``) bounds the loop, so a flaky generator just yields fewer examples.
    RE-ARC draws from Python's ``random``, which is seeded from ``rng``.
    """
    gens, vers = rearc_modules(root, download)
    gen = getattr(gens, f"generate_{task_id}", None)
    if gen is None:
        raise KeyError(f"RE-ARC has no generator for task {task_id!r}")
    ver = getattr(vers, f"verify_{task_id}", None) if verify else None
    pyrandom.seed(int(rng.integers(2**31 - 1)))
    max_attempts = 10 * n + 20 if max_attempts is None else max_attempts
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    use_alarm = _alarm_available() and timeout > 0

    def _on_alarm(signum, frame):
        raise _Timeout()

    old_handler = signal.signal(signal.SIGALRM, _on_alarm) if use_alarm else None
    try:
        for _ in range(max_attempts):
            if len(out) >= n:
                break
            try:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, timeout)
                ex = gen(float(diff[0]), float(diff[1]))
                if ver is not None and ver(ex["input"]) != ex["output"]:
                    continue
                if ex["input"] == ex["output"]:
                    continue
                inp, tgt = as_grid(ex["input"]), as_grid(ex["output"])
            except Exception:  # noqa: BLE001 - RE-ARC generators may fail on some draws
                continue
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            key = inp.shape.__repr__().encode() + inp.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((inp, tgt))
    finally:
        if use_alarm:
            signal.signal(signal.SIGALRM, old_handler)
    return out


def rearc_cache_path(root: str, task_id: str, n: int, seed: int) -> str:
    return os.path.join(arc_dir(root), "rearc", f"{task_id}_n{n}_seed{seed}.json")


def _rearc_cached(root: str, task_id: str, n: int, seed: int, download: bool) -> list:
    path = rearc_cache_path(root, task_id, n, seed)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [(as_grid(p["input"]), as_grid(p["output"])) for p in json.load(f)]
    pairs = rearc_examples(task_id, n, np.random.default_rng(seed), root, download=download)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"input": a.tolist(), "output": b.tolist()} for a, b in pairs], f)
    return pairs


# ----------------------------------------------------------------------------------------------
# Synthetic tasks (offline fallback for tests and smoke runs)
# ----------------------------------------------------------------------------------------------


def _place_rects(
    rng: np.random.Generator, h: int, w: int, n: int, size_range: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """``n`` random non-touching rectangles ``(i, j, rh, rw)`` inside ``h x w`` (1-cell gap)."""
    occupied = np.zeros((h + 2, w + 2), dtype=bool)
    rects = []
    for _ in range(200):
        if len(rects) == n:
            break
        rh = int(rng.integers(size_range[0], min(size_range[1], h) + 1))
        rw = int(rng.integers(size_range[0], min(size_range[1], w) + 1))
        i = int(rng.integers(h - rh + 1))
        j = int(rng.integers(w - rw + 1))
        if occupied[i : i + rh + 2, j : j + rw + 2].any():
            continue
        occupied[i : i + rh + 2, j : j + rw + 2] = True
        rects.append((i, j, rh, rw))
    return rects


def _syn_fill_enclosed(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rectangle outlines of colour 3 on black; output fills their interior with colour 4."""
    h, w = (int(v) for v in rng.integers(6, 13, size=2))
    inp = np.zeros((h, w), dtype=np.int64)
    out = inp.copy()
    for i, j, rh, rw in _place_rects(rng, h, w, int(rng.integers(1, 4)), (3, 6)):
        inp[i : i + rh, j : j + rw] = 3
        inp[i + 1 : i + rh - 1, j + 1 : j + rw - 1] = 0
        out[i : i + rh, j : j + rw] = 3
        out[i + 1 : i + rh - 1, j + 1 : j + rw - 1] = 4
    return inp, out


def _syn_recolor_largest(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Solid rectangles of colour 1 with distinct areas; output recolours the largest in 2."""
    for _ in range(100):
        h, w = (int(v) for v in rng.integers(6, 13, size=2))
        rects = _place_rects(rng, h, w, int(rng.integers(2, 5)), (1, 5))
        areas = [rh * rw for _, _, rh, rw in rects]
        if len(rects) >= 2 and areas.count(max(areas)) == 1:
            break
    inp = np.zeros((h, w), dtype=np.int64)
    for i, j, rh, rw in rects:
        inp[i : i + rh, j : j + rw] = 1
    out = inp.copy()
    i, j, rh, rw = rects[int(np.argmax(areas))]
    out[i : i + rh, j : j + rw] = 2
    return inp, out


def _syn_mirror_h(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random coloured grid; output is the input followed by its horizontal mirror image
    (the width doubles), a classic ARC pattern with an output size different from the input."""
    h, w = int(rng.integers(3, 11)), int(rng.integers(3, 9))
    inp = rng.integers(1, N_COLORS, size=(h, w)) * (rng.random((h, w)) < 0.5)
    inp = inp.astype(np.int64)
    return inp, np.concatenate([inp, inp[:, ::-1]], axis=1)


SYNTHETIC_RULES = {
    "fill_enclosed": _syn_fill_enclosed,
    "recolor_largest": _syn_recolor_largest,
    "mirror_h": _syn_mirror_h,
}


def synthetic_tasks(n_tasks: int, seed: int = 0) -> list[tuple[str, dict]]:
    """``n_tasks`` synthetic tasks ``(name, task)`` cycling through `SYNTHETIC_RULES` (3-5
    demonstration pairs and 2 test pairs each); names are ``synthetic_<rule>`` (``_<k>`` suffix
    when a rule is used more than once)."""
    rules = list(SYNTHETIC_RULES.items())
    tasks = []
    for t in range(n_tasks):
        name, fn = rules[t % len(rules)]
        full = f"synthetic_{name}" + (f"_{t // len(rules)}" if t >= len(rules) else "")
        rng = np.random.default_rng([seed, 7, t])
        n_demo = int(rng.integers(3, 6))
        pairs = [fn(rng) for _ in range(n_demo + 2)]
        task = {
            "train": [{"input": a, "output": b} for a, b in pairs[:n_demo]],
            "test": [{"input": a, "output": b} for a, b in pairs[n_demo:]],
        }
        tasks.append((full, normalize_task(task)))
    return tasks


# ----------------------------------------------------------------------------------------------
# Task selection
# ----------------------------------------------------------------------------------------------


def select_task_names(names: list[str], n_tasks: int, seed: int) -> list[str]:
    """``n_tasks`` names drawn without replacement with `seed` (sorted, so that ids are stable)."""
    names = sorted(names)
    if n_tasks >= len(names):
        return names
    idx = np.sort(np.random.default_rng([seed, 11]).choice(len(names), size=n_tasks, replace=False))
    return [names[i] for i in idx]


def official_tasks(
    root: str, split: str = "training", download: bool = True, names: list[str] | None = None
) -> list[tuple[str, dict]]:
    """``(name, task)`` for every (or the given) task of an official split."""
    fetch_arc(root, download)
    available = list_tasks(root, split)
    if not available:
        raise FileNotFoundError(
            f"no ARC {split} tasks under {os.path.join(data_dir(root), split)} (the evaluation "
            "split needs the repository archive)"
        )
    names = available if names is None else list(names)
    missing = sorted(set(names) - set(available))
    if missing:
        raise FileNotFoundError(f"unknown ARC {split} tasks {missing[:5]}...")
    return [(n, load_task_json(task_path(root, split, n))) for n in names]


def evaluation_tasks(
    root: str, n: int | None = None, seed: int = 0, download: bool = True
) -> list[tuple[str, dict]]:
    """``n`` (all when None) ARC evaluation tasks chosen by ``seed``, for test-time training."""
    fetch_arc(root, download)
    names = list_tasks(root, "evaluation")
    if n is not None:
        names = select_task_names(names, n, seed)
    return official_tasks(root, "evaluation", download, names)


# ----------------------------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------------------------


class ARCTask(Task):
    """ARC-AGI-1 on a 32 x 32 canvas (see the module docstring).

    ``n_tasks`` training tasks (chosen by ``task_seed``), ``rearc_per_task`` RE-ARC extras per
    task; online augmentation = random scale + offset (``online_scale``), optionally random
    dihedral symmetry (``online_dihedral``) and colour permutation (``online_color``).
    """

    name = "arc"
    vocab_size = 12
    output_tokens = tuple(range(12))
    token_names = (
        "black",
        "blue",
        "red",
        "green",
        "yellow",
        "grey",
        "magenta",
        "orange",
        "cyan",
        "maroon",
        "pad",
        "border",
    )
    main_metric = "exact"
    # the input slice is read-only on the whole canvas (``given`` all True) and the prediction is
    # read from the separate output slice on every cell: nothing is copied from the inputs
    fill_given = False

    def __init__(
        self,
        n_tasks: int = 30,
        rearc_per_task: int = 0,
        task_seed: int = 0,
        online_scale: bool = True,
        online_dihedral: bool = False,
        online_color: bool = False,
        canvas: int = CANVAS,
    ) -> None:
        self.n_tasks = int(n_tasks)
        self.rearc_per_task = int(rearc_per_task)
        self.task_seed = int(task_seed)
        self.online_scale = bool(online_scale)
        self.online_dihedral = bool(online_dihedral)
        self.online_color = bool(online_color)
        self.canvas = int(canvas)

    # ------------------------------------------------------------------ loading
    def _tasks(self, root: str, download: bool, source: str, task_ids) -> tuple[list, str]:
        if source == "synthetic":
            tasks = synthetic_tasks(self.n_tasks, self.task_seed)
            if task_ids is not None:
                tasks = [(n, t) for n, t in tasks if n in set(task_ids)]
            return tasks, "synthetic"
        if task_ids is None:
            fetch_arc(root, download)
            names = select_task_names(list_tasks(root, "training"), self.n_tasks, self.task_seed)
        else:
            names = list(task_ids)
        return official_tasks(root, "training", download, names), "official"

    def load(
        self,
        split: str,
        root: str,
        n: int | None = None,
        seed: int = 0,
        download: bool = True,
        source: str = "auto",
        task_ids=None,
        **kwargs,
    ) -> Boards:
        """ "train": every demonstration pair of the selected training tasks (+ RE-ARC extras);
        "test": the test pairs of the same tasks. Boards are rendered at scale 1 / offset (0, 0)
        (`augment` re-renders online). ``n`` subsamples the boards with ``seed``. ``task_ids`` is
        an explicit list of task names overriding the seeded selection. meta: see `make_boards`,
        plus ``pair_index`` (N,), ``is_rearc`` (N,) bool, ``source`` str, ``task_names`` tuple."""
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if source not in ("auto", "official", "synthetic"):
            raise ValueError("source must be 'auto', 'official' or 'synthetic'")
        tasks, used = self._tasks(root, download, source, task_ids)
        pairs, tids, names, pair_index, is_rearc = [], [], [], [], []
        for tid, (name, task) in enumerate(tasks):
            demos = [
                (p["input"], p["output"]) for p in task["train" if split == "train" else "test"]
            ]
            if split == "train" and self.rearc_per_task > 0 and used == "official":
                try:
                    seed_t = int(np.random.default_rng([self.task_seed, 3, tid]).integers(2**31))
                    extra = _rearc_cached(root, name, self.rearc_per_task, seed_t, download)
                except Exception as e:  # noqa: BLE001 - RE-ARC is optional
                    _log(f"RE-ARC examples unavailable for {name} ({type(e).__name__}: {e})")
                    extra = []
                else:
                    if len(extra) < self.rearc_per_task:
                        _log(f"RE-ARC: {len(extra)}/{self.rearc_per_task} examples for {name}")
            else:
                extra = []
            for k, p in enumerate(demos + extra):
                pairs.append(p)
                tids.append(tid)
                names.append(name)
                pair_index.append(k)
                is_rearc.append(k >= len(demos))
        if not pairs:
            raise ValueError(f"no {split} pairs in the selected ARC tasks")
        extra_meta = {
            "pair_index": np.asarray(pair_index, dtype=np.int64),
            "is_rearc": np.asarray(is_rearc, dtype=bool),
            "source": used,
            "split": split,
            "task_names": tuple(n for n, _ in tasks),
            "n_tasks": len(tasks),
        }
        boards = make_boards(pairs, tids, names, self.canvas, extra_meta=extra_meta)
        if n is not None and n < len(boards):
            idx = np.sort(np.random.default_rng(seed).choice(len(boards), size=n, replace=False))
            boards = boards.subset(idx)
        _log(
            f"{split}: {len(boards)} boards from {len(tasks)} {used} tasks "
            f"({int(boards.meta['is_rearc'].sum())} RE-ARC)"
        )
        return boards

    # ------------------------------------------------------------------ augmentation
    def augment(self, boards: Boards, rng: np.random.Generator) -> Boards:
        """Re-render every board from its original grids (``meta["pairs"]``) with fresh random
        scale / offset (``online_scale``), dihedral symmetry (``online_dihedral``) and colour
        permutation (``online_color``), composed on top of the board's base transform
        (``base_dihedral_k`` / ``base_color_perm`` when present, e.g. a TTT variant).
        ``dihedral_k`` / ``color_perm`` in the result are the *total* transforms from the original
        task frame, which `map_back` undoes."""
        N = len(boards)
        meta = boards.meta
        base_k = meta.get("base_dihedral_k", np.zeros(N, dtype=np.int64))
        base_perm = meta.get("base_color_perm", np.tile(IDENTITY_PERM, (N, 1)))
        ks = np.asarray(base_k, dtype=np.int64).copy()
        perms = np.asarray(base_perm, dtype=np.int64).copy()
        if self.online_dihedral:
            online = rng.integers(8, size=N)
            ks = np.array(
                [compose_dihedral(int(a), int(b)) for a, b in zip(ks, online, strict=True)]
            )
        if self.online_color:
            for i in range(N):
                perms[i] = random_color_perm(rng)[perms[i]]
        scale, offset = (None, None) if self.online_scale else (1, (0, 0))
        keep = {k: v for k, v in meta.items() if k not in _RENDER_META}
        out = make_boards(
            list(meta["pairs"]),
            meta["task_id"],
            meta["task_name"],
            self.canvas,
            dihedral_k=ks,
            color_perm=perms,
            scale=scale,
            offset=offset,
            rng=rng,
            extra_meta=keep,
        )
        return out

    # ------------------------------------------------------------------ metrics
    def eval_mask(self, boards: Boards) -> np.ndarray:
        """The cells carrying a target: the loss mask (target grid + its border ring); all False
        for a board without target (``has_target``). This is what the accuracy traces and the
        training log score (it matches ``exact_with_border``); ``exact`` / ``cell_acc`` below
        look at the target grid region only."""
        mask = np.asarray(boards.meta["loss_mask"], dtype=bool)
        return mask & np.asarray(boards.meta["has_target"], dtype=bool)[:, None, None]

    def metrics(self, pred: np.ndarray, boards: Boards) -> dict[str, np.ndarray]:
        """ "exact": every cell of the target grid region matches; "exact_with_border": every
        loss-mask cell (grid + border ring) matches; "cell_acc": fraction of target grid cells
        correct. Boards without a target (hidden test output) get False / 0."""
        pred = np.asarray(pred)
        N = len(boards)
        meta = boards.meta
        exact = np.zeros(N, dtype=bool)
        exact_b = np.zeros(N, dtype=bool)
        cell_acc = np.zeros(N, dtype=float)
        eq = pred == boards.targets
        for i in range(N):
            if not meta["has_target"][i]:
                continue
            rs, cs = grid_region(meta["scale"][i], meta["offset"][i], meta["out_shape"][i])
            region = eq[i, rs, cs]
            exact[i] = bool(region.all())
            cell_acc[i] = float(region.mean())
            exact_b[i] = bool(eq[i][meta["loss_mask"][i]].all())
        return {"exact": exact, "exact_with_border": exact_b, "cell_acc": cell_acc}

    def cell_accuracy(self, pred: np.ndarray, boards: Boards) -> np.ndarray:
        return self.metrics(pred, boards)["cell_acc"]

    # ------------------------------------------------------------------ test-time training
    def ttt_variants(
        self,
        task_json: dict,
        n_variants: int = 51,
        rng: np.random.Generator | None = None,
        canvas: int | None = None,
        task_name: str = "",
    ) -> tuple[Boards, Boards]:
        """Offline augmented variants of one task for test-time training (paper C.3).

        Variant 0 is the task itself; every other variant applies a random dihedral symmetry and
        colour permutation (distinct (k, perm) combinations) to both grids of every pair. Returns
        ``(train, test)``: the demonstration pairs, resp. the test pair(s), of every variant with
        ``task_id`` = variant index, rendered at scale 1 / offset (0, 0). meta carries ``variant``,
        ``dihedral_k``, ``color_perm`` (also as ``base_*`` so that `augment` composes on top) and
        for the test boards ``test_index``; `map_back` maps a predicted canvas to the original
        frame. Test targets are rendered when the JSON provides them (``has_target``).
        """
        rng = np.random.default_rng() if rng is None else rng
        canvas = self.canvas if canvas is None else canvas
        task = normalize_task(task_json)
        transforms = [(0, IDENTITY_PERM.copy())]
        seen = {(0, tuple(IDENTITY_PERM.tolist()))}
        tries = 0
        while len(transforms) < n_variants and tries < 100 * n_variants:
            tries += 1
            k, perm = int(rng.integers(8)), random_color_perm(rng)
            key = (k, tuple(perm.tolist()))
            if key in seen:
                continue
            seen.add(key)
            transforms.append((k, perm))
        out = []
        for split in ("train", "test"):
            pairs, tids, ks, perms, variant, index = [], [], [], [], [], []
            for v, (k, perm) in enumerate(transforms):
                for j, p in enumerate(task[split]):
                    pairs.append((p["input"], p["output"]))
                    tids.append(v)
                    ks.append(k)
                    perms.append(perm)
                    variant.append(v)
                    index.append(j)
            ks_arr = np.asarray(ks, dtype=np.int64)
            perms_arr = np.asarray(perms, dtype=np.int64).reshape(len(pairs), N_COLORS)
            extra = {
                "variant": np.asarray(variant, dtype=np.int64),
                "base_dihedral_k": ks_arr,
                "base_color_perm": perms_arr,
                "pair_index": np.asarray(index, dtype=np.int64),
                "source": "ttt",
                "split": split,
                "task_names": (task_name,),
                "n_tasks": len(transforms),
            }
            if split == "test":
                extra["test_index"] = np.asarray(index, dtype=np.int64)
            out.append(
                make_boards(
                    pairs,
                    tids,
                    [task_name] * len(pairs),
                    canvas,
                    dihedral_k=ks_arr,
                    color_perm=perms_arr,
                    extra_meta=extra,
                )
            )
        return out[0], out[1]


# meta keys recomputed by `make_boards` (everything else is carried through `augment`)
_RENDER_META = {
    "task_id",
    "task_name",
    "pairs",
    "in_shape",
    "out_shape",
    "scale",
    "offset",
    "dihedral_k",
    "color_perm",
    "loss_mask",
    "has_target",
}


# ----------------------------------------------------------------------------------------------
# Rendering for figures
# ----------------------------------------------------------------------------------------------

# the usual ARC palette, plus pad (light grey) and border (dark grey)
_COLORS = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    PAD: (235, 235, 235),
    BORDER: (90, 90, 90),
}


def render_arc(tokens: np.ndarray) -> np.ndarray:
    """RGB picture ``(H, W, 3)`` uint8 of a canvas or grid with the ARC palette."""
    lut = np.zeros((BORDER + 1, 3), dtype=np.uint8)
    for k, c in _COLORS.items():
        lut[k] = c
    return lut[np.clip(np.asarray(tokens), 0, BORDER)]


ASCII = "0123456789.#"


def decode_ascii(tokens: np.ndarray) -> str:
    """Text rendering, one char per cell: colour digit, ``.`` pad, ``#`` border."""
    return "\n".join("".join(ASCII[int(v)] for v in row) for row in np.asarray(tokens))
