"""Figures and animations. Only `scripts/make_figures.py` and `scripts/animate.py` import this.

Colours follow one fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet,
red) and one sequential hue (blues, light to dark) for ordered series such as difficulty levels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


def seq_colors(n: int):
    """`n` steps of one sequential hue (light to dark blue)."""
    import matplotlib as mpl

    cmap = mpl.colormaps["Blues"]
    return [cmap(0.35 + 0.6 * i / max(1, n - 1)) for i in range(n)]


def band(ax, x, mean, std, label=None, color=None, **kw):
    """Mean curve with a +-1 std shaded band."""
    mean, std = np.asarray(mean), np.asarray(std)
    (line,) = ax.plot(x, mean, label=label, color=color, **kw)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)
    return line


def finish(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------- boards as images
MAZE_COLORS = {
    0: (255, 255, 255),
    1: (42, 120, 214),
    2: (40, 40, 40),
    3: (0, 131, 0),
    4: (227, 73, 72),
}


def maze_image(tokens: np.ndarray, scale: int = 1) -> np.ndarray:
    """(H, W) maze tokens -> (H*scale, W*scale, 3) uint8 image."""
    H, W = tokens.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for t, c in MAZE_COLORS.items():
        img[tokens == t] = c
    if scale > 1:
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
    return img


def sudoku_image(
    tokens: np.ndarray,
    given: np.ndarray | None = None,
    solution: np.ndarray | None = None,
    cell: int = 24,
) -> np.ndarray:
    """Render a 9x9 Sudoku as an image with digits (PIL), given cells in bold grey, wrong cells red."""
    from PIL import Image, ImageDraw

    H, W = tokens.shape
    img = Image.new("RGB", (W * cell + 1, H * cell + 1), "white")
    d = ImageDraw.Draw(img)
    for i in range(H + 1):
        w = 2 if i % 3 == 0 else 1
        d.line([(0, i * cell), (W * cell, i * cell)], fill=(40, 40, 40), width=w)
    for j in range(W + 1):
        w = 2 if j % 3 == 0 else 1
        d.line([(j * cell, 0), (j * cell, H * cell)], fill=(40, 40, 40), width=w)
    for i in range(H):
        for j in range(W):
            t = int(tokens[i, j])
            if t == 0:
                continue
            is_given = given is not None and given[i, j]
            wrong = solution is not None and not is_given and t != solution[i, j]
            color = (90, 90, 90) if is_given else ((227, 73, 72) if wrong else (42, 120, 214))
            d.text((j * cell + cell * 0.33, i * cell + cell * 0.2), str(t), fill=color)
    return np.asarray(img)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def board_strip(
    frames: list[np.ndarray], steps: list[int], path: str | Path, title: str = ""
) -> None:
    """A row of snapshots (Figure 2 style)."""
    import matplotlib.pyplot as plt

    set_style()
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    for ax, f, s in zip(np.atleast_1d(axes), frames, steps, strict=True):
        ax.imshow(f, interpolation="nearest")
        ax.set_title(f"t = {s}")
        ax.axis("off")
        ax.grid(False)
    if title:
        fig.suptitle(title)
    finish(fig, path)
