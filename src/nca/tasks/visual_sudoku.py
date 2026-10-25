"""Pixel-input compute fixture for Visual Sudoku, explicitly not the MNIST benchmark."""

import numpy as np

from .base import Boards, Task


class PixelBoards(Boards):
    def __post_init__(self):
        if self.inputs.shape != self.targets.shape or self.inputs.shape != self.given.shape:
            raise ValueError("pixel board shapes differ")
        if self.inputs.ndim != 3:
            raise ValueError("pixel boards must have shape N,H,W")
        self.inputs = self.inputs.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        self.given = self.given.astype(bool)


class VisualSudoku(Task):
    name = "visual_sudoku"
    vocab_size = 1
    output_tokens = (0,)
    fill_given = False
    main_metric = "pixel_mse"

    def load(self, split, root, n=4, seed=0, source="generated", **kwargs):
        if source != "generated":
            raise ValueError("Official MNIST Visual Sudoku data pipeline is not implemented")
        from PIL import Image, ImageDraw, ImageFont

        rng = np.random.default_rng(seed)
        font = ImageFont.load_default(size=23)
        inputs, targets = [], []
        for _ in range(n):
            digits = rng.permutation(np.arange(1, 10))
            board = np.array(
                [[digits[(i * 3 + i // 3 + j) % 9] for j in range(9)] for i in range(9)]
            )
            clues = rng.random((9, 9)) < 0.35
            images = []
            for only_clues in (True, False):
                image = Image.new("L", (256, 256))
                draw = ImageDraw.Draw(image)
                for i in range(9):
                    for j in range(9):
                        if not only_clues or clues[i, j]:
                            draw.text(
                                (j * 28 + 7 + int(rng.integers(-1, 2)), i * 28 + 2),
                                str(board[i, j]),
                                font=font,
                                fill=255,
                            )
                images.append(np.asarray(image, dtype=np.float32) / 255)
            inputs.append(images[0])
            targets.append(images[1])
        a, b = np.stack(inputs), np.stack(targets)
        return PixelBoards(
            a,
            b,
            np.ones_like(a, dtype=bool),
            {"source": "generated font glyphs, not MNIST", "seed": seed},
        )

    def eval_mask(self, boards):
        return np.ones_like(boards.given)

    def metrics(self, pred, boards):
        return {"pixel_mse": ((pred - boards.targets) ** 2).mean(axis=(1, 2))}
