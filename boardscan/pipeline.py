"""Find the board in an image, read it, and return the position.

The two models are passed in as plain functions, so the same steps run on
ONNX files here and on PyTorch models during training:

    finder(view)      256 x 256 x 3 uint8 -> (corners 4 x 2 as fractions, logit that a whole board is in view)
    reader(crops)     N x 256 x 256 x 3 uint8 -> (square probabilities N x 13 x 8 x 8,
                                                   quarter-turn probabilities N x 4,
                                                   probability that White is at the bottom, N)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import around, cut_view, from_view, rectify, whole_photo

SYMBOLS = ("", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")


@dataclass
class Scan:
    corners: np.ndarray  # 4 x 2 in image pixels: top-left, top-right, bottom-right, bottom-left as displayed
    board_in_view: float  # 0..1, from the finder's last look
    quarter_turns: int  # anticlockwise quarter turns that bring the crop upright
    white_at_bottom: bool
    seen: np.ndarray  # 8 x 8 class numbers of the upright board, top row first
    confidence: np.ndarray  # 8 x 8, probability of the chosen class per square

    def placement(self) -> str:
        """The piece placement field of a FEN."""
        grid = self.seen if self.white_at_bottom else self.seen[::-1, ::-1]
        rows = []
        for row in grid:
            text, gap = "", 0
            for value in row:
                if value:
                    text += (str(gap) if gap else "") + SYMBOLS[int(value)]
                    gap = 0
                else:
                    gap += 1
            rows.append(text + (str(gap) if gap else ""))
        return "/".join(rows)


def find_board(finder, image: np.ndarray, looks: int = 2) -> tuple[np.ndarray, float]:
    """The board's corners in image pixels and how sure the finder is that a whole board is in view.

    The first look takes in the whole image. Each further look takes a closer
    window round the board found so far, which sharpens the corners.
    """
    matrix = whole_photo(image)
    corners, logit = None, 0.0
    for _ in range(looks):
        fractions, logit = finder(cut_view(image, matrix))
        corners = from_view(fractions, matrix)
        matrix = around(corners)
    return corners.astype(np.float32), float(1.0 / (1.0 + np.exp(-logit)))


def read_board(reader, image: np.ndarray, corners: np.ndarray, turns: tuple[int, ...] = (0, 1, 2, 3)):
    """Read the board inside `corners`, averaging readings of the crop at each of `turns`.

    Returns (upright 8 x 8 classes, upright 8 x 8 confidence, quarter turns, White at bottom).
    """
    crop = rectify(image, corners)
    squares, rotation, white = reader(np.stack([np.rot90(crop, extra) for extra in turns]))
    lying = np.mean([np.rot90(squares[index], -extra, axes=(1, 2)) for index, extra in enumerate(turns)], axis=0)
    quarter_turns = int(np.mean([np.roll(rotation[index], extra) for index, extra in enumerate(turns)], axis=0).argmax())
    upright = np.rot90(lying, quarter_turns, axes=(1, 2))
    return upright.argmax(axis=0), upright.max(axis=0), quarter_turns, bool(np.mean(white) >= 0.5)


def scan(finder, reader, image: np.ndarray) -> Scan:
    """Everything in one call, for an RGB image of any size."""
    corners, in_view = find_board(finder, image)
    seen, confidence, quarter_turns, white_at_bottom = read_board(reader, image, corners)
    return Scan(corners, in_view, quarter_turns, white_at_bottom, seen, confidence)


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = np.exp(values - values.max(axis=axis, keepdims=True))
    return shifted / shifted.sum(axis=axis, keepdims=True)


def onnx_finder(path: str | Path):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def finder(view: np.ndarray):
        corners, present = session.run(["corners", "present"], {"image": np.ascontiguousarray(view.transpose(2, 0, 1)[None], dtype=np.float32)})
        return corners[0], float(present[0])

    return finder


def onnx_reader(path: str | Path):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def reader(crops: np.ndarray):
        squares, white, rotation = session.run(["squares", "white_at_bottom", "rotation"], {"image": np.ascontiguousarray(crops.transpose(0, 3, 1, 2), dtype=np.float32)})
        return _softmax(squares, 1), _softmax(rotation, 1), 1.0 / (1.0 + np.exp(-white))

    return reader
