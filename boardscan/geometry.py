"""Windows and warps shared by training and by the scanning pipeline.

Pixel centres sit at whole numbers everywhere; positions inside a view are
given as fractions of its width and height, so that 0 and 1 are its edges.
"""

from __future__ import annotations

import cv2
import numpy as np

VIEW = 256


def window_transform(centre: np.ndarray, side: float, degrees: float = 0.0, size: int = VIEW) -> np.ndarray:
    """The 2 x 3 matrix that maps a square window of an image onto a view.

    The window is `side` pixels across, centred on `centre`, and turned by
    `degrees`.
    """
    scale = size / side
    radians = np.deg2rad(degrees)
    turn = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]) * scale
    shift = (size - 1) / 2 - turn @ np.asarray(centre, np.float64)
    return np.concatenate([turn, shift[:, None]], axis=1)


def cut_view(image: np.ndarray, matrix: np.ndarray, border: tuple[float, float, float] = (0, 0, 0), size: int = VIEW) -> np.ndarray:
    """Cut the window described by `matrix` out of `image`, shrinking in two steps to avoid aliasing."""
    scale = float(np.linalg.norm(matrix[:, 0]))
    if scale < 0.6:
        shrink = scale * 1.25
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        # Pixel centre p of the original lands at (p + 0.5) * shrink - 0.5 in the shrunk image.
        linear = matrix[:, :2] / shrink
        matrix = np.concatenate([linear, (matrix[:, 2] + linear @ np.full(2, 0.5 - 0.5 * shrink))[:, None]], axis=1)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def to_view(points: np.ndarray, matrix: np.ndarray, size: int = VIEW) -> np.ndarray:
    """Image points as fractions of the view."""
    return ((points @ matrix[:, :2].T + matrix[:, 2]) + 0.5) / size


def from_view(fractions: np.ndarray, matrix: np.ndarray, size: int = VIEW) -> np.ndarray:
    """Fractions of the view as image points: the reverse of to_view."""
    return (np.asarray(fractions, np.float64) * size - 0.5 - matrix[:, 2]) @ np.linalg.inv(matrix[:, :2]).T


def whole_photo(image: np.ndarray, margin: float = 1.12) -> np.ndarray:
    """The window for a first look: the whole image, centred, padded to a square.

    The margin leaves a border all round, so that a board filling the whole
    image, as in a tightly cropped screenshot, still has its corners inside
    the view.
    """
    height, width = image.shape[:2]
    return window_transform(np.array([(width - 1) / 2, (height - 1) / 2]), float(max(width, height)) * margin)


def around(corners: np.ndarray, margin: float = 1.35) -> np.ndarray:
    """The window for a second look: a square round a board that was found."""
    low, high = corners.min(axis=0), corners.max(axis=0)
    return window_transform((low + high) / 2, float(max((high - low).max(), 8.0) * margin))


def rectify(image: np.ndarray, corners: np.ndarray, size: int = VIEW) -> np.ndarray:
    """Warp the board inside `corners` to a size x size square, as the classifier expects it."""
    quad = np.asarray(corners, np.float32)
    target = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]], np.float32)
    side = float(np.linalg.norm(quad[1] - quad[0]))
    if side > size * 1.5:  # soften before a strong reduction to limit aliasing
        image = cv2.GaussianBlur(image, (0, 0), side / size * 0.5)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
