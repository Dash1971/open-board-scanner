"""Training views for the board finder, cut from whole scenes.

A view is what the finder sees: a 256 x 256 image and, when a whole board is
in it, the board's corners as fractions of the view. Three kinds are drawn:

- wide: the whole photo, shrunk and padded to a square, as on a first pass;
- zoom: a window round the board with some margin, as on a second pass that
  looks again at the area the first pass picked out;
- none: a window that cuts the board or sits inside it, so that the finder
  learns to say when no whole board is in view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from boardscan.geometry import cut_view, to_view, whole_photo, window_transform


@dataclass
class Scene:
    path: Path | None
    corners: np.ndarray  # 4 x 2 in the scene's pixels: top-left, top-right, bottom-right, bottom-left
    image: np.ndarray | None = None  # kept in memory for real photos; synthetic scenes are read from disk

    def pixels(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        return cv2.cvtColor(cv2.imread(str(self.path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_scenes(folder: str | Path) -> list[Scene]:
    """Every scene written by train.make_scenes into `folder`."""
    scenes = []
    for labels in sorted(Path(folder).glob("labels-*.json")):
        for entry in json.loads(labels.read_text()):
            scenes.append(Scene(Path(folder) / entry["file"], np.asarray(entry["corners"], np.float32)))
    return scenes


def sample_view(scene: Scene, rng: np.random.Generator, allow_none: bool = True):
    """One training view: (image, corners as fractions or zeros, whether a whole board is in view)."""
    image = scene.pixels()
    height, width = image.shape[:2]
    corners = scene.corners.astype(np.float64)
    low, high = corners.min(axis=0), corners.max(axis=0)
    middle, extent = (low + high) / 2, float((high - low).max())
    border = (0, 0, 0) if rng.random() < 0.6 else tuple(float(v) for v in rng.uniform(0, 255) * rng.uniform(0.8, 1.0, 3))
    kind = rng.choice(["wide", "zoom", "none"], p=[0.4, 0.45, 0.15] if allow_none else [0.47, 0.53, 0.0])

    if kind == "none":
        if rng.random() < 0.5:  # a window inside the board
            side = extent * rng.uniform(0.3, 0.85)
            centre = middle + rng.uniform(-0.25, 0.25, 2) * extent
        else:  # a window that cuts the board
            side = extent * rng.uniform(0.8, 1.6)
            angle = rng.uniform(0, 2 * np.pi)
            centre = middle + np.array([np.cos(angle), np.sin(angle)]) * extent * rng.uniform(0.45, 0.8)
        matrix = window_transform(centre, side, rng.uniform(-4, 4))
        inside = to_view(corners, matrix)
        if ((inside > -0.06) & (inside < 1.06)).all():  # too nearly whole to call absent
            return sample_view(scene, rng, allow_none=False)
        return cut_view(image, matrix, border), np.zeros((4, 2), np.float32), False

    for _ in range(6):
        if kind == "wide":
            side = max(width, height) * rng.uniform(1.0, 1.3)
            centre = np.array([(width - 1) / 2, (height - 1) / 2]) + rng.uniform(-0.5, 0.5, 2) * (side - np.array([width, height])).clip(0)
            degrees = rng.uniform(-3, 3)
        else:
            side = extent * rng.uniform(1.06, 2.4)
            centre = middle + rng.uniform(-0.45, 0.45, 2) * (side - extent)
            degrees = rng.uniform(-5, 5)
        matrix = window_transform(centre, side, degrees)
        inside = to_view(corners, matrix)
        if ((inside > 0.012) & (inside < 0.988)).all():
            return cut_view(image, matrix, border), inside.astype(np.float32), True
    matrix = whole_photo(image)  # padded, so the corners are inside even when the board fills the scene
    return cut_view(image, matrix, border), to_view(corners, matrix).astype(np.float32), True


def batch_of(scenes: list[Scene], count: int, rng: np.random.Generator, pool=None):
    """`count` views as (images, corners, present). With a thread pool the views are cut in parallel."""
    picks = rng.integers(len(scenes), size=count)
    seeds = rng.integers(2**62, size=count)
    make = lambda job: sample_view(scenes[int(job[0])], np.random.default_rng(int(job[1])))  # noqa: E731
    views = list(pool.map(make, zip(picks, seeds))) if pool else [make(job) for job in zip(picks, seeds)]
    images, corners, present = zip(*views)
    return np.stack(images), np.stack(corners), np.array(present, np.float32)
