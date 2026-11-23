"""Checks for the board finder and its training views; skipped when PyTorch is not installed."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.finder import GRID, BoardFinder, corners_from, parameter_count  # noqa: E402
from train.finder_data import Scene, batch_of, sample_view  # noqa: E402
from train.train_finder import heatmap_targets, score_finder  # noqa: E402

QUAD = np.array([[300, 200], [700, 230], [680, 620], [280, 590]], np.float32)
COLOURS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))


def scene() -> Scene:
    image = np.full((900, 1200, 3), 128, np.uint8)
    for corner, colour in zip(QUAD, COLOURS):
        cv2.circle(image, tuple(int(v) for v in corner), 12, colour, -1)
    return Scene(None, QUAD, image)


def test_outputs_have_the_documented_shapes_and_size():
    with torch.no_grad():
        corners, present, logits = BoardFinder().eval()(torch.rand(2, 3, 256, 256) * 255)
    assert corners.shape == (2, 4, 2) and present.shape == (2,) and logits.shape == (2, 4, GRID, GRID)
    assert 0 < float(corners.min()) and float(corners.max()) < 1
    assert parameter_count(BoardFinder()) < 2_000_000


def test_corners_are_read_from_where_the_heatmaps_point():
    target = torch.tensor([[[0.31, 0.22], [0.84, 0.19], [0.80, 0.77], [0.27, 0.81]]])
    blobs = heatmap_targets(target)
    assert torch.allclose(blobs.sum(dim=(2, 3)), torch.ones(1, 4))
    centres = (torch.arange(GRID, dtype=torch.float32) + 0.5) / GRID
    assert torch.allclose(corners_from(torch.log(blobs + 1e-12), centres), target, atol=1e-3)


def test_views_keep_corners_on_their_marks():
    rng = np.random.default_rng(0)
    whole = cut = 0
    for _ in range(60):
        image, corners, present = sample_view(scene(), rng)
        if not present:
            cut += 1
            assert not corners.any()
            continue
        whole += 1
        assert corners.min() > 0.01 and corners.max() < 0.99
        for corner, colour in zip(corners, COLOURS):
            x, y = np.round(corner * 256 - 0.5).astype(int)
            patch = image[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].reshape(-1, 3).mean(axis=0)
            assert np.abs(patch - np.array(colour)).max() < 90, (corner, colour, patch)
    assert whole > 35 and cut > 2
    images, corners, present = batch_of([scene()], 5, rng)
    assert images.shape == (5, 256, 256, 3) and corners.shape == (5, 4, 2) and present.shape == (5,)


def test_finder_scores_are_shares_of_the_board_side():
    def near(view):  # a finder that is always right would need the scene; this one always answers the middle
        return np.full((4, 2), 0.5), 3.0

    report = score_finder(near, [scene()], looks=1)
    assert report["scenes"] == 1 and report["within_tolerance"] == 0 and report["median_error"] > 0.3 and report["board_reported_in_view"] == 1
