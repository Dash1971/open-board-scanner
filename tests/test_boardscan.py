"""Checks for the scanning pipeline's geometry and bookkeeping (no models needed)."""

import cv2
import numpy as np

from boardgen import rectify as generator_rectify
from boardscan.geometry import around, cut_view, from_view, rectify, to_view, whole_photo, window_transform
from boardscan.pipeline import Scan, find_board, read_board

QUAD = np.array([[300, 200], [700, 230], [680, 620], [280, 590]], np.float32)


def photo_with_quad(height=900, width=1200) -> np.ndarray:
    image = np.full((height, width, 3), 235, np.uint8)
    cv2.fillPoly(image, [QUAD.astype(np.int32)], (220, 20, 20))
    return image


def dark_quad_finder(view: np.ndarray):
    """Stands in for the model: the extreme points of the red area, as fractions of the view."""
    ys, xs = np.nonzero((view[..., 0] > 150) & (view[..., 1] < 90))
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    total, difference = points.sum(axis=1), points[:, 0] - points[:, 1]
    corners = points[[total.argmin(), difference.argmax(), total.argmax(), difference.argmin()]]
    return (corners + 0.5) / view.shape[0], 4.0


def test_view_fractions_round_trip():
    matrix = window_transform(np.array([400.0, 300.0]), 500.0, degrees=7.0)
    points = np.array([[400.0, 300.0], [150.0, 320.0], [610.5, 77.25]])
    assert np.allclose(from_view(to_view(points, matrix), matrix), points)
    assert np.allclose(to_view(points[:1], matrix), 0.5)  # the window's centre is the view's centre


def test_cut_view_puts_scene_points_where_to_view_says():
    image = np.zeros((800, 1200, 3), np.uint8)
    cv2.circle(image, (900, 500), 14, (255, 255, 255), -1)
    for matrix in (whole_photo(image), window_transform(np.array([880.0, 480.0]), 300.0, 4.0), window_transform(np.array([700.0, 400.0]), 2600.0)):
        view = cut_view(image, matrix)
        ys, xs = np.nonzero(view[..., 0] > 100)
        expected = to_view(np.array([[900.0, 500.0]]), matrix)[0] * 256 - 0.5
        assert abs(xs.mean() - expected[0]) < 0.6 and abs(ys.mean() - expected[1]) < 0.6


def test_rectify_matches_the_generator():
    image = photo_with_quad()
    assert np.array_equal(rectify(image, QUAD), generator_rectify(image, QUAD, 256))


def test_second_look_sharpens_the_corners():
    image = photo_with_quad()
    one_look, _ = find_board(dark_quad_finder, image, looks=1)
    two_looks, in_view = find_board(dark_quad_finder, image, looks=2)
    assert np.abs(one_look - QUAD).max() < 8 and np.abs(two_looks - QUAD).max() < 4  # the stand-in finder is only good to a view pixel
    assert np.abs(two_looks - QUAD).max() <= np.abs(one_look - QUAD).max() and in_view > 0.9
    assert np.allclose(to_view(QUAD, around(QUAD)).mean(axis=0), 0.5, atol=0.01)


def test_turned_readings_are_combined_into_the_upright_board():
    rng = np.random.default_rng(0)
    upright = rng.integers(0, 13, (8, 8))
    quarter_turns = 3
    lying = np.rot90(upright, -quarter_turns)  # how the board lies in the crop

    def reader(crops: np.ndarray):
        squares, rotation = [], []
        for extra in range(len(crops)):  # the pipeline sends the crop at 0, 1, 2 and 3 extra turns
            seen = np.rot90(lying, extra)
            squares.append(np.eye(13)[seen].transpose(2, 0, 1) * 0.9 + 0.1 / 13)
            rotation.append(np.eye(4)[(quarter_turns - extra) % 4])
        return np.stack(squares), np.stack(rotation), np.full(len(crops), 0.2)

    seen, confidence, turns, white = read_board(reader, photo_with_quad(), QUAD)
    assert (seen == upright).all() and turns == quarter_turns and white is False and confidence.min() > 0.8


def test_placement_follows_the_viewing_side():
    seen = np.zeros((8, 8), int)
    seen[0, 0], seen[7, 7] = 12, 4  # a black king top-left, a white rook bottom-right
    common = dict(corners=QUAD, board_in_view=1.0, quarter_turns=0, seen=seen, confidence=np.ones((8, 8)))
    assert Scan(white_at_bottom=True, **common).placement() == "k7/8/8/8/8/8/8/7R"
    assert Scan(white_at_bottom=False, **common).placement() == "R7/8/8/8/8/8/8/7k"
