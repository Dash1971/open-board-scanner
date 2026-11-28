"""Checks that samples are reproducible and that their labels match the image."""

import dataclasses

import cv2
import numpy as np
import pytest

from boardgen import CLASSES, KINDS, generate, rectify
from boardgen.board import render_book_board, sample_book_style
from boardgen.camera import Curl
from boardgen.positions import to_fen


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_and_index_give_identical_samples(kind):
    first, second = generate(11, 3, kind), generate(11, 3, kind)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.corners, second.corners)
    assert first.labels() == second.labels()


def test_different_indices_give_different_samples():
    assert generate(11, 3).placement != generate(11, 4).placement or not np.array_equal(
        generate(11, 3).image, generate(11, 4).image
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("index", range(6))
def test_labels_are_consistent(kind, index):
    sample = generate(5, index, kind)
    height, width = sample.image.shape[:2]
    assert sample.image.dtype == np.uint8 and sample.image.shape[2] == 3
    assert max(height, width) <= 1024

    # Corners lie inside the image and run clockwise from the top-left.
    assert (sample.corners[:, 0] >= 0).all() and (sample.corners[:, 0] <= width - 1).all()
    assert (sample.corners[:, 1] >= 0).all() and (sample.corners[:, 1] <= height - 1).all()
    assert cv2.isContourConvex(sample.corners)
    assert cv2.contourArea(sample.corners, oriented=True) > 0
    assert np.argmin(sample.corners.sum(axis=1)) == 0

    # The squares as displayed agree with the position and the viewing side.
    assert len(sample.squares) == 64 and set(sample.squares) <= set(CLASSES)
    assert sample.quarter_turns in (0, 1, 2, 3)
    assert to_fen(sample.position()) == sample.placement


def _square_contrast(sample) -> tuple[np.ndarray, np.ndarray]:
    """Pixel spread inside each square of the rectified board, split by label."""
    crop = rectify(sample.image, sample.corners, 256).astype(np.float32).mean(axis=2)
    cells = crop.reshape(8, 32, 8, 32).swapaxes(1, 2)[:, :, 6:26, 6:26]
    spread = cells.std(axis=(2, 3)).ravel()
    occupied = np.array([bool(symbol) for symbol in sample.squares])
    return spread[occupied], spread[~occupied]


@pytest.mark.parametrize("kind", ["screenshot", "monitor"])
@pytest.mark.parametrize("index", range(8))
def test_corners_and_squares_line_up_with_the_pixels(kind, index):
    """On clean boards, squares labelled as occupied contain a piece and most others are bare.

    Arrows and highlights may cross a few empty squares, so the empty side is
    judged by its median. A label grid shifted by even one square fails this.
    """
    sample = generate(21, index, kind, degrade=False)
    occupied, empty = _square_contrast(sample)
    if len(occupied) == 0 or len(empty) == 0:
        pytest.skip("needs both occupied and empty squares")
    bare = float(np.median(empty))
    assert (occupied > 2 * bare + 2.0).mean() >= 0.95
    assert (empty < np.median(occupied)).mean() >= 0.6


@pytest.mark.parametrize("index", range(8))
def test_ebook_corners_line_up_with_the_squares(index):
    """On a clean e-book photo the corners frame the printed squares.

    Dark squares are patterned, so pieces cannot be told from texture as on
    screens. Instead, the inside of each empty light square must be plain
    paper and each empty dark square darker than that. Only a sixteenth of a
    square is left out at each edge, so a grid shifted by a tenth of a square
    puts pattern into the light squares and fails.
    """
    sample = generate(21, index, "ebook", degrade=False, overlays=False)
    crop = rectify(sample.image, sample.corners, 256).astype(np.float32).mean(axis=2)
    cells = crop.reshape(8, 32, 8, 32).swapaxes(1, 2)[:, :, 2:30, 2:30]
    grid = np.array(sample.squares, dtype=object).reshape(8, 8)
    dark = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(bool)
    if sample.quarter_turns % 2:  # a quarter turn moves h1 (light) to a dark corner position
        dark = ~dark
    empty = grid == ""
    light_cells, dark_cells = cells[empty & ~dark], cells[empty & dark]
    if len(light_cells) < 4 or len(dark_cells) < 4:
        pytest.skip("needs empty squares of both colours")
    paper = np.median(light_cells.mean(axis=(1, 2)))
    assert (light_cells.std(axis=(1, 2)) < 6).mean() >= 0.9
    assert (dark_cells.mean(axis=(1, 2)) < paper - 8).mean() >= 0.9


def test_speckled_squares_are_reproducible_and_cover_the_planned_share():
    paper, ink = np.full(3, 240, np.float32), np.full(3, 20, np.float32)
    style = sample_book_style(np.random.default_rng(1), 64, paper, ink)
    style = dataclasses.replace(style, dark="speckle", spacing=2.0, grey=0.45, grid=False, coords=False, marker=False)
    first = render_book_board(np.random.default_rng(2), style, [""] * 64, True)
    again = render_book_board(np.random.default_rng(2), style, [""] * 64, True)
    other = render_book_board(np.random.default_rng(3), style, [""] * 64, True)
    assert np.array_equal(first.image, again.image)
    assert not np.array_equal(first.image, other.image)
    m = style.margin
    area = first.image[m : m + 512, m : m + 512].mean(axis=2)
    ink_share = (240 - area) / 220
    squares = ink_share.reshape(8, 64, 8, 64).mean(axis=(1, 3))
    dark = np.add.outer(np.arange(8), np.arange(8)) % 2 == 1
    assert np.allclose(squares[dark], 0.45, atol=0.06)
    assert squares[~dark].max() < 0.01


def test_speckle_style_is_drawn_for_some_books():
    styles = {generate(9, index, "book").meta["dark_squares"] for index in range(40)}
    assert "speckle" in styles


@pytest.mark.parametrize("kind", ["screenshot", "monitor", "ebook"])
def test_overlays_change_pixels_but_not_labels(kind):
    """Pointers and badges have their own random stream: without them the
    sample is the same picture, with the same labels."""
    found = 0
    for index in range(60):
        sample = generate(13, index, kind, degrade=False)
        if not (sample.meta.get("pointer") or sample.meta.get("badges")):
            continue
        bare = generate(13, index, kind, degrade=False, overlays=False)
        labels, bare_labels = sample.labels(), bare.labels()
        overlay_keys = {"pointer", "badges"}
        assert {k: v for k, v in labels["meta"].items() if k not in overlay_keys} == bare_labels["meta"]
        labels.pop("meta"), bare_labels.pop("meta")
        assert labels == bare_labels
        changed = np.abs(sample.image.astype(np.int16) - bare.image.astype(np.int16)).max(axis=2) > 0
        assert 0 < changed.mean() < 0.02
        assert generate(13, index, kind, degrade=False).labels() == sample.labels()
        found += 1
        if found == 3:
            break
    assert found, "expected an overlay in sixty samples"


def test_curl_solver_finds_where_page_points_land():
    curl = Curl(cos=0.98, sin=0.2, centre=300.0, width=400.0, lift=35.0, squeeze=-20.0)
    points = np.array([[100.0, 120.0], [500.0, 90.0], [520.0, 480.0], [80.0, 510.0]])
    landed = curl.invert(points)
    dx, dy = curl.offset(landed[:, 0], landed[:, 1])
    assert np.allclose(landed + np.stack([dx, dy], axis=1), points, atol=1e-6)


def test_turned_photos_are_the_upright_photo_rotated():
    """A sideways sample shows the same picture and labels as its upright twin, rotated."""
    turned = [(index, generate(31, index, "book")) for index in range(12)]
    turned = [(index, sample) for index, sample in turned if sample.quarter_turns]
    assert turned, "expected at least one sideways book photo in twelve"
    for index, sample in turned:
        upright = generate(31, index, "book", turns=False)
        restored = np.rot90(sample.image, sample.quarter_turns)
        assert np.array_equal(restored, upright.image)
        assert sample.position() == upright.position()
        # The crop cut with the turned corners is the upright crop, turned.
        crop = np.rot90(rectify(sample.image, sample.corners, 128), sample.quarter_turns)
        difference = np.abs(crop.astype(np.int16) - rectify(upright.image, upright.corners, 128).astype(np.int16))
        assert difference.mean() < 2.0
