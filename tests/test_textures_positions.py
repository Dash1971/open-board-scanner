"""Checks for textured board surfaces and real positions, with and without fetched data."""

import chess
import numpy as np
import pytest

from boardgen import generate, positions, textures
from boardgen.board import checker


@pytest.fixture
def nothing_fetched(tmp_path, monkeypatch):
    """Point both loaders at an empty folder, as on a machine without network."""
    monkeypatch.setattr(textures, "BOARD_DIR", tmp_path / "boards")
    monkeypatch.setattr(positions, "REAL_POSITIONS", tmp_path / "positions" / "lichess-puzzles.txt")


needs_textures = pytest.mark.skipif(not textures.available(), reason="run tools/fetch_boards.py first")
needs_positions = pytest.mark.skipif(not positions.real_positions(), reason="run tools/fetch_positions.py first")


@needs_textures
@pytest.mark.parametrize("name", textures.available() or ["none"])
def test_texture_surfaces_have_the_right_shape_and_lighter_light_squares(name):
    light, dark = np.array([235.0, 225.0, 200.0]), np.array([120.0, 90.0, 60.0])
    dark_squares = checker(37) == 1
    for seed in range(3):  # covers whole-board turns, tinting and offsets
        surface, _ = textures.surface(np.random.default_rng(seed), 37, light, dark, name)
        assert surface.shape == (296, 296, 3) and surface.dtype == np.float32
        assert surface.min() >= 0 and surface.max() <= 255
        luma = surface.mean(axis=2)
        assert luma[~dark_squares].mean() > luma[dark_squares].mean() + 10


@needs_textures
def test_textured_samples_are_deterministic():
    found = 0
    for index in range(40):
        sample = generate(17, index, "screenshot")
        if "texture" not in sample.meta:
            continue
        again = generate(17, index, "screenshot")
        assert np.array_equal(sample.image, again.image) and sample.labels() == again.labels()
        found += 1
    assert found >= 5, "expected textured boards in about a third of screen samples"


@needs_positions
def test_real_positions_are_deterministic_and_legal_placements():
    found = 0
    for index in range(40):
        sample = generate(19, index, "screenshot", degrade=False)
        if sample.meta["position"] != "real":
            continue
        again = generate(19, index, "screenshot", degrade=False)
        assert np.array_equal(sample.image, again.image) and sample.labels() == again.labels()
        board = chess.Board(sample.placement + " w - - 0 1")
        assert len(board.pieces(chess.KING, chess.WHITE)) == 1 and len(board.pieces(chess.KING, chess.BLACK)) == 1
        found += 1
    assert found >= 5, "expected real positions in about a third of samples"


def test_generation_works_without_textures_or_positions(nothing_fetched):
    assert textures.available() == () and positions.real_positions() == ()
    methods = set()
    for index in range(30):
        sample = generate(23, index, "screenshot")
        assert "texture" not in sample.meta
        methods.add(sample.meta["position"])
        again = generate(23, index, "screenshot")
        assert np.array_equal(sample.image, again.image)
    assert "real" not in methods and "game" in methods
