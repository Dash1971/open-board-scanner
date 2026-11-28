"""Textured board surfaces from the images fetched by tools/fetch_boards.py.

Chess sites and programs draw boards from three kinds of image, all listed in
assets/boards.json:

- "board": a picture of a whole 8x8 board (Lichess), resized to fit;
- "pair": a light and a dark square texture (PyChess), cut square by square;
- "stacked": a light texture above a dark one in one image (ChessX), handled
  as a pair.

Any square texture may also be used as a "single" texture: its grain is kept
and it is tinted with a light and a dark colour, which gives many colourings
from a few images.

If no images have been fetched, `available()` is empty and callers keep to
flat colours.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import assets

MANIFEST = assets.MANIFEST_DIR / "boards.json"
BOARD_DIR = assets.ASSET_DIR / "boards"  # tests point this at an empty folder
TINT_PROBABILITY = 0.3  # of square textures, how often only their grain is used
MAX_BOARD = 1024
LIGHT_FLOOR, DARK_FLOOR = 150.0, 80.0  # lowest mean brightness of square textures

# Turns and reflections that keep a light square at the top left: the identity,
# a half turn, and reflection in either diagonal. A quarter turn or a mirror
# alone would swap light and dark squares.
_KEEP_COLOURS = (
    lambda a: a,
    lambda a: a[::-1, ::-1],
    lambda a: a.transpose(1, 0, 2),
    lambda a: a[::-1, ::-1].transpose(1, 0, 2),
)


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _files(entry: dict) -> list[str]:
    return [entry["light"], entry["dark"]] if entry["kind"] == "pair" else [entry["file"]]


@lru_cache(maxsize=8)
def _available_in(directory: Path) -> tuple[str, ...]:
    boards = manifest()["boards"]
    return tuple(
        sorted(name for name, entry in boards.items() if all((directory / entry["source"] / f).exists() for f in _files(entry)))
    )


def available() -> tuple[str, ...]:
    """Names of the textures whose files are present, in sorted order."""
    return _available_in(Path(BOARD_DIR))


@lru_cache(maxsize=None)
def _image(path: Path) -> np.ndarray:
    """An image file as read-only float32 RGB in 0..255."""
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    pixels.setflags(write=False)
    return pixels


@lru_cache(maxsize=None)
def _board_image(path: Path) -> np.ndarray:
    """A whole-board image, at most MAX_BOARD pixels a side to keep resizing quick."""
    image = _image(path)
    if image.shape[0] > MAX_BOARD:  # sizes are multiples of 8, so the grid stays exact
        image = cv2.resize(image, (MAX_BOARD, MAX_BOARD), interpolation=cv2.INTER_AREA)
        image.setflags(write=False)
    return image


def _square_textures(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The light and dark texture of a pair or stacked entry."""
    entry = manifest()["boards"][name]
    folder = Path(BOARD_DIR) / entry["source"]
    if entry["kind"] == "pair":
        return _image(folder / entry["light"]), _image(folder / entry["dark"])
    both = _image(folder / entry["file"])  # ChessX: light on top, dark below
    half = both.shape[0] // 2
    return both[:half], both[half : 2 * half]


def _tiles(rng: np.random.Generator, texture: np.ndarray, sq: int, count: int, scale: float) -> list[np.ndarray]:
    """`count` sq x sq crops of a texture at random offsets.

    The texture is resized once so that a square covers 1/scale of it, so the
    grain has the same size on every square of a board, as in real programs.
    Crops are only flipped, never given a quarter turn, so the grain runs the
    same way across the board; a board-wide quarter turn is drawn once.
    """
    side = max(sq, int(round(sq * scale)))
    resized = cv2.resize(np.ascontiguousarray(texture), (side, side), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        resized = resized.transpose(1, 0, 2)
    tiles = []
    for _ in range(count):
        y, x = rng.integers(0, side - sq + 1, size=2)
        tile = resized[y : y + sq, x : x + sq]
        if rng.random() < 0.5:
            tile = tile[::-1, ::-1]
        tiles.append(tile)
    return tiles


def _tinted(texture: np.ndarray, colour: np.ndarray, strength: float) -> np.ndarray:
    """A texture's grain applied to a flat colour.

    The grain is rescaled to a relative spread of `strength` whatever the
    source, so a strongly figured wood does not wash out the difference
    between light and dark squares.
    """
    luma = texture @ np.array([0.299, 0.587, 0.114], np.float32)
    relative = luma / (luma.mean() + 1e-3) - 1.0
    grain = 1.0 + np.clip(relative / (relative.std() + 1e-3) * strength, -3 * strength, 3 * strength)
    return np.asarray(colour, np.float32) * grain[..., None]


def surface(rng: np.random.Generator, sq: int, light, dark, name: str | None = None) -> tuple[np.ndarray, str]:
    """An 8x8 board surface of 8*sq pixels a side, light square at the top left.

    light and dark are the flat colours used when a texture is tinted. Returns
    float32 RGB and a description such as "lichess-wood" or "tint:pychess-maple".
    """
    names = available()
    if not names:
        raise FileNotFoundError(f"no board textures in {BOARD_DIR}; run tools/fetch_boards.py first")
    name = name or names[int(rng.integers(len(names)))]
    entry = manifest()["boards"][name]
    n = 8 * sq

    if entry["kind"] == "board":
        image = _board_image(Path(BOARD_DIR) / entry["source"] / entry["file"])
        # Every source image is a whole number of pixels per square, so a uniform
        # resize keeps the texture's own grid on ours.
        board = cv2.resize(np.ascontiguousarray(image), (n, n), interpolation=cv2.INTER_AREA if n < image.shape[0] else cv2.INTER_CUBIC)
        board = _KEEP_COLOURS[int(rng.integers(len(_KEEP_COLOURS)))](board)
        return np.ascontiguousarray(board, dtype=np.float32), name

    light_texture, dark_texture = _square_textures(name)
    # A few woods (cherry, lanta) are so dark that black pieces vanish on them;
    # they are brightened just enough to keep pieces readable, grain intact.
    light_texture = light_texture * max(1.0, LIGHT_FLOOR / float(light_texture.mean()))
    dark_texture = dark_texture * max(1.0, DARK_FLOOR / float(dark_texture.mean()))
    tint = rng.random() < TINT_PROBABILITY
    if tint:
        strength = float(rng.uniform(0.02, 0.08))
        source = light_texture if rng.random() < 0.5 else dark_texture
        light_texture = _tinted(source, light, strength)
        dark_texture = _tinted(source, dark, strength)
    scale = float(rng.uniform(1.0, 2.5))
    light_tiles = _tiles(rng, light_texture, sq, 32, scale)
    dark_tiles = _tiles(rng, dark_texture, sq, 32, scale)
    board = np.empty((n, n, 3), np.float32)
    for row in range(8):
        for col in range(8):
            tiles = dark_tiles if (row + col) % 2 else light_tiles
            board[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq] = tiles[(row * 8 + col) // 2]
    return np.clip(board, 0, 255), ("tint:" if tint else "") + name
