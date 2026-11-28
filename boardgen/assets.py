"""Loading of piece sets and fonts fetched by tools/fetch_assets.py."""

from __future__ import annotations

import io
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import resvg_py
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "assets"
ASSET_DIR = ROOT / "data" / "assets"
BASE_PIECE_SIZE = 192
_VIEWBOX = re.compile(r'viewBox\s*=\s*"([^"]+)"')

# Fonts with Cyrillic coverage, used for non-Latin page text.
CYRILLIC_FONTS = ("Arimo.ttf", "EBGaramond.ttf", "PTSerif-Bold.ttf", "PTSerif-Regular.ttf", "SourceSans3.ttf")


@lru_cache(maxsize=1)
def piece_sets() -> dict[str, dict]:
    """Piece-set names mapped to their manifest entries, in sorted order.

    SVG sets come from piece_sets.json; sets drawn from a font's chess glyphs
    come from glyph_fonts.json and carry the font file and character lists.
    """
    sets = dict(json.loads((MANIFEST_DIR / "piece_sets.json").read_text())["sets"])
    sets.update(json.loads((MANIFEST_DIR / "glyph_fonts.json").read_text())["fonts"])
    return {name: sets[name] for name in sorted(sets)}


@lru_cache(maxsize=1)
def print_piece_sets() -> tuple[str, ...]:
    return tuple(name for name, entry in piece_sets().items() if entry["print"])


@lru_cache(maxsize=1)
def open_piece_sets() -> tuple[str, ...]:
    """Sets under a stated open licence.

    Left out: sets marked noncommercial, noderivatives (no modified copies
    allowed), freeware with no licence text, and unlicensed sets, for which
    no terms are stated at all.
    """
    closed = ("noncommercial", "noderivatives", "freeware", "unlicensed")
    return tuple(name for name, entry in piece_sets().items() if not any(entry.get(flag) for flag in closed))


@lru_cache(maxsize=1)
def font_names() -> tuple[str, ...]:
    fonts = json.loads((MANIFEST_DIR / "fonts.json").read_text())["fonts"]
    return tuple(sorted(fonts))


def _glyph_piece(entry: dict, symbol: str) -> np.ndarray:
    """Draw a piece from a font glyph and fill its enclosed areas with white.

    Fonts draw white pieces as outlines with nothing inside. Printed diagrams
    show them with a white body, so every area the ink encloses is filled.
    """
    path = ASSET_DIR / "glyph_fonts" / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/fetch_assets.py first")
    size = BASE_PIECE_SIZE
    # Older chess fonts map their glyphs only through a symbol or Mac Roman table,
    # which FreeType does not pick by default; such entries name the table to use.
    encoding = entry.get("encoding", "")
    probe = ImageFont.truetype(str(path), 100, encoding=encoding)
    left, top, right, bottom = probe.getbbox(entry["white"][0], anchor="ls")  # the king, relative to the baseline
    scale = 0.84 * size / max(bottom - top, right - left)
    font = ImageFont.truetype(str(path), max(8, int(round(100 * scale))), encoding=encoding)
    characters = entry["white"] if symbol.isupper() else entry["black"]
    canvas = Image.new("L", (size, size), 0)
    baseline = size * 0.92 - bottom * scale
    ImageDraw.Draw(canvas).text((size / 2, baseline), characters["KQRBNP".index(symbol.upper())], font=font, fill=255, anchor="ms")
    ink = np.asarray(canvas, dtype=np.float32) / 255.0
    open_area = np.pad((ink < 0.5).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(open_area, None, (0, 0), 2)  # everything reachable from the border becomes 2
    enclosed = (open_area[1:-1, 1:-1] == 1).astype(np.float32)
    alpha = np.maximum(ink, enclosed)
    white = (alpha - ink)[..., None] * 255.0
    premultiplied = np.concatenate([np.repeat(white, 3, axis=2), alpha[..., None]], axis=2).astype(np.float32)
    premultiplied.setflags(write=False)
    return premultiplied


def _premultiply(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    return np.concatenate([rgba[..., :3] * alpha, alpha], axis=2)


def _fitted_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """The size that fits a width x height picture into `scale` of the square, keeping its shape."""
    factor = scale * BASE_PIECE_SIZE / max(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))


def _letterbox(art: np.ndarray, scale: float) -> np.ndarray:
    """Place premultiplied artwork, already fitted, in the square.

    It is centred across. Vertically it keeps the margin that `scale` leaves
    below it, so artwork wider than tall stands on the same baseline as square
    sets instead of floating in the middle of the square.
    """
    size = BASE_PIECE_SIZE
    height, width = art.shape[:2]
    bottom = size - int(round(size * (1.0 - scale) / 2))
    top, left = bottom - height, (size - width) // 2
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    canvas[top:bottom, left : left + width] = art
    return canvas


def _widened_svg(path: Path, scale: float) -> str:
    """The SVG's text with its viewBox widened on every side, so the drawing fills `scale` of it.

    Unlike shrinking a finished render, this also shows artwork that reaches
    past the original viewBox, such as a halo the original clips.
    """
    text = path.read_text(encoding="utf-8")
    root = text.find("<svg")
    match = _VIEWBOX.search(text, root, text.find(">", root)) if root >= 0 else None
    if match is None:
        raise ValueError(f"{path} has no viewBox to widen")
    x, y, width, height = (float(value) for value in match.group(1).replace(",", " ").split())
    pad_x, pad_y = width * (1.0 / scale - 1.0) / 2, height * (1.0 / scale - 1.0) / 2
    widened = f"{x - pad_x:g} {y - pad_y:g} {width + 2 * pad_x:g} {height + 2 * pad_y:g}"
    return text[: match.start(1)] + widened + text[match.end(1) :]


def _svg_piece(path: Path, scale: float) -> np.ndarray:
    source = {"svg_path": str(path)} if scale == 1.0 else {"svg_string": _widened_svg(path, scale), "resources_dir": str(path.parent)}
    # An explicit DPI is required: with the default, sizes in mm or pt are rejected.
    # resvg fits the drawing inside width x height and keeps its shape, so
    # artwork that is not square comes back smaller on one side.
    png = resvg_py.svg_to_bytes(**source, width=BASE_PIECE_SIZE, height=BASE_PIECE_SIZE, dpi=96.0)
    return _letterbox(_premultiply(Image.open(io.BytesIO(bytes(png)))), 1.0)


def _raster_piece(path: Path, scale: float) -> np.ndarray:
    """A piece from a bitmap (WebP, PNG), with its alpha channel kept."""
    with Image.open(path) as image:
        art = _premultiply(image)
    width, height = _fitted_size(art.shape[1], art.shape[0], scale)
    interpolation = cv2.INTER_AREA if width < art.shape[1] else cv2.INTER_CUBIC
    art = cv2.resize(art, (width, height), interpolation=interpolation)
    art[..., 3] = np.clip(art[..., 3], 0.0, 1.0)
    art[..., :3] = np.clip(art[..., :3], 0.0, art[..., 3:] * 255.0)
    return _letterbox(art, scale)


@lru_cache(maxsize=None)
def _base_piece(set_name: str, symbol: str) -> np.ndarray:
    """A piece at the base size.

    SVG and bitmap artwork keeps its shape: what is not square is letterboxed.
    A set's `scale` (default 1) shrinks its artwork inside the square, for
    drawings that reach or cross the edge, such as pieces with a halo; for SVG
    it widens the viewBox, so that what the original clips is drawn.
    """
    entry = piece_sets()[set_name]
    if "file" in entry:
        return _glyph_piece(entry, symbol)
    colour = "w" if symbol.isupper() else "b"
    extension = entry.get("format", "svg")
    path = ASSET_DIR / "pieces" / set_name / f"{colour}{symbol.upper()}.{extension}"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/fetch_assets.py first")
    scale = float(entry.get("scale", 1.0))
    premultiplied = _svg_piece(path, scale) if extension == "svg" else _raster_piece(path, scale)
    premultiplied.setflags(write=False)
    return premultiplied


def piece(set_name: str, symbol: str, size: int) -> np.ndarray:
    """A piece as premultiplied float32 RGBA: colour in 0..255, alpha in 0..1."""
    base = _base_piece(set_name, symbol)
    interpolation = cv2.INTER_AREA if size < BASE_PIECE_SIZE else cv2.INTER_CUBIC
    resized = cv2.resize(base, (size, size), interpolation=interpolation)
    resized[..., 3] = np.clip(resized[..., 3], 0.0, 1.0)
    return resized


@lru_cache(maxsize=512)
def font(name: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    path = ASSET_DIR / "fonts" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/fetch_assets.py first")
    loaded = ImageFont.truetype(str(path), max(4, size))
    try:
        loaded.set_variation_by_axes([weight])
    except OSError:
        pass  # not a variable font
    return loaded
