"""Loading of piece sets and fonts fetched by tools/fetch_assets.py."""

from __future__ import annotations

import io
import json
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
    """Sets under an open licence: not marked noncommercial, and not freeware with no licence text."""
    return tuple(name for name, entry in piece_sets().items() if not entry.get("noncommercial") and not entry.get("freeware"))


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
    probe = ImageFont.truetype(str(path), 100)
    left, top, right, bottom = probe.getbbox(entry["white"][0], anchor="ls")  # the king, relative to the baseline
    scale = 0.84 * size / max(bottom - top, right - left)
    font = ImageFont.truetype(str(path), max(8, int(round(100 * scale))))
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


@lru_cache(maxsize=None)
def _base_piece(set_name: str, symbol: str) -> np.ndarray:
    entry = piece_sets()[set_name]
    if "file" in entry:
        return _glyph_piece(entry, symbol)
    colour = "w" if symbol.isupper() else "b"
    path = ASSET_DIR / "pieces" / set_name / f"{colour}{symbol.upper()}.svg"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/fetch_assets.py first")
    # An explicit DPI is required: with the default, sizes in mm or pt are rejected.
    png = resvg_py.svg_to_bytes(svg_path=str(path), width=BASE_PIECE_SIZE, height=BASE_PIECE_SIZE, dpi=96.0)
    rgba = np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    premultiplied = np.concatenate([rgba[..., :3] * alpha, alpha], axis=2)
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
