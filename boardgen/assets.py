"""Loading of piece sets and fonts fetched by tools/fetch_assets.py."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import resvg_py
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "assets"
ASSET_DIR = ROOT / "data" / "assets"
BASE_PIECE_SIZE = 192

# Fonts with Cyrillic coverage, used for non-Latin page text.
CYRILLIC_FONTS = ("Arimo.ttf", "EBGaramond.ttf", "PTSerif-Bold.ttf", "PTSerif-Regular.ttf", "SourceSans3.ttf")


@lru_cache(maxsize=1)
def piece_sets() -> dict[str, dict]:
    """Piece-set names mapped to their manifest entries, in sorted order."""
    sets = json.loads((MANIFEST_DIR / "piece_sets.json").read_text())["sets"]
    return {name: sets[name] for name in sorted(sets)}


@lru_cache(maxsize=1)
def print_piece_sets() -> tuple[str, ...]:
    return tuple(name for name, entry in piece_sets().items() if entry["print"])


@lru_cache(maxsize=1)
def font_names() -> tuple[str, ...]:
    fonts = json.loads((MANIFEST_DIR / "fonts.json").read_text())["fonts"]
    return tuple(sorted(fonts))


@lru_cache(maxsize=None)
def _base_piece(set_name: str, symbol: str) -> np.ndarray:
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
