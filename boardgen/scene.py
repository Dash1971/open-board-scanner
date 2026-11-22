"""Surroundings for a board: a printed page, or an app or website on a screen.

Each builder returns a float32 RGB scene and the board's playing-area corners
in scene coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import assets, positions
from .board import BoardImage, BookStyle, draw_texts, render_book_board, smooth_field

_WORDS = (
    "the of and to in is that it for as with was his on be at by this had not are but from or have an they "
    "which one you were all her she there would their we him been has when who will more no if out so said "
    "what up its about into than them can only other new some could time these two may then do first any my "
    "now such like our over man me even most made after also did many before must through back years where "
    "much your way well down should because each just those people how too little state good very make world "
    "still own see men work long get here between both life being under never day same another know while "
    "last might us great old year off come since against go came right used take three white black move "
    "king queen rook bishop knight pawn attack defence position variation better played wins draw sacrifice"
).split()
_CYRILLIC = "абвгдежзийклмнопрстуфхцчшщыьэюя"
_PIECE_LETTERS = ("", "", "N", "B", "R", "Q", "K")


def _word(rng: np.random.Generator, cyrillic: bool) -> str:
    roll = rng.random()
    if roll < 0.18:
        number = f"{int(rng.integers(1, 60))}." if rng.random() < 0.5 else ""
        piece = _PIECE_LETTERS[int(rng.integers(len(_PIECE_LETTERS)))]
        capture = "x" if rng.random() < 0.2 else ""
        suffix = str(rng.choice(["", "", "", "+", "!", "?", "#"]))
        return f"{number}{piece}{capture}{'abcdefgh'[int(rng.integers(8))]}{int(rng.integers(1, 9))}{suffix}"
    if cyrillic:
        return "".join(_CYRILLIC[int(i)] for i in rng.integers(0, len(_CYRILLIC), size=int(rng.integers(1, 10))))
    word = _WORDS[int(rng.integers(len(_WORDS)))]
    return word.capitalize() if roll > 0.95 else word


def _free_segments(left: float, right: float, y0: float, y1: float, blocks: list[tuple], min_width: float):
    """Horizontal spans of a text line that avoid the blocked rectangles."""
    segments = [(left, right)]
    for bx0, by0, bx1, by1, allow_beside in blocks:
        if y1 <= by0 or y0 >= by1:
            continue
        updated = []
        for x0, x1 in segments:
            if x1 <= bx0 or x0 >= bx1:
                updated.append((x0, x1))
            elif allow_beside:
                updated.extend([(x0, bx0), (bx1, x1)])
        segments = updated
    return [(x0, x1) for x0, x1 in segments if x1 - x0 >= min_width]


def _text_coverage(rng, width: int, height: int, sq: int, blocks: list[tuple], captions: list[tuple]) -> tuple[np.ndarray, str]:
    """Ink coverage (0..1) of body text flowing around the blocked rectangles, and the script used."""
    layer = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(layer)
    cyrillic = rng.random() < 0.12
    names = assets.CYRILLIC_FONTS if cyrillic else assets.font_names()
    font_name = str(rng.choice(names))
    size = max(6, int(round(sq * rng.uniform(0.36, 0.8))))
    weight = int(rng.choice([400, 400, 500]))
    font = assets.font(font_name, size, weight)
    leading = size * rng.uniform(1.15, 1.55)
    space = font.getlength(" ") * rng.uniform(0.9, 1.3)
    left = width * rng.uniform(0.0, 0.08)
    right = width * (1 - rng.uniform(0.0, 0.08))
    y = height * rng.uniform(0.0, 0.06)
    indent_next = False
    while y + size < height:
        stop_early = rng.random() < 0.1
        for x0, x1 in _free_segments(left, right, y, y + size, blocks, 6 * size):
            x = x0 + (size * 1.5 if indent_next else 0)
            limit = x0 + (x1 - x0) * rng.uniform(0.2, 0.8) if stop_early else x1
            while True:
                word = _word(rng, cyrillic)
                length = font.getlength(word)
                if x + length > limit:
                    break
                drawer.text((x, y), word, font=font, fill=255)
                x += length + space
        indent_next = stop_early
        y += leading * (2 if stop_early and rng.random() < 0.3 else 1)
    caption_font = assets.font(font_name, max(6, int(size * rng.uniform(0.8, 1.0))), 400)
    for x, y, text in captions:
        drawer.text((x, y), text, font=caption_font, fill=255, anchor="ma")
    return np.asarray(layer, dtype=np.float32) / 255.0, "cyrillic" if cyrillic else "latin"


def make_book_page(
    rng: np.random.Generator,
    style: BookStyle,
    board: BoardImage,
    board_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Place the board on a page of text; board_fraction is its planned share of the photo width."""
    sq = style.sq
    side = board.image.shape[0]
    width = int(side * np.clip(rng.uniform(0.9, 1.4) / board_fraction, 1.25, 3.2))
    height = int(width * rng.uniform(1.0, 1.4))
    page = np.empty((height, width, 3), np.float32)
    page[:] = style.paper
    page += (smooth_field(rng, height, width, sq * 2.0) * rng.uniform(0, 5))[..., None]

    bx = int(rng.uniform(0.02, 0.98) * (width - side))
    by = int(rng.uniform(0.02, 0.98) * (height - side))
    pad = sq * rng.uniform(0.0, 0.5)
    beside = bool(rng.random() < 0.45)
    captions = []
    caption_room = 0.0
    if rng.random() < 0.5:
        text = str(rng.choice(["White to move", "Black to move", "Diagram", "Position after", "Ход белых"]))
        if rng.random() < 0.5:
            text = f"{text} {int(rng.integers(1, 400))}"
        captions.append((bx + side / 2, by + side + pad * 0.3, text))
        caption_room = sq * 0.9
    blocks = [(bx - pad, by - pad, bx + side + pad, by + side + pad + caption_room, beside)]

    # A second diagram nearby, as a distractor for the localizer.
    other = None
    if rng.random() < 0.25:
        placement, _ = positions.sample_placement(rng)
        other = render_book_board(rng, style, placement, True)
        for _ in range(8):
            ox = int(rng.uniform(0, 1) * (width - side))
            oy = int(rng.uniform(0, 1) * (height - side))
            if abs(ox - bx) >= side + sq * 0.5 or abs(oy - by) >= side + sq * 0.5:
                blocks.append((ox - pad, oy - pad, ox + side + pad, oy + side + pad, beside))
                break
        else:
            other = None

    coverage, script = _text_coverage(rng, width, height, sq, blocks, captions)
    page = page * (1 - coverage[..., None]) + style.ink * coverage[..., None]
    if rng.random() < 0.3:  # print showing through from the other side of the leaf
        ghost = cv2.GaussianBlur(coverage[:, ::-1], (0, 0), max(0.8, sq * 0.04)) * rng.uniform(0.03, 0.1)
        page = page * (1 - ghost[..., None]) + style.ink * ghost[..., None]

    if other is not None:
        ox, oy = int(blocks[-1][0] + pad), int(blocks[-1][1] + pad)
        page[oy : oy + side, ox : ox + side] = other.image
    page[by : by + side, bx : bx + side] = board.image
    corners = board.corners + np.array([bx, by], np.float32)
    return page, corners, {"second_diagram": other is not None, "script": script}


def _panel(scene: np.ndarray, x0: float, y0: float, x1: float, y1: float, colour) -> None:
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), scene.shape[1]), min(int(y1), scene.shape[0])
    if x1 > x0 and y1 > y0:
        scene[y0:y1, x0:x1] = np.asarray(colour, np.float32)


def make_screen(rng: np.random.Generator, board: BoardImage, sq: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Place the board inside a generic chess app or website layout."""
    side = board.image.shape[0]
    layout = str(rng.choice(["phone", "desktop", "tight"], p=[0.45, 0.35, 0.20]))
    if layout == "phone":
        width = int(side * rng.uniform(1.0, 1.12))
        height = int(width * rng.uniform(1.5, 2.1))
    elif layout == "desktop":
        width = int(side * rng.uniform(1.6, 2.8))
        height = int(side * rng.uniform(1.1, 1.5))
    else:
        width = int(side * rng.uniform(1.0, 1.2))
        height = int(side * rng.uniform(1.0, 1.25))
    dark_ui = rng.random() < 0.6
    base = rng.uniform(18, 55) if dark_ui else rng.uniform(232, 255)
    background = np.clip(base + rng.normal(0, 4, 3), 0, 255).astype(np.float32)
    text_colour = np.full(3, 215.0 if dark_ui else 45.0, np.float32)
    panel_colour = np.clip(background + (18 if dark_ui else -14), 0, 255)
    scene = np.empty((height, width, 3), np.float32)
    scene[:] = background

    bx = int(rng.uniform(0, 1) * (width - side))
    by = int(rng.uniform(0.1, 0.9) * (height - side))
    bar = int(sq * rng.uniform(0.55, 0.9))
    items = []
    font_name = str(rng.choice(assets.font_names()))
    font = assets.font(font_name, max(6, int(sq * rng.uniform(0.26, 0.4))), 500)

    if layout != "tight" and rng.random() < 0.8:  # player bars with names and clocks
        for y in (by - bar, by + side):
            _panel(scene, bx, y, bx + side, y + bar, panel_colour)
            name = "".join(chr(int(c)) for c in rng.integers(97, 123, size=int(rng.integers(4, 12))))
            items.append((bx + sq * 0.3, y + bar / 2, f"{name} ({int(rng.integers(600, 2900))})", font, text_colour, "lm"))
            clock = np.clip(panel_colour + (25 if dark_ui else -25), 0, 255)
            _panel(scene, bx + side - sq * 1.6, y + bar * 0.15, bx + side - sq * 0.15, y + bar * 0.85, clock)
            time = f"{int(rng.integers(0, 60))}:{int(rng.integers(0, 60)):02d}"
            items.append((bx + side - sq * 0.87, y + bar / 2, time, font, text_colour, "mm"))
    if rng.random() < 0.3:  # evaluation bar
        split = by + int(side * rng.uniform(0.1, 0.9))
        ex = bx - int(sq * 0.45) if bx > sq * 0.6 else bx + side + int(sq * 0.1)
        _panel(scene, ex, by, ex + sq * 0.32, split, (60, 60, 60))
        _panel(scene, ex, split, ex + sq * 0.32, by + side, (240, 240, 240))

    # Side or bottom panels with a move list and buttons.
    regions = []
    if bx + side + sq * 3 < width:
        regions.append((bx + side + sq * 0.6, by, width - sq * 0.3, by + side))
    if bx - sq * 3 > 0:
        regions.append((sq * 0.3, by, bx - sq * 0.7, by + side))
    if by + side + bar + sq * 2 < height:
        regions.append((sq * 0.2, by + side + bar + sq * 0.3, width - sq * 0.2, height - sq * 0.2))
    for x0, y0, x1, y1 in regions:
        if rng.random() < 0.25:
            continue
        _panel(scene, x0, y0, x1, y1, panel_colour)
        y = y0 + sq * 0.3
        move = int(rng.integers(1, 40))
        while y + sq * 0.5 < y1:
            if rng.random() < 0.12:
                button = np.clip(panel_colour + rng.normal(0, 25, 3), 0, 255)
                _panel(scene, x0 + sq * 0.3, y, min(x1 - sq * 0.3, x0 + sq * rng.uniform(2, 5)), y + sq * 0.45, button)
            else:
                line, x = f"{move}.", x0 + sq * 0.3
                move += 1
                while x + font.getlength(line) + sq < x1 and rng.random() < 0.9:
                    line += " " + _word(rng, False)
                items.append((x, y, line, font, text_colour, "la"))
            y += sq * rng.uniform(0.5, 0.75)

    scene = draw_texts(scene, items)
    scene[by : by + side, bx : bx + side] = board.image
    corners = board.corners + np.array([bx, by], np.float32)
    return scene, corners, {"layout": layout}

