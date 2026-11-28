"""Surroundings for a board: a printed page, or an app or website on a screen.

Each builder returns a float32 RGB scene and the board's playing-area corners
in scene coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import assets, positions
from .board import BoardImage, BookStyle, draw_texts, paste_premultiplied, render_book_board, smooth_field

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
    printed: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Place the board on a page of text; board_fraction is its planned share of the photo width.

    printed=False gives a page as an e-book shows it: no paper texture and no
    print showing through.
    """
    sq = style.sq
    side = board.image.shape[0]
    width = int(side * np.clip(rng.uniform(0.9, 1.4) / board_fraction, 1.25, 3.2))
    height = int(width * rng.uniform(1.0, 1.4))
    page = np.empty((height, width, 3), np.float32)
    page[:] = style.paper
    if printed:
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
    if printed and rng.random() < 0.3:  # print showing through from the other side of the leaf
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



def _toolbar(rng: np.random.Generator, width: int, height: int, dark: bool) -> np.ndarray:
    """A document viewer's toolbar: icons, a page counter and a zoom level."""
    colour = np.clip((rng.uniform(40, 70) if dark else rng.uniform(232, 250)) + rng.normal(0, 3, 3), 0, 255)
    ink = np.full(3, rng.uniform(190, 230) if dark else rng.uniform(40, 90), np.float32)
    strip = np.empty((height, width, 3), np.float32)
    strip[:] = colour
    strip[-max(1, height // 30) :] = np.clip(colour + (25 if dark else -30), 0, 255)  # separator line
    icon = height * rng.uniform(0.3, 0.45)
    x = height * 0.4
    for _ in range(int(rng.integers(2, 7))):
        top, left = int(height / 2 - icon / 2), int(x)
        if rng.random() < 0.5:
            _panel(strip, left, top, left + icon, top + icon, ink)
            if rng.random() < 0.6:  # outlined rather than solid
                edge = max(1, int(icon / 7))
                _panel(strip, left + edge, top + edge, left + icon - edge, top + icon - edge, colour)
        else:
            cv2.circle(strip, (int(x + icon / 2), int(height / 2)), int(icon / 2), tuple(float(v) for v in ink), -1, cv2.LINE_AA)
        x += icon * rng.uniform(1.6, 2.6)
    font = assets.font(str(rng.choice(assets.font_names())), max(6, int(height * rng.uniform(0.32, 0.45))))
    total = int(rng.integers(20, 600))
    items = [(width * rng.uniform(0.4, 0.55), height / 2, f"{int(rng.integers(1, total))} / {total}", font, ink, "mm")]
    if rng.random() < 0.7:
        items.append((width - height * 0.5, height / 2, f"{int(rng.choice([75, 90, 100, 110, 125, 150, 200]))}%", font, ink, "rm"))
    return draw_texts(strip, items)


def make_reader(rng: np.random.Generator, page: np.ndarray, corners: np.ndarray, sq: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Show a page in a PDF or e-book viewer: either zoomed so that the page fills
    the window, or whole on the application's background; sometimes under a toolbar."""
    height, width = page.shape[:2]
    dark_app = rng.random() < 0.5
    lo, hi = corners.min(axis=0) - sq * 0.3, corners.max(axis=0) + sq * 0.3
    framed = rng.random() < 0.45
    if framed:
        background = np.clip((rng.uniform(35, 95) if dark_app else rng.uniform(190, 235)) + rng.normal(0, 3, 3), 0, 255)
        side_pad, top_pad = int(width * rng.uniform(0.03, 0.4)), int(sq * rng.uniform(0.2, 1.5))
        bottom = int(sq * rng.uniform(0.2, 1.5))
        scene = np.empty((top_pad + height + bottom, width + 2 * side_pad, 3), np.float32)
        scene[:] = background
        edge = max(1, sq // 40)  # the page's outline or shadow
        _panel(scene, side_pad - edge, top_pad - edge, side_pad + width + edge, top_pad + height + edge, background * 0.8)
        scene[top_pad : top_pad + height, side_pad : side_pad + width] = page
        offset = np.array([side_pad, top_pad], np.float32)
    else:  # zoomed in: the window shows part of the page around the board
        x0 = int(rng.uniform(0, 1) * np.clip(lo[0], 0, 0.15 * width))
        x1 = int(width - rng.uniform(0, 1) * np.clip(width - hi[0], 0, 0.15 * width))
        y0 = int(rng.uniform(0, 1) * max(0.0, lo[1]))
        y1 = int(height - rng.uniform(0, 1) * max(0.0, height - hi[1]))
        scene = page[y0:y1, x0:x1]
        offset = np.array([-x0, -y0], np.float32)
    bar = int(sq * rng.uniform(0.45, 0.85)) if rng.random() < 0.5 else 0
    if bar:
        scene = np.vstack([_toolbar(rng, scene.shape[1], bar, dark_app), scene])
        offset = offset + [0, bar]
    return scene, corners + offset, {"reader": "framed" if framed else "zoomed", "toolbar": bool(bar)}


# --- things drawn over the board on screens ---------------------------------

# The usual arrow pointer, tip at the origin, one unit tall.
_POINTER = np.array([[0, 0], [0, 0.8], [0.19, 0.63], [0.32, 0.94], [0.43, 0.89], [0.31, 0.6], [0.55, 0.6]])
_BADGES = {  # end-of-game badges: disc colour and white symbol
    "win": ((129, 182, 76), "crown"),
    "checkmated": ((202, 52, 49), "#"),
    "resigned": ((95, 95, 95), "flag"),
    "draw": ((140, 140, 140), "\u00bd"),
}
_K = 4  # overlays are drawn at this many times the resolution, then reduced, for smooth edges


def _composite(scene: np.ndarray, colour: Image.Image, alpha: Image.Image, left: int, top: int) -> None:
    """Reduce a supersampled drawing and composite it onto the scene with its top-left at (left, top)."""
    size = (colour.width // _K, colour.height // _K)
    a = np.asarray(alpha, np.float32) / 255.0
    rgb = cv2.resize(np.asarray(colour, np.float32) * a[..., None], size, interpolation=cv2.INTER_AREA)
    paste_premultiplied(scene, rgb, cv2.resize(a, size, interpolation=cv2.INTER_AREA), left, top)


def add_pointer(rng: np.random.Generator, scene: np.ndarray, corners: np.ndarray, sq: int) -> None:
    """A mouse pointer anywhere over or near the board: a dark arrow with a light edge, or the reverse."""
    height = sq * rng.uniform(0.3, 0.55)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    x, y = rng.uniform(lo - sq * 0.4, hi + sq * 0.1)
    fill, edge = ((10, 10, 10), (250, 250, 250)) if rng.random() < 0.75 else ((250, 250, 250), (10, 10, 10))
    left, top, size = int(x) - 2, int(y) - 2, int(height) + 6
    colour = Image.new("RGB", (size * _K, size * _K), 0)
    alpha = Image.new("L", (size * _K, size * _K), 0)
    outline = [tuple(point) for point in (_POINTER * height + [x - left, y - top]) * _K]
    if rng.random() < 0.4:  # a soft shadow down and to the right
        shift = height * 0.06 * _K
        ImageDraw.Draw(alpha).polygon([(px + shift, py + shift) for px, py in outline], fill=60)
    width = int(max(1.0, height / 14) * _K)
    ImageDraw.Draw(colour).polygon(outline, fill=fill, outline=edge, width=width)
    ImageDraw.Draw(alpha).polygon(outline, fill=255, outline=255, width=width)
    _composite(scene, colour, alpha, left, top)


def _badge(rng: np.random.Generator, scene: np.ndarray, x: float, y: float, diameter: float, result: str) -> None:
    """One round badge centred at (x, y), measured from the edge of the scene's first pixel."""
    disc, symbol = _BADGES[result]
    disc = tuple(int(v) for v in np.clip(np.array(disc) + rng.normal(0, 8, 3), 0, 255))
    left, top, size = int(x - diameter / 2) - 2, int(y - diameter / 2) - 2, int(diameter) + 5
    colour = Image.new("RGB", (size * _K, size * _K), disc)
    alpha = Image.new("L", (size * _K, size * _K), 0)
    cx, cy, r = (x - left) * _K, (y - top) * _K, diameter / 2 * _K
    box = (cx - r, cy - r, cx + r, cy + r)
    ImageDraw.Draw(alpha).ellipse(box, fill=255)
    drawer = ImageDraw.Draw(colour)
    if rng.random() < 0.5:  # a thin white rim
        drawer.ellipse(box, fill=disc, outline=(255, 255, 255), width=int(max(1, r * 0.12)))
    white = (255, 255, 255)
    if symbol == "crown":
        s = r * 0.5
        drawer.polygon(
            [(cx - s, cy + s * 0.6), (cx - s, cy - s * 0.5), (cx - s * 0.5, cy), (cx, cy - s * 0.8),
             (cx + s * 0.5, cy), (cx + s, cy - s * 0.5), (cx + s, cy + s * 0.6)],
            fill=white,
        )
    elif symbol == "flag":
        s = r * 0.5
        drawer.rectangle((cx - s * 0.75, cy - s, cx - s * 0.5, cy + s), fill=white)
        drawer.polygon([(cx - s * 0.5, cy - s), (cx + s * 0.8, cy - s * 0.55), (cx - s * 0.5, cy - s * 0.1)], fill=white)
    else:
        font = assets.font("Arimo.ttf", max(6, int(r * 1.3)), 700)
        drawer.text((cx, cy), symbol, font=font, fill=white, anchor="mm")
    _composite(scene, colour, alpha, left, top)


def add_badges(rng: np.random.Generator, scene: np.ndarray, corners: np.ndarray, sq: int, squares: list[str]) -> int:
    """End-of-game badges over a corner of one or both kings' squares, as chess sites draw them.

    corners are the board's axis-aligned playing-area corners in the scene and
    squares the 64 symbols as displayed. Returns the number of badges drawn.
    """
    kings = [index for index, symbol in enumerate(squares) if symbol in ("K", "k")]
    if not kings:
        return 0
    if len(kings) >= 2 and rng.random() < 0.6:
        chosen = [int(i) for i in rng.choice(kings, size=2, replace=False)]
        draw = rng.random() < 0.25
        results = ["draw", "draw"] if draw else ["win", str(rng.choice(["checkmated", "resigned"]))]
    else:
        chosen = [kings[int(rng.integers(len(kings)))]]
        results = [str(rng.choice(list(_BADGES)))]
    diameter = sq * rng.uniform(0.3, 0.45)
    origin = corners[0] + 0.5  # top-left edge of the playing area, counting from the edge of the first pixel
    for index, result in zip(chosen, results):
        row, col = divmod(index, 8)
        inset = rng.uniform(0.0, 0.4, 2) * diameter
        right = rng.random() < 0.85
        x = origin[0] + (col + 1) * sq - inset[0] if right else origin[0] + col * sq + inset[0]
        y = origin[1] + row * sq + inset[1]
        _badge(rng, scene, x, y, diameter, result)
    return len(chosen)
