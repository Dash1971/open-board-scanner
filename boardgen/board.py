"""Flat renders of a chess board in printed-book and on-screen styles.

Images are float32 RGB in 0..255. Corner coordinates use OpenCV's convention,
where integer coordinates are pixel centres, and mark the outer corners of the
8x8 playing area (not the frame or the coordinate margin) in the order
top-left, top-right, bottom-right, bottom-left as displayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import assets

SCREEN_THEMES = (
    ((240, 217, 181), (181, 136, 99)),
    ((238, 238, 210), (118, 150, 86)),
    ((222, 227, 230), (140, 162, 173)),
    ((232, 235, 239), (125, 135, 150)),
    ((240, 240, 240), (160, 160, 160)),
    ((255, 255, 221), (134, 166, 102)),
    ((236, 218, 185), (174, 138, 104)),
    ((220, 220, 220), (100, 130, 170)),
    ((245, 219, 195), (187, 87, 70)),
    ((230, 230, 250), (150, 120, 180)),
    ((255, 228, 178), (209, 139, 71)),
    ((200, 200, 200), (90, 90, 90)),
)
HIGHLIGHT_COLOURS = ((205, 210, 106), (155, 199, 0), (247, 236, 116), (100, 170, 220), (240, 150, 80))
ARROW_COLOURS = ((21, 120, 27), (136, 32, 32), (0, 48, 136), (230, 143, 0), (255, 170, 0))


@dataclass
class BoardImage:
    image: np.ndarray
    corners: np.ndarray
    meta: dict = field(default_factory=dict)


# --- drawing helpers ---------------------------------------------------------


def paste_premultiplied(dst: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """Composite a premultiplied patch onto dst at integer position (x, y)."""
    height, width = alpha.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, dst.shape[1]), min(y + height, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y : y1 - y, x0 - x : x1 - x, None]
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1.0 - a) + rgb[y0 - y : y1 - y, x0 - x : x1 - x]


def blend(dst: np.ndarray, colour, coverage: np.ndarray) -> None:
    """Blend a flat colour into dst using a 0..1 coverage map of the same size."""
    c = coverage[..., None]
    dst[:] = dst * (1.0 - c) + np.asarray(colour, np.float32) * c


def draw_texts(image: np.ndarray, items: list[tuple]) -> np.ndarray:
    """Draw (x, y, text, font, colour, anchor) items; returns a new float32 image."""
    if not items:
        return image
    canvas = Image.fromarray(np.clip(image + 0.5, 0, 255).astype(np.uint8))
    drawer = ImageDraw.Draw(canvas)
    for x, y, text, font, colour, anchor in items:
        drawer.text((x, y), text, font=font, fill=tuple(int(v) for v in colour), anchor=anchor)
    return np.asarray(canvas, dtype=np.float32).copy()


def smooth_field(rng: np.random.Generator, height: int, width: int, scale: float) -> np.ndarray:
    """Low-frequency noise in -1..1 with features about `scale` pixels across."""
    shape = (max(2, int(height / scale) + 2), max(2, int(width / scale) + 2))
    field = cv2.resize(rng.normal(0, 1, shape).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return field / (np.abs(field).max() + 1e-6)


def checker(sq: int) -> np.ndarray:
    """1.0 on dark squares, 0.0 on light, for an 8x8 board of sq-pixel squares."""
    dark = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(np.float32)
    return np.kron(dark, np.ones((sq, sq), np.float32))


def line_pattern(n: int, spacing: float, thickness: float, angle: float, offset: float) -> np.ndarray:
    """Antialiased parallel lines as a 0..1 ink-coverage map."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    theta = np.deg2rad(angle)
    distance = (xx * np.cos(theta) + yy * np.sin(theta) + offset) % spacing
    return np.clip(thickness / 2 - np.abs(distance - spacing / 2) + 0.5, 0.0, 1.0)


def dot_pattern(n: int, spacing: float, radius: float, angle: float) -> np.ndarray:
    """Antialiased dot lattice as a 0..1 ink-coverage map."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    theta = np.deg2rad(angle)
    u = (xx * np.cos(theta) + yy * np.sin(theta)) % spacing - spacing / 2
    v = (-xx * np.sin(theta) + yy * np.cos(theta)) % spacing - spacing / 2
    return np.clip(radius - np.hypot(u, v) + 0.5, 0.0, 1.0)


def speckle_pattern(rng: np.random.Generator, n: int, grain: float, share: float) -> np.ndarray:
    """Irregular ink specks about `grain` pixels across covering about `share` of the area, as a 0..1 map.

    Some books and scanned e-books print dark squares this way. Blurred white
    noise cut at a quantile gives blobs of the right size and coverage, with
    edges about a pixel soft.
    """
    noise = rng.standard_normal((n, n)).astype(np.float32)
    if grain > 1.0:
        noise = cv2.GaussianBlur(noise, (0, 0), grain * 0.4)
    spread = float(noise.std()) + 1e-6
    cut = float(np.quantile(noise[:: max(1, n // 64), :: 3], 1.0 - share))
    return np.clip((noise - cut) / (spread * 0.35) + 0.5, 0.0, 1.0)


def _playing_area_corners(origin: int, n: int) -> np.ndarray:
    lo, hi = origin - 0.5, origin + n - 0.5
    return np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)


def _coordinate_labels(white_at_bottom: bool, upper: bool) -> tuple[list[str], list[str]]:
    files = list("ABCDEFGH" if upper else "abcdefgh")
    ranks = list("87654321")
    return (files, ranks) if white_at_bottom else (files[::-1], ranks[::-1])


# --- printed book diagrams ---------------------------------------------------


@dataclass
class BookStyle:
    sq: int
    paper: np.ndarray
    ink: np.ndarray
    piece_set: str
    piece_scale: float
    piece_contrast: float
    piece_width: float
    piece_height: float
    stroke: float
    dark: str
    spacing: float
    thickness: float
    angle: float
    grey: float
    halo: float
    frame: int
    double_frame: bool
    grid: bool
    coords: bool
    coords_far_sides: bool
    coords_upper: bool
    coord_font: str
    coord_size: int
    marker: bool
    margin: int


def sample_book_style(
    rng: np.random.Generator, sq: int, paper: np.ndarray, ink: np.ndarray, piece_sets: tuple[str, ...] | None = None
) -> BookStyle:
    """Draw the look of a printed diagram; piece_sets limits which sets may be used."""
    every_set = [name for name in assets.piece_sets() if piece_sets is None or name in piece_sets]
    print_sets = [name for name in assets.print_piece_sets() if name in every_set] or every_set
    dark = str(rng.choice(["hatch", "grey", "halftone", "cross", "dots", "speckle"], p=[0.37, 0.27, 0.11, 0.07, 0.07, 0.11]))
    if dark == "speckle":
        spacing = sq * rng.uniform(0.012, 0.035)  # grain size
        thickness = 1.0  # not used: grey sets the share of ink
    elif dark == "halftone":
        spacing = sq / rng.uniform(9, 18)
        thickness = spacing * rng.uniform(0.25, 0.42)  # dot radius
    elif dark == "dots":
        spacing = sq / rng.uniform(4, 8)
        thickness = spacing * rng.uniform(0.12, 0.3)
    else:
        spacing = sq / rng.uniform(4.5, 11)
        thickness = spacing * rng.uniform(0.15, 0.45)
    piece_set = str(rng.choice(print_sets)) if rng.random() < 0.8 else str(rng.choice(every_set))
    patterned = dark in ("hatch", "cross", "dots", "speckle")
    # Typefaces differ in proportions and line weight far more than the available
    # artwork does, so pieces are stretched and their strokes thickened or thinned.
    stretch = float(rng.uniform(0.8, 1.25)) if rng.random() < 0.6 else 1.0
    stroke = float(rng.uniform(-1.0, 1.6)) if rng.random() < 0.45 else 0.0
    return BookStyle(
        sq=sq,
        paper=paper,
        ink=ink,
        piece_set=piece_set,
        piece_scale=float(rng.uniform(0.8, 1.0)),
        piece_contrast=float(rng.uniform(1.0, 2.5)),
        piece_width=min(1.0, 1.0 / stretch),
        piece_height=min(1.0, stretch),
        stroke=stroke,
        dark=dark,
        spacing=float(max(spacing, 2.0)),
        thickness=float(max(thickness, 0.6)),
        angle=float(rng.choice([45.0, 135.0]) + rng.normal(0, 1.5)),
        grey=float(rng.uniform(0.3, 0.55) if dark == "speckle" else rng.uniform(0.12, 0.5)),  # ink share
        halo=float(rng.uniform(0.03, 0.09)) if patterned and rng.random() < 0.75 else 0.0,
        frame=int(max(1, round(sq * rng.uniform(0.02, 0.09)))) if rng.random() < 0.85 else 0,
        double_frame=bool(rng.random() < 0.2),
        grid=bool(rng.random() < 0.12),
        coords=bool(rng.random() < 0.6),
        coords_far_sides=bool(rng.random() < 0.1),
        coords_upper=bool(rng.random() < 0.15),
        coord_font=str(rng.choice(assets.font_names())),
        coord_size=int(round(sq * rng.uniform(0.28, 0.44))),
        marker=bool(rng.random() < 0.2),
        margin=int(round(sq * rng.uniform(0.55, 0.95))),
    )


def _print_piece(style: BookStyle, symbol: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """A piece in ink on paper: returns premultiplied RGB and alpha."""
    patch = assets.piece(style.piece_set, symbol, size)
    alpha = patch[..., 3]
    luma = (patch[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    straight = np.where(alpha > 1e-3, luma / np.maximum(alpha, 1e-3), 1.0)
    straight = np.clip((straight - 0.5) * style.piece_contrast + 0.5, 0.0, 1.0)
    ink = alpha * (1.0 - straight)
    radius = int(round(abs(style.stroke) * size / 64))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        if style.stroke > 0:
            ink = cv2.dilate(ink, kernel)
            alpha = np.maximum(alpha, ink)
        else:
            ink = cv2.erode(ink, kernel)
    width, height = max(4, int(round(size * style.piece_width))), max(4, int(round(size * style.piece_height)))
    if (width, height) != (size, size):
        ink = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    white = (alpha - ink)[..., None]
    rgb = style.paper * white + style.ink * ink[..., None]
    return rgb, alpha


def render_book_board(
    rng: np.random.Generator, style: BookStyle, visual: list[str], white_at_bottom: bool
) -> BoardImage:
    sq, m = style.sq, style.margin
    n = 8 * sq
    size = n + 2 * m
    image = np.empty((size, size, 3), np.float32)
    image[:] = style.paper
    area = image[m : m + n, m : m + n]

    dark = checker(sq)
    offset = float(rng.uniform(0, style.spacing))
    if style.dark == "hatch":
        coverage = line_pattern(n, style.spacing, style.thickness, style.angle, offset)
    elif style.dark == "cross":
        coverage = np.maximum(
            line_pattern(n, style.spacing, style.thickness, style.angle, offset),
            line_pattern(n, style.spacing, style.thickness, style.angle + 90, offset),
        )
    elif style.dark == "speckle":
        coverage = speckle_pattern(rng, n, style.spacing, style.grey)
    elif style.dark in ("dots", "halftone"):
        coverage = dot_pattern(n, style.spacing, style.thickness, 45.0 if style.dark == "halftone" else 0.0)
    else:
        coverage = np.full((n, n), style.grey, np.float32)
    blend(area, style.ink, coverage * dark)

    if style.grid:
        for i in range(1, 8):
            area[i * sq, :] = style.ink
            area[:, i * sq] = style.ink

    if style.frame:
        t = style.frame
        rings = [(0, t)]
        if style.double_frame:
            gap = t + max(1, int(sq * 0.06))
            rings.append((gap, max(1, t // 2 + 1) + t))
        for gap, thickness in rings:
            lo, hi = m - gap - thickness, m + n + gap + thickness
            if lo < 0:
                continue
            image[lo : lo + thickness, lo:hi] = style.ink
            image[hi - thickness : hi, lo:hi] = style.ink
            image[lo:hi, lo : lo + thickness] = style.ink
            image[lo:hi, hi - thickness : hi] = style.ink

    if style.marker:
        side = max(3, int(sq * 0.28))
        x = m + n + max(2, (m - side) // 2)
        y = m + (int(sq * 0.3) if rng.random() < 0.5 else n - side - int(sq * 0.3))
        image[y : y + side, x : x + side] = style.ink
        if rng.random() < 0.5 and side > 4:
            image[y + 1 : y + side - 1, x + 1 : x + side - 1] = style.paper

    if style.coords:
        files, ranks = _coordinate_labels(white_at_bottom, style.coords_upper)
        font = assets.font(style.coord_font, style.coord_size)
        items = []
        for i in range(8):
            centre = m + i * sq + sq / 2
            items.append((centre, m + n + m * 0.5, files[i], font, style.ink, "mm"))
            items.append((m * 0.45, centre, ranks[i], font, style.ink, "mm"))
            if style.coords_far_sides:
                items.append((centre, m * 0.45, files[i], font, style.ink, "mm"))
                items.append((m + n + m * 0.55, centre, ranks[i], font, style.ink, "mm"))
        image = draw_texts(image, items)

    piece_size = max(6, int(round(sq * style.piece_scale)))
    shift_x, shift_y = rng.uniform(-0.03, 0.03, size=2) * sq
    halo_radius = int(round(sq * style.halo))
    for index, symbol in enumerate(visual):
        if not symbol:
            continue
        row, col = divmod(index, 8)
        rgb, alpha = _print_piece(style, symbol, piece_size)
        x = int(round(m + col * sq + (sq - alpha.shape[1]) / 2 + shift_x))
        y = int(round(m + row * sq + (sq - alpha.shape[0]) / 2 + shift_y))
        if halo_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_radius + 1,) * 2)
            padded = cv2.copyMakeBorder(alpha, *(halo_radius,) * 4, cv2.BORDER_CONSTANT, value=0)
            halo = cv2.dilate(padded, kernel)
            paste_premultiplied(image, halo[..., None] * style.paper, halo, x - halo_radius, y - halo_radius)
        paste_premultiplied(image, rgb, alpha, x, y)

    meta = {"style": "book", "dark_squares": style.dark, "piece_set": style.piece_set, "coords": style.coords}
    return BoardImage(image, _playing_area_corners(m, n), meta)


# --- on-screen boards --------------------------------------------------------


def _square_slice(origin: int, sq: int, row: int, col: int) -> tuple[slice, slice]:
    return slice(origin + row * sq, origin + (row + 1) * sq), slice(origin + col * sq, origin + (col + 1) * sq)


def render_screen_board(
    rng: np.random.Generator,
    sq: int,
    visual: list[str],
    white_at_bottom: bool,
    piece_sets: tuple[str, ...] | None = None,
) -> BoardImage:
    n = 8 * sq
    if rng.random() < 0.85:
        light, dark_colour = SCREEN_THEMES[int(rng.integers(len(SCREEN_THEMES)))]
        light = np.clip(np.array(light, np.float32) + rng.normal(0, 5, 3), 0, 255)
        dark_colour = np.clip(np.array(dark_colour, np.float32) + rng.normal(0, 7, 3), 0, 255)
    else:
        tint = rng.uniform(0.6, 1.0, 3)
        light = (255 - 45 * (1 - tint)).astype(np.float32) * rng.uniform(0.88, 1.0)
        dark_colour = (rng.uniform(70, 170) * tint).astype(np.float32)

    coords = str(rng.choice(["none", "inside", "outside"], p=[0.4, 0.45, 0.15]))
    m = int(sq * rng.uniform(0.45, 0.7)) if coords == "outside" else int(sq * rng.uniform(0, 0.25) * (rng.random() < 0.4))
    size = n + 2 * m
    surround = np.clip(dark_colour * rng.uniform(0.4, 1.3), 0, 255) if rng.random() < 0.5 else light
    image = np.empty((size, size, 3), np.float32)
    image[:] = surround
    area = image[m : m + n, m : m + n]
    dark = checker(sq)[..., None]
    area[:] = light * (1 - dark) + dark_colour * dark
    if rng.random() < 0.3:
        area += (smooth_field(rng, n, n, sq * rng.uniform(0.15, 1.5)) * rng.uniform(3, 12))[..., None]

    occupied = [i for i, symbol in enumerate(visual) if symbol]
    empty = [i for i, symbol in enumerate(visual) if not symbol]

    def tint_square(index: int, colour, strength: float) -> None:
        rows, cols = _square_slice(0, sq, *divmod(index, 8))
        area[rows, cols] = area[rows, cols] * (1 - strength) + np.asarray(colour, np.float32) * strength

    if rng.random() < 0.5:
        colour = HIGHLIGHT_COLOURS[int(rng.integers(len(HIGHLIGHT_COLOURS)))]
        strength = rng.uniform(0.35, 0.6)
        for index in rng.choice(64, size=2, replace=False):
            tint_square(int(index), colour, strength)
    if rng.random() < 0.12:
        tint_square(int(rng.integers(64)), HIGHLIGHT_COLOURS[int(rng.integers(len(HIGHLIGHT_COLOURS)))], 0.5)
    kings = [i for i in occupied if visual[i] in "Kk"]
    if kings and rng.random() < 0.08:
        row, col = divmod(kings[int(rng.integers(len(kings)))], 8)
        yy, xx = np.mgrid[0:sq, 0:sq].astype(np.float32)
        glow = np.clip(1.1 - np.hypot(xx - sq / 2, yy - sq / 2) / (sq * 0.6), 0, 1)
        rows, cols = _square_slice(0, sq, row, col)
        blend(area[rows, cols], (235, 40, 30), glow * 0.85)

    if coords != "none":
        files, ranks = _coordinate_labels(white_at_bottom, upper=False)
        font_name = str(rng.choice(assets.font_names()))
        items = []
        if coords == "inside":
            font = assets.font(font_name, int(round(sq * rng.uniform(0.19, 0.27))), weight=600)
            pad = sq * 0.07
            ranks_right, files_right = rng.random() < 0.5, rng.random() < 0.5
            for i in range(8):
                rank_col, file_row = (7 if ranks_right else 0), 7
                rank_colour = light if (i + rank_col) % 2 else dark_colour
                file_colour = light if (file_row + i) % 2 else dark_colour
                x = m + n - pad if ranks_right else m + pad
                items.append((x, m + i * sq + pad, ranks[i], font, rank_colour, "ra" if ranks_right else "la"))
                x = m + (i + 1) * sq - pad if files_right else m + i * sq + pad
                items.append((x, m + n - pad, files[i], font, file_colour, "rs" if files_right else "ls"))
        else:
            font = assets.font(font_name, int(round(sq * rng.uniform(0.26, 0.4))))
            colour = (235, 235, 235) if surround.mean() < 128 else (40, 40, 40)
            for i in range(8):
                centre = m + i * sq + sq / 2
                items.append((centre, m + n + m * 0.5, files[i], font, colour, "mm"))
                items.append((m * 0.5, centre, ranks[i], font, colour, "mm"))
        image = draw_texts(image, items)
        area = image[m : m + n, m : m + n]

    piece_set = str(rng.choice([name for name in assets.piece_sets() if piece_sets is None or name in piece_sets]))
    piece_size = max(6, int(round(sq * rng.uniform(0.84, 1.0))))
    for index in occupied:
        row, col = divmod(index, 8)
        patch = assets.piece(piece_set, visual[index], piece_size)
        x = m + col * sq + (sq - piece_size) // 2
        y = m + row * sq + (sq - piece_size) // 2
        paste_premultiplied(image, patch[..., :3], patch[..., 3], x, y)

    # Legal-move hints: dots on empty squares, rings on occupied ones.
    if rng.random() < 0.12:
        yy, xx = np.mgrid[0:sq, 0:sq].astype(np.float32)
        radius = np.hypot(xx - (sq - 1) / 2, yy - (sq - 1) / 2)
        dot = np.clip(sq * 0.16 - radius + 0.5, 0, 1)
        ring = np.clip(sq * 0.5 - radius + 0.5, 0, 1) * np.clip(radius - sq * 0.4 + 0.5, 0, 1)
        targets = rng.choice(64, size=int(rng.integers(2, 12)), replace=False)
        for index in targets:
            rows, cols = _square_slice(0, sq, *divmod(int(index), 8))
            blend(area[rows, cols], (20, 60, 20), (ring if visual[int(index)] else dot) * 0.3)

    if rng.random() < 0.2 and len(empty) + len(occupied) >= 2:
        mask = np.zeros((n, n), np.uint8)
        colour = ARROW_COLOURS[int(rng.integers(len(ARROW_COLOURS)))]
        for _ in range(int(rng.integers(1, 4))):
            a, b = rng.choice(64, size=2, replace=False)
            (r0, c0), (r1, c1) = divmod(int(a), 8), divmod(int(b), 8)
            start = (int(c0 * sq + sq / 2), int(r0 * sq + sq / 2))
            end = (int(c1 * sq + sq / 2), int(r1 * sq + sq / 2))
            length = max(1.0, float(np.hypot(end[0] - start[0], end[1] - start[1])))
            cv2.arrowedLine(mask, start, end, 255, max(2, int(sq * 0.16)), cv2.LINE_AA, tipLength=min(0.5, sq * 0.45 / length))
        blend(area, colour, mask.astype(np.float32) / 255.0 * rng.uniform(0.55, 0.85))
    if rng.random() < 0.1:
        mask = np.zeros((n, n), np.uint8)
        for index in rng.choice(64, size=int(rng.integers(1, 4)), replace=False):
            row, col = divmod(int(index), 8)
            centre = (int(col * sq + sq / 2), int(row * sq + sq / 2))
            cv2.circle(mask, centre, int(sq * 0.44), 255, max(2, int(sq * 0.07)), cv2.LINE_AA)
        blend(area, ARROW_COLOURS[int(rng.integers(len(ARROW_COLOURS)))], mask.astype(np.float32) / 255.0 * 0.8)

    if rng.random() < 0.05:
        image[:] = (image @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]

    meta = {"style": "screen", "piece_set": piece_set, "coords": coords != "none"}
    return BoardImage(np.clip(image, 0, 255), _playing_area_corners(m, n), meta)
