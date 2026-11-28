"""One labelled training sample, reproducible from (seed, index)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import board as board_module
from . import camera, positions, scene

# Stored data refers to kinds by position, so new kinds go at the end.
KINDS = ("book", "screenshot", "monitor", "ebook")
KIND_WEIGHTS = (0.45, 0.22, 0.13, 0.20)
POINTER_PROBABILITY = 0.25  # on screens of any kind
BADGE_PROBABILITY = 0.06  # on chess-site boards


@dataclass
class Sample:
    """A generated image and its labels.

    image: uint8 RGB.
    corners: float32 (4, 2), outer corners of the playing area in the image as
        top-left, top-right, bottom-right, bottom-left, in OpenCV pixel-centre
        coordinates.
    squares: 64 piece symbols ("" for empty) as seen in the image, row by row
        from the top-left square.
    placement: the FEN piece-placement field of the real position.
    quarter_turns: how many quarter turns counter-clockwise bring the image
        upright (0 to 3). Photos are often taken with the phone turned.
    white_at_bottom: False if the upright board is shown from Black's side.

    To recover the position: turn the 8x8 grid of `squares` counter-clockwise
    by `quarter_turns`, then turn it a further half turn if `white_at_bottom`
    is False. See `Sample.position()`.
    """

    image: np.ndarray
    corners: np.ndarray
    squares: list[str]
    placement: str
    white_at_bottom: bool
    kind: str
    quarter_turns: int = 0
    meta: dict = field(default_factory=dict)

    def position(self) -> list[str]:
        """The 64 squares in FEN order (a8 first), undoing rotation and viewing side."""
        grid = np.rot90(np.array(self.squares, dtype=object).reshape(8, 8), self.quarter_turns)
        squares = [str(symbol) for symbol in grid.ravel()]
        return squares if self.white_at_bottom else squares[::-1]

    def labels(self) -> dict:
        """Everything except the image, ready for JSON."""
        return {
            "kind": self.kind,
            "placement": self.placement,
            "white_at_bottom": self.white_at_bottom,
            "quarter_turns": self.quarter_turns,
            "squares": self.squares,
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners],
            "size": [int(self.image.shape[1]), int(self.image.shape[0])],
            "meta": self.meta,
        }


def generate(
    seed: int,
    index: int,
    kind: str | None = None,
    degrade: bool = True,
    long_side: int = 1024,
    board_fraction: tuple[float, float] = (0.35, 0.9),
    piece_sets: tuple[str, ...] | None = None,
    turns: bool = True,
    overlays: bool = True,
) -> Sample:
    """Generate sample `index` of the stream identified by `seed`.

    kind fixes the sample type; by default it is drawn from KIND_WEIGHTS.
    degrade=False skips lighting, blur, noise and compression (for tests).
    board_fraction is the range of the board's width as a share of the photo's
    shorter side. piece_sets limits which piece sets may be drawn. turns=False
    keeps every photo upright. overlays=False leaves out mouse pointers and
    end-of-game badges on screens.
    """
    rng = np.random.default_rng([seed, index])
    drawn = str(rng.choice(KINDS, p=KIND_WEIGHTS))
    kind = kind or drawn
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")

    placement, method = positions.sample_placement(rng)
    white_at_bottom = bool(rng.random() >= (0.4 if kind in ("screenshot", "monitor") else 0.15))
    visual = placement if white_at_bottom else placement[::-1]
    meta = {"seed": seed, "index": index, "position": method}

    # Overlays use their own random stream, so that a sample drawn without them
    # is the same picture with the same labels, minus the overlays.
    overlay_rng = np.random.default_rng([seed, index, 8])

    def add_overlays(view: np.ndarray, corners: np.ndarray, sq: int) -> dict:
        if not overlays:
            return {}
        added = {}
        if kind != "ebook" and overlay_rng.random() < BADGE_PROBABILITY:
            added["badges"] = scene.add_badges(overlay_rng, view, corners, sq, visual)
        if overlay_rng.random() < POINTER_PROBABILITY:
            scene.add_pointer(overlay_rng, view, corners, sq)
            added["pointer"] = True
        return added

    if kind == "screenshot":
        sq = int(rng.integers(28, 110))
        flat = board_module.render_screen_board(rng, sq, visual, white_at_bottom, piece_sets)
        view, corners, scene_meta = scene.make_screen(rng, flat, sq)
        scene_meta |= add_overlays(view, corners, sq)
        if degrade:
            image, scale = camera.degrade_screenshot(rng, view, long_side)
        else:
            image, scale = np.clip(view + 0.5, 0, 255).astype(np.uint8), 1.0
        corners = (corners + 0.5) * scale - 0.5
        meta.update(flat.meta | scene_meta)
        return Sample(image, corners.astype(np.float32), visual, positions.to_fen(placement), white_at_bottom, kind, 0, meta)

    portrait = rng.random() < 0.7
    width, height = (long_side * 3 // 4, long_side) if portrait else (long_side, long_side * 3 // 4)
    fraction = rng.uniform(*board_fraction)
    side = fraction * min(width, height)
    sq = max(14, int(round(side / 8 * rng.uniform(1.0, 1.25))))

    if kind in ("book", "ebook"):
        printed = kind == "book"
        warmth = rng.uniform(0, 1) if printed else rng.uniform(0, 0.4)
        lightness = rng.uniform(205, 252) if printed else rng.uniform(215, 255)
        paper = (lightness * np.array([1.0, 1 - 0.04 * warmth, 1 - 0.13 * warmth])).astype(np.float32)
        ink = (rng.uniform(8, 75) * np.array([1.0, 0.98, 0.95])).astype(np.float32)
        style = board_module.sample_book_style(rng, sq, paper, ink, piece_sets)
        flat = board_module.render_book_board(rng, style, visual, white_at_bottom)
        view, corners, scene_meta = scene.make_book_page(rng, style, flat, fraction, printed)
        if rng.random() < 0.7:  # slight ink spread and softness from printing or scanning
            view = cv2.GaussianBlur(view, (0, 0), max(0.3, sq / 48 * rng.uniform(0.3, 0.9)))
        if not printed:
            view, corners, reader_meta = scene.make_reader(rng, view, corners, sq)
            scene_meta |= reader_meta
    else:
        flat = board_module.render_screen_board(rng, sq, visual, white_at_bottom, piece_sets)
        view, corners, scene_meta = scene.make_screen(rng, flat, sq)

    if kind == "book":
        background = rng.uniform(20, 200) * rng.uniform(0.7, 1.0, 3)
        max_skew, curl_probability = 0.09, 0.5
    else:  # a monitor, photographed
        scene_meta |= add_overlays(view, corners, sq)
        bezel = int(sq * rng.uniform(0.2, 1.2))
        view = cv2.copyMakeBorder(view, bezel, bezel, bezel, bezel, cv2.BORDER_CONSTANT, value=(12.0, 12.0, 12.0))
        corners = corners + bezel
        background = rng.uniform(10, 160) * rng.uniform(0.7, 1.0, 3)
        max_skew, curl_probability = 0.12, 0.0

    photo, corners, in_scene, camera_meta = camera.photograph(
        rng, view, corners, width, height, side, max_skew, curl_probability, background
    )
    if degrade:
        image, degrade_meta = camera.degrade_photo(rng, photo, None if kind == "book" else in_scene)
    else:
        image, degrade_meta = np.clip(photo + 0.5, 0, 255).astype(np.uint8), {}
    meta.update(flat.meta | scene_meta | camera_meta | degrade_meta)

    # Phones are often held sideways. The turn uses its own random stream so
    # that everything else about a sample is the same with or without it.
    applied = 0
    if turns:
        roll = np.random.default_rng([seed, index, 7]).random()
        sideways, upside_down = (0.35, 0.03) if kind == "book" else (0.10, 0.02)
        applied = 1 if roll < sideways / 2 else 3 if roll < sideways else 2 if roll < sideways + upside_down else 0
    squares = visual
    for _ in range(applied):  # one quarter turn counter-clockwise at a time
        width = image.shape[1]
        image = np.ascontiguousarray(np.rot90(image))
        corners = np.roll(np.stack([corners[:, 1], (width - 1) - corners[:, 0]], axis=1), -1, axis=0)
        squares = [str(symbol) for symbol in np.rot90(np.array(squares, dtype=object).reshape(8, 8)).ravel()]
    return Sample(
        image, corners.astype(np.float32), squares, positions.to_fen(placement), white_at_bottom, kind, (4 - applied) % 4, meta
    )


def rectify(
    image: np.ndarray,
    corners: np.ndarray,
    size: int = 256,
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
    max_margin: float = 0.0,
) -> np.ndarray:
    """Warp the board to a size x size square, as the classifier will see it.

    With an rng, each corner is moved by up to `jitter` of the board's side and
    the crop is widened by up to `max_margin`, imitating an imperfect localizer.
    """
    quad = corners.astype(np.float64)
    if rng is not None:
        side = np.linalg.norm(quad[1] - quad[0])
        quad = quad + rng.uniform(-jitter, jitter, (4, 2)) * side
        centre = quad.mean(axis=0)
        quad = centre + (quad - centre) * (1 + rng.uniform(0, max_margin))
    target = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]], np.float32)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    side = float(np.linalg.norm(quad[1] - quad[0]))
    source = image
    if side > size * 1.5:  # soften before a strong reduction to limit aliasing
        source = cv2.GaussianBlur(image, (0, 0), side / size * 0.5)
    return cv2.warpPerspective(source, transform, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
