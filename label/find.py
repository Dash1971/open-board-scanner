"""Find board outlines in a photo with plain contour detection.

This is a labelling aid, not the localizer the scanner will ship with: it only
has to be right often enough that few outlines need dragging into place.
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    quad = quad.reshape(4, 2).astype(np.float32)
    total, difference = quad.sum(axis=1), quad[:, 0] - quad[:, 1]
    return np.array([quad[total.argmin()], quad[difference.argmax()], quad[total.argmax()], quad[difference.argmin()]], np.float32)


def _same_outline(first: np.ndarray, second: np.ndarray) -> bool:
    side = np.linalg.norm(first[1] - first[0])
    return bool(np.abs(first - second).max() < 0.08 * side)


def find_boards(image: np.ndarray, limit: int = 6) -> list[np.ndarray]:
    """Candidate board outlines, largest first, each 4 x 2 in photo pixels."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape
    smooth = cv2.GaussianBlur(grey, (5, 5), 0)
    found: list[tuple[float, np.ndarray]] = []
    for block in (31, 61, 101):
        binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.04 * height * width < area < 0.9 * height * width:
                continue
            outline = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(outline) != 4 or not cv2.isContourConvex(outline):
                continue
            quad = order_corners(outline)
            sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
            if min(sides) / max(sides) < 0.7:
                continue
            found.append((area, quad))
    boards: list[np.ndarray] = []
    for _, quad in sorted(found, key=lambda item: -item[0]):
        if not any(_same_outline(quad, kept) for kept in boards):
            boards.append(quad)
        if len(boards) == limit:
            break
    return boards


def default_outline(image: np.ndarray) -> np.ndarray:
    """A centred square to drag into place when nothing was found."""
    height, width = image.shape[:2]
    half = 0.3 * min(height, width)
    cx, cy = width / 2, height / 2
    return np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]], np.float32)
