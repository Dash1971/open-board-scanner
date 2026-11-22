"""Turning a flat scene into a phone photo: page curl, perspective and image degradations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .board import smooth_field


@dataclass
class Curl:
    """A smooth bulge across the page, as near the spine of an open book.

    A photo pixel that would show page point p instead shows p + offset(p).
    """

    cos: float
    sin: float
    centre: float
    width: float
    lift: float
    squeeze: float

    def offset(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = (x * self.cos + y * self.sin - self.centre) / self.width
        bump = np.exp(-t * t)
        along = self.squeeze * t * bump
        across = self.lift * bump
        return along * self.cos - across * self.sin, along * self.sin + across * self.cos

    def invert(self, points: np.ndarray) -> np.ndarray:
        """Where scene points end up after the curl (fixed-point solve of p + offset(p) = c)."""
        solved = points.astype(np.float64).copy()
        for _ in range(40):
            dx, dy = self.offset(solved[:, 0], solved[:, 1])
            solved = points - np.stack([dx, dy], axis=1)
        return solved


def sample_curl(rng: np.random.Generator, corners: np.ndarray) -> Curl:
    side = float(np.linalg.norm(corners[1] - corners[0]))
    angle = np.deg2rad(rng.normal(0, 10))
    centre_point = corners.mean(axis=0) + rng.uniform(-1.2, 1.2, 2) * side
    return Curl(
        cos=float(np.cos(angle)),
        sin=float(np.sin(angle)),
        centre=float(centre_point[0] * np.cos(angle) + centre_point[1] * np.sin(angle)),
        width=float(side * rng.uniform(0.6, 1.5)),
        lift=float(side * rng.uniform(-0.07, 0.07)),
        squeeze=float(side * rng.uniform(-0.05, 0.05)),
    )


def _target_quad(rng: np.random.Generator, width: int, height: int, side: float, skew: float) -> np.ndarray:
    half = side / 2
    margin = 0.04 * min(width, height)
    cx = rng.uniform(half + margin, max(half + margin + 1, width - half - margin))
    cy = rng.uniform(half + margin, max(half + margin + 1, height - half - margin))
    aspect = rng.uniform(0.93, 1.07)
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) * [1.0, aspect]
    angle = np.deg2rad(np.clip(rng.normal(0, 3.5), -12, 12))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    quad = square @ rotation.T + rng.uniform(-skew, skew, (4, 2)) * side
    return (quad + [cx, cy]).astype(np.float32)


def photograph(
    rng: np.random.Generator,
    scene: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
    side: float,
    max_skew: float,
    curl_probability: float,
    background,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Warp the scene into a width x height photo in which the board spans about `side` pixels.

    Returns the photo, the board corners in it, a mask of the pixels that show
    the scene rather than the background, and a description of the warp.
    """
    margin = 0.015 * min(width, height)
    for attempt in range(12):
        skew = rng.uniform(0, max_skew) * (0.5 ** (attempt // 4))
        curl = sample_curl(rng, corners) if rng.random() < curl_probability and attempt < 8 else None
        quad = _target_quad(rng, width, height, side, skew)
        homography = cv2.getPerspectiveTransform(corners.astype(np.float32), quad)
        curled = curl.invert(corners) if curl else corners.astype(np.float64)
        photo_corners = cv2.perspectiveTransform(curled.reshape(1, 4, 2), homography.astype(np.float64))[0]
        inside = (
            (photo_corners[:, 0] > margin).all()
            and (photo_corners[:, 0] < width - 1 - margin).all()
            and (photo_corners[:, 1] > margin).all()
            and (photo_corners[:, 1] < height - 1 - margin).all()
        )
        if inside and cv2.isContourConvex(photo_corners.astype(np.float32)):
            break
    else:  # fall back to a plain centred view
        curl, skew = None, 0.0
        half = min(side, 0.9 * min(width, height)) / 2
        centre = np.array([width / 2, height / 2])
        quad = (centre + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])).astype(np.float32)
        homography = cv2.getPerspectiveTransform(corners.astype(np.float32), quad)
        photo_corners = quad.astype(np.float64)

    inverse = np.linalg.inv(homography)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    w = inverse[2, 0] * xs + inverse[2, 1] * ys + inverse[2, 2]
    map_x = (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]) / w
    map_y = (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]) / w
    if curl:
        dx, dy = curl.offset(map_x, map_y)
        map_x, map_y = map_x + dx, map_y + dy
    photo = cv2.remap(
        scene,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in background),
    )
    in_scene = (map_x >= 0) & (map_x <= scene.shape[1] - 1) & (map_y >= 0) & (map_y <= scene.shape[0] - 1)
    meta = {"curl": curl is not None, "skew": round(float(skew), 4)}
    return photo, photo_corners.astype(np.float32), in_scene, meta


# --- degradations ------------------------------------------------------------


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade_photo(
    rng: np.random.Generator, photo: np.ndarray, screen: np.ndarray | None
) -> tuple[np.ndarray, dict]:
    """Lighting, shadow, moire, blur, noise and JPEG compression. Returns uint8 RGB.

    screen is the mask of pixels showing a monitor, or None for a printed page.
    """
    monitor = screen is not None
    height, width = photo.shape[:2]
    image = photo.astype(np.float32)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    applied = {}

    if monitor and rng.random() < 0.6:  # moire between the screen's pixel grid and the sensor
        period = rng.uniform(4, 40)
        angle = rng.uniform(0, np.pi)
        phase = (xs * np.cos(angle) + ys * np.sin(angle)) * (2 * np.pi / period)
        wobble = smooth_field(rng, height, width, 200) * rng.uniform(0, 6)
        amplitude = rng.uniform(0.02, 0.12)
        shifts = rng.uniform(0, 2.0, 3) if rng.random() < 0.5 else np.zeros(3)
        for channel in range(3):
            image[..., channel] *= 1 + amplitude * np.sin(phase + wobble + shifts[channel]) * screen
        applied["moire"] = True

    # Uneven lighting: a linear falloff plus a broad random field.
    direction = rng.uniform(0, 2 * np.pi)
    ramp = ((xs / width - 0.5) * np.cos(direction) + (ys / height - 0.5) * np.sin(direction)) * rng.uniform(0, 0.5)
    lighting = 1 + ramp + smooth_field(rng, height, width, 300) * rng.uniform(0, 0.12)
    image *= lighting[..., None]

    if rng.random() < 0.3:  # cast shadow with a soft edge, e.g. from the phone or a hand
        direction = rng.uniform(0, 2 * np.pi)
        distance = (xs - width * rng.uniform(0.2, 0.8)) * np.cos(direction) + (ys - height * rng.uniform(0.2, 0.8)) * np.sin(direction)
        softness = rng.uniform(15, 120)
        shade = 1 / (1 + np.exp(-distance / softness))
        image *= (1 - shade * rng.uniform(0.15, 0.5))[..., None]
        applied["shadow"] = True

    if rng.random() < (0.3 if monitor else 0.08):  # glare
        gx, gy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(0.1, 0.4) * max(width, height)
        image += (np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * radius**2)) * rng.uniform(20, 90))[..., None]
        applied["glare"] = True

    image *= rng.uniform(0.9, 1.1, 3) * rng.uniform(0.75, 1.1)  # white balance and exposure
    image = 255.0 * np.power(np.clip(image, 0, 255) / 255.0, rng.uniform(0.8, 1.3))

    if rng.random() < 0.65:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.7))
    if rng.random() < 0.1:  # motion blur
        length = int(rng.integers(3, 10))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        rotation = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        image = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))
        applied["motion_blur"] = True
    if rng.random() < 0.2:  # a low-resolution capture, upscaled
        factor = rng.uniform(0.5, 0.9)
        small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    sigma = rng.uniform(0, 7)
    image += rng.normal(0, sigma, (height, width, 1)).astype(np.float32)
    image += rng.normal(0, sigma * 0.4, image.shape).astype(np.float32)
    result = jpeg(np.clip(image + 0.5, 0, 255).astype(np.uint8), int(rng.integers(40, 96)))
    return result, applied


def degrade_screenshot(rng: np.random.Generator, image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Screenshots stay clean apart from rescaling and, sometimes, JPEG. Returns uint8 RGB and the scale used."""
    height, width = image.shape[:2]
    scale = min(rng.uniform(0.6, 1.3), max_side / max(height, width))
    if abs(scale - 1) > 0.02:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    else:
        scale = 1.0
    result = np.clip(image + 0.5, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        result = jpeg(result, int(rng.integers(55, 96)))
    return result, float(scale)
