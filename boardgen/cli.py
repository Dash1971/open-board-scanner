"""Command line: write a dataset, or a contact sheet for visual review.

    python -m boardgen sheet --seed 1 --count 12 --out docs/samples/sheet.jpg
    python -m boardgen dump --seed 1 --count 1000 --out data/gen/seed1
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import assets
from .sample import generate, rectify

CELL = 384


def _fit(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Scale an image to fit a size x size cell; returns the cell, the scale and the offsets."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    cell = np.full((size, size, 3), 30, np.uint8)
    ox, oy = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
    cell[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    return cell, scale, ox, oy


def _sheet_row(args: tuple[int, int, str | None]) -> np.ndarray:
    seed, index, kind = args
    sample = generate(seed, index, kind)
    cell, scale, ox, oy = _fit(sample.image, CELL)
    outline = ((sample.corners + 0.5) * scale - 0.5 + [ox, oy]).round().astype(np.int32)
    cv2.polylines(cell, [outline], True, (0, 230, 0), 1, cv2.LINE_AA)
    cv2.circle(cell, tuple(int(v) for v in outline[0]), 4, (230, 0, 0), -1, cv2.LINE_AA)

    crop_rng = np.random.default_rng([seed, index, 1])
    crop = cv2.resize(rectify(sample.image, sample.corners, 256, crop_rng, jitter=0.02, max_margin=0.05), (CELL, CELL))
    for i in range(1, 8):  # the 8x8 grid the classifier assumes, over a deliberately imperfect crop
        cv2.line(crop, (i * CELL // 8, 0), (i * CELL // 8, CELL - 1), (0, 230, 0), 1)
        cv2.line(crop, (0, i * CELL // 8), (CELL - 1, i * CELL // 8), (0, 230, 0), 1)

    caption = Image.new("RGB", (2 * CELL, 34), (30, 30, 30))
    side = "white" if sample.white_at_bottom else "black"
    text = f"#{index} {sample.kind} | {side} at bottom | {sample.placement}"
    ImageDraw.Draw(caption).text((6, 17), text, font=assets.font("Arimo.ttf", 15), fill=(235, 235, 235), anchor="lm")
    return np.vstack([np.hstack([cell, crop]), np.asarray(caption)])


def _dump_one(args: tuple[int, int, str | None, str]) -> str:
    seed, index, kind, out = args
    sample = generate(seed, index, kind)
    name = f"{index:07d}.jpg"
    cv2.imwrite(str(Path(out) / "images" / name), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return json.dumps({"file": f"images/{name}", **sample.labels()}, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m boardgen", description="Synthetic chess-diagram generator")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sheet", "dump"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--start", type=int, default=0, help="first sample index")
        command.add_argument("--count", type=int, default=12)
        command.add_argument("--kind", choices=["book", "screenshot", "monitor"], default=None)
        command.add_argument("--out", required=True)
        command.add_argument("--workers", type=int, default=4)
    commands.choices["sheet"].add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    indices = range(args.start, args.start + args.count)

    if args.command == "sheet":
        with Pool(args.workers) as pool:
            rows = pool.map(_sheet_row, [(args.seed, i, args.kind) for i in indices])
        while len(rows) % args.columns:
            rows.append(np.full_like(rows[0], 30))
        sheet = np.vstack([np.hstack(rows[i : i + args.columns]) for i in range(0, len(rows), args.columns)])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"wrote {args.out} ({sheet.shape[1]}x{sheet.shape[0]})")
    else:
        out = Path(args.out)
        (out / "images").mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            lines = pool.map(_dump_one, [(args.seed, i, args.kind, str(out)) for i in indices], chunksize=16)
        (out / "labels.jsonl").write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} samples to {out}")


if __name__ == "__main__":
    main()
