"""Turn the "Chess Positions" board images into crop shards.

The images come from tools/fetch_chess_positions.py (Hugging Face mirror of
koryakinp's Kaggle dataset, CC0). Each image is a whole 400 x 400 board seen
from White's side and named by the piece-placement part of its FEN, with "-"
for "/". The mirror holds only the Kaggle training images, so the split is
made here from a hash of the position: no position lands in both parts.

Each image gives one 256 x 256 crop. Test crops are the board resized with
area interpolation; training crops go through `rectify` with the same small
outline error as train.make_crops, so the model does not learn that boards
are always cut perfectly. Labels follow train.make_crops: `kind` is
"screenshot", and every crop points at one extra piece-set name,
"chess-positions", appended to the current set names, because the dataset
does not name its board and piece styles. Shards hold 1,000 crops; existing
shards are skipped, so a run can be resumed.

    python -m train.import_chess_positions
    python -m train.import_chess_positions --source data/external/chess-positions --out data/crops
"""

from __future__ import annotations

import argparse
import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from boardgen import KINDS, rectify
from train.make_crops import CLASS_INDEX, CROP, set_names

ROOT = Path(__file__).resolve().parent.parent
SET_NAME = "chess-positions"
TEST_PERCENT = 20
SHARD = 1000
SEED = 4242


def squares_from_name(name: str) -> list[int]:
    """Class numbers of the 64 squares, top-left first, from a file name such as
    `1B1B4-8-8-8-8-8-8-k6K.jpeg` (the board is drawn with White at the bottom)."""
    rows = Path(name).name.split(".")[0].split("-")
    if len(rows) != 8:
        raise ValueError(f"{name}: expected 8 rows, got {len(rows)}")
    squares: list[int] = []
    for row in rows:
        cells: list[int] = []
        for character in row:
            if character in "12345678":
                cells.extend([0] * int(character))
            elif character in CLASS_INDEX and character:
                cells.append(CLASS_INDEX[character])
            else:
                raise ValueError(f"{name}: unexpected character {character!r}")
        if len(cells) != 8:
            raise ValueError(f"{name}: row {row!r} does not have 8 squares")
        squares.extend(cells)
    return squares


def is_test(name: str) -> bool:
    """Whether a position belongs to the test part, decided by a hash of the position alone."""
    position = Path(name).name.split(".")[0]
    return int(hashlib.sha256(position.encode()).hexdigest()[:8], 16) % 100 < TEST_PERCENT


def _make_shard(job: tuple) -> tuple[str, int, float]:
    part, start, paths, out, names = job
    images_path = Path(out) / f"images-{start:07d}.npy"
    labels_path = Path(out) / f"labels-{start:07d}.npz"
    if labels_path.exists():
        return part, start, 0.0
    began = time.time()
    count = len(paths)
    images = np.empty((count, CROP, CROP, 3), np.uint8)
    squares = np.empty((count, 64), np.uint8)
    for row, path in enumerate(paths):
        picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if picture is None:
            raise ValueError(f"cannot read {path}")
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        if part == "train":
            height, width = picture.shape[:2]
            corners = np.array([[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]])
            rng = np.random.default_rng([SEED, start + row])
            images[row] = rectify(picture, corners, CROP, rng, jitter=0.025, max_margin=0.06)
        else:
            images[row] = cv2.resize(picture, (CROP, CROP), interpolation=cv2.INTER_AREA)
        squares[row] = squares_from_name(path.name)
    np.save(images_path, images)
    np.savez(
        labels_path,
        squares=squares,
        white_at_bottom=np.ones(count, np.uint8),
        kind=np.full(count, KINDS.index("screenshot"), np.uint8),
        piece_set=np.full(count, names.index(SET_NAME), np.uint8),
        quarter_turns=np.zeros(count, np.uint8),
        set_names=np.array(names),
    )
    return part, start, time.time() - began


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(ROOT / "data" / "external" / "chess-positions"))
    parser.add_argument("--out", default=str(ROOT / "data" / "crops"), help="parent folder of chess-positions-train and -test")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    files = sorted(path for path in Path(args.source).glob("*/*.jp*g"))
    if not files:
        raise SystemExit(f"no images under {args.source}; run tools/fetch_chess_positions.py first")
    names = set_names() + [SET_NAME]
    if len(names) > 255:
        raise SystemExit("too many set names for a uint8 piece_set")
    parts = {"train": [path for path in files if not is_test(path.name)], "test": [path for path in files if is_test(path.name)]}
    jobs = []
    for part, paths in parts.items():
        out = Path(args.out) / f"chess-positions-{part}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{part}: {len(paths)} images -> {out}")
        jobs += [(part, start, paths[start : start + SHARD], str(out), names) for start in range(0, len(paths), SHARD)]
    began = time.time()
    with Pool(args.workers) as pool:
        for done, (part, start, seconds) in enumerate(pool.imap_unordered(_make_shard, jobs), 1):
            print(f"{part} shard {start:07d} done in {seconds:4.0f}s  ({done}/{len(jobs)}, {time.time() - began:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
