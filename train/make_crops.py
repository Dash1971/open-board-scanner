"""Pre-render rectified board crops for classifier training.

Samples come from boardgen in "crop mode": a smaller photo in which the board
is large, so rendering is fast and the 256-pixel crop is not upscaled. Each
sample yields several crops with different simulated localizer error.

Output is a folder of shards. Shard k holds `images-k.npy` (uint8, N x 256 x
256 x 3) and `labels-k.npz`. Crops keep the photo's own orientation, so a
sideways photo gives a sideways crop, labelled with `quarter_turns`. Existing shards are skipped, so a run can be
resumed or extended.

    python -m train.make_crops --seed 1 --count 24000 --holdout --out data/crops/train
    python -m train.make_crops --seed 1000 --count 3000 --crops 1 --out data/crops/test
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from boardgen import CLASSES, KINDS, assets, generate, rectify

CROP = 256
# Piece sets kept out of training so that accuracy on unseen artwork can be measured.
HELD_OUT_SETS = ("fantasy", "mpchess")
CLASS_INDEX = {symbol: index for index, symbol in enumerate(CLASSES)}


def set_names() -> list[str]:
    return list(assets.piece_sets())


def _make_shard(job: tuple) -> tuple[int, float]:
    seed, start, count, crops, holdout, out = job
    images_path = Path(out) / f"images-{start:07d}.npy"
    labels_path = Path(out) / f"labels-{start:07d}.npz"
    if labels_path.exists():
        return start, 0.0
    began = time.time()
    names = set_names()
    allowed = tuple(name for name in names if name not in HELD_OUT_SETS) if holdout else None
    images = np.empty((count * crops, CROP, CROP, 3), np.uint8)
    squares = np.empty((count * crops, 64), np.uint8)
    white_at_bottom = np.empty(count * crops, np.uint8)
    kind = np.empty(count * crops, np.uint8)
    piece_set = np.empty(count * crops, np.uint8)
    quarter_turns = np.empty(count * crops, np.uint8)
    for offset in range(count):
        index = start + offset
        sample = generate(seed, index, long_side=640, board_fraction=(0.55, 0.92), piece_sets=allowed)
        for crop in range(crops):
            row = offset * crops + crop
            rng = np.random.default_rng([seed, index, 1 + crop])
            images[row] = rectify(sample.image, sample.corners, CROP, rng, jitter=0.025, max_margin=0.06)
            squares[row] = [CLASS_INDEX[symbol] for symbol in sample.squares]
            white_at_bottom[row] = sample.white_at_bottom
            kind[row] = KINDS.index(sample.kind)
            piece_set[row] = names.index(sample.meta["piece_set"])
            quarter_turns[row] = sample.quarter_turns
    np.save(images_path, images)
    np.savez(
        labels_path,
        squares=squares,
        white_at_bottom=white_at_bottom,
        kind=kind,
        piece_set=piece_set,
        quarter_turns=quarter_turns,
        set_names=np.array(names),
    )
    return start, time.time() - began


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard", type=int, default=500, help="samples per shard")
    parser.add_argument("--crops", type=int, default=2, help="crops per sample")
    parser.add_argument("--holdout", action="store_true", help="exclude the held-out piece sets")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [
        (args.seed, start, min(args.shard, args.start + args.count - start), args.crops, args.holdout, args.out)
        for start in range(args.start, args.start + args.count, args.shard)
    ]
    began = time.time()
    with Pool(args.workers) as pool:
        for done, (start, seconds) in enumerate(pool.imap_unordered(_make_shard, jobs), 1):
            print(f"shard {start:07d} done in {seconds:5.0f}s  ({done}/{len(jobs)}, {time.time() - began:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
