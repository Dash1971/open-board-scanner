"""Pre-render whole photos and screenshots for training the board finder.

Each sample is a full scene from boardgen with the board somewhere in it,
saved as a JPEG. Labels go in one JSON file per shard: the board's corners
(top-left, top-right, bottom-right, bottom-left as displayed), the kind of
scene and the piece set. Existing shards are skipped, so a run can be resumed
or extended.

    python -m train.make_scenes --seed 5 --count 12000 --holdout --out data/scenes/train
    python -m train.make_scenes --seed 5000 --count 600 --out data/scenes/test
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

from boardgen import assets, generate

from .make_crops import HELD_OUT_SETS

LONG_SIDE = 640


def _make_shard(job: tuple) -> tuple[int, float]:
    seed, start, count, holdout, out = job
    labels_path = Path(out) / f"labels-{start:07d}.json"
    if labels_path.exists():
        return start, 0.0
    began = time.time()
    allowed = tuple(name for name in assets.piece_sets() if name not in HELD_OUT_SETS) if holdout else None
    labels = []
    for index in range(start, start + count):
        sample = generate(seed, index, long_side=LONG_SIDE, piece_sets=allowed)
        name = f"scene-{index:07d}.jpg"
        cv2.imwrite(str(Path(out) / name), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        labels.append({"file": name, "corners": sample.corners.round(2).tolist(), "kind": sample.kind, "piece_set": sample.meta["piece_set"]})
    labels_path.write_text(json.dumps(labels) + "\n")
    return start, time.time() - began


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard", type=int, default=200)
    parser.add_argument("--holdout", action="store_true", help="leave out the held-out piece sets")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed, start, min(args.shard, args.start + args.count - start), args.holdout, args.out) for start in range(args.start, args.start + args.count, args.shard)]
    began = time.time()
    with Pool(args.workers) as pool:
        for done, (start, seconds) in enumerate(pool.imap_unordered(_make_shard, jobs), 1):
            print(f"shard {start:7d}  {seconds:5.0f}s  {done}/{len(jobs)}  {(time.time() - began) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
