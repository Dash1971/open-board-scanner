"""Score the pipeline on a folder of rendered boards with a manifest, by piece set.

    python -m train.evaluate_renders --data data/external/boards --finder runs/finder-v1/boardfinder-int8.onnx \\
        --reader runs/boardnet-v3/boardnet-int8.onnx

The folder holds `manifest.jsonl`, one JSON object per image with:

    image                     path of the image inside the folder
    piece_set                 name used to group the results
    board_box_xyxy            [left, top, right, bottom] of the board, right and bottom exclusive
    screen_labels_row_major   64 symbols as displayed, top-left first; "." is an empty square
    orientation               "white_bottom" or "black_bottom"

Each board is read three ways: from its given box at one turn, from its given
box at four turns, and from the whole image with the finder.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from boardscan.pipeline import SYMBOLS, find_board, onnx_finder, onnx_reader, read_board

from .train_finder import TOLERANCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--finder", required=True)
    parser.add_argument("--reader", required=True)
    args = parser.parse_args()
    finder, reader = onnx_finder(args.finder), onnx_reader(args.reader)
    results: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in (args.data / "manifest.jsonl").read_text().splitlines():
        item = json.loads(line)
        loaded = cv2.imread(str(args.data / item["image"]), cv2.IMREAD_COLOR)
        if loaded is None:
            raise SystemExit(f"cannot read {item['image']}")
        image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        truth = np.array([0 if symbol == "." else SYMBOLS.index(symbol) for symbol in item["screen_labels_row_major"]]).reshape(8, 8)
        left, top, right, bottom = item["board_box_xyxy"]
        box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32) - 0.5
        row = results[item["piece_set"]]
        row["boards"] += 1
        once, _, turns, _ = read_board(reader, image, box, turns=(0,))
        row["exact_given_box_one_turn"] += int(turns == 0 and (once == truth).all())
        voted, _, turns, white = read_board(reader, image, box)
        row["exact_given_box_four_turns"] += int(turns == 0 and (voted == truth).all())
        row["viewing_side_right"] += int(white == (item["orientation"] == "white_bottom"))
        corners, _ = find_board(finder, image)
        row["outline_within_tolerance"] += int(np.linalg.norm(corners - box, axis=1).max() / (right - left) <= TOLERANCE)
        seen, _, turns, _ = read_board(reader, image, corners)
        row["exact_from_whole_image"] += int(turns == 0 and (seen == truth).all())
    print(json.dumps({name: dict(results[name]) for name in sorted(results)}, indent=1))


if __name__ == "__main__":
    main()
