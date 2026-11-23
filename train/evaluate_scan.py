"""Score the whole pipeline on labelled real photos: photo in, position out.

    python -m train.evaluate_scan --finder runs/finder-v1/boardfinder-int8.onnx --reader runs/boardnet-v2/boardnet-int8.onnx \\
        --labels eval/private/labels/my-photos.json --photos eval/private/my-photos

Nothing but the photo is given to the models. A board counts as exact when
every square of the upright board matches the label. The same reader is also
run on the labelled outline, to show what the finder costs.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from boardscan.pipeline import find_board, onnx_finder, onnx_reader, read_board

from .real import load_labelled_sets
from .train_finder import TOLERANCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--finder", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--photos", action="append", required=True)
    parser.add_argument("--source", action="append", default=[], help="only these sources (repeatable)")
    parser.add_argument("--fold", type=int, nargs=2, metavar=("I", "K"), help="only the boards train.finetune held out with the same --fold and --seed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    finder, reader = onnx_finder(args.finder), onnx_reader(args.reader)
    boards = load_labelled_sets(args.labels, args.photos)
    if args.fold:
        from .finetune import split

        boards = split(boards, [], tuple(args.fold), args.seed)[1]
    boards = [board for board in boards if not args.source or board.source in args.source]

    rows, seconds = [], 0.0
    for board in boards:
        truth = np.rot90(board.squares, board.quarter_turns)
        began = time.perf_counter()
        corners, in_view = find_board(finder, board.image)
        seen, _, turns, _ = read_board(reader, board.image, corners)
        seconds += time.perf_counter() - began
        given, _, _, _ = read_board(reader, board.image, board.corners)
        side = np.mean([np.linalg.norm(board.corners[i] - board.corners[(i + 1) % 4]) for i in range(4)])
        # The finder may name the corners starting from a different one when the photo is sideways; compare as a set.
        offset = min(np.linalg.norm(np.roll(corners, shift, axis=0) - board.corners, axis=1).max() for shift in range(4)) / side
        rows.append((board.source, int((seen != truth).sum()), int((given != truth).sum()), float(offset), turns == board.quarter_turns, in_view))

    def summary(chosen: list) -> dict:
        return {
            "boards": len(chosen),
            "exact_from_photo": sum(row[1] == 0 for row in chosen),
            "wrong_squares_from_photo": sum(row[1] for row in chosen),
            "exact_with_labelled_outline": sum(row[2] == 0 for row in chosen),
            "outline_within_tolerance": sum(row[3] <= TOLERANCE for row in chosen),
            "worst_outline_error": round(max(row[3] for row in chosen), 4),
            "turned_upright": sum(bool(row[4]) for row in chosen),
        }

    report = {"all": summary(rows), "by_source": {source: summary([row for row in rows if row[0] == source]) for source in sorted({row[0] for row in rows})}}
    report["milliseconds_per_photo"] = round(1000 * seconds / max(1, len(rows)), 1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
