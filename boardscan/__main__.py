"""Scan images from the command line.

    python -m boardscan --finder runs/finder-v1/boardfinder-int8.onnx --reader runs/boardnet-v2/boardnet-int8.onnx photo.jpg
"""

from __future__ import annotations

import argparse

import cv2

from .pipeline import onnx_finder, onnx_reader, scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the position found in each image.")
    parser.add_argument("--finder", required=True, help="BoardFinder ONNX file")
    parser.add_argument("--reader", required=True, help="BoardNet ONNX file")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    finder, reader = onnx_finder(args.finder), onnx_reader(args.reader)
    for path in args.images:
        loaded = cv2.imread(path, cv2.IMREAD_COLOR)
        if loaded is None:
            print(f"{path}: not a readable image")
            continue
        result = scan(finder, reader, cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB))
        doubtful = int((result.confidence < 0.8).sum())
        print(f"{path}: {result.placement()}  board in view {result.board_in_view:.2f}, {doubtful} doubtful squares, turned {result.quarter_turns}")


if __name__ == "__main__":
    main()
