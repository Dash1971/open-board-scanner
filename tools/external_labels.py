#!/usr/bin/env python3
"""Write the other labelled image sets in the labelling tool's format.

Training and scoring read labelled boards from the JSON files the labelling
tool writes (see label/store.py). This turns the sets that were labelled
another way into the same form, so they can be passed as --labels/--photos:

    wikisource   printed diagrams with Wikisource transcriptions
                 -> data/external/wikisource/labels.json, photos in images/
    chesscom     phone screenshots whose readings were checked by eye
                 -> eval/private/labels/chesscom-2026-11-23.store.json
    lichess      browser captures of known positions
                 -> data/external/lichess-shots/labels.json, photos in images/

    python tools/external_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boardscan.pipeline import SYMBOLS, find_board, onnx_finder, onnx_reader, read_board  # noqa: E402
from label.store import grid_to_rows, rows_to_grid  # noqa: E402


def record(photo: str, corners, seen: str, white_at_bottom: bool, source: str, method: str) -> dict:
    return {
        "photo": photo, "corners": np.asarray(corners, np.float64).round(1).tolist(), "quarter_turns": 0,
        "seen": grid_to_rows(rows_to_grid(seen)), "white_at_bottom": bool(white_at_bottom), "source": source,
        "status": "done", "found": True, "method": method,
    }  # fmt: skip


def write(path: Path, photos: str, boards: dict) -> None:
    path.write_text(json.dumps({"version": 1, "photos": photos, "boards": boards}, indent=1) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(boards)} boards")


def wikisource() -> None:
    folder = ROOT / "data" / "external" / "wikisource"
    if not (folder / "manifest.json").exists():
        return
    boards = {}
    for entry in json.loads((folder / "manifest.json").read_text()):
        if entry["status"] not in ("agrees", "checked"):
            continue
        photo = Path(entry["image"]).name
        key = photo if entry["index_on_page"] == 0 else f"{photo}#{entry['index_on_page'] + 1}"
        # Where the printed side cannot be told, White at the bottom is by far the likelier.
        boards[key] = record(photo, entry["corners"], entry["squares"], entry["white_at_bottom"] is not False, entry["book"], "wikisource")
    write(folder / "labels.json", "images", boards)


def chesscom() -> None:
    labels = ROOT / "eval" / "private" / "labels" / "chesscom-2026-11-23.json"
    if not labels.exists():
        return
    boards = {name: record(name, entry["corners"], entry["seen"], entry["white_at_bottom"], "chess.com app", "sheets") for name, entry in json.loads(labels.read_text()).items()}
    write(labels.with_suffix(".store.json"), "chesscom-2026-11-23", boards)


def lichess(finder, reader) -> None:
    folder = ROOT / "data" / "external" / "lichess-shots"
    if not (folder / "cases.json").exists():
        return
    boards = {}
    for case in json.loads((folder / "cases.json").read_text()):
        path = folder / "images" / f"{case['case']}.jpg"
        if not path.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        corners, _ = find_board(finder, image)
        seen = read_board(reader, image, corners)[0]
        cells = [0 if s == "." else SYMBOLS.index(s) for row in case["fen"].split()[0].split("/") for item in row for s in ("." * int(item) if item.isdigit() else item)]
        position = np.array(cells).reshape(8, 8)
        # The site shows the side to move at the bottom; the reading says which way up this capture is.
        flipped = (seen != position[::-1, ::-1]).sum() < (seen != position).sum()
        boards[path.name] = record(path.name, corners, grid_to_rows(position[::-1, ::-1] if flipped else position), not flipped, "lichess site", "known position")
    write(folder / "labels.json", "images", boards)


def main() -> None:
    wikisource()
    chesscom()
    lichess(onnx_finder(ROOT / "runs/finder-v1/boardfinder-int8.onnx"), onnx_reader(ROOT / "runs/boardnet-v4/boardnet-int8.onnx"))


if __name__ == "__main__":
    main()
