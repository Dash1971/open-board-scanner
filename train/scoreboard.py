"""Score one pair of models on every test the project has, and write one file.

    python -m train.scoreboard --name boardnet-v4 --finder runs/finder-v1/boardfinder-int8.onnx \\
        --reader runs/boardnet-v4/boardnet-int8.onnx

Running it again for a later model gives a second file with the same rows, so
the two can be compared line by line. A test whose data is not on this
machine is left out. Every row counts boards read exactly: all 64 squares.

Rows:
    photos            private labelled photos, from the whole photo, by source
    wikisource        printed diagrams from public-domain books, outline given
    archive scans     printed diagrams from Internet Archive scans, outline given
    chess.com app     phone screenshots, from the whole screenshot
    lichess           browser captures, from the whole screenshot
    rendered boards   clean boards in many piece sets, from the whole image
    chess positions set   the public Chess Positions boards held out of training, board given
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from boardscan.pipeline import SYMBOLS, find_board, onnx_finder, onnx_reader, read_board

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "eval" / "private"
EXTERNAL = ROOT / "data" / "external"


def grid_of(rows: str) -> np.ndarray:
    cells = [0 if symbol == "." else SYMBOLS.index(symbol) for row in rows.split()[0].split("/") for item in row for symbol in ("." * int(item) if item.isdigit() else item)]
    return np.array(cells).reshape(8, 8)


def load(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Tally:
    def __init__(self) -> None:
        self.groups: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])

    def add(self, group: str, exact: bool) -> None:
        self.groups[group][0] += 1
        self.groups[group][1] += int(exact)

    def report(self) -> dict:
        groups = {name: {"boards": boards, "exact": exact} for name, (boards, exact) in sorted(self.groups.items())}
        return {"boards": sum(g["boards"] for g in groups.values()), "exact": sum(g["exact"] for g in groups.values()), "by_group": groups}


def photos(finder, reader) -> dict | None:
    from .real import load_labelled

    tally = Tally()
    for labels in sorted((PRIVATE / "labels").glob("telegram-*.json")):
        folder = PRIVATE / labels.stem
        if not folder.is_dir():
            continue
        for board in load_labelled(labels, folder):
            corners, _ = find_board(finder, board.image)
            seen = read_board(reader, board.image, corners)[0]
            tally.add(board.source, bool((seen == np.rot90(board.squares, board.quarter_turns)).all()))
    return tally.report() if tally.groups else None


def wikisource(finder, reader) -> dict | None:
    manifest = EXTERNAL / "wikisource" / "manifest.json"
    if not manifest.exists():
        return None
    tally, pages = Tally(), {}
    for entry in json.loads(manifest.read_text()):
        if entry["status"] not in ("agrees", "checked"):
            continue
        if entry["image"] not in pages:
            pages[entry["image"]] = load(ROOT / entry["image"])
        seen = read_board(reader, pages[entry["image"]], np.asarray(entry["corners"], np.float32))[0]
        tally.add(entry["book"], bool((seen == grid_of(entry["squares"])).all()))
    return tally.report()


def archive(finder, reader) -> dict | None:
    from label.store import rows_to_grid

    labels = EXTERNAL / "ia" / "labels.json"
    if not labels.exists():
        return None
    tally, page, name = Tally(), None, None
    for record in json.loads(labels.read_text())["boards"].values():
        if record["status"] != "done":
            continue
        if record["photo"] != name:  # boards of one page follow each other
            name, page = record["photo"], load(EXTERNAL / "ia" / record["photo"])
        seen = read_board(reader, page, np.asarray(record["corners"], np.float32))[0]
        tally.add(record["source"], bool((seen == rows_to_grid(record["seen"])).all()))
    return tally.report()


def chesscom(finder, reader) -> dict | None:
    labels = PRIVATE / "labels" / "chesscom-2026-11-23.json"
    if not labels.exists():
        return None
    tally = Tally()
    for name, entry in json.loads(labels.read_text()).items():
        image = load(PRIVATE / "chesscom-2026-11-23" / name)
        seen = read_board(reader, image, find_board(finder, image)[0])[0]
        tally.add("default pieces and board", bool((seen == grid_of(entry["seen"])).all()))
    return tally.report()


def lichess(finder, reader) -> dict | None:
    folder = EXTERNAL / "lichess-shots"
    if not (folder / "cases.json").exists():
        return None
    tally = Tally()
    for case in json.loads((folder / "cases.json").read_text()):
        path = folder / "images" / f"{case['case']}.jpg"
        if not path.exists():
            continue
        image = load(path)
        seen = read_board(reader, image, find_board(finder, image)[0])[0]
        truth = grid_of(case["fen"])  # the site shows the side to move at the bottom, so either way up is right
        tally.add(case["look"].replace("leave the piece set and board as they are", "default").replace("keep ", ""), bool((seen == truth).all() or (seen == truth[::-1, ::-1]).all()))
    return tally.report() if tally.groups else None


def rendered(finder, reader) -> dict | None:
    folder = EXTERNAL / "piece-sets-zip" / "chess-piece-sets-33x50"
    if not (folder / "manifest.jsonl").exists():
        return None
    tally = Tally()
    for line in (folder / "manifest.jsonl").read_text().splitlines():
        item = json.loads(line)
        image = load(folder / item["image"])
        seen, _, turns, _ = read_board(reader, image, find_board(finder, image)[0])
        truth = np.array([0 if symbol == "." else SYMBOLS.index(symbol) for symbol in item["screen_labels_row_major"]]).reshape(8, 8)
        tally.add(item["piece_set"], bool(turns == 0 and (seen == truth).all()))
    return tally.report()


def chess_positions(finder, reader) -> dict | None:
    from .data import load_all, shard_starts

    folder = ROOT / "data" / "crops" / "kaggle" / "chess-positions-test"
    if not shard_starts(folder):
        return None
    data = load_all(folder)
    box = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32) - 0.5
    tally = Tally()
    for image, squares in zip(data["images"], data["squares"]):
        seen, _, turns, _ = read_board(reader, image, box)
        tally.add("32 piece styles, 28 board styles", bool(turns == 0 and (seen.reshape(64) == squares).all()))
    return tally.report()


TESTS = (
    ("photos", photos),
    ("wikisource", wikisource),
    ("archive scans", archive),
    ("chess.com app", chesscom),
    ("lichess", lichess),
    ("rendered boards", rendered),
    ("chess positions set", chess_positions),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="a short name for this pair of models; also the output file name")
    parser.add_argument("--finder", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--note", default="", help="what these models were trained on, so trained-on rows can be told from unseen ones")
    parser.add_argument("--only", action="append", default=[], help="run just this test and keep the file's other rows (repeatable)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "scoreboard")
    args = parser.parse_args()
    finder, reader = onnx_finder(args.finder), onnx_reader(args.reader)
    path = args.out / f"{args.name}.json"
    board = json.loads(path.read_text()) if args.only and path.exists() else {"name": args.name, "note": args.note, "tests": {}}
    for title, test in TESTS:
        if args.only and title not in args.only:
            continue
        result = test(finder, reader)
        if result is None:
            print(f"{title}: data not here, left out")
            continue
        board["tests"][title] = result
        print(f"{title}: {result['exact']} of {result['boards']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=1) + "\n")


if __name__ == "__main__":
    main()
