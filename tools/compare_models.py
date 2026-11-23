#!/usr/bin/env python3
"""Run this project's models and three other projects' models on the same boards.

    python tools/compare_models.py --others data/external/others --renders data/external/boards \\
        --labels eval/private/labels/a.json --photos eval/private/a --fold 0 5 --out results/comparison

Every model is given the same input: the board already cut out, upright, as a
512 x 512 image. A board counts as exact when all 64 squares match, either
as they appear or turned half round, so no model is judged on which side it
thinks is at the bottom. The two models
that can take a whole photo are also scored that way on the real photos.

The other projects' files are not part of this repository. The versions used,
with the hashes they were checked against, are recorded in the Mobile Maia
study this follows:
https://github.com/Dash1971/mobile-maia-research/tree/main/studies/offline-board-recognition-2026-11-13

    chessq-lite-v4-int8.onnx      ChessQ Lite V4, as served by Lichess
    fenshot-chess-tiles-v2.onnx   Fenshot
    fenify-model.pt               Fenify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boardscan.geometry import rectify  # noqa: E402
from boardscan.pipeline import SYMBOLS, find_board, onnx_finder, onnx_reader, read_board  # noqa: E402

SIDE = 512
BOX = np.array([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]], np.float32) - 0.5
IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def session(path: Path):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def ours(reader):
    return lambda crop: read_board(reader, crop, BOX)[0]


def chessq(path: Path):
    """ChessQ Lite: any image resized to 644 x 644, ImageNet normalisation, 64 x 13 logits in our class order."""
    net = session(path)
    name = net.get_inputs()[0].name

    def read(image: np.ndarray) -> np.ndarray:
        resized = np.asarray(Image.fromarray(image).resize((644, 644), Image.Resampling.BILINEAR), np.float32) / 255.0
        tensor = ((resized - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        outputs = dict(zip([output.name for output in net.get_outputs()], net.run(None, {name: tensor})))
        return outputs["logits"][0].argmax(axis=-1).reshape(8, 8)

    return read


def fenshot(path: Path):
    """Fenshot: grey 256 x 256 board cut into 32 x 32 tiles, first tile at a1; classes "1KQRBNPkqrbnp"."""
    net = session(path)
    order = np.array([SYMBOLS.index("" if symbol == "1" else symbol) for symbol in "1KQRBNPkqrbnp"])

    def read(crop: np.ndarray) -> np.ndarray:
        grey = crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        board = cv2.resize(grey, (256, 256), interpolation=cv2.INTER_LINEAR) / 255.0
        tiles = np.stack([board[(7 - rank) * 32 : (8 - rank) * 32, file * 32 : (file + 1) * 32].reshape(1024) for rank in range(8) for file in range(8)])
        probabilities = net.run(["probs"], {"tiles": tiles.astype(np.float32)})[0]
        return order[probabilities.argmax(axis=1)].reshape(8, 8)[::-1]

    return read


def fenify(path: Path):
    """Fenify: grey 300 x 300 image in three channels, ImageNet normalisation; 64 x 13 outputs starting at a1."""
    import torch

    torch.set_num_threads(2)
    net = torch.jit.load(str(path), map_location="cpu").eval()

    def read(crop: np.ndarray) -> np.ndarray:
        grey = Image.fromarray(crop).convert("L").resize((300, 300), Image.Resampling.BILINEAR)
        image = np.repeat(np.asarray(grey, np.float32)[..., None] / 255.0, 3, axis=2)
        tensor = torch.from_numpy(((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32))
        with torch.no_grad():
            return net(tensor).squeeze(0).argmax(dim=1).numpy().reshape(8, 8)[::-1]

    return read


def wrong(seen: np.ndarray, truth: np.ndarray) -> int:
    """Wrong squares, taking the better of the board as seen and turned half round.

    ChessQ writes a board shown from Black's side as if seen from White's.
    That is a correct reading of the position, so either way up is accepted,
    for every model.
    """
    return int(min((seen != truth).sum(), (seen[::-1, ::-1] != truth).sum()))


def tally(rows: list[tuple[str, int, float]]) -> dict:
    groups = sorted({row[0] for row in rows})
    summary = lambda chosen: {"boards": len(chosen), "exact": sum(row[1] == 0 for row in chosen), "wrong_squares": sum(row[1] for row in chosen)}  # noqa: E731
    return {**summary(rows), "milliseconds_per_board": round(1000 * sum(row[2] for row in rows) / max(1, len(rows)), 1), "by_group": {group: summary([row for row in rows if row[0] == group]) for group in groups}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--others", type=Path, required=True, help="folder holding the three other projects' model files")
    parser.add_argument("--finder", default=str(ROOT / "runs/finder-v1/boardfinder-int8.onnx"))
    parser.add_argument("--reader", default=str(ROOT / "runs/boardnet-v4/boardnet-int8.onnx"))
    parser.add_argument("--renders", type=Path, help="rendered boards with manifest.jsonl (see train/evaluate_renders.py)")
    parser.add_argument("--labels", action="append", default=[])
    parser.add_argument("--photos", action="append", default=[])
    parser.add_argument("--fold", type=int, nargs=2, metavar=("I", "K"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    finder, reader = onnx_finder(args.finder), onnx_reader(args.reader)
    whole_chessq = chessq(args.others / "chessq-lite-v4-int8.onnx")
    models = {
        "open-board-scanner": ours(reader),
        "chessq-lite-v4-int8": whole_chessq,
        "fenshot": fenshot(args.others / "fenshot-chess-tiles-v2.onnx"),
        "fenify": fenify(args.others / "fenify-model.pt"),
    }
    cases: dict[str, list] = {"real_photos": [], "rendered_boards": []}  # (group, crop, truth, whole image or None)
    if args.labels:
        from train.finetune import split
        from train.real import load_labelled_sets

        boards = load_labelled_sets(args.labels, args.photos)
        if args.fold:
            boards = split(boards, [], tuple(args.fold), 0)[1]
        for board in boards:
            crop = np.ascontiguousarray(np.rot90(rectify(board.image, board.corners, SIDE), board.quarter_turns))
            cases["real_photos"].append((board.source, crop, np.rot90(board.squares, board.quarter_turns), board.image))
    if args.renders:
        for line in (args.renders / "manifest.jsonl").read_text().splitlines():
            item = json.loads(line)
            image = cv2.cvtColor(cv2.imread(str(args.renders / item["image"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            left, top, right, bottom = item["board_box_xyxy"]
            crop = cv2.resize(image[top:bottom, left:right], (SIDE, SIDE), interpolation=cv2.INTER_AREA if right - left > SIDE else cv2.INTER_LINEAR)
            truth = np.array([0 if symbol == "." else SYMBOLS.index(symbol) for symbol in item["screen_labels_row_major"]]).reshape(8, 8)
            cases["rendered_boards"].append((item["piece_set"], crop, truth, None))

    report: dict = {"input": "board cut out, upright, 512 x 512", "threads": 2}
    for test, items in cases.items():
        if not items:
            continue
        report[test] = {}
        for name, read in models.items():
            rows = []
            for group, crop, truth, _ in items:
                began = time.perf_counter()
                seen = read(crop)
                rows.append((group, wrong(seen, truth), time.perf_counter() - began))
            report[test][name] = tally(rows)
            print(test, name, report[test][name]["exact"], "of", len(rows), flush=True)
    if cases["real_photos"]:
        report["real_photos_whole_image"] = {}
        routes = {
            "open-board-scanner": lambda image: read_board(reader, image, find_board(finder, image)[0]),
            "chessq-lite-v4-int8": lambda image: (whole_chessq(image), None, 0, None),
        }
        for name, route in routes.items():
            rows = []
            for board, (group, _, truth, image) in zip(boards, cases["real_photos"]):
                began = time.perf_counter()
                if name == "chessq-lite-v4-int8":  # it does not turn the board, so it is given the photo upright
                    image = np.ascontiguousarray(np.rot90(image, board.quarter_turns))
                seen = route(image)[0]
                rows.append((group, wrong(seen, truth), time.perf_counter() - began))
            report["real_photos_whole_image"][name] = tally(rows)
            print("whole image", name, report["real_photos_whole_image"][name]["exact"], "of", len(rows), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
