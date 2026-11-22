"""Check many readings at a glance, and record the outcome as labels.

`sheets` draws each unlabelled board with the proposed reading written onto
it: a tag in the corner of every square that holds a piece, yellow for White
and black for Black, and a red frame round squares the models were unsure of.
Checking a board then means looking for tags that do not match the piece
under them, and for pieces without a tag.

    python -m label.review sheets --photos eval/private/my-photos --labels eval/private/labels/my-photos.json \\
        --checkpoint runs/boardnet-v2/model.pt --out eval/private/review

`apply` turns the outcome into labels. It reads the proposals written by
`sheets` and a JSON file that names boards by their number on the sheets:

    {"ok": [0, 1, 4],                          accepted as proposed
     "fix": {"2": {"e4": "P", "h1": ""}},      accepted with these squares changed
     "skip": [3],                              unusable
     "black_at_bottom": [4],
     "source": {"some book": [0, 1, 2, 4]}}

Squares are named as the board appears on the sheet, whichever side is at
the bottom: files a to h from the left, ranks 8 down to 1 from the top.
Boards not mentioned are left unlabelled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from boardgen import CLASSES, rectify

from .store import Reader, Store, grid_to_rows, rows_to_grid

TILE = 640
MARGIN = 26
SUSPECT_BELOW = 0.9


def propose(models, device, image: np.ndarray, corners: np.ndarray, rng: np.random.Generator, jitters: int = 2) -> dict:
    """Read one board with every model, at all four turns and with the outline nudged.

    Returns the upright reading, its confidence per square, and the squares
    on which any single reading disagreed with the consensus.
    """
    import torch

    crops, extras = [], []
    for extra in range(4):
        for jitter in range(jitters + 1):
            crop = rectify(image, corners, 256, rng if jitter else None, jitter=0.015, max_margin=0.03)
            crops.append(np.rot90(crop, extra))
            extras.append(extra)
    batch = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float()
    squares, turns, white = [], [], []
    with torch.no_grad():
        for model in models:
            logits, orientation, rotation = model(batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            turn_probabilities = torch.softmax(rotation, dim=1).cpu().numpy()
            for index, extra in enumerate(extras):
                squares.append(np.rot90(probabilities[index], -extra, axes=(1, 2)))  # back to how the board lies
                turns.append(np.roll(turn_probabilities[index], extra))
            white.append(torch.sigmoid(orientation).mean().item())
    squares = np.stack(squares)
    mean = squares.mean(axis=0)
    consensus = mean.argmax(axis=0)
    suspect = (squares.argmax(axis=1) != consensus).any(axis=0) | (mean.max(axis=0) < SUSPECT_BELOW)
    quarter_turns = int(np.mean(turns, axis=0).argmax())
    return {
        "quarter_turns": quarter_turns,
        "seen": grid_to_rows(np.rot90(consensus, quarter_turns)),
        "suspect": np.rot90(suspect, quarter_turns).astype(int).tolist(),
        "white_at_bottom": bool(np.mean(white) >= 0.5),
    }


def problems(grid: np.ndarray) -> list[str]:
    """Reasons a reading cannot be a chess position. An empty list proves nothing."""
    found = []
    symbols = [CLASSES[int(value)] for value in grid.reshape(64)]
    for king in "Kk":
        if symbols.count(king) != 1:
            found.append(f"{symbols.count(king)}{king}")
    for pawn in "Pp":
        if symbols.count(pawn) > 8:
            found.append(f"{symbols.count(pawn)}{pawn}")
    if any(symbol in ("P", "p") for symbol in symbols[:8] + symbols[56:]):
        found.append("pawn on an end row")
    return found


def oddities(grid: np.ndarray) -> list[str]:
    """Material that is legal but rare, and so worth a second look."""
    found = []
    symbols = [CLASSES[int(value)] for value in grid.reshape(64)]
    for piece, usual in (("Q", 1), ("R", 2), ("B", 2), ("N", 2)):
        for symbol in (piece, piece.lower()):
            if symbols.count(symbol) > usual:
                found.append(f"{symbols.count(symbol)}{symbol}")
    for symbol in "Bb":
        shades = [(index // 8 + index % 8) % 2 for index, value in enumerate(symbols) if value == symbol]
        if len(shades) == 2 and shades[0] == shades[1]:
            found.append(f"both {symbol} on one colour")
    return found


def draw_tile(image: np.ndarray, corners: np.ndarray, proposal: dict, title: str) -> np.ndarray:
    board = np.ascontiguousarray(np.rot90(rectify(image, corners, TILE), proposal["quarter_turns"]))
    grid = rows_to_grid(proposal["seen"])
    suspect = np.asarray(proposal["suspect"], bool)
    square = TILE // 8
    for row in range(8):
        for column in range(8):
            x, y = column * square, row * square
            if suspect[row, column]:
                cv2.rectangle(board, (x + 1, y + 1), (x + square - 2, y + square - 2), (255, 0, 0), 2)
            symbol = CLASSES[int(grid[row, column])]
            if symbol:
                white = symbol.isupper()  # yellow tags for White, black for Black: unmistakable at a glance
                cv2.rectangle(board, (x + 2, y + 2), (x + 22, y + 22), (255, 225, 0) if white else (0, 0, 0), -1)
                cv2.putText(board, symbol.upper(), (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if white else (255, 255, 255), 2, cv2.LINE_AA)
    tile = np.full((TILE + 2 * MARGIN, TILE + MARGIN, 3), 255, np.uint8)
    tile[MARGIN : MARGIN + TILE, MARGIN:] = board
    for index in range(8):
        cv2.putText(tile, "87654321"[index], (7, MARGIN + index * square + square // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, "abcdefgh"[index], (MARGIN + index * square + square // 2 - 5, MARGIN + TILE + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(tile, title, (MARGIN, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2, cv2.LINE_AA)
    return tile


def sheets(args: argparse.Namespace) -> None:
    import torch

    from train.evaluate import pick_device
    from train.model import BoardNet

    device = pick_device()
    models = []
    for checkpoint in args.checkpoint:
        model = BoardNet().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        models.append(model.eval())
    store = Store(args.photos, args.labels, Reader(None))
    rng = np.random.default_rng(0)
    chosen = [board_id for board_id in store.ids() if store.boards[board_id]["status"] == args.status][args.first :][: args.count]
    args.out.mkdir(parents=True, exist_ok=True)
    proposals, tiles = {}, []
    for number, board_id in enumerate(chosen):
        record = store.boards[board_id]
        image = store.image(record["photo"])
        corners = np.asarray(record["corners"], np.float32)
        proposal = propose(models, device, image, corners, rng)
        proposals[str(number)] = {"id": board_id, **proposal}
        grid = rows_to_grid(proposal["seen"])
        flags, odd = problems(grid), oddities(grid)
        title = f"#{number}" + (f"  IMPOSSIBLE: {', '.join(flags)}" if flags else "") + (f"  odd: {', '.join(odd)}" if odd else "")
        tiles.append(draw_tile(image, corners, proposal, title))
    per = args.per_sheet
    columns = 2 if per > 1 else 1
    for start in range(0, len(tiles), per):
        group = tiles[start : start + per]
        group += [np.full_like(group[0], 255)] * (-len(group) % columns)
        rows = [np.hstack(group[i : i + columns]) for i in range(0, len(group), columns)]
        cv2.imwrite(str(args.out / f"sheet-{start // per:03d}.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (args.out / "proposals.json").write_text(json.dumps(proposals, indent=1) + "\n")
    print(f"{len(tiles)} boards on {-(-len(tiles) // per)} sheets in {args.out}")


def square_index(name: str) -> tuple[int, int]:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"{name!r} is not a square")
    return 8 - int(name[1]), "abcdefgh".index(name[0])


def apply(args: argparse.Namespace) -> None:
    proposals = json.loads(args.proposals.read_text())
    outcome = json.loads(args.corrections.read_text())
    fixes = {str(number): changes for number, changes in outcome.get("fix", {}).items()}
    accepted = {str(number) for number in outcome.get("ok", [])} | set(fixes)
    skipped = {str(number) for number in outcome.get("skip", [])}
    black = {str(number) for number in outcome.get("black_at_bottom", [])}
    sources = {str(number): name for name, numbers in outcome.get("source", {}).items() for number in numbers}
    unknown = (accepted | skipped | black | set(sources)) - set(proposals)
    if unknown or accepted & skipped:
        raise SystemExit(f"unknown board numbers {sorted(unknown)}; both accepted and skipped {sorted(accepted & skipped)}")
    store = Store(args.photos, args.labels, Reader(None))
    for number in sorted(accepted | skipped, key=int):
        proposal = proposals[number]
        record = store.boards[proposal["id"]]
        grid = rows_to_grid(proposal["seen"])
        for name, symbol in fixes.get(number, {}).items():
            if symbol not in CLASSES:
                raise SystemExit(f"board {number}: {symbol!r} is not a piece")
            grid[square_index(name)] = CLASSES.index(symbol)
        store.update(
            proposal["id"], record["corners"], proposal["quarter_turns"], grid_to_rows(grid),
            proposal["white_at_bottom"] and number not in black, sources.get(number, record["source"]),
            "skip" if number in skipped else "done", method="sheets",
        )  # fmt: skip
    print(f"{len(accepted)} boards labelled ({len(fixes)} with changes), {len(skipped)} marked unusable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("sheets", help="draw proposed readings onto the boards")
    make.add_argument("--photos", type=Path, required=True)
    make.add_argument("--labels", type=Path, required=True)
    make.add_argument("--checkpoint", action="append", required=True, help="model to read with; repeat to combine several")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--status", default="todo", help="which boards to draw")
    make.add_argument("--first", type=int, default=0)
    make.add_argument("--count", type=int, default=10**6)
    make.add_argument("--per-sheet", type=int, default=4)
    make.set_defaults(run=sheets)
    record = commands.add_parser("apply", help="record checked readings as labels")
    record.add_argument("--photos", type=Path, required=True)
    record.add_argument("--labels", type=Path, required=True)
    record.add_argument("--proposals", type=Path, required=True)
    record.add_argument("--corrections", type=Path, required=True)
    record.set_defaults(run=apply)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
