#!/usr/bin/env python3
"""Turn the Wikisource chess pages into a test set of printed diagrams.

Reads what tools/fetch_wikisource.py downloaded and works in three steps:

    python tools/wikisource_manifest.py read     find and read every diagram, draw sheets of the disagreements
    (look at the sheets, write the verdicts into decisions.json)
    python tools/wikisource_manifest.py build    write manifest.json

`read` parses the {{Chess diagram}} templates on each page
(tools/chess_diagram_template.py), finds the board outlines on the scan, and
matches them to the templates in reading order: top to bottom, then left to
right. Outlines come from label.find.find_boards, refined by the ONNX board
finder looking at a window round each, and from the finder looking at the
whole page with the boards already found painted out; each must also show
alternating light and dark squares where an 8 x 8 grid inside it puts them. A page where the number of outlines is not the number of templates is
left unmatched: no guessing. Each matched board is cut out with
boardscan.geometry.rectify and read with the ONNX reader, and the reading is
compared square by square with the template, both as printed (the reading
before it is turned upright). Results go to readings.json, and every board
where the two differ is drawn on sheets/, the differing squares framed in red
and listed underneath as square:template/reading.

decisions.json holds the verdicts given by eye, by diagram id:

    {"<id>": {"right": "model" | "wikisource" | "neither",
              "rows": "<placement as printed, when neither was right>",
              "exclude": "<reason, instead of a verdict>",
              "note": "..."}}

`build` writes manifest.json with one entry per diagram, the squares as
printed in FEN style (top row first, whichever side is at the bottom), and
whether White is at the bottom: from a verdict, else from the BLACK / WHITE
captions some books print round the diagram, else from where the pawns and
kings stand (side_from_position), else unknown. Status is `agrees` when the
reading and the template matched, `checked` when they differed and a verdict
settled it, `excluded` with a reason otherwise. Diagrams on unmatched pages,
diagrams with fairy codes, arrows or markers, and disagreements without a
verdict are all excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from boardscan.geometry import around, cut_view, from_view, rectify  # noqa: E402
from boardscan.pipeline import SYMBOLS, find_board, onnx_finder, onnx_reader, read_board  # noqa: E402
from chess_diagram_template import _template_end, parse  # noqa: E402
from label.find import find_boards  # noqa: E402

DEST = ROOT / "data" / "external" / "wikisource"
FINDER = ROOT / "runs" / "finder-v1" / "boardfinder-int8.onnx"
READER = ROOT / "runs" / "boardnet-v4" / "boardnet-int8.onnx"
BOOKS = {
    "Chess fundamentals (IA chessfundamental00capa).pdf": "Capablanca, Chess Fundamentals (1921)",
    "My Chess Career (Capablanca).djvu": "Capablanca, My Chess Career (1920)",
    "The chess-player's text book.djvu": "Staunton, The Chess-Player's Text Book (1849)",
    "How to Play Chess (Rogers).djvu": "Rogers, How to Play Chess (1907)",
    "Collier's Cyclopedia of Commercial and Social Information.djvu": "Collier's Cyclopedia of Commercial and Social Information (1882)",
    "Chess Player's Chronicle - Series 1 - Volume 1.djvu": "The Chess Player's Chronicle, series 1, volume 1 (1841)",
    "Chess Player's Chronicle - Series 3 - Volume 1.djvu": "The Chess Player's Chronicle, series 3, volume 1 (1877)",
    "Through the Looking-Glass, and What Alice Found There.djvu": "Carroll, Through the Looking-Glass (1872)",
}
GRID_CONTRAST = 1.0  # alternating squares must differ by this many spreads of the square-to-square noise
AGREE_SIDES = 0.06  # finder and contour outlines agree when no corner is further apart than this share of a side
SQUARISH = 0.85


def grid_to_rows(grid) -> list[str]:
    return ["".join(SYMBOLS[int(value)] or "." for value in row) for row in grid]


def placement(rows: list[str]) -> str:
    return "/".join(re.sub(r"\.+", lambda run: str(len(run.group())), row) for row in rows)


def unplace(text: str) -> list[str]:
    rows = [re.sub(r"\d", lambda digit: "." * int(digit.group()), row) for row in text.split("/")]
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        raise ValueError(f"not 8 rows of 8: {text}")
    return rows


def side(quad: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))


def squarish(quad: np.ndarray) -> bool:
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return min(sides) / max(sides) >= SQUARISH


def refine(finder, image: np.ndarray, quad: np.ndarray, looks: int = 2) -> tuple[np.ndarray, float]:
    """Ask the finder for the board in a window round `quad`, then once more round what it found."""
    corners, logit = quad.astype(np.float64), 0.0
    for _ in range(looks):
        matrix = around(corners)
        fractions, logit = finder(cut_view(image, matrix))
        corners = from_view(fractions, matrix)
    return corners.astype(np.float32), float(1.0 / (1.0 + np.exp(-logit)))


def grid_contrast(image: np.ndarray, corners: np.ndarray, shift: float = 0.0) -> float:
    """How clearly the inside of `corners` alternates light and dark like an 8 x 8 board.

    The median brightness of the middle of each square is taken; the gap
    between the two colours of square is divided by the spread within each
    colour. With `shift` the grid is moved by that share of a square, which
    should spoil the pattern when the outline is right.
    """
    crop = rectify(image, np.asarray(corners, np.float32), 256).mean(axis=2)
    crop = np.roll(crop, (-int(shift * 32), -int(shift * 32)), axis=(0, 1))
    values: list[list[float]] = [[], []]
    for row in range(8):
        for column in range(8):
            values[(row + column) % 2].append(float(np.median(crop[row * 32 + 5 : row * 32 + 27, column * 32 + 5 : column * 32 + 27])))
    first, second = np.array(values[0]), np.array(values[1])
    spread = np.std(np.concatenate([first - np.median(first), second - np.median(second)])) + 3.0
    return float(abs(np.median(first) - np.median(second)) / spread)


def looks_like_board(image: np.ndarray, corners: np.ndarray) -> bool:
    aligned = grid_contrast(image, corners)
    return bool(squarish(corners) and aligned >= GRID_CONTRAST and aligned > 3 * grid_contrast(image, corners, 0.5))


def outlines(finder, image: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    """Board outlines on a page, in reading order, with notes on the candidates looked at.

    First each contour candidate is refined by the finder looking at a window
    round it. Then the finder looks at the whole page, again and again, with
    the boards found so far painted over, until it finds nothing more. Every
    outline must pass the light-and-dark check of grid_contrast.
    """
    width = image.shape[1]
    found, notes = [], []

    def consider(corners: np.ndarray, in_view: float, note: dict) -> bool:
        board = bool(in_view >= 0.5 and looks_like_board(image, corners))
        note.update(finder=corners.round(1).tolist(), in_view=round(in_view, 3), grid=round(grid_contrast(image, corners), 2), board=board)
        notes.append(note)
        if not board or any(float(np.abs(corners - kept).max()) < 0.15 * side(kept) for kept in found):
            return False
        if any(_overlap(corners, kept) > 0.2 for kept in found):
            note["overlaps"] = True
            return False
        found.append(corners)
        return True

    for quad in find_boards(image, limit=16):
        if not (squarish(quad) and 0.12 * width < side(quad) < 0.95 * width):
            continue
        corners, in_view = refine(finder, image, quad)
        gap = float(np.abs(corners - quad).max()) / side(quad)
        if gap <= AGREE_SIDES:
            consider(corners, in_view, {"from": "contour", "contour": quad.round(1).tolist(), "gap": round(gap, 3)})
    page = image.copy()
    background = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    for _ in range(8):
        for kept in found:
            low, high = kept.min(axis=0), kept.max(axis=0)
            margin = 0.06 * (high - low)
            (x0, y0), (x1, y1) = np.maximum(low - margin, 0).astype(int), (high + margin).astype(int)
            page[y0 : y1 + 1, x0 : x1 + 1] = background
        corners, in_view = find_board(finder, page)
        if not consider(corners, in_view, {"from": "page"}):
            break
    # Reading order: a board belongs to the same line as another when their heights overlap by half.
    found.sort(key=lambda quad: quad[:, 1].mean())
    lines: list[list[np.ndarray]] = []
    for quad in found:
        top, bottom = quad[:, 1].min(), quad[:, 1].max()
        if lines:
            last = lines[-1][0]
            overlap = min(bottom, last[:, 1].max()) - max(top, last[:, 1].min())
            if overlap > 0.5 * (bottom - top):
                lines[-1].append(quad)
                continue
        lines.append([quad])
    return [quad for line in lines for quad in sorted(line, key=lambda quad: quad[:, 0].mean())], notes


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Overlap of the two outlines' bounding boxes, as a share of the smaller one."""
    low, high = np.maximum(first.min(axis=0), second.min(axis=0)), np.minimum(first.max(axis=0), second.max(axis=0))
    inter = float(np.prod(np.clip(high - low, 0, None)))
    area = min(float(np.prod(first.max(axis=0) - first.min(axis=0))), float(np.prod(second.max(axis=0) - second.min(axis=0))))
    return inter / max(area, 1.0)


def column_order(quads: list[np.ndarray]) -> list[int]:
    """Indices of `quads` (given in reading order) taken down each column, columns from left to right."""
    columns: list[list[int]] = []
    for index in sorted(range(len(quads)), key=lambda at: quads[at][:, 0].mean()):
        left, right = quads[index][:, 0].min(), quads[index][:, 0].max()
        if columns:
            last = quads[columns[-1][0]]
            if min(right, last[:, 0].max()) - max(left, last[:, 0].min()) > 0.5 * (right - left):
                columns[-1].append(index)
                continue
        columns.append([index])
    return [index for column in columns for index in sorted(column, key=lambda at: quads[at][:, 1].mean())]


def _count_differ(first: list[str], second: list[str]) -> int:
    return sum(a != b for a, b in zip("".join(first), "".join(second)))


def _caption_cells(text: str) -> list[str]:
    """Short pieces of text between pipes and line breaks, with formatting templates and quotes removed."""
    text = re.sub(r"\{\{[A-Za-z ]+\|", "|", text).replace("}}", "|").replace("{{", "|").replace("\'\'", "")
    return [cell.strip() for cell in re.split(r"[|\n]", text) if cell.strip()]


def caption_side(wikitext: str, offset: int) -> bool | None:
    """True when the page prints a caption 'Black' just above the diagram and 'White' just below, as several of these books do.

    Only a caption standing on its own counts, such as {{sc|BLACK.}}; the
    words in running text do not.
    """
    end = _template_end(wikitext, offset)
    caption = re.compile(r"^(black|white)\.?$", re.IGNORECASE)
    above = [cell.lower().rstrip(".") for cell in _caption_cells(wikitext[max(0, offset - 250) : offset]) if caption.match(cell)]
    below = [cell.lower().rstrip(".") for cell in _caption_cells(wikitext[end : end + 250]) if caption.match(cell)]
    if above and below and above[-1] != below[0]:
        return below[0] == "white"
    return None


def read(args: argparse.Namespace) -> None:
    finder, reader = onnx_finder(FINDER), onnx_reader(READER)
    pages = json.loads((DEST / "pages.json").read_text())
    readings = []
    for number, page in enumerate(pages):
        if not (DEST / page["image"]).exists() or not (DEST / page["wikitext"]).exists():
            continue
        wikitext = (DEST / page["wikitext"]).read_text(encoding="utf-8")
        diagrams = parse(wikitext)
        image = np.asarray(Image.open(DEST / page["image"]).convert("RGB"))
        quads, notes = outlines(finder, image)
        matched = len(quads) == len(diagrams)
        looks = []
        for corners in quads if matched else []:
            seen, confidence, quarter_turns, white = read_board(reader, image, corners)
            looks.append((corners, grid_to_rows(np.rot90(seen, -quarter_turns)), quarter_turns, white, confidence))  # rows as printed
        order, order_name = list(range(len(looks))), "rows"
        columns = column_order(quads) if matched else order
        if columns != order:
            # Boards side by side: the transcriber may have gone down each column instead of along each line.
            cost = lambda chosen: sum(_count_differ(looks[at][1], diagram.rows) for at, diagram in zip(chosen, diagrams))
            if cost(columns) * 2 < cost(order):
                order, order_name = columns, "columns"
        for index, diagram in enumerate(diagrams):
            entry = {
                "id": f"{Path(page['image']).stem}_{index}",
                "book": BOOKS.get(page["file"], page["file"]),
                "file": page["file"],
                "page_title": page["title"],
                "image": page["image"],
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "index_on_page": index,
                "diagrams_on_page": len(diagrams),
                "outlines_on_page": len(quads),
                "wikisource": diagram.placement(),
                "template_source": diagram.source,
                "template_reverse": diagram.reverse,
                "markers": diagram.markers,
                "template_problems": diagram.problems,
                "caption_white_at_bottom": caption_side(wikitext, diagram.offset),
                "matched": matched,
            }
            if matched:
                corners, model, quarter_turns, white, confidence = looks[order[index]]
                entry["page_order"] = order_name
                differ = [f"{'abcdefgh'[c]}{8 - r}" for r in range(8) for c in range(8) if model[r][c] != diagram.rows[r][c]]
                entry.update(
                    corners=corners.round(1).tolist(),
                    model=placement(model),
                    model_quarter_turns=quarter_turns,
                    model_white_at_bottom=white,
                    model_min_confidence=round(float(confidence.min()), 3),
                    differ=differ,
                )
            entry["outline_notes"] = notes
            readings.append(entry)
        if number % 20 == 0:
            print(f"{number + 1}/{len(pages)} pages", flush=True)
    (DEST / "readings.json").write_text(json.dumps(readings, indent=1, ensure_ascii=False))
    disagree = [entry for entry in readings if entry.get("differ")]
    for number, entry in enumerate(disagree):
        entry["sheet_number"] = number
    (DEST / "readings.json").write_text(json.dumps(readings, indent=1, ensure_ascii=False))
    print(f"{len(readings)} diagrams, {sum(entry['matched'] for entry in readings)} matched, {len(disagree)} differ")
    draw_sheets(disagree, DEST / "sheets", "differ")


TILE, MARGIN = 400, 24


def draw_sheets(entries: list[dict], folder: Path, prefix: str, per_row: int = 3, rows_per_sheet: int = 3) -> None:
    """Each board cut out, with the squares where the template and the reading differ framed in red.

    Under each board the differences are listed as square: template / reading,
    so a glance at the framed squares settles which is right.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{prefix}-*.jpg"):
        old.unlink()
    per_sheet = per_row * rows_per_sheet
    square = TILE // 8
    for start in range(0, len(entries), per_sheet):
        tiles = []
        for entry in entries[start : start + per_sheet]:
            image = np.asarray(Image.open(DEST / entry["image"]).convert("RGB"))
            board = cv2.resize(rectify(image, np.array(entry["corners"], np.float32), 512), (TILE, TILE), interpolation=cv2.INTER_AREA).copy()
            template, model = unplace(entry["wikisource"]), unplace(entry["model"])
            lines = []
            for name in entry["differ"]:
                column, row = "abcdefgh".index(name[0]), 8 - int(name[1])
                cv2.rectangle(board, (column * square + 1, row * square + 1), ((column + 1) * square - 2, (row + 1) * square - 2), (230, 0, 0), 2)
                lines.append(f"{name}:{template[row][column]}/{model[row][column]}")
            tile = np.full((TILE + 3 * MARGIN + 18, TILE + MARGIN, 3), 255, np.uint8)
            tile[MARGIN : MARGIN + TILE, MARGIN:] = board
            for index in range(8):
                cv2.putText(tile, "87654321"[index], (6, MARGIN + index * square + square // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(tile, "abcdefgh"[index], (MARGIN + index * square + square // 2 - 5, MARGIN + TILE + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, f"{start + len(tiles)} {entry['id'][-28:]}", (MARGIN, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1, cv2.LINE_AA)
            text = " ".join(lines)
            for part in range(2):
                cv2.putText(tile, text[part * 52 : (part + 1) * 52], (4, MARGIN + TILE + 38 + part * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % per_row:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.concatenate([np.concatenate(tiles[i : i + per_row], axis=1) for i in range(0, len(tiles), per_row)], axis=0)
        Image.fromarray(sheet).save(folder / f"{prefix}-{start // per_sheet:03d}.jpg", quality=88)
    print(f"{len(entries)} boards drawn on sheets in {folder}")


def side_from_position(rows: list[str]) -> bool | None:
    """White at the bottom, as the pieces show it, or None when they do not show it clearly.

    White's pawns standing lower on the board than Black's is one vote, and
    White's king standing lower than Black's is another. The pawn vote is
    required (kings alone wander too much in endings) and the king vote, when
    there is one, must agree with it.
    """
    def mean_row(symbol: str) -> float | None:
        found = [index for index, row in enumerate(rows) for char in row if char == symbol]
        return sum(found) / len(found) if found else None

    votes = []
    for white, black in (("P", "p"), ("K", "k")):
        low, high = mean_row(white), mean_row(black)
        if low is not None and high is not None and low != high:
            votes.append(low > high)
    pawns_counted = mean_row("P") is not None and mean_row("p") is not None and mean_row("P") != mean_row("p")
    return votes[0] if pawns_counted and all(vote == votes[0] for vote in votes) else None


def build(args: argparse.Namespace) -> None:
    readings = json.loads((DEST / "readings.json").read_text())
    path = DEST / "decisions.json"
    decisions = json.loads(path.read_text()) if path.exists() else {}
    manifest, counts = [], {}
    for entry in readings:
        decision = decisions.get(entry["id"], {})
        out = {key: entry[key] for key in ("id", "book", "file", "page_title", "image", "image_size", "index_on_page", "diagrams_on_page")}
        out["image"] = "data/external/wikisource/" + entry["image"]
        out["corners"] = entry.get("corners")
        out["wikisource"] = entry["wikisource"]
        out["model"] = entry.get("model")
        if "exclude" in decision:
            out.update(status="excluded", reason=decision["exclude"])
        elif not entry["matched"]:
            out.update(status="excluded", reason=f"page has {entry['diagrams_on_page']} templates but {entry['outlines_on_page']} outlines were found")
        elif entry["markers"]:
            out.update(status="excluded", reason="diagram carries arrows, crosses or labels: " + " ".join(entry["markers"]))
        elif decision.get("right") in ("model", "wikisource", "neither"):
            # A verdict given by eye also settles boards whose template is malformed.
            right = decision["right"]
            squares = entry["model"] if right == "model" else entry["wikisource"] if right == "wikisource" else placement(unplace(decision["rows"]))
            out.update(status="checked", right=right, squares=squares)
        elif entry["template_problems"]:
            out.update(status="excluded", reason="template: " + "; ".join(entry["template_problems"]))
        elif not entry["differ"]:
            out.update(status="agrees", squares=entry["wikisource"])
        else:
            out.update(status="excluded", reason="model and transcription differ and no verdict was given")
        wikitext = (DEST / "wikitext" / (Path(entry["image"]).stem + ".txt")).read_text(encoding="utf-8")
        side, source = decision.get("white_at_bottom"), "decision"
        if side is None:
            side, source = caption_side(wikitext, parse(wikitext)[entry["index_on_page"]].offset), "caption"
        if side is None and out.get("squares"):
            side, source = side_from_position(unplace(out["squares"])), "position"
        out["white_at_bottom"] = side
        out["white_at_bottom_from"] = source if side is not None else None
        if entry.get("model_white_at_bottom") is not None:
            out["model_white_at_bottom"] = entry["model_white_at_bottom"]
        if decision.get("note"):
            out["note"] = decision["note"]
        if out["status"] != "excluded":
            out["model_exact"] = out["model"] == out["squares"]
        counts[out["status"]] = counts.get(out["status"], 0) + 1
        manifest.append(out)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["read", "build"])
    args = parser.parse_args()
    {"read": read, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
