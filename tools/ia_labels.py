#!/usr/bin/env python3
"""Labels for the diagrams collected from Internet Archive scans.

tools/fetch_ia_diagrams.py finds the diagrams but cannot say what they show.
The positions were read by a model, then every board was checked by eye on
contact sheets (label/review.py) and every corrected board was checked a
second time; boards that could not be read with confidence, or on which the
two checks disagreed, are marked unusable. The outcome is kept in the
repository as assets/archive_diagrams.json, so the labelled set can be
rebuilt from the scans:

    python tools/fetch_ia_diagrams.py fetch     # the page images
    python tools/ia_labels.py restore           # data/external/ia/labels.json

    python tools/ia_labels.py start             # unlabelled records for new diagrams
    python tools/ia_labels.py export            # labels.json -> assets/archive_diagrams.json

labels.json is in the labelling tool's format (label/store.py) and is what
training and scoring read. `start` and `restore` keep records already there.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "external" / "ia"
PUBLISHED = ROOT / "assets" / "archive_diagrams.json"
EMPTY = "8/8/8/8/8/8/8/8"


def read_boards() -> dict:
    labels = FOLDER / "labels.json"
    return json.loads(labels.read_text())["boards"] if labels.exists() else {}


def write_boards(boards: dict, added: int) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "labels.json").write_text(json.dumps({"version": 1, "photos": FOLDER.name, "boards": boards}, indent=1) + "\n")
    print(f"{FOLDER / 'labels.json'}: {len(boards)} boards, {added} added")


def blank(photo: str, corners: list, source: str) -> dict:
    return {
        "photo": photo, "corners": corners, "quarter_turns": 0, "seen": EMPTY, "white_at_bottom": True,
        "source": source[:80], "status": "todo", "found": True, "read": {"model": "none", "corners": corners},
    }  # fmt: skip


def start() -> None:
    boards, on_page, added = read_boards(), Counter(), 0
    for entry in json.loads((FOLDER / "manifest.json").read_text()):
        photo = entry["image"]
        on_page[photo] += 1
        key = photo if on_page[photo] == 1 else f"{photo}#{on_page[photo]}"
        if key not in boards:
            corners = [[round(float(x), 1), round(float(y), 1)] for x, y in entry["corners"]]
            boards[key] = blank(photo, corners, f"{entry['title'].rstrip(';')} ({entry['year']})")
            added += 1
    write_boards(boards, added)


def export() -> None:
    rows = [
        {"id": key, "source": record["source"], "corners": record["corners"], "status": record["status"]}
        | ({"seen": record["seen"], "white_at_bottom": record["white_at_bottom"], "quarter_turns": record["quarter_turns"]} if record["status"] == "done" else {})
        for key, record in read_boards().items()
        if record["status"] in ("done", "skip")
    ]
    PUBLISHED.write_text("[\n" + ",\n".join(" " + json.dumps(row, ensure_ascii=False) for row in rows) + "\n]\n")
    print(f"{PUBLISHED.relative_to(ROOT)}: {sum(row['status'] == 'done' for row in rows)} labelled, {sum(row['status'] == 'skip' for row in rows)} unusable")


def restore() -> None:
    boards, added = read_boards(), 0
    for row in json.loads(PUBLISHED.read_text()):
        if row["id"] in boards and boards[row["id"]]["status"] != "todo":
            continue
        record = blank(row["id"].partition("#")[0], row["corners"], row["source"])
        record["status"] = row["status"]
        if row["status"] == "done":
            record.update(seen=row["seen"], white_at_bottom=row["white_at_bottom"], quarter_turns=row["quarter_turns"], method="sheets")
        boards[row["id"]] = record
        added += 1
    write_boards(boards, added)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("start", "export", "restore"))
    {"start": start, "export": export, "restore": restore}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
