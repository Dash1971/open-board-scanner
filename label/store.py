"""Label records for a folder of photos, and the model reading that pre-fills them.

One record describes one board in one photo:

    photo           file name inside the photo folder
    corners         the board's outline in the photo: top-left, top-right,
                    bottom-right, bottom-left as the board lies in the photo
    quarter_turns   anticlockwise quarter turns that bring the crop upright
    seen            the upright board as seen, top row first, written like the
                    piece placement of a FEN. It is the position itself only
                    when White is at the bottom; see position()
    white_at_bottom whether White plays up the upright board
    source          free text naming the book or program, so that whole
                    sources can be kept out of training
    status          "todo", "done" or "skip"
    found           whether the outline came from the board finder
    method          how the label was made: "page" for squares corrected by hand
                    on the labelling page, "sheets" for a model reading checked
                    by eye on contact sheets (see label/review.py)
    read            what the model made of the crop, kept for scoring

Records live in one JSON file, rewritten whole on every change.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from boardgen import CLASSES, rectify
from boardgen.positions import to_fen

from .find import default_outline, find_boards

STATUSES = ("todo", "done", "skip")
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
MODEL_SIZE = 256


def grid_to_rows(grid: np.ndarray) -> str:
    """An 8 x 8 grid of class numbers, top row first, as FEN-style rows."""
    return to_fen([CLASSES[int(value)] for value in np.asarray(grid).reshape(64)])


def rows_to_grid(rows: str) -> np.ndarray:
    """The reverse of grid_to_rows. Raises ValueError for anything but 8 x 8 squares."""
    parts = rows.split("/")
    if len(parts) != 8:
        raise ValueError(f"expected 8 rows, got {len(parts)}")
    grid = np.zeros((8, 8), np.int64)
    for row, part in enumerate(parts):
        column = 0
        for character in part:
            if character.isdigit() and character != "0":
                column += int(character)
            elif character in CLASSES[1:]:
                if column > 7:
                    raise ValueError(f"row {part!r} is too long")
                grid[row, column] = CLASSES.index(character)
                column += 1
            else:
                raise ValueError(f"unexpected {character!r} in row {part!r}")
        if column != 8:
            raise ValueError(f"row {part!r} has {column} squares")
    return grid


def position(record: dict) -> str:
    """The piece placement of the position a record shows."""
    grid = rows_to_grid(record["seen"])
    return grid_to_rows(grid if record["white_at_bottom"] else grid[::-1, ::-1])


def crop_as_it_lies(record: dict) -> np.ndarray:
    """A record's labels turned back to how the board lies in the photo: 8 x 8 class numbers."""
    return np.rot90(rows_to_grid(record["seen"]), -record["quarter_turns"])


class Reader:
    """Runs the exported classifier on one crop. Without a model, reads every board as empty."""

    def __init__(self, model_path: Path | None) -> None:
        self.session = None
        self.tag = "none"
        if model_path is not None:
            import onnxruntime

            self.tag = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
            self.session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def read(self, image: np.ndarray, corners: np.ndarray) -> dict:
        if self.session is None:
            grid, confidence, turns, white = np.zeros((8, 8), np.int64), np.ones((8, 8)), 0, 1.0
        else:
            crop = rectify(image, np.asarray(corners, np.float32), MODEL_SIZE)
            batch = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)
            logits, orientation, rotation = self.session.run(None, {"image": batch})
            logits = logits[0] - logits[0].max(axis=0)
            probabilities = np.exp(logits) / np.exp(logits).sum(axis=0)
            turns = int(rotation[0].argmax())
            grid = np.rot90(probabilities.argmax(axis=0), turns)
            confidence = np.rot90(probabilities.max(axis=0), turns)
            white = float(1.0 / (1.0 + np.exp(-float(orientation[0]))))
        return {
            "model": self.tag,
            "corners": np.asarray(corners, np.float64).round(1).tolist(),
            "quarter_turns": turns,
            "seen": grid_to_rows(grid),
            "confidence": [round(float(value), 3) for value in confidence.reshape(64)],
            "white_at_bottom": round(white, 3),
        }


def _inside(point: np.ndarray, quad: np.ndarray) -> bool:
    return cv2.pointPolygonTest(np.asarray(quad, np.float32), (float(point[0]), float(point[1])), False) >= 0


class Store:
    def __init__(self, photos: Path, labels: Path, reader: Reader) -> None:
        self.photos = photos
        self.labels = labels
        self.reader = reader
        self.lock = threading.RLock()
        self._images: OrderedDict[str, np.ndarray] = OrderedDict()
        self.boards: dict[str, dict] = {}
        if labels.exists():
            self.boards = json.loads(labels.read_text())["boards"]

    def photo_names(self) -> list[str]:
        return sorted(path.name for path in self.photos.iterdir() if path.suffix.lower() in PHOTO_SUFFIXES)

    def image(self, photo: str) -> np.ndarray:
        """A photo as RGB. The last few are kept in memory."""
        with self.lock:
            if photo in self._images:
                self._images.move_to_end(photo)
                return self._images[photo]
        loaded = cv2.imread(str(self.photos / photo), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"{photo} is not a readable image")
        image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        with self.lock:
            self._images[photo] = image
            while len(self._images) > 8:
                self._images.popitem(last=False)
        return image

    def ids(self) -> list[str]:
        def key(board_id: str) -> tuple[str, int]:
            photo, _, number = board_id.partition("#")
            return photo, int(number or 1)

        return sorted(self.boards, key=key)

    def _fill(self, record: dict, image: np.ndarray) -> None:
        """Set a record's labels to the model's reading of its outline."""
        reading = self.reader.read(image, np.asarray(record["corners"], np.float32))
        record["read"] = reading
        record["quarter_turns"] = reading["quarter_turns"]
        record["seen"] = reading["seen"]
        record["white_at_bottom"] = reading["white_at_bottom"] >= 0.5

    def _new(self, photo: str, corners: np.ndarray, found: bool) -> dict:
        record = {"photo": photo, "corners": np.asarray(corners, np.float64).round(1).tolist(), "source": "", "status": "todo", "found": found}
        self._fill(record, self.image(photo))
        return record

    def prepare(self, progress=None) -> int:
        """Add a record for every new photo and refresh unlabelled readings made by another model.

        Returns the number of records added or refreshed.
        """
        changed = 0
        have = {record["photo"] for record in self.boards.values()}
        names = self.photo_names()
        for count, photo in enumerate(names, 1):
            if photo not in have:
                image = self.image(photo)
                outlines = find_boards(image)
                self.boards[photo] = self._new(photo, outlines[0] if outlines else default_outline(image), bool(outlines))
                changed += 1
            if progress and count % 50 == 0:
                progress(count, len(names))
        for record in self.boards.values():
            stale = record["status"] == "todo" and record["read"]["model"] != self.reader.tag
            if stale and (self.photos / record["photo"]).exists():
                self._fill(record, self.image(record["photo"]))
                changed += 1
        if changed:
            self.save()
        return changed

    def add_board(self, photo: str) -> str:
        """Add a record for one more board in a photo and return its id."""
        with self.lock:
            image = self.image(photo)
            taken = [np.asarray(record["corners"]) for record in self.boards.values() if record["photo"] == photo]
            fresh = [quad for quad in find_boards(image) if not any(_inside(quad.mean(axis=0), other) for other in taken)]
            number = 2
            while f"{photo}#{number}" in self.boards:
                number += 1
            board_id = f"{photo}#{number}"
            self.boards[board_id] = self._new(photo, fresh[0] if fresh else default_outline(image), bool(fresh))
            self.save()
            return board_id

    def read(self, board_id: str, corners: np.ndarray) -> dict:
        return self.reader.read(self.image(self.boards[board_id]["photo"]), corners)

    def update(self, board_id: str, corners, quarter_turns: int, seen: str, white_at_bottom: bool, source: str, status: str, method: str = "page") -> None:
        corners = np.asarray(corners, np.float64)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("corners must be four points")
        if quarter_turns not in (0, 1, 2, 3):
            raise ValueError("quarter_turns must be 0 to 3")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        seen = grid_to_rows(rows_to_grid(seen))
        with self.lock:
            record = self.boards[board_id]
            rounded = corners.round(1).tolist()
            if rounded != record["read"]["corners"]:
                record["read"] = self.read(board_id, corners)
            record.update(corners=rounded, quarter_turns=int(quarter_turns), seen=seen, white_at_bottom=bool(white_at_bottom), source=str(source).strip()[:80], status=status, method=method)
            self.save()

    def save(self) -> None:
        with self.lock:
            self.labels.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.labels.with_suffix(".tmp")
            temporary.write_text(json.dumps({"version": 1, "photos": self.photos.name, "boards": {key: self.boards[key] for key in self.ids()}}, indent=1) + "\n")
            os.replace(temporary, self.labels)
