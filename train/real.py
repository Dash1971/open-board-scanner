"""Labelled real photos as training and test data.

Labels come from the labelling tool (see label/store.py). Each board keeps its
whole photo, so that every training crop can be cut afresh with the outline
moved a little, as an imperfect localizer would.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from boardgen import rectify
from label.store import crop_as_it_lies

from .model import BoardNet


@dataclass
class RealBoard:
    board_id: str
    source: str
    image: np.ndarray  # the whole photo, RGB
    corners: np.ndarray  # 4 x 2, as the board lies in the photo
    squares: np.ndarray  # 8 x 8 class numbers, as the board lies in the photo
    quarter_turns: int
    white_at_bottom: bool

    def crop(self, rng: np.random.Generator | None = None, size: int = 256):
        """One crop with its labels: (image, 64 squares, white_at_bottom, quarter_turns).

        With an rng the outline is jittered and the crop is given a random
        extra quarter turn, with the labels turned to match.
        """
        if rng is None:
            return rectify(self.image, self.corners, size), self.squares.reshape(64), self.white_at_bottom, self.quarter_turns
        extra = int(rng.integers(4))
        image = np.rot90(rectify(self.image, self.corners, size, rng, jitter=0.025, max_margin=0.06), extra)
        squares = np.rot90(self.squares, extra)
        return np.ascontiguousarray(image), squares.reshape(64), self.white_at_bottom, (self.quarter_turns - extra) % 4


def load_labelled(labels: str | Path, photos: str | Path) -> list[RealBoard]:
    """Every board marked as labelled, in the order of the label file."""
    boards = []
    cache: dict[str, np.ndarray] = {}
    for board_id, record in json.loads(Path(labels).read_text())["boards"].items():
        if record["status"] != "done":
            continue
        photo = record["photo"]
        if photo not in cache:
            loaded = cv2.imread(str(Path(photos) / photo), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"{photo} is listed in {labels} but cannot be read from {photos}")
            cache[photo] = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        boards.append(
            RealBoard(
                board_id=board_id,
                source=record["source"],
                image=cache[photo],
                corners=np.asarray(record["corners"], np.float32),
                squares=np.ascontiguousarray(crop_as_it_lies(record)),
                quarter_turns=int(record["quarter_turns"]),
                white_at_bottom=bool(record["white_at_bottom"]),
            )
        )
    return boards


def turn_crops(images: np.ndarray, squares: np.ndarray, quarter_turns: np.ndarray, rng: np.random.Generator):
    """Give every crop a random extra quarter turn, with its labels turned to match.

    Without this a model only meets sideways boards in the kinds of scene
    that are photographed sideways, and reading a screenshot at all four
    turns, as the scanner does, would show it artwork it has never seen
    turned.
    """
    images, squares, quarter_turns = images.copy(), squares.copy(), quarter_turns.astype(np.int64).copy()
    extras = rng.integers(4, size=len(images))
    for index, extra in enumerate(extras):
        if extra:
            images[index] = np.rot90(images[index], extra)
            squares[index] = np.rot90(squares[index].reshape(8, 8), extra).reshape(64)
    return images, squares, (quarter_turns - extras) % 4


def load_labelled_sets(labels: list[str], photos: list[str]) -> list[RealBoard]:
    """Labelled boards from several label files, each paired with its photo folder."""
    if len(labels) != len(photos):
        raise SystemExit("give one --photos folder for every --labels file")
    return [board for pair in zip(labels, photos) for board in load_labelled(*pair)]


def real_batch(boards: list[RealBoard], count: int, rng: np.random.Generator):
    """`count` jittered crops, with each source drawn in proportion to the square root of its size.

    Drawing by board would let a large book drown a small one; drawing evenly
    by source would show a source with one board as often as one with a
    hundred. The square root sits between the two.
    """
    sources = sorted({board.source for board in boards})
    by_source = {source: [board for board in boards if board.source == source] for source in sources}
    weights = np.sqrt([len(by_source[source]) for source in sources])
    crops = []
    for choice in rng.choice(len(sources), size=count, p=weights / weights.sum()):
        pool = by_source[sources[int(choice)]]
        crops.append(pool[int(rng.integers(len(pool)))].crop(rng))
    images, squares, white, turns = zip(*crops)
    return np.stack(images), np.stack(squares).astype(np.int64), np.array(white, np.float32), np.array(turns, np.int64)


@torch.no_grad()
def read_boards(model: BoardNet, boards: list[RealBoard], device: torch.device, turns: tuple[int, ...] = (0,)):
    """Read real boards cut at their labelled outlines.

    Returns the predicted classes (N x 64, as each board lies) and quarter
    turns (N). With several entries in `turns`, the crop is read once at each
    of those extra quarter turns and the readings are averaged.
    """
    model.eval()
    images = torch.from_numpy(np.stack([board.crop()[0] for board in boards])).to(device).permute(0, 3, 1, 2).float()
    squares, rotation = 0.0, 0.0
    for extra in turns:
        logits, _, turn_logits = model(torch.rot90(images, extra, dims=(2, 3)))
        squares = squares + torch.rot90(torch.softmax(logits, dim=1), -extra, dims=(2, 3))
        rotation = rotation + torch.roll(torch.softmax(turn_logits, dim=1), extra, dims=1)
    return squares.argmax(dim=1).flatten(1).cpu().numpy(), rotation.argmax(dim=1).cpu().numpy()


def _real_summary(boards: list[RealBoard], predicted: np.ndarray, turns: np.ndarray) -> dict:
    truth = np.stack([board.squares.reshape(64) for board in boards])
    wrong = (predicted != truth).sum(axis=1)
    return {
        "boards": len(boards),
        "exact_boards": int((wrong == 0).sum()),
        "boards_with_at_most_1_error": int((wrong <= 1).sum()),
        "wrong_squares": int(wrong.sum()),
        "square_accuracy": round(float((predicted == truth).mean()), 5),
        "quarter_turns_right": int((turns == np.array([board.quarter_turns for board in boards])).sum()),
    }


def score_real(model: BoardNet, boards: list[RealBoard], device: torch.device, turns: tuple[int, ...] = (0,)) -> dict:
    """Score a model on real boards, overall and for each source."""
    if not boards:
        return {"boards": 0}
    predicted, read_turns = read_boards(model, boards, device, turns)
    report = _real_summary(boards, predicted, read_turns)
    truth = np.stack([board.squares.reshape(64) for board in boards])
    report["wrong_by_board"] = {board.board_id: int(count) for board, count in zip(boards, (predicted != truth).sum(axis=1)) if count}
    report["by_source"] = {}
    for source in sorted({board.source for board in boards}):
        chosen = [index for index, board in enumerate(boards) if board.source == source]
        report["by_source"][source] = _real_summary([boards[i] for i in chosen], predicted[chosen], read_turns[chosen])
    return report
