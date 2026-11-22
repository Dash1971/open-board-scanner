"""Score a classifier on a folder of labelled crops.

    python -m train.evaluate --checkpoint runs/boardnet-v0/model.pt --data data/crops/test
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from boardgen import CLASSES, KINDS

from .data import load_all
from .make_crops import HELD_OUT_SETS
from .model import BoardNet


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: BoardNet, images: np.ndarray, device: torch.device, batch_size: int = 256):
    """Return predicted classes (N x 64), White-at-bottom probabilities (N) and quarter turns (N)."""
    model.eval()
    squares, white, turns = [], [], []
    for offset in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[offset : offset + batch_size]).to(device).permute(0, 3, 1, 2).float()
        logits, orientation, rotation = model(batch)
        squares.append(logits.argmax(dim=1).flatten(1).cpu().numpy())
        white.append(torch.sigmoid(orientation).cpu().numpy())
        turns.append(rotation.argmax(dim=1).cpu().numpy())
    return np.concatenate(squares), np.concatenate(white), np.concatenate(turns)


def to_positions(squares: np.ndarray, white_at_bottom: np.ndarray, quarter_turns: np.ndarray) -> np.ndarray:
    """Turn grids as seen in the image (N x 64) into positions in FEN order."""
    grids = squares.reshape(-1, 8, 8)
    upright = np.empty_like(grids)
    for turns in range(4):
        chosen = quarter_turns == turns
        upright[chosen] = np.rot90(grids[chosen], turns, axes=(1, 2))
    flat = upright.reshape(-1, 64)
    return np.where(white_at_bottom.astype(bool)[:, None], flat, flat[:, ::-1])


def _summary(predicted: np.ndarray, truth: np.ndarray) -> dict:
    wrong = (predicted != truth).sum(axis=1)
    return {
        "boards": int(len(truth)),
        "exact_boards": round(float((wrong == 0).mean()), 4),
        "boards_with_at_most_1_error": round(float((wrong <= 1).mean()), 4),
        "square_accuracy": round(float((predicted == truth).mean()), 5),
        "wrong_squares_per_board": round(float(wrong.mean()), 3),
    }


def score(model: BoardNet, data: dict[str, np.ndarray], device: torch.device) -> dict:
    predicted, white, turns = predict(model, data["images"], device)
    truth = data["squares"]
    names = list(data["set_names"])
    held_out = np.isin(data["piece_set"], [names.index(name) for name in HELD_OUT_SETS])
    report = {"all": _summary(predicted, truth)}
    report["seen_piece_sets"] = _summary(predicted[~held_out], truth[~held_out])
    report["held_out_piece_sets"] = _summary(predicted[held_out], truth[held_out])
    for index, kind in enumerate(KINDS):
        mask = data["kind"] == index
        report[kind] = _summary(predicted[mask], truth[mask])
        report[f"{kind}_held_out"] = _summary(predicted[mask & held_out], truth[mask & held_out])
    # Rotation and viewing side are only knowable when the position is not empty.
    known = (truth != 0).any(axis=1)
    report["white_at_bottom_accuracy"] = round(float(((white > 0.5) == data["white_at_bottom"].astype(bool))[known].mean()), 4)
    report["quarter_turn_accuracy"] = round(float((turns == data["quarter_turns"])[known].mean()), 4)
    sideways = known & (data["quarter_turns"] != 0)
    report["quarter_turn_accuracy_on_turned_crops"] = round(float((turns == data["quarter_turns"])[sideways].mean()), 4) if sideways.any() else None
    # The end result: the position after undoing the predicted rotation and viewing side.
    true_positions = to_positions(truth, data["white_at_bottom"], data["quarter_turns"])
    read_positions = to_positions(predicted, white > 0.5, turns)
    report["exact_positions"] = round(float((true_positions == read_positions).all(axis=1).mean()), 4)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    np.add.at(confusion, (truth.ravel(), predicted.ravel()), 1)
    mistakes = [
        (int(confusion[t, p]), CLASSES[t] or "empty", CLASSES[p] or "empty")
        for t in range(len(CLASSES))
        for p in range(len(CLASSES))
        if t != p and confusion[t, p]
    ]
    report["commonest_mistakes"] = [f"{count}x {t} read as {p}" for count, t, p in sorted(mistakes, reverse=True)[:8]]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    device = pick_device()
    model = BoardNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(json.dumps(score(model, load_all(args.data), device), indent=2))


if __name__ == "__main__":
    main()
