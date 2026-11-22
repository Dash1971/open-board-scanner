"""Checks for the contact-sheet review of proposed readings."""

import argparse
import json

import cv2
import numpy as np
import pytest

from label.review import TILE, apply, draw_tile, oddities, problems, square_index
from label.store import Reader, Store, rows_to_grid

from test_label import QUAD, page_with_board

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_squares_are_named_as_they_appear_on_the_sheet():
    assert square_index("a8") == (0, 0) and square_index("h1") == (7, 7) and square_index("e4") == (4, 4)
    for bad in ("i1", "a9", "a", "e44", ""):
        with pytest.raises(ValueError):
            square_index(bad)


def test_impossible_and_odd_material_is_reported():
    assert problems(rows_to_grid(START)) == [] and oddities(rows_to_grid(START)) == []
    assert problems(rows_to_grid("8/8/8/8/8/8/8/8")) == ["0K", "0k"]
    assert "pawn on an end row" in problems(rows_to_grid("k6P/8/8/8/8/8/8/K7"))
    assert problems(rows_to_grid("k7/8/8/8/8/8/8/KK6")) == ["2K"]
    assert oddities(rows_to_grid("k7/8/8/8/8/8/8/KQQ5")) == ["2Q"]
    assert oddities(rows_to_grid("k7/8/8/8/8/8/8/KB1B4")) == ["both B on one colour"]


def test_tile_shows_the_board_with_room_for_coordinates():
    proposal = {"quarter_turns": 1, "seen": START, "suspect": np.eye(8, dtype=int).tolist()}
    tile = draw_tile(page_with_board(), QUAD, proposal, "#0")
    assert tile.shape[0] > TILE and tile.shape[1] > TILE and tile.dtype == np.uint8
    assert (tile[:, :, 0].astype(int) - tile[:, :, 2] > 150).any()  # yellow tags and red frames were drawn


def test_checked_readings_become_labels(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        cv2.imwrite(str(photos / name), cv2.cvtColor(page_with_board(), cv2.COLOR_RGB2BGR))
    labels = tmp_path / "labels.json"
    Store(photos, labels, Reader(None)).prepare()
    proposal = {"quarter_turns": 2, "seen": "8/8/8/8/8/8/8/K6k", "suspect": np.zeros((8, 8), int).tolist(), "white_at_bottom": True}
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps({str(number): {"id": name, **proposal} for number, name in enumerate(("a.png", "b.png", "c.png"))}))
    corrections = tmp_path / "checked.json"
    corrections.write_text(json.dumps({"ok": [0], "fix": {"1": {"a8": "q", "a1": ""}}, "black_at_bottom": [1], "source": {"a book": [0, 1]}}))
    apply(argparse.Namespace(photos=photos, labels=labels, proposals=proposals, corrections=corrections))
    boards = json.loads(labels.read_text())["boards"]
    assert (boards["a.png"]["status"], boards["a.png"]["seen"], boards["a.png"]["quarter_turns"], boards["a.png"]["method"]) == ("done", "8/8/8/8/8/8/8/K6k", 2, "sheets")
    assert (boards["b.png"]["seen"], boards["b.png"]["white_at_bottom"], boards["b.png"]["source"]) == ("q7/8/8/8/8/8/8/7k", False, "a book")
    assert boards["c.png"]["status"] == "todo"  # not mentioned, so left alone

    corrections.write_text(json.dumps({"ok": [7]}))
    with pytest.raises(SystemExit):
        apply(argparse.Namespace(photos=photos, labels=labels, proposals=proposals, corrections=corrections))


def test_proposals_combine_turned_readings():
    torch = pytest.importorskip("torch")
    from label.review import propose
    from train.model import BoardNet

    torch.manual_seed(0)
    proposal = propose([BoardNet().eval()], torch.device("cpu"), page_with_board(), QUAD, np.random.default_rng(0), jitters=1)
    assert rows_to_grid(proposal["seen"]).shape == (8, 8) and proposal["quarter_turns"] in (0, 1, 2, 3)
    assert np.asarray(proposal["suspect"]).shape == (8, 8) and isinstance(proposal["white_at_bottom"], bool)
