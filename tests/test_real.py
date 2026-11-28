"""Checks for training on labelled real photos; skipped when PyTorch is not installed."""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from label.store import rows_to_grid  # noqa: E402
from train.finetune import split  # noqa: E402
from train.real import RealBoard, load_labelled, real_batch  # noqa: E402


def board(source="a", quarter_turns=1, seen="K7/8/8/8/8/8/8/7p", name="one"):
    image = np.zeros((200, 200, 3), np.uint8)
    image[20:40, 20:40] = 255  # a mark in the photo's top-left square of the board
    corners = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float32)
    squares = np.rot90(rows_to_grid(seen), -quarter_turns)
    return RealBoard(name, source, image, corners, np.ascontiguousarray(squares), quarter_turns, True)


def test_plain_crop_keeps_labels_as_the_board_lies():
    image, squares, white, turns = board().crop()
    assert image.shape == (256, 256, 3) and squares.shape == (64,) and white is True and turns == 1
    assert image[:30, :30].mean() > 200 and image[40:, 40:].mean() < 5


def test_turned_crops_keep_image_and_labels_together():
    example = board()
    upright = rows_to_grid("K7/8/8/8/8/8/8/7p")
    top_left = np.zeros((8, 8), int)
    top_left[0, 0] = 1
    rng = np.random.default_rng(0)
    seen_turns = set()
    for _ in range(40):
        image, squares, _, turns = example.crop(rng)
        seen_turns.add(turns)
        # Bringing the crop upright must always give the labelled board.
        assert (np.rot90(squares.reshape(8, 8), turns) == upright).all()
        # The bright mark must sit on the same square in the image as in the plain crop's labels.
        row, column = np.argwhere(np.rot90(top_left, (example.quarter_turns - turns) % 4))[0]
        assert image[row * 32 + 8 : row * 32 + 24, column * 32 + 8 : column * 32 + 24].mean() > 150
    assert seen_turns == {0, 1, 2, 3}


def test_batches_favour_small_sources_without_drowning_large_ones():
    boards = [board("big", name=str(i)) for i in range(9)] + [board("small", seen="8/8/8/8/8/8/8/8", name="s")]
    images, squares, white, turns = real_batch(boards, 400, np.random.default_rng(1))
    assert images.shape == (400, 256, 256, 3) and squares.shape == (400, 64)
    from_small = int(((squares != 0).sum(axis=1) == 0).sum())
    assert 60 < from_small < 140  # one board in ten, drawn a quarter of the time


def test_split_by_source_and_by_fold():
    boards = [board("a", name=f"a{i}") for i in range(6)] + [board("b", name=f"b{i}") for i in range(4)]
    train, test = split(boards, ["b"], None, 0)
    assert {b.source for b in train} == {"a"} and {b.source for b in test} == {"b"}
    held = []
    for index in range(5):
        train, test = split(boards, [], (index, 5), 0)
        assert len(train) + len(test) == 10 and not {b.board_id for b in train} & {b.board_id for b in test}
        held += [b.board_id for b in test]
    assert sorted(held) == sorted(b.board_id for b in boards)  # every board is held out exactly once
    assert split(boards, [], None, 0) == (boards, [])
    with pytest.raises(SystemExit):
        split(boards, ["missing"], None, 0)


def test_only_labelled_boards_are_loaded(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "p.png"), np.zeros((60, 80, 3), np.uint8))
    record = {"photo": "p.png", "corners": [[1, 1], [50, 1], [50, 50], [1, 50]], "quarter_turns": 3, "seen": "8/8/8/8/8/8/8/Q7", "white_at_bottom": False, "source": "x"}
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"boards": {"p.png": {**record, "status": "done"}, "p.png#2": {**record, "status": "todo"}, "p.png#3": {**record, "status": "skip"}}}))
    loaded = load_labelled(labels, photos)
    assert [b.board_id for b in loaded] == ["p.png"]
    assert loaded[0].quarter_turns == 3 and loaded[0].white_at_bottom is False and loaded[0].image.shape == (60, 80, 3)
    assert (np.rot90(loaded[0].squares, 3) == rows_to_grid("8/8/8/8/8/8/8/Q7")).all()


def test_synthetic_crops_are_turned_with_their_labels():
    from train.real import turn_crops

    rng = np.random.default_rng(3)
    images = rng.integers(0, 255, (12, 256, 256, 3), dtype=np.uint8)
    squares = rng.integers(0, 13, (12, 64))
    turns = rng.integers(0, 4, 12)
    new_images, new_squares, new_turns = turn_crops(images, squares, turns, np.random.default_rng(4))
    changed = 0
    for index in range(12):
        extra = (turns[index] - new_turns[index]) % 4
        changed += extra != 0
        assert np.array_equal(new_images[index], np.rot90(images[index], extra))
        # Bringing the crop upright gives the same board before and after the extra turn.
        before = np.rot90(squares[index].reshape(8, 8), turns[index])
        after = np.rot90(new_squares[index].reshape(8, 8), new_turns[index])
        assert np.array_equal(before, after)
    assert changed > 4


def test_sources_and_a_fold_can_be_held_out_together():
    first = [board("a", name=f"a{i}") for i in range(10)]
    later = [board("c", name=f"c{i}") for i in range(4)] + [board("d", name=f"d{i}") for i in range(3)]
    earlier_test = {b.board_id for b in split(first, [], (0, 5), 0)[1]}
    train, test = split(first + later, ["d"], (0, 5), 0, fold_pool=len(first))
    held = {b.board_id for b in test}
    # The same boards of the first set as before, the whole of source d, and nothing of source c.
    assert held == earlier_test | {"d0", "d1", "d2"}
    assert len(train) + len(test) == 17 and not held & {b.board_id for b in train}


def test_keeping_only_the_board_gives_the_same_crop(tmp_path):
    from train.real import around_board
    from boardgen import rectify

    rng = np.random.default_rng(3)
    page = cv2.resize(rng.integers(0, 255, (90, 70, 3), dtype=np.uint8), (700, 900), interpolation=cv2.INTER_CUBIC)
    corners = np.array([[210, 330], [520, 338], [514, 650], [204, 640]], np.float32)
    part, moved = around_board(page, corners)
    assert part.shape[0] < 500 and part.shape[1] < 500
    whole, kept = rectify(page, corners, 256).astype(int), rectify(part, moved, 256).astype(int)
    assert np.abs(whole - kept).mean() < 0.5  # the same pixels, up to rounding in the warp
    # A board at the very edge of the photo is kept whole.
    edge = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], np.float32)
    part, moved = around_board(page, edge)
    assert np.array_equal(moved, edge) and part.shape[:2] == (361, 361)


def test_a_large_board_is_shrunk_and_still_cuts_the_same():
    from train.real import around_board
    from boardgen import rectify

    rng = np.random.default_rng(4)
    page = cv2.resize(rng.integers(0, 255, (30, 24, 3), dtype=np.uint8), (1600, 2000), interpolation=cv2.INTER_CUBIC)
    corners = np.array([[300, 500], [1210, 520], [1195, 1430], [290, 1405]], np.float32)
    part, moved = around_board(page, corners, max_side=320)
    assert max(part.shape[:2]) < 480 and abs(np.linalg.norm(moved[1] - moved[0]) - 320) < 2
    whole, kept = rectify(page, corners, 256).astype(int), rectify(part, moved, 256).astype(int)
    assert np.abs(whole - kept).mean() < 2.0
