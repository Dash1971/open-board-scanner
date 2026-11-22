"""Checks for the labelling tool: label conventions, the board finder, the store and the server."""

import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from label.find import default_outline, find_boards
from label.server import make_server
from label.store import Reader, Store, crop_as_it_lies, grid_to_rows, position, rows_to_grid

QUAD = np.array([[210, 120], [590, 140], [570, 500], [200, 480]], np.float32)


def page_with_board() -> np.ndarray:
    """A plain page with one framed, slightly skewed chequered board at QUAD."""
    board = np.full((400, 400, 3), 255, np.uint8)
    for row in range(8):
        for column in range(8):
            if (row + column) % 2:
                board[row * 50 : (row + 1) * 50, column * 50 : (column + 1) * 50] = 120
    cv2.rectangle(board, (0, 0), (399, 399), (0, 0, 0), 5)
    flat = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32)
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(flat, QUAD), (800, 600), borderValue=(245, 245, 245))


@pytest.fixture
def store(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "page.png"), cv2.cvtColor(page_with_board(), cv2.COLOR_RGB2BGR))
    store = Store(photos, tmp_path / "labels" / "photos.json", Reader(None))
    store.prepare()
    return store


def test_rows_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        grid = rng.integers(0, 13, (8, 8)) * (rng.random((8, 8)) < 0.4)
        assert (rows_to_grid(grid_to_rows(grid)) == grid).all()
    assert grid_to_rows(np.zeros((8, 8), int)) == "8/8/8/8/8/8/8/8"


@pytest.mark.parametrize("rows", ["8/8/8", "9/8/8/8/8/8/8/8", "7/8/8/8/8/8/8/8", "8/8/8/8/8/8/8/7x", "pppppppp1/8/8/8/8/8/8/8", "08/8/8/8/8/8/8/8"])
def test_bad_rows_are_rejected(rows):
    with pytest.raises(ValueError):
        rows_to_grid(rows)


def test_position_follows_the_viewing_side():
    seen = "k7/8/8/8/8/8/8/7R"
    assert position({"seen": seen, "white_at_bottom": True}) == seen
    assert position({"seen": seen, "white_at_bottom": False}) == "R7/8/8/8/8/8/8/7k"


def test_labels_turn_back_to_the_crop_as_it_lies():
    # A crop that needs one anticlockwise quarter turn shows the upright top-left square at its top right.
    record = {"seen": "K7/8/8/8/8/8/8/8", "quarter_turns": 1}
    lying = crop_as_it_lies(record)
    assert lying[0, 7] != 0 and (lying != 0).sum() == 1
    assert (np.rot90(lying, record["quarter_turns"]) == rows_to_grid(record["seen"])).all()


def test_finder_locates_a_framed_board():
    image = page_with_board()
    outlines = find_boards(image)
    assert outlines, "no outline found"
    assert np.abs(outlines[0] - QUAD).max() < 6
    assert find_boards(np.full((600, 800, 3), 240, np.uint8)) == []
    assert default_outline(image).shape == (4, 2)


def test_store_prepares_updates_and_reloads(store):
    assert store.ids() == ["page.png"]
    record = store.boards["page.png"]
    assert record["found"] and record["status"] == "todo" and record["seen"] == "8/8/8/8/8/8/8/8"
    store.update("page.png", record["corners"], 3, "8/8/8/8/8/8/8/K6k", False, "  my book ", "done")
    extra = store.add_board("page.png")
    assert extra == "page.png#2" and not store.boards[extra]["found"]
    reloaded = Store(store.photos, store.labels, Reader(None))
    assert reloaded.ids() == ["page.png", "page.png#2"]
    saved = reloaded.boards["page.png"]
    assert (saved["quarter_turns"], saved["seen"], saved["white_at_bottom"], saved["source"], saved["status"]) == (3, "8/8/8/8/8/8/8/K6k", False, "my book", "done")
    assert reloaded.prepare() == 0  # nothing new, and labelled boards are left alone


def test_store_rejects_bad_labels(store):
    corners = store.boards["page.png"]["corners"]
    for arguments in (
        (corners[:3], 0, "8/8/8/8/8/8/8/8", True, "", "done"),
        (corners, 4, "8/8/8/8/8/8/8/8", True, "", "done"),
        (corners, 0, "8/8/8/8/8/8/8", True, "", "done"),
        (corners, 0, "8/8/8/8/8/8/8/8", True, "", "finished"),
    ):
        with pytest.raises(ValueError):
            store.update("page.png", *arguments)
    assert store.boards["page.png"]["status"] == "todo"


def test_server_answers_only_its_own_page(store):
    server = make_server(store, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    try:
        status, body = call("/api/boards")
        assert status == 200 and json.loads(body) == [{"id": "page.png", "status": "todo", "source": ""}]
        assert call("/")[0] == 200 and call("/photo?id=page.png")[0] == 200
        assert call("/crop?id=page.png&corners=1,1,90,1,90,90,1,90&turns=1")[0] == 200
        assert call("/photo?id=../secret")[0] == 400
        assert call("/api/boards", headers={"Host": "evil.example"})[0] == 400
        save = {"id": "page.png", "corners": store.boards["page.png"]["corners"], "quarter_turns": 0, "seen": "8/8/8/8/8/8/8/8", "white_at_bottom": True, "status": "skip"}
        assert call("/api/save", save)[0] == 400  # no X-Label header, as from another site
        assert store.boards["page.png"]["status"] == "todo"
        assert call("/api/save", save, {"X-Label": "1"})[0] == 200
        assert store.boards["page.png"]["status"] == "skip"
    finally:
        server.shutdown()
        server.server_close()
