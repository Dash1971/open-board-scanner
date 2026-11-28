"""The published Archive labels and the label file rebuilt from them."""

import importlib.util
import json
from pathlib import Path

from label.store import rows_to_grid

ROOT = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location("ia_labels", ROOT / "tools" / "ia_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_labels_are_well_formed():
    rows = json.loads((ROOT / "assets" / "archive_diagrams.json").read_text())
    assert len({row["id"] for row in rows}) == len(rows) > 800
    for row in rows:
        assert row["status"] in ("done", "skip") and len(row["corners"]) == 4
        if row["status"] == "done":
            assert rows_to_grid(row["seen"]).shape == (8, 8)


def test_export_then_restore_gives_the_same_labels(tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "FOLDER", tmp_path / "ia")
    monkeypatch.setattr(tool, "PUBLISHED", tmp_path / "published.json")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    corners = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]
    done = tool.blank("pages/book/0001.jpg", corners, "A book (1900)") | {"status": "done", "seen": "8/8/8/3k4/8/8/4K3/8", "method": "sheets"}
    skipped = tool.blank("pages/book/0001.jpg", corners, "A book (1900)") | {"status": "skip"}
    todo = tool.blank("pages/book/0002.jpg", corners, "A book (1900)")
    tool.write_boards({"pages/book/0001.jpg": done, "pages/book/0001.jpg#2": skipped, "pages/book/0002.jpg": todo}, 3)
    tool.export()
    (tmp_path / "ia" / "labels.json").unlink()
    tool.restore()
    boards = tool.read_boards()
    assert set(boards) == {"pages/book/0001.jpg", "pages/book/0001.jpg#2"}  # unlabelled boards are not published
    assert boards["pages/book/0001.jpg"]["seen"] == "8/8/8/3k4/8/8/4K3/8" and boards["pages/book/0001.jpg"]["status"] == "done"
    assert boards["pages/book/0001.jpg#2"]["status"] == "skip" and boards["pages/book/0001.jpg#2"]["photo"] == "pages/book/0001.jpg"
