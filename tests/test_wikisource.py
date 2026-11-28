"""Checks for reading {{Chess diagram}} templates from Wikisource pages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from chess_diagram_template import parse  # noqa: E402

EMPTY = "|  |  |  |  |  |  |  |  \n"


def rows(diagram) -> list[str]:
    return ["".join(row) for row in diagram.rows]


def test_positional_squares_are_read_in_printed_order():
    text = "Before {{Chess diagram|tright|\n|kd|  |  |  |  |  |  |rl\n" + EMPTY * 6 + "|  |  |  |  |kl|  |  |pd\n|A caption}} after"
    (diagram,) = parse(text)
    assert rows(diagram)[0] == "k......R" and rows(diagram)[7] == "....K..p"
    assert rows(diagram)[1:7] == ["........"] * 6


def test_fen_form_and_several_diagrams_on_a_page():
    text = "{{Chess diagram|fen=8/8/8/3k4/8/8/8/R3K3}} text {{chess diagram|fen=k7/8/8/8/8/8/8/7K}}"
    first, second = parse(text)
    assert rows(first)[3] == "...k...." and rows(first)[7] == "R...K..."
    assert rows(second)[0] == "k......." and rows(second)[7] == ".......K"


def test_pages_without_a_diagram_give_nothing():
    assert parse("No diagram here, only {{other template|kd|rl}}.") == []
