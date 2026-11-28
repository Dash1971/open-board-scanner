"""Checks for reading positions from the names of the "Chess Positions" images."""

import pytest

from boardgen import CLASSES
from train.import_chess_positions import is_test, squares_from_name


def test_squares_from_name_reads_rows_top_first():
    squares = squares_from_name("train/1B1B4-8-8-8-8-8-pP6-k6K.jpeg")
    assert len(squares) == 64
    symbols = [CLASSES[index] for index in squares]
    assert symbols[:8] == ["", "B", "", "B", "", "", "", ""]
    assert symbols[48:56] == ["p", "P", "", "", "", "", "", ""]
    assert symbols[56] == "k" and symbols[63] == "K"
    assert sum(1 for symbol in symbols if symbol) == 6


@pytest.mark.parametrize("name", ["8-8-8-8-8-8-8.jpeg", "9-8-8-8-8-8-8-8.jpeg", "7x-8-8-8-8-8-8-8.jpeg", "7-8-8-8-8-8-8-8.jpeg"])
def test_squares_from_name_rejects_bad_names(name):
    with pytest.raises(ValueError):
        squares_from_name(name)


def test_split_depends_only_on_the_position():
    name = "1B1B4-8-8-8-8-8-pP6-k6K"
    assert is_test(f"train/{name}.jpeg") == is_test(f"test/{name}.jpg")
