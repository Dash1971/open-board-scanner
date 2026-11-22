"""Shape and size checks for the classifier; skipped when PyTorch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from train.model import BoardNet, parameter_count  # noqa: E402


def test_outputs_one_cell_per_square_and_one_orientation_logit():
    squares, white_at_bottom = BoardNet().eval()(torch.rand(2, 3, 256, 256) * 255)
    assert squares.shape == (2, 13, 8, 8)
    assert white_at_bottom.shape == (2,)


def test_stays_within_the_size_budget():
    # One byte per parameter as INT8: the whole model must fit well inside 10 MB.
    assert parameter_count(BoardNet()) < 3_000_000
