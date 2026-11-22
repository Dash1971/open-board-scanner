"""Shape and size checks for the classifier; skipped when PyTorch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from train.model import BoardNet, parameter_count  # noqa: E402


def test_outputs_have_the_documented_shapes():
    squares, white_at_bottom, rotation = BoardNet().eval()(torch.rand(2, 3, 256, 256) * 255)
    assert squares.shape == (2, 13, 8, 8)
    assert white_at_bottom.shape == (2,)
    assert rotation.shape == (2, 4)


def test_stays_within_the_size_budget():
    # One byte per parameter as INT8: the whole model must fit well inside 10 MB.
    assert parameter_count(BoardNet()) < 3_000_000


def test_positions_are_recovered_from_turned_and_flipped_grids():
    np = pytest.importorskip("numpy")
    from train.evaluate import to_positions

    position = np.arange(64)
    as_seen = np.stack([
        position,                                   # upright, White at the bottom
        position[::-1],                             # upright, Black at the bottom
        np.rot90(position.reshape(8, 8), 1).ravel(),  # needs three more quarter turns
    ])
    restored = to_positions(as_seen, np.array([1, 0, 1]), np.array([0, 0, 3]))
    assert (restored == position).all()
