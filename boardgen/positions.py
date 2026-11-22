"""Random piece placements.

A placement is a list of 64 strings in FEN order (a8 first, h1 last). Each
entry is a piece symbol (PNBRQK for White, pnbrqk for Black) or "" if empty.
"""

from __future__ import annotations

import chess
import numpy as np

CLASSES = ("", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
_SCATTER_SYMBOLS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
_SCATTER_WEIGHTS = np.array([8, 2, 2, 2, 1, 1] * 2, dtype=np.float64)
_SCATTER_WEIGHTS /= _SCATTER_WEIGHTS.sum()


def _from_board(board: chess.Board) -> list[str]:
    placement = []
    for rank in range(7, -1, -1):
        for file in range(8):
            found = board.piece_at(chess.square(file, rank))
            placement.append(found.symbol() if found else "")
    return placement


def _game(rng: np.random.Generator) -> list[str]:
    """A position reached by random legal moves, with captures favoured."""
    board = chess.Board()
    for _ in range(int(rng.integers(0, 100))):
        moves = list(board.legal_moves)
        if not moves:
            break
        captures = [move for move in moves if board.is_capture(move)]
        pool = captures if captures and rng.random() < 0.3 else moves
        board.push(pool[int(rng.integers(len(pool)))])
    return _from_board(board)


def _sparse(rng: np.random.Generator) -> list[str]:
    """Two kings and up to six other pieces, as in endgames and studies."""
    extras = int(rng.integers(0, 7))
    symbols = ["K", "k"] + [str(s) for s in rng.choice(_SCATTER_SYMBOLS[:5] + _SCATTER_SYMBOLS[6:11], size=extras)]
    squares = rng.choice(64, size=len(symbols), replace=False)
    placement = [""] * 64
    for square, symbol in zip(squares, symbols):
        placement[int(square)] = symbol
    return placement


def _scatter(rng: np.random.Generator) -> list[str]:
    """Pieces on random squares, ignoring legality, at a random density."""
    density = rng.uniform(0.05, 0.6)
    occupied = rng.random(64) < density
    symbols = rng.choice(_SCATTER_SYMBOLS, size=64, p=_SCATTER_WEIGHTS)
    return [str(symbol) if filled else "" for symbol, filled in zip(symbols, occupied)]


def sample_placement(rng: np.random.Generator) -> tuple[list[str], str]:
    """Return a placement and the name of the method that produced it."""
    roll = rng.random()
    if roll < 0.55:
        return _game(rng), "game"
    if roll < 0.75:
        return _sparse(rng), "sparse"
    if roll < 0.95:
        return _scatter(rng), "scatter"
    if roll < 0.98:
        return _from_board(chess.Board()), "start"
    return [""] * 64, "empty"


def to_fen(placement: list[str]) -> str:
    """The piece-placement field of a FEN string."""
    rows = []
    for row in range(8):
        text, gap = "", 0
        for symbol in placement[row * 8 : row * 8 + 8]:
            if symbol:
                text += (str(gap) if gap else "") + symbol
                gap = 0
            else:
                gap += 1
        rows.append(text + (str(gap) if gap else ""))
    return "/".join(rows)
