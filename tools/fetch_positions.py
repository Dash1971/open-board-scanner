#!/usr/bin/env python3
"""Take real positions from the start of the Lichess puzzle database.

Only the first part of the compressed file is read: the download is streamed
through the `zstd` command and stopped after --rows rows, so the multi-gigabyte
whole is never fetched. Each puzzle is stored as it is shown to the solver,
after the opponent's first move, as a FEN piece-placement field, one per line
and without duplicates. Source and licence are in assets/positions.json.

    python tools/fetch_positions.py              # about 60,000 rows
    python tools/fetch_positions.py --rows 5000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets" / "positions.json"
CHUNK = 1 << 16


def _pump(response, sink, stop: threading.Event) -> None:
    """Copy the download into zstd until told to stop or the pipe closes."""
    try:
        while not stop.is_set():
            block = response.read(CHUNK)
            if not block:
                break
            sink.write(block)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            sink.close()
        except OSError:
            pass


def shown_placement(fen: str, moves: str) -> str | None:
    """The placement after the first move of the puzzle line, or None if it does not parse."""
    try:
        board = chess.Board(fen)
        first = moves.split()[0]
        board.push(chess.Move.from_uci(first))
    except (ValueError, IndexError):
        return None
    return board.board_fen()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=60_000, help="rows to read before stopping")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    zstd = shutil.which("zstd")
    if zstd is None:
        print("the zstd command is needed (for example: brew install zstd)")
        return 1
    target = ROOT / manifest["file"]

    request = urllib.request.Request(manifest["url"], headers={"User-Agent": "fetch-positions"})
    stop = threading.Event()
    with urllib.request.urlopen(request, timeout=60) as response:
        process = subprocess.Popen([zstd, "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pump = threading.Thread(target=_pump, args=(response, process.stdin, stop), daemon=True)
        pump.start()
        seen: dict[str, None] = {}  # keeps first-seen order
        rows = 0
        reader = csv.reader(io.TextIOWrapper(process.stdout, encoding="utf-8", newline=""))
        for row in reader:
            if rows == 0 and row and row[0] == "PuzzleId":
                continue  # header
            rows += 1
            if len(row) > 2:
                placement = shown_placement(row[1], row[2])
                if placement:
                    seen.setdefault(placement)
            if rows >= args.rows:
                break
        stop.set()
        process.stdout.close()
        process.kill()
        process.wait()
        pump.join(timeout=10)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{placement}\n" for placement in seen))
    manifest["rows_read"] = rows
    manifest["positions"] = len(seen)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{rows} rows read, {len(seen)} distinct positions written to {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
