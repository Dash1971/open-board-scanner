"""A page on this computer for correcting the model's reading of each photo.

    python -m label --photos eval/private/my-photos

It listens on 127.0.0.1 only. Labels are written to eval/private/labels/, which
git ignores, as is everything else under eval/private/.
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from boardgen import rectify
from boardgen.assets import ASSET_DIR

from .store import Reader, Store

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("page.html")
DEFAULT_MODEL = ROOT / "runs" / "boardnet-v1" / "boardnet-fp32.onnx"
PIECES = {symbol: f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg" for symbol in "KQRBNPkqrbnp"}
MAX_BODY = 20_000


class BadRequest(Exception):
    pass


def make_handler(store: Store, hosts: set[str]):
    """A request handler for one store. `hosts` holds the Host headers it answers to."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status: int = 200) -> None:
            self._send(json.dumps(value).encode(), "application/json", status)

        def _record(self, query: dict) -> tuple[str, dict]:
            board_id = query.get("id", [""])[0]
            if board_id not in store.boards:
                raise BadRequest(f"no board {board_id!r}")
            return board_id, store.boards[board_id]

        def _summary(self) -> list[dict]:
            return [{"id": key, "status": store.boards[key]["status"], "source": store.boards[key]["source"]} for key in store.ids()]

        def _handle(self, method: str) -> None:
            # Requests must name this server, so that another site cannot reach it through a DNS trick.
            if self.headers.get("Host") not in hosts:
                raise BadRequest("unexpected Host header")
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if method == "GET":
                self._get(url.path, query)
            else:
                # A header that pages on other sites cannot send without permission.
                if self.headers.get("X-Label") != "1":
                    raise BadRequest("missing X-Label header")
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    raise BadRequest("bad request size")
                self._post(url.path, json.loads(self.rfile.read(length)))

        def _get(self, path: str, query: dict) -> None:
            if path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/boards":
                self._json(self._summary())
            elif path == "/api/board":
                board_id, record = self._record(query)
                height, width = store.image(record["photo"]).shape[:2]
                self._json({"id": board_id, "width": width, "height": height, **record})
            elif path == "/photo":
                _, record = self._record(query)
                self._send((store.photos / record["photo"]).read_bytes(), "image/jpeg", cache=True)
            elif path == "/crop":
                _, record = self._record(query)
                try:
                    corners = np.array([float(value) for value in query["corners"][0].split(",")], np.float32).reshape(4, 2)
                    turns = int(query["turns"][0]) % 4
                except (KeyError, ValueError) as error:
                    raise BadRequest(f"bad crop request: {error}") from error
                crop = np.rot90(rectify(store.image(record["photo"]), corners, 512), turns)
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._send(encoded.tobytes(), "image/jpeg")
            elif path.startswith("/piece/") and path[7:] in PIECES:
                self._send((ASSET_DIR / "pieces" / "cburnett" / PIECES[path[7:]]).read_bytes(), "image/svg+xml", cache=True)
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def _post(self, path: str, body: dict) -> None:
            board_id = body.get("id", "")
            if board_id not in store.boards:
                raise BadRequest(f"no board {board_id!r}")
            if path == "/api/read":
                self._json(store.read(board_id, np.asarray(body["corners"], np.float32).reshape(4, 2)))
            elif path == "/api/save":
                store.update(board_id, body["corners"], body["quarter_turns"], body["seen"], body["white_at_bottom"], body.get("source", ""), body["status"])
                self._json({"ok": True})
            elif path == "/api/add":
                self._json({"id": store.add_board(store.boards[board_id]["photo"]), "boards": self._summary()})
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def _guarded(self, method: str) -> None:
            try:
                self._handle(method)
            except (BadRequest, KeyError, ValueError, TypeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_GET(self):
            self._guarded("GET")

        def do_POST(self):
            self._guarded("POST")

    return Handler


def make_server(store: Store, port: int) -> ThreadingHTTPServer:
    """A server on 127.0.0.1. Port 0 picks a free port."""
    hosts: set[str] = set()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, hosts))
    bound = server.server_address[1]
    hosts.update({f"127.0.0.1:{bound}", f"localhost:{bound}"})
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=Path, required=True, help="folder of photos to label")
    parser.add_argument("--labels", type=Path, help="label file (default: eval/private/labels/<folder name>.json)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="ONNX classifier used to pre-fill readings")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    args = parser.parse_args()

    photos = args.photos.resolve()
    labels = args.labels or ROOT / "eval" / "private" / "labels" / f"{photos.name}.json"
    if not args.model.exists():
        print(f"{args.model} not found: boards will start empty")
    store = Store(photos, labels, Reader(args.model if args.model.exists() else None))
    changed = store.prepare(lambda done, total: print(f"  looked at {done} of {total} photos", flush=True))
    done = sum(record["status"] != "todo" for record in store.boards.values())
    print(f"{len(store.boards)} boards ({changed} read just now, {done} already labelled); labels in {labels}")
    server = make_server(store, args.port)
    print(f"Open http://127.0.0.1:{args.port}/   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
