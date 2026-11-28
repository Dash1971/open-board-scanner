#!/usr/bin/env python3
"""Collect printed chess diagrams from old books scanned by the Internet Archive.

For each item the IIIF manifest lists the pages. Pages are looked at in an
order that spreads them over the whole book (the first look is the middle,
then the quarters, then the eighths, and so on), each at a small width.
Board outlines are found by contour detection (label.find.find_boards) and
kept only when the inside looks like an 8 x 8 board of alternating squares
and the ONNX board finder agrees that a whole board is in view. Pages that
pass are fetched again at full width, the outlines are found again there,
and the page image is saved. Positions are not read or labelled.

Everything lands under data/external/ia/:

    items/<id>.json         title, year, language and rights fields of each item
    cache/<id>.manifest.json  IIIF manifests, so a rerun does not ask again
    state/<id>.json         pages looked at and the boards found on each (restart point)
    pages/<id>/<page>.jpg   pages with diagrams, 1800 px wide or the scan's own width if less
    manifest.json           one entry per diagram
    rejected.json           diagrams removed by eye, kept out of later rebuilds
    sheets/                 contact sheets of the cut-out boards

Requests go to archive.org only, at most one a second, with a plain
User-Agent. Rate-limit or server errors make the script wait and then stop;
run it again later and it carries on where it left off.

    python tools/fetch_ia_diagrams.py meta ID [ID ...]      # show title, date, rights
    python tools/fetch_ia_diagrams.py search 'subject:chess AND year:[1800 TO 1930]'
    python tools/fetch_ia_diagrams.py fetch [ID ...]        # default: the ITEMS list below
    python tools/fetch_ia_diagrams.py sheets                # contact sheets for checking
    python tools/fetch_ia_diagrams.py drop ID:PAGE:N [...]  # remove diagrams by eye
    python tools/fetch_ia_diagrams.py build                 # rewrite manifest.json from state
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boardscan.geometry import around, cut_view, from_view, rectify  # noqa: E402
from label.find import find_boards  # noqa: E402

OUT = ROOT / "data" / "external" / "ia"
FINDER = ROOT / "runs" / "finder-v1" / "boardfinder-int8.onnx"
USER_AGENT = "chess-diagram-fetch/1.0"
GAP = 1.05  # seconds between requests, at least
SMALL, LARGE = 500, 1800
PAGES_PER_ITEM = 40
LOOKS_PER_ITEM = 260  # small pages looked at before giving up on an item
MIN_SIDE = 0.12  # of the small page's width
GRID_SCORE = 0.06  # contrast between the two colours of square, 0..1
IN_VIEW = 0.5  # finder probability that a whole board is in view
RIGHTS = re.compile(r"copyright|rights|licen[cs]e|restrict|access|public.?domain|possible", re.I)

# Items chosen for different diagram typefaces; see the report for each.
ITEMS = [
    "chessfundamental00capa",
    "chessstrategy00lask",
    "modernchessinstr00steirich",
    "cu31924029919820",
    "championshipmat00hoffgoog",
    "derschachwettka00tarrgoog",
    "chessplayerstext00stau_0",
    "americanchessnu00gilbgoog",
    "TavernerChessProblemsMadeEasy",
    "bub_gb_wRNBAAAAYAAJ",
    "palamde00unkngoog",
    "RevistaDeAjedrezVol.11889",
    "nordiskskaktide00unkngoog",
    "middlegameinches00znos",
    "the-british-chess-magazine-volume-12-1892",
    "the-british-chess-magazine-volume-27-1907",
    "the-british-chess-magazine-volume-46-1926",
]


class Stop(Exception):
    """The server refused or kept failing: stop and try again later."""


class Client:
    def __init__(self) -> None:
        self.last = 0.0
        self.requests = 0
        self.refusals = 0

    def get(self, url: str, kind: str) -> bytes | None:
        """Fetch one URL, waiting so that requests are at least GAP apart.

        Returns None for a missing page. Rate limits and server errors are
        waited out twice; a third failure raises Stop. A web page where an
        image or JSON was expected (a bot check, say) also raises Stop.
        """
        if urllib.parse.urlparse(url).hostname not in ("archive.org", "iiif.archive.org"):
            raise ValueError(f"not an archive.org address: {url}")
        for attempt in range(3):
            wait = self.last + GAP - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=90) as response:
                    data, content_type = response.read(), response.headers.get("Content-Type", "")
                self.last = time.monotonic()
            except urllib.error.HTTPError as error:
                self.last = time.monotonic()
                if error.code in (400, 404, 410):  # 400: a size the page cannot be served at
                    print(f"  HTTP {error.code} for {url}", flush=True)
                    return None
                if error.code in (403,):
                    raise Stop(f"{error.code} refused for {url}")
                self.refusals += 1
                pause = float(error.headers.get("Retry-After") or 0) or 60.0 * (attempt + 1)
                print(f"  HTTP {error.code}, waiting {pause:.0f} s", flush=True)
                time.sleep(min(pause, 600))
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                self.refusals += 1
                print(f"  {error}, waiting 60 s", flush=True)
                time.sleep(60)
                continue
            if "html" in content_type.lower():
                raise Stop(f"got a web page for {url}: possibly a bot check")
            if kind == "image" and not content_type.startswith("image/"):
                raise Stop(f"expected an image, got {content_type} for {url}")
            return data
        raise Stop(f"gave up on {url} after three failures")


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)


def item_record(client: Client, identifier: str) -> dict:
    """Title, year, language and every rights-like field of an item, cached."""
    path = OUT / "items" / f"{identifier}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = client.get(f"https://archive.org/metadata/{identifier}", "json")
    metadata = json.loads(data or b"{}").get("metadata", {})
    if not metadata:
        return {"identifier": identifier, "missing": True}
    record = {
        "identifier": identifier,
        "title": metadata.get("title"),
        "creator": metadata.get("creator"),
        "date": metadata.get("date"),
        "year": metadata.get("year"),
        "language": metadata.get("language"),
        "publisher": metadata.get("publisher"),
        "rights": {key: value for key, value in metadata.items() if RIGHTS.search(key) and key != "identifier-access"},
        "imagecount": metadata.get("imagecount"),
    }
    write_json(path, record)
    return record


def year_of(record: dict) -> int | None:
    found = re.findall(r"1[5-9]\d\d|20\d\d", " ".join(str(record.get(key) or "") for key in ("date", "year")))
    return max(int(year) for year in found) if found else None


def page_services(client: Client, identifier: str) -> list[tuple[str, int]]:
    """The image service URL and full width of every page, in reading order."""
    path = OUT / "cache" / f"{identifier}.manifest.json"
    if not path.exists():
        data = client.get(f"https://iiif.archive.org/iiif/{identifier}/manifest.json", "json")
        if data is None:
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = json.loads(path.read_text())
    services = []
    for canvas in manifest.get("items", []):
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                service = annotation.get("body", {}).get("service", [{}])[0]
                if service.get("id"):
                    services.append((service["id"], int(canvas.get("width") or LARGE)))
    return services


def spread_order(count: int) -> list[int]:
    """Page indices ordered so that any prefix is spread over the whole book."""
    order, seen = [], set()
    depth = 1
    while len(order) < count:
        for step in range(1, 2**depth, 2):
            index = int(step * count / 2**depth)
            if index not in seen:
                seen.add(index)
                order.append(index)
        depth += 1
        if 2**depth > 4 * count:
            order += [index for index in range(count) if index not in seen]
            break
    return order


def fetch_image(client: Client, service: str, width: int) -> np.ndarray | None:
    data = client.get(f"{service}/full/{width},/0/default.jpg", "image")
    if data is None:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def grid_score(image: np.ndarray, corners: np.ndarray) -> float:
    """How strongly the inside alternates like a chess board, 0..1.

    Squares of one colour are compared with squares of the other, using the
    median of each so that pieces standing on some of them matter little.
    """
    crop = cv2.cvtColor(rectify(image, corners, 256), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    cells = crop.reshape(8, 32, 8, 32)[:, 6:26, :, 6:26].mean(axis=(1, 3))
    parity = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(bool)
    return float(abs(np.median(cells[parity]) - np.median(cells[~parity])))


def edge_score(image: np.ndarray, corners: np.ndarray) -> float:
    """Darkness of the inner grid lines between squares minus that of square centres.

    High for a board drawn with lines and no shading, which grid_score misses.
    """
    crop = cv2.cvtColor(rectify(image, corners, 256), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    lines = np.concatenate([crop[:, 30:34].ravel(), crop[30:34, :].ravel()])
    return float(crop[8:24, 8:24].mean() - np.median(lines))


class Finder:
    def __init__(self) -> None:
        from boardscan.pipeline import onnx_finder

        self.finder = onnx_finder(FINDER)

    def check(self, image: np.ndarray, corners: np.ndarray, looks: int = 2) -> tuple[np.ndarray, float]:
        """The finder's corners and whole-board probability, starting from a window round `corners`."""
        matrix = around(corners)
        found, logit = corners, 0.0
        for _ in range(looks):
            fractions, logit = self.finder(cut_view(image, matrix, border=(255, 255, 255)))
            found = from_view(fractions, matrix)
            matrix = around(found)
        return found.astype(np.float32), float(1.0 / (1.0 + np.exp(-logit)))


def boards_on(image: np.ndarray, finder: Finder) -> list[dict]:
    """Board outlines on a page image that look like chess diagrams."""
    height, width = image.shape[:2]
    kept = []
    for quad in find_boards(image, limit=8):
        sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        if sides.min() < MIN_SIDE * width:
            continue
        if any(np.abs(quad - np.asarray(other["corners"])).max() < 0.08 * sides.mean() for other in kept):
            continue
        # A thick frame gives an outer and an inner outline; a board inside a
        # larger box (a page border, a table) must not swallow it, so every
        # outline is judged on its own.
        grid, edges = grid_score(image, quad), edge_score(image, quad)
        if grid < GRID_SCORE and edges < 0.15:
            continue
        found, in_view = finder.check(image, quad)
        if in_view < IN_VIEW or np.abs(found - quad).max() > 0.15 * sides.mean():
            continue
        kept.append({"corners": quad.round(1).tolist(), "grid": round(grid, 3), "lines": round(edges, 3), "in_view": round(in_view, 3)})
    # Drop an outline that holds another kept one (a box drawn round a diagram).
    inner = []
    for board in kept:
        quad = np.asarray(board["corners"])
        holds = any(
            other is not board
            and cv2.pointPolygonTest(quad.astype(np.float32), tuple(map(float, np.mean(other["corners"], axis=0))), False) > 0
            and cv2.contourArea(np.asarray(other["corners"], np.float32)) < 0.8 * cv2.contourArea(quad.astype(np.float32))
            for other in kept
        )
        if not holds:
            inner.append(board)
    return inner


def refine(image: np.ndarray, small: dict, scale: float, finder: Finder) -> dict | None:
    """Find the outline again on the full page, near where the small page put it."""
    guess = np.asarray(small["corners"], np.float32) * scale
    side = float(np.linalg.norm(guess[1] - guess[0]))
    low = np.maximum(guess.min(axis=0) - 0.2 * side, 0).astype(int)
    high = np.minimum(guess.max(axis=0) + 0.2 * side, image.shape[1::-1]).astype(int)
    window = image[low[1] : high[1], low[0] : high[0]]
    best = None
    for quad in find_boards(window, limit=8):
        quad = (quad + low).astype(np.float32)
        if np.abs(quad - guess).max() > 0.06 * side:
            continue
        if best is None or cv2.contourArea(quad) < cv2.contourArea(best):
            best = quad  # the innermost of the near matches: the playing area, not the frame
    corners = best if best is not None else guess
    found, in_view = finder.check(image, corners)
    return {
        "corners": np.asarray(corners).round(1).tolist(),
        "refined": best is not None,
        "grid": round(grid_score(image, corners), 3),
        "lines": round(edge_score(image, corners), 3),
        "in_view": round(in_view, 3),
    }


def fetch_item(client: Client, finder: Finder, identifier: str, pages_wanted: int, looks: int) -> None:
    record = item_record(client, identifier)
    if record.get("missing"):
        print(f"{identifier}: no such item")
        return
    year = year_of(record)
    if year is None or year >= 1931:
        print(f"{identifier}: date {record.get('date')!r} is not before 1931, skipped")
        return
    services = page_services(client, identifier)
    state_path = OUT / "state" / f"{identifier}.json"
    state = read_json(state_path, {"looked": [], "pages": {}})
    looked = set(state["looked"])
    order = spread_order(len(services))
    print(f"{identifier}: {len(services)} pages, {len(state['pages'])} with diagrams so far", flush=True)
    for index in order:
        if len(state["pages"]) >= pages_wanted or len(looked) >= looks:
            break
        if index in looked:
            continue
        service, full_width = services[index]
        small = fetch_image(client, service, min(SMALL, full_width))
        boards = boards_on(small, finder) if small is not None else []
        if boards:
            large = fetch_image(client, service, min(LARGE, full_width))  # the server does not enlarge
            if large is not None:
                scale = large.shape[1] / small.shape[1]
                refined = [board for board in (refine(large, small_board, scale, finder) for small_board in boards) if board]
                if refined:
                    name = f"pages/{identifier}/{index:04d}.jpg"
                    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(OUT / name), cv2.cvtColor(large, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    state["pages"][str(index)] = {"image": name, "size": [large.shape[1], large.shape[0]], "boards": refined}
                    print(f"  page {index}: {len(refined)} board(s)", flush=True)
        looked.add(index)
        state["looked"] = sorted(looked)
        write_json(state_path, state)
    print(f"{identifier}: {len(state['pages'])} diagram pages after looking at {len(looked)}", flush=True)


def overlapping(boards: list[dict]) -> set[int]:
    """Numbers of boards to leave out because a better-looking outline overlaps them.

    Two outlines overlap when the centre of either lies inside the other; the
    one whose inside alternates more like a board (grid plus lines) is kept.
    """
    def inside(quad, point) -> bool:
        return cv2.pointPolygonTest(np.asarray(quad, np.float32), tuple(map(float, point)), False) > 0

    worse = set()
    for first, one in enumerate(boards):
        for second, other in enumerate(boards):
            if first < second and (inside(one["corners"], np.mean(other["corners"], axis=0)) or inside(other["corners"], np.mean(one["corners"], axis=0))):
                worse.add(second if one["grid"] + one["lines"] >= other["grid"] + other["lines"] else first)
    return worse


def build() -> list[dict]:
    """Rewrite manifest.json from the state files, leaving out rejected diagrams."""
    rejected = set(read_json(OUT / "rejected.json", {}).keys())
    entries = []
    for state_path in sorted((OUT / "state").glob("*.json")):
        identifier = state_path.stem
        record = read_json(OUT / "items" / f"{identifier}.json", {})
        for index, page in sorted(json.loads(state_path.read_text())["pages"].items(), key=lambda item: int(item[0])):
            worse = overlapping(page["boards"])
            for number, board in enumerate(page["boards"]):
                key = f"{identifier}:{index}:{number}"
                if key in rejected or number in worse:
                    continue
                entries.append(
                    {
                        "id": key,
                        "item": identifier,
                        "page": int(index),
                        "image": page["image"],
                        "corners": board["corners"],
                        "title": record.get("title"),
                        "year": year_of(record),
                        "language": record.get("language"),
                        "rights": record.get("rights"),
                    }
                )
    write_json(OUT / "manifest.json", entries)
    return entries


def sheets(per_sheet: int = 48, columns: int = 8, cell: int = 160) -> None:
    """Contact sheets of the cut-out boards, one set per item, each labelled with its id."""
    entries = build()
    folder = OUT / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    by_item: dict[str, list[dict]] = {}
    for entry in entries:
        by_item.setdefault(entry["item"], []).append(entry)
    for identifier, group in by_item.items():
        for start in range(0, len(group), per_sheet):
            chunk = group[start : start + per_sheet]
            rows = (len(chunk) + columns - 1) // columns
            sheet = np.full((rows * (cell + 18), columns * cell, 3), 255, np.uint8)
            for position, entry in enumerate(chunk):
                image = cv2.imread(str(OUT / entry["image"]))
                crop = cv2.resize(rectify(image, np.asarray(entry["corners"], np.float32), 256), (cell - 6, cell - 6), interpolation=cv2.INTER_AREA)
                y, x = (position // columns) * (cell + 18), (position % columns) * cell
                sheet[y + 3 : y + cell - 3, x + 3 : x + cell - 3] = crop
                cv2.putText(sheet, f"{entry['page']}:{entry['id'].rsplit(':', 1)[1]}", (x + 4, y + cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
            cv2.imwrite(str(folder / f"{identifier}-{start // per_sheet + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(entries)} diagrams on sheets in {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    meta = commands.add_parser("meta", help="show title, date and rights of items")
    meta.add_argument("ids", nargs="+")
    search = commands.add_parser("search", help="search the archive")
    search.add_argument("query")
    search.add_argument("--rows", type=int, default=50)
    fetch = commands.add_parser("fetch", help="find and save diagram pages")
    fetch.add_argument("ids", nargs="*")
    fetch.add_argument("--pages", type=int, default=PAGES_PER_ITEM, help="diagram pages wanted per item")
    fetch.add_argument("--looks", type=int, default=LOOKS_PER_ITEM, help="small pages looked at per item, at most")
    commands.add_parser("sheets", help="contact sheets for checking by eye")
    drop = commands.add_parser("drop", help="remove diagrams found wanting by eye")
    drop.add_argument("keys", nargs="+", help="ITEM:PAGE:N, or ITEM:PAGE for every diagram on a page")
    drop.add_argument("--why", default="")
    commands.add_parser("build", help="rewrite manifest.json")
    args = parser.parse_args()
    client = Client()

    try:
        if args.command == "meta":
            for identifier in args.ids:
                record = item_record(client, identifier)
                print(json.dumps(record, ensure_ascii=False, indent=1))
        elif args.command == "search":
            query = urllib.parse.urlencode(
                [("q", args.query), ("fl[]", "identifier"), ("fl[]", "title"), ("fl[]", "year"), ("fl[]", "language"), ("rows", args.rows), ("output", "json")]
            )
            data = client.get(f"https://archive.org/advancedsearch.php?{query}", "json")
            for doc in json.loads(data or b"{}").get("response", {}).get("docs", []):
                print(doc.get("identifier"), doc.get("year"), doc.get("language"), "|", str(doc.get("title"))[:90])
        elif args.command == "fetch":
            finder = Finder()
            for identifier in args.ids or ITEMS:
                fetch_item(client, finder, identifier, args.pages, args.looks)
            print(f"{len(build())} diagrams in manifest.json")
        elif args.command == "sheets":
            sheets()
        elif args.command == "drop":
            rejected = read_json(OUT / "rejected.json", {})
            for key in args.keys:
                identifier, index, *number = key.split(":")
                page = read_json(OUT / "state" / f"{identifier}.json", {"pages": {}})["pages"].get(index)
                count = len(page["boards"]) if page else 0
                for n in [int(number[0])] if number else range(count):
                    rejected[f"{identifier}:{index}:{n}"] = args.why
            write_json(OUT / "rejected.json", rejected)
            print(f"{len(build())} diagrams in manifest.json, {len(rejected)} rejected")
        elif args.command == "build":
            print(f"{len(build())} diagrams in manifest.json")
    except Stop as stop:
        print(f"STOPPED: {stop}. Run again later to carry on.")
        return 2
    finally:
        print(f"{client.requests} requests, {client.refusals} refused or failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
