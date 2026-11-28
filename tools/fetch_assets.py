#!/usr/bin/env python3
"""Download the piece sets and fonts listed in the assets/ manifests.

Files come from pinned commits, or from a fixed URL where the source is not a
git repository (a font inside a zip archive is named by `member`), and are
checked against the SHA-256 hashes in the manifests. A piece set from another
repository gives its own `source` and, when its files are not named wK.svg
and so on, a `files` map from our names to its own. A set drawn as bitmaps
names their file type in `format` (for example "webp"); the default is SVG.
Nothing is downloaded that a manifest does not name.

    python tools/fetch_assets.py          # download and verify
    python tools/fetch_assets.py --pin    # record hashes for files without one
    python tools/fetch_assets.py --only pieces --pin    # one manifest only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIECES_MANIFEST = ROOT / "assets" / "piece_sets.json"
FONTS_MANIFEST = ROOT / "assets" / "fonts.json"
GLYPHS_MANIFEST = ROOT / "assets" / "glyph_fonts.json"
DEST = ROOT / "data" / "assets"
PIECES = [(colour, piece) for colour in "wb" for piece in "KQRBNP"]
LONG_NAMES = {"w": "white", "b": "black", "K": "King", "Q": "Queen", "R": "Rook", "B": "Bishop", "N": "Knight", "P": "Pawn"}
MAX_BYTES = 4_000_000


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fetch-assets"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
    return data


def raw_url(source: dict, path: str) -> str:
    return f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{urllib.parse.quote(path)}"


class Fetcher:
    def __init__(self, pin: bool) -> None:
        self.pin = pin
        self.failures = 0

    def get(self, url: str, target: Path, hashes: dict, key: str, member: str | None = None) -> None:
        """Fetch one file, or reuse the local copy, and check or record its hash.

        With `member`, the URL is a zip archive and the named file inside it is
        kept. Nothing in the archive is run, and nothing else is unpacked.
        """
        if target.exists():
            data = target.read_bytes()
        else:
            data = fetch(url)
            if member is not None:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    data = archive.read(member)
        digest = hashlib.sha256(data).hexdigest()
        expected = hashes.get(key)
        if expected is None and self.pin:
            hashes[key] = digest
        elif expected != digest:
            print(f"HASH MISMATCH {target.name}: got {digest}, want {expected}")
            self.failures += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="record missing hashes")
    parser.add_argument("--only", choices=("pieces", "fonts", "glyphs"), help="handle one manifest and leave the others untouched")
    args = parser.parse_args()
    fetcher = Fetcher(args.pin)
    wanted = (args.only,) if args.only else ("pieces", "fonts", "glyphs")
    changed = []

    if "pieces" in wanted:
        pieces = json.loads(PIECES_MANIFEST.read_text())
        for name, entry in pieces["sets"].items():
            source = entry.get("source", pieces["source"])
            folder = source["path"] if "source" in entry else f"{source['path']}/{name}"
            hashes = entry.setdefault("sha256", {})
            extension = entry.get("format", "svg")  # bitmap sets name theirs, for example "webp" or "png"
            for colour, piece in PIECES:
                local = f"{colour}{piece}.{extension}"
                if "files" in entry:  # the set's own file names, keyed by ours without the extension
                    remote = entry["files"][f"{colour}{piece}"]
                else:
                    remote = f"{LONG_NAMES[colour]}{LONG_NAMES[piece]}.{extension}" if entry.get("naming") == "long" else local
                fetcher.get(raw_url(source, f"{folder}/{remote}"), DEST / "pieces" / name / local, hashes, local)
            print(f"pieces/{name}: done")
        changed.append((PIECES_MANIFEST, pieces))

    if "fonts" in wanted:
        fonts = json.loads(FONTS_MANIFEST.read_text())
        for name, entry in fonts["fonts"].items():
            fetcher.get(raw_url(fonts["source"], entry["path"]), DEST / "fonts" / name, entry, "sha256")
            print(f"fonts/{name}: done")
        changed.append((FONTS_MANIFEST, fonts))

    if "glyphs" in wanted:
        glyphs = json.loads(GLYPHS_MANIFEST.read_text())
        for name, entry in glyphs["fonts"].items():
            url = entry["url"] if "url" in entry else raw_url(entry, entry["path"])
            fetcher.get(url, DEST / "glyph_fonts" / entry["file"], entry, "sha256", entry.get("member"))
            print(f"glyph_fonts/{name}: done")
        changed.append((GLYPHS_MANIFEST, glyphs))

    if args.pin:
        for path, manifest in changed:
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if fetcher.failures:
        print(f"{fetcher.failures} file(s) failed verification")
    return 1 if fetcher.failures else 0


if __name__ == "__main__":
    sys.exit(main())
