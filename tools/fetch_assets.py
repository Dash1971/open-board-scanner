#!/usr/bin/env python3
"""Download the piece sets and fonts listed in the assets/ manifests.

Files come from pinned commits and are checked against the SHA-256 hashes in
the manifests. Nothing is downloaded that a manifest does not name.

    python tools/fetch_assets.py          # download and verify
    python tools/fetch_assets.py --pin    # record hashes for files without one
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIECES_MANIFEST = ROOT / "assets" / "piece_sets.json"
FONTS_MANIFEST = ROOT / "assets" / "fonts.json"
DEST = ROOT / "data" / "assets"
PIECE_FILES = [f"{c}{p}.svg" for c in "wb" for p in "KQRBNP"]
MAX_BYTES = 4_000_000


def fetch(repo: str, commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
    request = urllib.request.Request(url, headers={"User-Agent": "fetch-assets"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
    return data


class Fetcher:
    def __init__(self, pin: bool) -> None:
        self.pin = pin
        self.failures = 0

    def get(self, source: dict, path: str, target: Path, hashes: dict, key: str) -> None:
        """Fetch one file, or reuse the local copy, and check or record its hash."""
        if target.exists():
            data = target.read_bytes()
        else:
            data = fetch(source["repo"], source["commit"], path)
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
    args = parser.parse_args()
    fetcher = Fetcher(args.pin)

    pieces = json.loads(PIECES_MANIFEST.read_text())
    for name, entry in pieces["sets"].items():
        hashes = entry.setdefault("sha256", {})
        for filename in PIECE_FILES:
            path = f"{pieces['source']['path']}/{name}/{filename}"
            fetcher.get(pieces["source"], path, DEST / "pieces" / name / filename, hashes, filename)
        print(f"pieces/{name}: done")

    fonts = json.loads(FONTS_MANIFEST.read_text())
    for name, entry in fonts["fonts"].items():
        fetcher.get(fonts["source"], entry["path"], DEST / "fonts" / name, entry, "sha256")
        print(f"fonts/{name}: done")

    if args.pin:
        for manifest_path, manifest in ((PIECES_MANIFEST, pieces), (FONTS_MANIFEST, fonts)):
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if fetcher.failures:
        print(f"{fetcher.failures} file(s) failed verification")
    return 1 if fetcher.failures else 0


if __name__ == "__main__":
    sys.exit(main())
