#!/usr/bin/env python3
"""Download the board textures listed in assets/boards.json.

Files come from the pinned commit of each source repository and are checked
against the SHA-256 hashes in the manifest. Nothing is downloaded that the
manifest does not name, and downloaded images are only ever read as pixels.

    python tools/fetch_boards.py          # download and verify
    python tools/fetch_boards.py --pin    # record hashes for files without one
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets" / "boards.json"
DEST = ROOT / "data" / "assets" / "boards"
MAX_BYTES = 4_000_000
PAUSE = 0.5  # seconds between downloads, to stay polite to the host


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fetch-boards"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
    return data


def raw_url(source: dict, path: str) -> str:
    return f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{urllib.parse.quote(path)}"


def files_of(entry: dict) -> list[str]:
    """The remote file names an entry needs."""
    return [entry["light"], entry["dark"]] if entry["kind"] == "pair" else [entry["file"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="record missing hashes")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for name, entry in manifest["boards"].items():
        source = manifest["sources"][entry["source"]]
        hashes = entry.setdefault("sha256", {})
        for remote in files_of(entry):
            # Sources reuse file names (wood.png), so each is kept under its source.
            target = DEST / entry["source"] / remote
            if target.exists():
                data = target.read_bytes()
            else:
                try:
                    data = fetch(raw_url(source, f"{source['path']}/{remote}"))
                except (OSError, ValueError) as error:
                    print(f"FAILED {name}/{remote}: {error}")
                    failures += 1
                    continue
                time.sleep(PAUSE)
            digest = hashlib.sha256(data).hexdigest()
            expected = hashes.get(remote)
            if expected is None and args.pin:
                hashes[remote] = digest
            elif expected != digest:
                print(f"HASH MISMATCH {name}/{remote}: got {digest}, want {expected}")
                failures += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(f"boards/{name}: done")
    if args.pin:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if failures:
        print(f"{failures} file(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
