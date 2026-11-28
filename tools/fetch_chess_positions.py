#!/usr/bin/env python3
"""Download the "Chess Positions" board images from their Hugging Face mirror.

The dataset is "Chess Positions" by koryakinp (Kaggle koryakinp/chess-positions,
CC0: Public Domain): rendered 400 x 400 JPEG boards, each named by the
piece-placement part of its FEN with "-" in place of "/". The Hugging Face
mirror smallchess/kaggle-chess-positions stores them as loose images under
`train/` (the Kaggle test split is not mirrored).

The file list is read from the Hugging Face tree API and kept in
`listing.json`; each image is then fetched from the pinned commit and checked
against the SHA-256 that the listing gives for it. Files already present with
the right size are kept, so an interrupted run can simply be started again.
Anonymous downloads are rate limited (about 3,000 files per five minutes), so
the script follows the server's RateLimit header and a full run takes a couple
of hours. When every file is present, `SHA256SUMS` is written next to them.
Nothing downloaded is ever run.

    python tools/fetch_chess_positions.py               # download everything
    python tools/fetch_chess_positions.py --limit 100   # a quick sample
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "external" / "chess-positions"
REPO = "smallchess/kaggle-chess-positions"
# The commit the import was built from; pinning it keeps the file list stable.
COMMIT = "35fed168fce84b775473c72f643543c0c64d695a"
API = f"https://huggingface.co/api/datasets/{REPO}"
USER_AGENT = "fetch-chess-positions"
MAX_BYTES = 2_000_000
FILE_NAME = re.compile(r"^(train|test)/[1-8KQRBNPkqrbnp]+(-[1-8KQRBNPkqrbnp]+){7}\.jpe?g$")


def request(url: str) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
        headers = {key.lower(): value for key, value in response.headers.items()}
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
    return data, headers


def list_files() -> list[dict]:
    """Every image in the mirror, as {"path", "size", "sha256"}, read page by page."""
    files = []
    url = f"{API}/tree/{COMMIT}?recursive=true&limit=1000"
    while url:
        data, headers = request(url)
        for entry in json.loads(data):
            if entry["type"] == "file" and "lfs" in entry and FILE_NAME.match(entry["path"]):
                files.append({"path": entry["path"], "size": entry["lfs"]["size"], "sha256": entry["lfs"]["oid"]})
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
        url = match.group(1) if match else None
        print(f"\rlisted {len(files)} images", end="", flush=True)
    print()
    return files


class Limiter:
    """Pause all threads when the server says the request budget is nearly spent."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.resume_at = 0.0

    def wait(self) -> None:
        while True:
            with self.lock:
                delay = self.resume_at - time.time()
            if delay <= 0:
                return
            time.sleep(min(delay, 5.0))

    def update(self, header: str | None, retry_after: float | None = None) -> None:
        pause = retry_after
        if header:
            remaining = re.search(r"r=(\d+)", header)
            reset = re.search(r"t=(\d+)", header)
            if remaining and reset and int(remaining.group(1)) <= 5:
                pause = int(reset.group(1)) + 1
        if pause:
            with self.lock:
                self.resume_at = max(self.resume_at, time.time() + pause)


def download(entry: dict, limiter: Limiter) -> str | None:
    """Fetch one image unless it is already present; return an error message on failure."""
    target = DEST / entry["path"]
    if target.exists() and target.stat().st_size == entry["size"]:
        return None
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{urllib.parse.quote(entry['path'])}"
    for attempt in range(6):
        limiter.wait()
        try:
            data, headers = request(url)
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504):
                retry = error.headers.get("Retry-After")
                limiter.update(error.headers.get("RateLimit"), float(retry) if retry and retry.isdigit() else 30.0 * (attempt + 1))
                continue
            return f"{entry['path']}: HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            time.sleep(5 * (attempt + 1))
            last = str(error)
            continue
        limiter.update(headers.get("ratelimit"))
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            return f"{entry['path']}: hash mismatch"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(data)
        partial.replace(target)
        return None
    return f"{entry['path']}: gave up after retries ({locals().get('last', 'rate limited')})"


def write_sums(files: list[dict]) -> int:
    """Hash every present file, check it against the listing and write SHA256SUMS."""
    bad = 0
    lines = []
    for entry in sorted(files, key=lambda item: item["path"]):
        digest = hashlib.sha256((DEST / entry["path"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            print(f"HASH MISMATCH {entry['path']}")
            bad += 1
        lines.append(f"{digest}  {entry['path']}\n")
    (DEST / "SHA256SUMS").write_text("".join(lines))
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="download only the first N images of the listing")
    parser.add_argument("--threads", type=int, default=2, help="downloads in flight at once")
    parser.add_argument("--relist", action="store_true", help="read the file list from the server again")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    listing = DEST / "listing.json"
    if listing.exists() and not args.relist:
        files = json.loads(listing.read_text())
    else:
        files = list_files()
        listing.write_text(json.dumps({"repo": REPO, "commit": COMMIT, "files": files}, indent=0))
        files = {"files": files}
    files = files["files"]
    splits = {}
    for entry in files:
        splits[entry["path"].split("/")[0]] = splits.get(entry["path"].split("/")[0], 0) + 1
    print(f"{len(files)} images in the mirror: {splits}")
    wanted = files[: args.limit] if args.limit else files
    limiter = Limiter()
    failures = []
    began = time.time()
    with ThreadPoolExecutor(args.threads) as pool:
        for done, error in enumerate(pool.map(lambda entry: download(entry, limiter), wanted), 1):
            if error:
                failures.append(error)
                print(error, flush=True)
            if done % 1000 == 0 or done == len(wanted):
                print(f"{done}/{len(wanted)} checked, {len(failures)} failed, {time.time() - began:.0f}s", flush=True)
    if failures:
        sys.exit(f"{len(failures)} downloads failed; run again to retry them")
    if all((DEST / entry["path"]).exists() for entry in files):
        bad = write_sums(files)
        print(f"wrote {DEST / 'SHA256SUMS'}" + (f" ({bad} mismatches)" if bad else ""))
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
