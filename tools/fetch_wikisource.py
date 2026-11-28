#!/usr/bin/env python3
"""Download proofread pages of scanned chess books from English Wikisource.

On Wikisource, volunteers proofread scanned books page by page. Where a page
shows a chess diagram, many of them encoded the diagram with the
{{Chess diagram}} template, so both the printed board and its position are
known. This fetches, for every such page in the Page: namespace, the wikitext
and a scan of the page. Wikimedia serves page images only at standard
widths: asked for 1600 px, DjVu scans come at the standard width at or below
their own (1920 px for most, 1280 px for My Chess Career, 500 px for one
title page), and PDF pages are asked for at 1920 px directly:

    data/external/wikisource/pages.json            one entry per page
    data/external/wikisource/files.json            rights metadata of each scanned book
    data/external/wikisource/wikitext/<name>.txt
    data/external/wikisource/images/<name>.jpg

    python tools/fetch_wikisource.py               fetch whatever is missing
    python tools/fetch_wikisource.py --list-only   only list the pages

It can be stopped and started again: files already on disk are not fetched
again. Requests go out at most about one a second, to the MediaWiki API of
en.wikisource.org and to upload.wikimedia.org only, with a plain User-Agent.

"My System.pdf" is skipped: its 1930 translation may still be in copyright
in some countries. Boards are checked against the template by
tools/wikisource_manifest.py.
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

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "external" / "wikisource"
API = "https://en.wikisource.org/w/api.php"
USER_AGENT = "chess-diagram-test-set/0.1 (python-urllib)"
WIDTH = 1600  # asked for; Wikimedia serves thumbnails only at standard widths and rounds
PDF_WIDTH = 1920  # PDF pages are rendered, so ask for the standard width above WIDTH directly
SKIP_FILES = {"My System.pdf"}
PAUSE = 1.05  # seconds between requests
MAX_BYTES = 20_000_000

_last_request = 0.0


def get(url: str, attempts: int = 4) -> bytes:
    """One polite GET: waits so requests stay about a second apart, backs off when asked to."""
    global _last_request
    for attempt in range(attempts):
        wait = PAUSE - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError(f"{url} is larger than {MAX_BYTES} bytes")
            return data
        except urllib.error.HTTPError as error:
            if error.code == 403:
                sys.exit(f"refused (403) for {url}: stopping rather than working round it")
            if error.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                delay = float(error.headers.get("Retry-After") or 0) or 10.0 * (attempt + 1)
                print(f"  {error.code}, waiting {delay:.0f} s", flush=True)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def api(**params: str) -> dict:
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    reply = json.loads(get(API + "?" + urllib.parse.urlencode(params)))
    if "error" in reply:
        raise RuntimeError(f"API error: {reply['error']}")
    return reply


def list_pages() -> list[str]:
    """Titles of Page: namespace pages that use {{Chess diagram}}."""
    titles, more = [], {}
    while True:
        reply = api(action="query", list="embeddedin", eititle="Template:Chess_diagram", einamespace="104", eilimit="500", **more)
        titles += [entry["title"] for entry in reply["query"]["embeddedin"]]
        if "continue" not in reply:
            return sorted(set(titles))
        more = {key: value for key, value in reply["continue"].items()}


def split_title(title: str) -> tuple[str, int]:
    """'Page:Some book.djvu/23' -> ('Some book.djvu', 23)."""
    match = re.fullmatch(r"Page:(.+)/(\d+)", title)
    if not match:
        raise ValueError(f"not a page of a scanned file: {title}")
    return match.group(1), int(match.group(2))


def local_name(file: str, number: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", file.rsplit(".", 1)[0]).strip("-").lower()
    return f"{stem}_{number:04d}"


def fetch_wikitext(entries: list[dict]) -> None:
    missing = [entry for entry in entries if not (DEST / entry["wikitext"]).exists()]
    for start in range(0, len(missing), 50):
        batch = missing[start : start + 50]
        reply = api(action="query", prop="revisions", rvprop="content|ids", rvslots="main", titles="|".join(entry["title"] for entry in batch))
        found = {page["title"]: page for page in reply["query"]["pages"]}
        for entry in batch:
            page = found.get(entry["title"])
            if not page or "revisions" not in page:
                print(f"  no wikitext for {entry['title']}")
                continue
            revision = page["revisions"][0]
            entry["revid"] = revision["revid"]
            (DEST / entry["wikitext"]).write_text(revision["slots"]["main"]["content"], encoding="utf-8")
        print(f"wikitext {start + len(batch)}/{len(missing)}", flush=True)


def fetch_file_metadata(files: list[str]) -> dict:
    """Licence and source fields of each scanned file, as its description page gives them."""
    path = DEST / "files.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    for file in files:
        if file in known:
            continue
        reply = api(action="query", prop="imageinfo", iiprop="url|extmetadata|size", titles=f"File:{file}")
        page = reply["query"]["pages"][0]
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        keep = ("LicenseShortName", "License", "UsageTerms", "Copyrighted", "Credit", "Artist", "DateTimeOriginal", "Attribution", "Restrictions")
        known[file] = {
            "repository": page.get("imagerepository"),
            "description_url": info.get("descriptionurl"),
            "pages": info.get("pagecount"),
            **{key: re.sub(r"<[^>]+>", "", meta[key]["value"]).strip() if isinstance(meta.get(key, {}).get("value"), str) else meta.get(key, {}).get("value") for key in keep if key in meta},
        }
        path.write_text(json.dumps(known, indent=1, ensure_ascii=False))
    return known


def fetch_image(entry: dict) -> None:
    target = DEST / entry["image"]
    if target.exists():
        return
    if not entry.get("image_url"):
        reply = api(action="query", prop="imageinfo", iiprop="url", iiurlwidth=str(WIDTH), iiurlparam=f"page{entry['page']}-{WIDTH}px", titles=f"File:{entry['file']}")
        info = (reply["query"]["pages"][0].get("imageinfo") or [{}])[0]
        url = info.get("thumburl")
        if not url:
            print(f"  no scan for {entry['title']}")
            return
        if entry["file"].lower().endswith(".pdf"):
            # The API never offers a width above the PDF's nominal page width
            # (827 points for one book, which comes back as 500 px), but the
            # thumbnail server renders the page at any standard width.
            url = re.sub(r"/page(\d+)-\d+px-", rf"/page\1-{PDF_WIDTH}px-", url)
        entry["image_url"] = url.split("?")[0]
    data = get(entry["image_url"])
    if not data[:3] == b"\xff\xd8\xff" and not data[:8] == b"\x89PNG\r\n\x1a\n":
        print(f"  {entry['title']}: not an image, skipped")
        return
    target.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-only", action="store_true", help="only list the pages")
    parser.add_argument("--limit", type=int, default=0, help="fetch at most this many scans (for a trial run)")
    args = parser.parse_args()

    (DEST / "wikitext").mkdir(parents=True, exist_ok=True)
    (DEST / "images").mkdir(parents=True, exist_ok=True)
    index_path = DEST / "pages.json"
    entries = json.loads(index_path.read_text()) if index_path.exists() else []
    if not entries:
        for title in list_pages():
            file, number = split_title(title)
            if file in SKIP_FILES:
                continue
            name = local_name(file, number)
            entries.append({"title": title, "file": file, "page": number, "wikitext": f"wikitext/{name}.txt", "image": f"images/{name}.jpg"})
        index_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["file"]] = counts.get(entry["file"], 0) + 1
    for file, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"{count:4d}  {file}")
    if args.list_only:
        return

    fetch_file_metadata(sorted(counts))
    fetch_wikitext(entries)
    index_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    done = 0
    for number, entry in enumerate(entries):
        if args.limit and done >= args.limit:
            break
        if (DEST / entry["image"]).exists():
            continue
        try:
            fetch_image(entry)
        except Exception as error:  # keep going; a later run tries again
            print(f"  {entry['title']}: {error}")
        done += 1
        if done % 10 == 0:
            index_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
            print(f"scans {number + 1}/{len(entries)}", flush=True)
    index_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    have = sum((DEST / entry["image"]).exists() for entry in entries)
    print(f"{have}/{len(entries)} scans on disk")


if __name__ == "__main__":
    main()
