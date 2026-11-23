#!/usr/bin/env python3
"""Draw the report's figures from results/comparison/comparison.json.

    python tools/make_report_figures.py

Writes light and dark SVG files to docs/report/. No packages are needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report"
THEMES = {
    "light": {"text": "#0b0b0b", "soft": "#52514e", "grid": "#e4e3df", "ours": "#2a78d6", "other": "#8f8e88", "surface": "#ffffff"},
    "dark": {"text": "#f0efec", "soft": "#c3c2b7", "grid": "#34343a", "ours": "#3987e5", "other": "#7d7c76", "surface": "#0d1117"},
}
MODELS = [  # key in the results, name, size in MB, licence in a few words
    ("open-board-scanner", "Open Board Scanner", 3.46, "noncommercial"),
    ("chessq-lite-v4-int8", "ChessQ Lite V4", 41.37, "noncommercial"),
    ("fenify", "Fenify", 127.15, "MIT"),
    ("fenshot", "Fenshot", 1.29, "MIT"),
]
FONT = 'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"'


def bar_end(x: float, y: float, width: float, height: float, radius: float = 4) -> str:
    """A bar that is square at the baseline and rounded at its data end."""
    radius = min(radius, width, height / 2)
    return f"M{x},{y} h{width - radius:.1f} a{radius},{radius} 0 0 1 {radius},{radius} v{height - 2 * radius:.1f} a{radius},{radius} 0 0 1 -{radius},{radius} h-{width - radius:.1f} z"


def bars(results: dict, theme: dict) -> str:
    panels = [("real_photos", "Real photos the model never trained on"), ("rendered_boards", "Clean boards in 33 Lichess piece sets")]
    left, track, row, top = 190, 380, 34, 58
    height = top + len(panels) * (len(MODELS) * row + 56)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 {height}" width="720" height="{height}" role="img" aria-label="Boards read exactly by each model on two tests">']
    parts.append(f'<text x="0" y="22" {FONT} font-size="17" font-weight="600" fill="{theme["text"]}">Boards read exactly, every square right</text>')
    parts.append(f'<text x="0" y="42" {FONT} font-size="13" fill="{theme["soft"]}">Each model was given the same board, already cut out and upright</text>')
    y = top
    for key, title in panels:
        total = results[key][MODELS[0][0]]["boards"]
        parts.append(f'<text x="0" y="{y + 22}" {FONT} font-size="14" font-weight="600" fill="{theme["text"]}">{title} ({total:,} boards)</text>')
        y += 36
        parts.append(f'<line x1="{left}" y1="{y - 4}" x2="{left}" y2="{y + len(MODELS) * row - 6}" stroke="{theme["grid"]}" stroke-width="1"/>')
        for name, label, _, _ in MODELS:
            score = results[key][name]
            share = score["exact"] / score["boards"]
            colour = theme["ours"] if name == MODELS[0][0] else theme["other"]
            weight = "600" if name == MODELS[0][0] else "400"
            parts.append(f'<text x="{left - 10}" y="{y + 15}" text-anchor="end" {FONT} font-size="13" font-weight="{weight}" fill="{theme["text"]}">{label}</text>')
            parts.append(f'<path d="{bar_end(left, y, max(2.0, share * track), 20)}" fill="{colour}"/>')
            parts.append(f'<text x="{left + max(2.0, share * track) + 8:.1f}" y="{y + 15}" {FONT} font-size="13" fill="{theme["text"]}"><tspan font-weight="600">{share:.0%}</tspan><tspan dx="6" fill="{theme["soft"]}">{score["exact"]:,} of {score["boards"]:,}</tspan></text>')
            y += row
        y += 20
    parts.append("</svg>")
    return "\n".join(parts)


def size_against_accuracy(results: dict, theme: dict) -> str:
    left, right, top, bottom, width, height = 64, 700, 64, 300, 720, 350
    low, high = math.log10(1), math.log10(200)
    x_of = lambda megabytes: left + (math.log10(megabytes) - low) / (high - low) * (right - left)  # noqa: E731
    y_of = lambda share: bottom - share * (bottom - top)  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Model size against real photos read exactly">']
    parts.append(f'<text x="0" y="22" {FONT} font-size="17" font-weight="600" fill="{theme["text"]}">Small and accurate</text>')
    parts.append(f'<text x="0" y="42" {FONT} font-size="13" fill="{theme["soft"]}">Real photos read exactly, against the size of the model files</text>')
    for share in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" y1="{y_of(share):.1f}" x2="{right}" y2="{y_of(share):.1f}" stroke="{theme["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y_of(share) + 4:.1f}" text-anchor="end" {FONT} font-size="12" fill="{theme["soft"]}">{share:.0%}</text>')
    for megabytes in (1, 3, 10, 30, 100):
        parts.append(f'<text x="{x_of(megabytes):.1f}" y="{bottom + 20}" text-anchor="middle" {FONT} font-size="12" fill="{theme["soft"]}">{megabytes} MB</text>')
    parts.append(f'<text x="{right}" y="{bottom + 42}" text-anchor="end" {FONT} font-size="12" fill="{theme["soft"]}">size of the model files (log scale)</text>')
    for name, label, megabytes, _ in MODELS:
        score = results["real_photos"][name]
        share = score["exact"] / score["boards"]
        ours = name == MODELS[0][0]
        x, y = x_of(megabytes), y_of(share)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{8 if ours else 6}" fill="{theme["ours"] if ours else theme["other"]}" stroke="{theme["surface"]}" stroke-width="2"/>')
        # Labels sit where they cannot collide: under the two dots on the top line, beside the others.
        if share > 0.9:
            anchor, lx, ly = ("start", x - 8, y + 26) if ours else ("middle", x, y + 26)
        else:
            anchor, lx, ly = ("end", x - 18, y + 4) if megabytes > 60 else ("start", x + 14, y + 4)
        size = f"{megabytes:.1f}" if megabytes < 10 else f"{megabytes:.0f}"
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" {FONT} font-size="13" font-weight="{"600" if ours else "400"}" fill="{theme["text"]}">{label}<tspan dx="8" fill="{theme["soft"]}" font-weight="400">{size} MB</tspan></text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    results = json.loads((ROOT / "results" / "comparison" / "comparison.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        (OUT / f"exact-boards-{mode}.svg").write_text(bars(results, theme) + "\n")
        (OUT / f"size-accuracy-{mode}.svg").write_text(size_against_accuracy(results, theme) + "\n")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
