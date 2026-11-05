"""Measure color grouping over the reference photos (provenance for the constants
in rosette/vision/colors.py).

Usage: .venv/Scripts/python.exe -m tools.color_report [photo stems...]

Prints one line per photo: chosen groups (rgb, P = pattern, area, texture) and the
smallest distance between cluster centers. Then the summary: highest plain-group
texture and lowest pattern-group texture (the PATTERN_T margin).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from rosette.vision import colors
from rosette.vision.prepare import prepare

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def main(stems: list[str]) -> None:
    paths = [PHOTOS / f"{s}.png" for s in stems] if stems else sorted(PHOTOS.glob("*.png"))
    seen: dict[str, float] = {}
    original = colors._cluster

    def spy(feat, max_groups):
        centers = original(feat, max_groups)
        seen["sep"] = colors._min_sep(centers) if len(centers) > 1 else float("inf")
        return centers

    colors._cluster = spy
    plain, pattern = [], []
    try:
        for path in paths:
            p = prepare(path.read_bytes())
            t = time.perf_counter()
            _, groups = colors.pixel_groups(p)
            dt = time.perf_counter() - t
            desc = " | ".join(f"{g.rgb}{'P' if g.pattern else ''} a={g.area:.2f} t={g.texture:.2f}"
                              for g in groups)
            print(f"{path.stem:9s} {dt:.2f}s sep={seen['sep']:.1f} k={len(groups)} | {desc}")
            for g in groups:
                (pattern if g.pattern else plain).append((g.texture, path.stem, g.rgb))
    finally:
        colors._cluster = original
    if plain:
        print("max plain texture:", max(plain))
    if pattern:
        print("min pattern texture:", min(pattern))
    print("PATTERN_T:", colors.PATTERN_T)


if __name__ == "__main__":
    main(sys.argv[1:])
