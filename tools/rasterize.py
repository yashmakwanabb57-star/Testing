"""Category-colored rasterization of designer SVGs (shared by tests and scoring)."""
from __future__ import annotations

import io
import re

import numpy as np
import resvg_py
from lxml import etree
from PIL import Image

from rosette.roles import pattern_layer

# Default fill per color category (1-4) when the caller gives none.
DEFAULT_PALETTE = {1: "#c0392b", 2: "#2c3e9e", 3: "#e8b923", 4: "#2e8b57"}

# Color classes the shop fills: stN_1, stN_2, stN_2_pattern, stN_3, stN_4.
_COLOR_CLASS = re.compile(r"^st\d+_([1-4])(?:_pattern)?$")
# Stroke-only classes hidden when scoring wants pure color regions.
_STROKE_CLASSES = {"st4", "st6", "st_circel"}


def _style(el, extra: str) -> None:
    old = el.get("style", "").strip().rstrip(";")
    el.set("style", f"{old};{extra}" if old else extra)


def rasterize(svg: str, height: int, category_colors: dict[int, str] | None) -> np.ndarray:
    """Render a designer SVG to an (H, W, 3) uint8 array on a white background.

    Color classes get their category color as an inline style (inline style
    beats the file's own `fill:none` rules), `#Layer_3` is hidden so the solid
    configuration shows (only when `#Layer_2_1` exists too, as in the shop: old-layout
    files without it show every group), and stroke classes are hidden only when explicit
    category colors are given.
    """
    palette = dict(DEFAULT_PALETTE)
    if category_colors:
        palette.update(category_colors)
    root = etree.fromstring(svg.encode("utf-8"), etree.XMLParser(remove_comments=True, huge_tree=True))
    hidden = pattern_layer(root)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el is hidden:
            _style(el, "display:none")
            continue
        tokens = (el.get("class") or "").split()
        for tok in tokens:
            m = _COLOR_CLASS.match(tok)
            if m:
                fill = palette[int(m.group(1))]
                _style(el, f"fill:{fill};fill-opacity:1" + (";stroke:none" if category_colors else ""))
                break
            if category_colors and tok in _STROKE_CLASSES:
                _style(el, "display:none")
                break
    png = resvg_py.svg_to_bytes(svg_string=etree.tostring(root, encoding="unicode"),
                                height=int(height), background="#ffffff")
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB"))
