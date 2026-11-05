"""Test helpers: truth from designer SVGs and synthetic photos rendered from them."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

from rosette.geometry import tails as tail_geometry
from rosette.roles import render_categories
from tools.truth import PINKED_DEPTH, svg_text, truth_from_svg  # noqa: F401 (re-exported)

REF = Path(__file__).resolve().parent.parent / "reference"
# Well separated category colors for synthetic photos.
SYNTH_COLORS = {1: "#d0312d", 2: "#2a3fa0", 3: "#f2c12e", 4: "#2e8b57"}


@lru_cache(maxsize=None)
def synthetic_photo(sid: str, height: int = 1150) -> bytes:
    """PNG of the designer SVG's solid configuration in category colors, no outlines."""
    img = render_categories(svg_text(sid), height, SYNTH_COLORS)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


FAN_COLORS = ("#d0312d", "#2a3fa0", "#f2c12e")  # stripe colors, left to right
FAN_HEAD = "#2e8b57"


@lru_cache(maxsize=None)
def fan_rosette_png(count: int, spread: float, top_y: float, top_width: float,
                    bottom_width: float, stripes: tuple, height: int = 1150) -> bytes:
    """PNG of a constructed rosette: a plain head disc (top at y=25) over `count`
    tails drawn by geometry.tails (bottoms at y=825), each stripe its own color.
    The viewBox is widened so wide fans stay inside the image."""
    import resvg_py

    head_cx, head_cy, head_r = 300.0, 175.0, 150.0
    polys = tail_geometry(top_y, 825.0, count, top_width, bottom_width, spread, list(stripes))
    body = []
    for row in polys:
        for i, shape in enumerate(row):
            body.append(f'<polygon points="{shape.attrs["points"]}" fill="{FAN_COLORS[i % 3]}"/>')
    body.append(f'<circle cx="{head_cx}" cy="{head_cy}" r="{head_r}" fill="{FAN_HEAD}"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-300 -10 1200 870">'
           + "".join(body) + "</svg>")
    png = resvg_py.svg_to_bytes(svg_string=svg, height=height, background="#ffffff")
    return bytes(png)
