from pathlib import Path

import numpy as np

from tools.rasterize import DEFAULT_PALETTE, rasterize

SVG = Path(__file__).resolve().parent.parent / "reference" / "svg" / "S35-19.svg"


def _share(img, hex_color):
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (img == rgb).all(axis=2).mean()


def test_category_colors_fill_regions_and_hide_strokes():
    colors = {1: "#ff0000", 2: "#0000ff", 3: "#00ff00"}
    img = rasterize(SVG.read_text(encoding="utf-8"), 850, colors)
    assert img.shape == (850, 600, 3) and img.dtype == np.uint8
    for c in colors.values():
        assert _share(img, c) > 0.05
    assert (img[0, 0] == 255).all()  # white background
    # strokes hidden: no near-black pixels left
    assert (img.max(axis=2) < 60).mean() < 0.001


def test_default_palette_keeps_outlines():
    img = rasterize(SVG.read_text(encoding="utf-8"), 850, None)
    assert _share(img, DEFAULT_PALETTE[1]) > 0.05
    assert (img.max(axis=2) < 60).mean() > 0.001


def test_layer_3_hidden():
    svg = SVG.read_text(encoding="utf-8")
    shown = rasterize(svg.replace('id="Layer_3"', 'id="Layer_3x"'), 850, {1: "#ff0000", 2: "#0000ff", 3: "#00ff00"})
    hidden = rasterize(svg, 850, {1: "#ff0000", 2: "#0000ff", 3: "#00ff00"})
    assert (shown != hidden).any(axis=2).mean() > 0.002
