"""Classify the color elements of a designer SVG by rosette part role.

classify_roles(svg) -> [(element, role)] for every color element (class stN_1..4)
shown in the solid configuration, i.e. outside `Layer_3` (whose elements are the
pattern twins of `Layer_2_1` ones) when the file has both toggle groups; old-layout
files without `Layer_2_1` show everything, as the shop does (see `pattern_layer`). Roles: head, ring, points, center, tail_stripe,
unknown. Used for truth in the part-detection tests and for mining color rules.

Each element is rendered alone (resvg), split into connected pieces and judged by its shape around the head
circle, which `prepare` localizes on the whole rasterized file:
- mostly below the head -> tail_stripe;
- reaching the head edge with a closed outer band -> head (cut-outs inside allowed);
- angular coverage partial with a few lobes, on its outer side -> points;
- otherwise full coverage: a disc around the middle -> center, an annulus -> ring;
- anything else (e.g. wholly outside the head) -> unknown.
"""
from __future__ import annotations

import copy
import io
import re
from dataclasses import dataclass

import numpy as np
import resvg_py
from lxml import etree
from PIL import Image
from scipy import ndimage

from rosette.vision.prepare import Prepared, prepare

# Rows of the render used for classification.
RENDER_H = 900
# White margin (share of the render height) added around renders so a design
# touching its viewBox still has a clean background border for `prepare`.
PAD_FRAC = 0.04
# An element with at least this share of its pixels below the head is a tail stripe.
TAIL_SHARE = 0.5
# Angular coverage at or above which a radius counts as fully covered.
FULL_COV = 0.85
# A radius is 'lobed' when coverage is partial and splits into 2..MAX_LOBES runs
# (pinked teeth give many more runs than any points ring).
MAX_LOBES = 30
# Share of an element's radial extent that must be lobed for it to be a points ring.
LOBED_SHARE = 0.35
# A fully covered element starting within this share of the head radius contains
# the center; reaching past HEAD_EDGE it is the head disc.
CENTER_RMIN, HEAD_EDGE = 0.1, 0.9
# Pieces smaller than this (canvas units^2) are rendering specks, not parts.
MIN_COMPONENT_UNITS2 = 20.0
# A shape lying wholly beyond this many head radii is not part of the head.
OUTSIDE = 1.05
# A piece filling this share of the head disc and as much again below it is split.
SPLIT_HEAD_SHARE = 0.1
# Angular samples per radius and radial samples per element.
N_THETA, N_RADII = 720, 48

_COLOR = re.compile(r"^st\d+_([1-4])(?:_pattern)?$")
_HIDE = {"st4", "st6", "st_circel"}


@dataclass
class Frame:
    """Render geometry shared by all elements of one file (pixels of the padded render)."""
    head: tuple[float, float, float]
    tails_top_y: float
    bottom_y: float
    scale: float  # render px per canvas unit
    prepared: Prepared


def _local(el) -> str:
    return etree.QName(el).localname


def _first_by_id(root, ident: str):
    """First descendant with this id, like `svgImg.querySelector('#id')` (root excluded)."""
    for el in root.iterdescendants():
        if isinstance(el.tag, str) and el.get("id") == ident:
            return el
    return None


def pattern_layer(root):
    """The group hidden in the solid configuration, or None.

    The shop toggles `#Layer_3` (pattern) against `#Layer_2_1` (solid) only when
    both exist; old-layout designer files (e.g. Layer_1 nested inside Layer_3, or a
    top-level Layer_3 holding the only copy of the Color 3 parts) have no
    `Layer_2_1`, so nothing is hidden and every color element shows."""
    pl, sl = _first_by_id(root, "Layer_3"), _first_by_id(root, "Layer_2_1")
    return pl if pl is not None and sl is not None else None


def _in_layer3(el, hidden) -> bool:
    return hidden is not None and any(a is hidden for a in el.iterancestors())


def _color_elements(root) -> list:
    out = []
    hidden = pattern_layer(root)
    for el in root.iter():
        if not isinstance(el.tag, str) or _local(el) == "g" or _in_layer3(el, hidden):
            continue
        if any(_COLOR.match(c) for c in (el.get("class") or "").split()):
            out.append(el)
    return out


def category(el) -> int:
    """Color category (1-4) of a color element."""
    for c in (el.get("class") or "").split():
        m = _COLOR.match(c)
        if m:
            return int(m.group(1))
    raise ValueError("not a color element")


def _render(root, height: int) -> np.ndarray:
    png = resvg_py.svg_to_bytes(svg_string=etree.tostring(root, encoding="unicode"),
                                height=int(height), background="#ffffff")
    img = np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB"))
    pad = int(round(PAD_FRAC * height))
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)


def _style(el, extra: str) -> None:
    old = el.get("style", "").strip().rstrip(";")
    el.set("style", f"{old};{extra}" if old else extra)


def render_categories(svg: str, height: int, colors: dict[int, str]) -> np.ndarray:
    """Solid configuration with every color element filled by its category color,
    strokes and the pattern layer hidden (`pattern_layer`), padded with a white margin
    (PAD_FRAC)."""
    root = etree.fromstring(svg.encode("utf-8"), etree.XMLParser(remove_comments=True, huge_tree=True))
    hidden = pattern_layer(root)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el is hidden:
            _style(el, "display:none")
            continue
        cls = (el.get("class") or "").split()
        if any(_COLOR.match(c) for c in cls) and _local(el) != "g":
            _style(el, f"fill:{colors[category(el)]};fill-opacity:1;stroke:none")
        elif any(c in _HIDE for c in cls):
            _style(el, "display:none")
    return _render(root, height)


def _png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def frame_for(svg: str, height: int = RENDER_H) -> Frame:
    img = render_categories(svg, height, {1: "#ff0000", 2: "#0000ff", 3: "#00c000", 4: "#ff00ff"})
    p = prepare(_png(img))
    k = img.shape[0] / p.rgb.shape[0]  # prepare works at its own height
    cx, cy, r = p.head
    return Frame(head=(cx * k, cy * k, r * k), tails_top_y=p.tails_top_y * k,
                 bottom_y=p.bottom_y * k, scale=p.scale * k, prepared=p)


def element_masks(svg: str, height: int = RENDER_H) -> list[tuple[object, np.ndarray]]:
    """(element, bool mask) per solid-configuration color element, each rendered alone
    in the padded frame of `render_categories`."""
    root = etree.fromstring(svg.encode("utf-8"), etree.XMLParser(remove_comments=True, huge_tree=True))
    els = _color_elements(root)
    out = []
    for i, el in enumerate(els):
        r2 = copy.deepcopy(root)
        hidden = pattern_layer(r2)
        for e in r2.iter():
            if not isinstance(e.tag, str):
                continue
            if e is hidden:
                _style(e, "display:none")
            elif _local(e) != "g" and (e.get("class") or "").split():
                _style(e, "display:none")
        target = _color_elements(r2)[i]
        _style(target, "display:inline;fill:#000000;fill-opacity:1;stroke:none")
        out.append((el, _render(r2, height).min(axis=2) < 128))
    return out


def polar_mask(mask: np.ndarray, cx: float, cy: float, radii: np.ndarray,
               n_theta: int = N_THETA) -> np.ndarray:
    """(len(radii), n_theta) samples of mask on circles; theta from +x, clockwise on screen."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    xs = np.rint(cx + radii[:, None] * np.cos(th)[None]).astype(int)
    ys = np.rint(cy + radii[:, None] * np.sin(th)[None]).astype(int)
    h, w = mask.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    out = np.zeros(xs.shape, mask.dtype if mask.dtype != bool else bool)
    if mask.dtype != bool:
        out[:] = -1
    out[ok] = mask[ys[ok], xs[ok]]
    return out


def circular_runs(row: np.ndarray) -> int:
    """Number of True runs on a circular boolean row."""
    if row.all():
        return 1
    return int(np.count_nonzero(row & ~np.roll(row, 1)))


def _classify(mask: np.ndarray, f: Frame) -> str:
    """Role of one connected shape (bool mask) in the frame `f`."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return "unknown"
    if (ys >= f.tails_top_y).mean() >= TAIL_SHARE:
        return "tail_stripe"
    cx, cy, r = f.head
    d = np.hypot(xs - cx, ys - cy)
    rmin, rmax = float(d.min()), float(d.max())
    if rmin > r * OUTSIDE:
        return "unknown"
    radii = np.linspace(rmin, rmax, N_RADII + 2)[1:-1]
    pol = polar_mask(mask, cx, cy, radii)
    cov = pol.mean(axis=1)
    runs = np.array([circular_runs(row) for row in pol])
    full = cov >= FULL_COV
    lobed = (~full) & (cov > 0) & (runs <= MAX_LOBES)
    outer = slice(N_RADII // 2, None)
    if rmax >= HEAD_EDGE * r and full[outer].any():
        return "head"  # disc or annulus whose outer band is closed (cut-outs inside allowed)
    if lobed.mean() >= LOBED_SHARE and lobed[outer].mean() >= 0.5:
        return "points"  # lobes on the outer side: tips pointing outward
    if full.any():
        return "center" if rmin <= CENTER_RMIN * r else "ring"
    return "unknown"


def _components(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return [lab == i + 1 for i in range(n) if sizes[i] >= min_px]


def _split_head(comp: np.ndarray, f: Frame) -> list[np.ndarray]:
    """A piece covering a large part of the head disc and also running down the tails
    (one path for head disc + tail) is cut at the head circle into two pieces."""
    cx, cy, r = f.head
    h, w = comp.shape
    yy, xx = np.ogrid[:h, :w]
    inside = comp & ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & (yy < f.tails_top_y)
    rest = comp & ~inside
    big = SPLIT_HEAD_SHARE * np.pi * r * r
    if inside.sum() >= big and rest.sum() >= big:
        return [inside, rest]
    return [comp]


def classify_components(svg: str) -> tuple[Frame, list[tuple[object, str, np.ndarray]]]:
    """(frame, [(element, role, component mask)]) - designers often merge parts into
    one path (e.g. the head disc and the tails), so each connected piece of an
    element is judged on its own."""
    f = frame_for(svg)
    min_px = max(1, int(MIN_COMPONENT_UNITS2 * f.scale ** 2))
    out = []
    for el, m in element_masks(svg):
        for comp in _components(m, min_px):
            for piece in _split_head(comp, f):
                out.append((el, _classify(piece, f), piece))
    return f, out


def classify_roles(svg: str) -> list[tuple[object, str]]:
    """[(lxml element, role)] for every solid-configuration color element; an element
    made of several pieces takes the role of its largest piece."""
    _, items = classify_components(svg)
    best: dict[int, tuple[int, object, str]] = {}
    order = []
    for el, role, comp in items:
        k = id(el)
        if k not in best:
            order.append(k)
        area = int(comp.sum())
        if k not in best or area > best[k][0]:
            best[k] = (area, el, role)
    return [(best[k][1], best[k][2]) for k in order]
