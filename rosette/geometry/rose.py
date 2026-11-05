from __future__ import annotations

import math
import re
from pathlib import Path

from lxml import etree

from ..shapes import Shape, fmt

ROSE_SVG = Path(__file__).resolve().parents[2] / "reference" / "rose.svg"
_NUM = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")
_ARGS = {"m": 2, "l": 2, "h": 1, "v": 1, "c": 6, "s": 4, "q": 4, "t": 2, "a": 7, "z": 0}


def _path_points(d: str) -> list[tuple[float, float]]:
    """Absolute coordinates of every endpoint/control point of a path."""
    pts: list[tuple[float, float]] = []
    cx = cy = 0.0
    for cmd, args in re.findall(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)", d):
        low = cmd.lower()
        n = _ARGS[low]
        if n == 0:
            continue
        rel = cmd.islower()
        nums = [float(x) for x in _NUM.findall(args)]
        for i in range(0, len(nums) - n + 1, n):
            a = nums[i:i + n]
            if low == "h":
                cx = cx + a[0] if rel else a[0]
                pts.append((cx, cy))
            elif low == "v":
                cy = cy + a[0] if rel else a[0]
                pts.append((cx, cy))
            elif low == "a":
                cx, cy = (cx + a[5], cy + a[6]) if rel else (a[5], a[6])
                pts.append((cx, cy))
            else:
                ox, oy = (cx, cy) if rel else (0.0, 0.0)
                for j in range(0, n, 2):
                    pts.append((ox + a[j], oy + a[j + 1]))
                cx, cy = ox + a[n - 2], oy + a[n - 1]
    return pts


def _elements():
    root = etree.parse(str(ROSE_SVG)).getroot()
    return [e for e in root.iter() if isinstance(e.tag, str)
            and etree.QName(e).localname in ("path", "polyline", "polygon")]


def place_rose(cx: float, cy: float, r: float) -> list[Shape]:
    """Rose elements with a matrix transform mapping the rose's bbox center to
    (cx, cy) and its bbox half-diagonal to r. Original d/points are kept verbatim."""
    els = _elements()
    pts: list[tuple[float, float]] = []
    for e in els:
        if "d" in e.attrib:
            pts += _path_points(e.get("d"))
        else:
            v = [float(x) for x in _NUM.findall(e.get("points", ""))]
            pts += list(zip(v[0::2], v[1::2]))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bx, by = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2
    s = r / half_diag
    tx, ty = cx - s * bx, cy - s * by
    transform = f"matrix({s:.6f} 0 0 {s:.6f} {fmt(tx)} {fmt(ty)})"
    out = []
    for e in els:
        tag = etree.QName(e).localname
        attrs = {"d": e.get("d")} if tag == "path" else {"points": e.get("points")}
        attrs["transform"] = transform
        out.append(Shape(tag, attrs, None, kind="rose"))
    return out
