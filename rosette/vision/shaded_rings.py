"""R32: head pleat rings read from hue when shading mislabels them.

Shaded pleats lose their brightness, so the color groups (which weigh lightness too) give the
dark pleats in shadow the label of a lighter ribbon of the same family (A3-15: the dark blue
outer pleats read as the mid blue). Their hue stays: per head radius, the median a*b* of the
visible circle is matched to the used group of the nearest hue angle (saturated colors;
shade lowers chroma, not hue), else of the nearest a*b*. When that
hue profile disagrees with the head's labels on at least HUE_DISAGREE of the radii between the
center and the rim, the head disc and rings are rebuilt from it: the disc takes the profile's
color at the rim, and every run of another color at least HUE_MIN_RUN units long between the
center's edge and the rim becomes a pinked ring of that color.
"""
from __future__ import annotations

import cv2
import numpy as np

from rosette.model import ColorGroup, Part
from rosette.vision.parts_common import Frame, r3
from rosette.vision.prepare import Prepared

HUE_DISAGREE, HUE_MIN_RUN, HUE_SMOOTH = 0.3, 8.0, 5
HUE_CHROMA = 15.0  # a*b* chroma from which a color is matched by hue angle
HUE_CONF = 0.5  # flagged: the user confirms the rebuilt rings
HUE_INFO = "Head pleats in shade: the rings were read by hue - check their colors."


def _ab(rgb) -> np.ndarray:
    px = np.array([[rgb]], np.float32) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0, 1:]


def _nearest(ab: np.ndarray, cents: np.ndarray) -> int:
    """Index of the group color nearest to ab: by hue angle among saturated colors when ab is
    saturated (shade lowers chroma, not hue), else by a*b* distance."""
    chroma = np.hypot(cents[:, 0], cents[:, 1])
    if np.hypot(*ab) >= HUE_CHROMA and (chroma >= HUE_CHROMA).any():
        ang = np.degrees(np.arctan2(cents[:, 1], cents[:, 0]))
        d = np.abs((ang - np.degrees(np.arctan2(ab[1], ab[0])) + 180) % 360 - 180)
        d[chroma < HUE_CHROMA] = np.inf
        return int(np.argmin(d))
    return int(np.argmin(np.linalg.norm(cents - ab, axis=1)))


def hue_profile(p: Prepared, groups: list[ColorGroup], used: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """(radii in canvas units, nearest used group per radius by a*b*) over the head."""
    fr = Frame.of(p)
    lab = cv2.cvtColor(p.rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    cx, cy, r = p.head
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    vis = ~((th > np.radians(55)) & (th < np.radians(125)))
    h, w = p.mask.shape
    ids = [g.id for g in groups if g.id in used]
    cents = np.array([_ab(g.rgb) for g in groups if g.id in used])
    radii = np.arange(1.0, fr.d(r), 1.0)
    out = []
    for ru in radii:
        rp = ru * fr.scale
        xs = np.clip(np.rint(cx + rp * np.cos(th)).astype(int), 0, w - 1)
        ys = np.clip(np.rint(cy + rp * np.sin(th)).astype(int), 0, h - 1)
        ok = vis & p.mask[ys, xs]
        if not ok.any():
            out.append(-1)
            continue
        ab = np.median(lab[ys[ok], xs[ok], 1:], axis=0)
        out.append(ids[_nearest(ab, cents)])
    prof = np.array(out)
    k = HUE_SMOOTH // 2  # majority over a few radii
    sm = prof.copy()
    for i in range(len(prof)):
        win = prof[max(0, i - k):i + k + 1]
        win = win[win >= 0]
        if win.size:
            sm[i] = np.bincount(win).argmax()
    return radii, sm


def shaded_rings(p: Prepared, parts: list[Part], groups: list[ColorGroup]) -> list[Part] | None:
    """R32: the parts with the head disc and rings rebuilt from hue, or None (module doc)."""
    head = next((q for q in parts if q.role == "head"), None)
    if head is None or any(q.role == "points" for q in parts):
        return None
    used = {q.color_group for q in parts}
    if len(used) < 3:
        return None
    radii, prof = hue_profile(p, groups, used)
    c = next((q for q in parts if q.role == "center"), None)
    r0 = float(c.params["r"]) if c is not None else 0.0
    rim = float(head.params["r"])
    zone = (radii > r0) & (radii < rim)
    if zone.sum() < 10:
        return None
    # the labels' reading per radius: the ring covering it, else the head disc
    cur = np.full(len(radii), head.color_group)
    for q in parts:
        if q.role == "ring":
            cur[(radii >= float(q.params["r_inner"])) & (radii < float(q.params["r_outer"]))] = q.color_group
    if (cur[zone] != prof[zone]).mean() < HUE_DISAGREE:
        return None
    outer = radii > 0.85 * rim
    disc_g = int(np.bincount(prof[outer & (prof >= 0)]).argmax()) if (outer & (prof >= 0)).any() else head.color_group
    head_roles = ("head", "ring", "center", "unknown", "points")
    before = {q.color_group for q in parts if q.role in head_roles}
    old_head = (head.color_group, head.confidence)
    out = [q for q in parts if q.role not in ("ring", "unknown")]
    head.color_group, head.confidence = disc_g, min(head.confidence, HUE_CONF)
    i, k, n = 0, 0, len(radii)
    idx = np.flatnonzero(zone)
    while i < len(idx):
        j = i
        g = prof[idx[i]]
        while j + 1 < len(idx) and prof[idx[j + 1]] == g:
            j += 1
        a, b = float(radii[idx[i]]), float(radii[idx[j]]) + 1.0
        if g >= 0 and g != disc_g and b - a >= HUE_MIN_RUN:
            out.append(Part(f"ring-hue-{k}", "ring", {"cx": head.params["cx"], "cy": head.params["cy"],
                                                      "r_inner": r3(a), "r_outer": r3(b), "pinked": 1},
                            int(g), 30 + k, HUE_CONF))
            k += 1
        i = j + 1
    after = {q.color_group for q in out if q.role in head_roles}
    if len(after) < len(before):
        head.color_group, head.confidence = old_head
        return None  # the rebuild loses a color the head shows: not shaded pleats (A10-15B, S68-19)
    return out


# R34 (`dark_loop_edges`): loops edged with a darker ribbon of their own hue (S10-19: navy edges
# on blue loops, gold-threaded) read as plain loops; the edging is too thin to form a color
# group in the head. On the middle 40 % of the loops' radii the loop's photo L* at its side
# edges (the outer tenth of its width on each side) is at least LOOP_EDGE_DARK below its
# middle third (S10-19 8.2; over the 76 photos every other loop layer reads 6.8 or less), and a
# used group darker than the loops (L* LOOP_EDGE_DARK or more lower) lies within
# LOOP_EDGE_HUE of their a*b*: that group is drawn as the loops' edge strips (the designer's
# S10-19 edges, Color 1; parts_trims EDGE_SHARE of the loop's width).
LOOP_EDGE_DARK, LOOP_EDGE_HUE = 8.0, 30.0


def dark_loop_edges(p: Prepared, labels: np.ndarray, parts: list[Part], groups: list[ColorGroup]) -> list[Part] | None:
    """R34: the parts with dark edge strips added along loops edged in a darker ribbon, or None."""
    from rosette.vision.parts_common import polar_sample
    from rosette.vision.parts_trims import EDGE_SHARE
    L = cv2.cvtColor(p.rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[..., 0]
    cx, cy, _ = p.head
    s = p.scale
    t = 720
    win = np.abs(np.degrees(np.linspace(0, 2 * np.pi, t, endpoint=False)) - 90) > 35
    used = {q.color_group for q in parts}
    lab = {g.id: cv2.cvtColor(np.array([[g.rgb]], np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0, 0] for g in groups}
    out, added = list(parts), False
    for q in parts:
        a = q.params
        if q.role != "points" or a.get("kind") != "bow" or a.get("edge_width") is not None:
            continue
        g = q.color_group
        ri, ro = float(a["r_inner"]), float(a["r_outer"])
        radii = np.arange((ri + 0.3 * (ro - ri)) * s, (ri + 0.7 * (ro - ri)) * s, s)
        pl = polar_sample(labels.astype(np.int32), cx, cy, radii, t)
        pL = polar_sample(L, cx, cy, radii, t, fill=np.nan)
        e, c = [], []
        for row, rl in zip(pl, pL):
            runs = np.flatnonzero(np.diff(np.r_[0, (row == g).astype(int), 0])).reshape(-1, 2)
            for i0, i1 in runs:
                w = i1 - i0
                if w < 12 or not win[(i0 + i1) // 2 % t]:
                    continue
                k = max(2, w // 10)
                e += list(rl[i0:i0 + k]) + list(rl[i1 - k:i1])
                c += list(rl[i0 + w // 3:i1 - w // 3])
        if not e or np.nanmedian(c) - np.nanmedian(e) < LOOP_EDGE_DARK:
            continue
        dark = [h for h in used if h != g and h in lab and lab[h][0] <= lab[g][0] - LOOP_EDGE_DARK
                and np.hypot(*(lab[h][1:] - lab[g][1:])) <= LOOP_EDGE_HUE]
        if not dark:
            continue
        h = min(dark, key=lambda h: np.hypot(*(lab[h][1:] - lab[g][1:])))
        w = float(a["width_ratio"]) * (ro - ri)
        out.append(Part(f"edge-{q.id}", "points", dict(a, edge_width=r3(EDGE_SHARE * w)), h, q.z + 1,
                        min(q.confidence, HUE_CONF)))
        added = True
    return out if added else None
