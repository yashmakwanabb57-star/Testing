"""R43: a head of stacked stars (S17-15): a back star with an edge, a star of another color inset
on it, points of a third color between its tips and a top star of the edge's color under the
center. Read along rays from the center: on the tip rays the top star's color runs out to its tip
and the inset color follows; on the rays between the tips the top star's color stops at its valley
and the third color follows. The hidden valleys take the designer's proportions."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..model import Part
from .layer_classify import KEEP_PARAM
from .parts_common import Frame

STACK_COUNTS = (8, 10, 12)  # 6-point heads are loops and petals (S26-15, S70-19 ...)
STACK_AGREE = 0.8  # share of tip (or valley) rays that show the same next color
STACK_MIN_SWING = 0.3  # the between color's share of the band swings this much around the head
STACK_BACK_VALLEY = 0.68  # back star valley / tip (designer S17-15)
STACK_BETWEEN_VALLEY = 0.62  # the between points' valley / tip, under the top star (designer S17-15)
STACK_TOP_VALLEY = 0.59  # the top star's valley / tip at most (designer S17-15: 0.42 / 0.716)
STACK_DIAG_VALLEY = 0.49  # 4 diagonal points on it (designer 0.35 / 0.716)
STACK_SMALL_TIP = 0.6  # a small 4-point star on the axes over them (designer 0.43 / 0.716)
STACK_ROSE = 0.27  # the center / the back star's tip at most (designer S17-15: 0.26)
STACK_EDGE = 6.0  # the back star's edge, canvas units (designer S17-15)
STACK_VIS_DEG = 45.0  # rays within this of straight down cross the tails


def _runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end) runs of a ray, end exclusive; -1 is background."""
    out, s = [], 0
    for i in range(1, len(row) + 1):
        if i == len(row) or row[i] != row[s]:
            out.append((int(row[s]), s, i))
            s = i
    return out


def _clean(runs: list, min_len: int = 3) -> list:
    """Short runs (glitter grain, edges) join their neighbors."""
    keep = [r for r in runs if r[2] - r[1] >= min_len or r[0] < 0]
    merged: list = []
    for r in keep:
        if merged and merged[-1][0] == r[0]:
            merged[-1] = (r[0], merged[-1][1], r[2])
        else:
            merged.append(r)
    return merged


def stacked_stars(p, labels: np.ndarray, parts: list) -> list | None:
    c = next((q for q in parts if q.role == "center"), None)
    if c is None or any(q.role == "points" and q.params.get("tips") for q in parts):
        return None
    fr = Frame.of(p)
    h, w = labels.shape
    hx, hy = int(round(p.head[0])), int(round(p.head[1]))
    g = int(labels[hy, hx])
    if g < 0 or g != c.color_group:
        return None
    lab, _ = ndimage.label(labels == g)
    blob = lab == lab[hy, hx]
    ys, xs = np.nonzero(blob)
    cx, cy = float(xs.mean()), float(ys.mean())
    head_u = fr.d(p.head[2])
    rc = float(np.sqrt(blob.sum() / np.pi)) / fr.scale
    if rc > 0.45 * head_u:
        return None
    n_th = 720
    th = np.arange(n_th) * 2 * np.pi / n_th  # from straight up, clockwise
    radii = np.arange(0.0, 1.15 * head_u, 1.0)
    X = np.clip(np.rint(cx + np.outer(np.sin(th), radii * fr.scale)).astype(int), 0, w - 1)
    Y = np.clip(np.rint(cy - np.outer(np.cos(th), radii * fr.scale)).astype(int), 0, h - 1)
    L = labels[Y, X]
    vis = np.abs(np.degrees(th) - 180.0) > STACK_VIS_DEG
    rays = []
    for i in range(n_th):
        if not vis[i]:
            rays.append(None)
            continue
        rr = _clean(_runs(L[i]))
        if len(rr) < 3 or rr[0][0] != g or rr[1][0] < 0 or rr[1][0] == g:
            rays.append(None)
            continue
        rays.append(rr)
    ok = np.array([r is not None for r in rays])
    if ok.sum() < 0.5 * vis.sum():
        return None
    top = np.bincount([r[1][0] for r in rays if r is not None]).argmax()
    r_top = np.array([float(r[1][2]) if r is not None and r[1][0] == top else np.nan for r in rays])
    good = ~np.isnan(r_top)
    # the count and phase from the points between the top star's tips: their color comes and
    # goes around a band just outside the top star's valleys (least squares on the visible rays)
    band = (radii > rc + 0.1 * head_u) & (radii < 0.75 * head_u)
    best = None
    for col in np.unique(L[np.ix_(vis, band)]):
        if col < 0 or col in (g, top):
            continue
        ind = (L[:, band] == col).mean(axis=1)[vis]
        for n in STACK_COUNTS:
            A = np.stack([np.ones(vis.sum()), np.cos(n * th[vis]), np.sin(n * th[vis])], axis=1)
            coef, *_ = np.linalg.lstsq(A, ind, rcond=None)
            amp = 2 * float(np.hypot(coef[1], coef[2]))
            if best is None or amp > best[1]:
                peak = (np.degrees(np.arctan2(coef[2], coef[1])) / n) % (360.0 / n)
                best = (n, amp, (peak + 180.0 / n) % (360.0 / n), int(col))
    if best is None:
        return None
    n, amp, phase, _ = best
    if amp < STACK_MIN_SWING:
        return None
    step = 360.0 / n
    deg = np.degrees(th)
    off = (deg - phase + step / 2) % step - step / 2  # 0 on a tip ray
    tip = good & (np.abs(off) <= step * 0.08)
    val = good & (np.abs(np.abs(off) - step / 2) <= step * 0.08)
    if tip.sum() < 6 or val.sum() < 6:
        return None

    def nxt(mask):
        cols = [rays[i][2][0] for i in np.flatnonzero(mask)]
        cols = [x for x in cols if x >= 0]
        if not cols:
            return None, 0.0
        b = np.bincount(cols)
        return int(b.argmax()), float(b.max() / len(cols))

    inset, a1 = nxt(tip)
    between, a2 = nxt(val)
    if inset is None or between is None or a1 < STACK_AGREE or a2 < STACK_AGREE:
        return None
    if len({int(top), inset, between, g}) < 3 or inset == between:
        return None
    # the photo's slant pushes the far side's valleys out: the near quarter reads them
    top_in = float(np.percentile(r_top[val], 25))
    top_out = float(np.median([r_top[i] for i in np.flatnonzero(tip) if rays[i][2][0] == inset]))
    btw_out = float(np.median([rays[i][2][2] for i in np.flatnonzero(val) if rays[i][2][0] == between]))
    outline = float(np.median([max(e for lb, s, e in rays[i] if lb >= 0) for i in np.flatnonzero(tip)]))
    if not (top_in < top_out < outline and top_in < btw_out < outline):
        return None
    edge = sum(1 for i in np.flatnonzero(tip) if [r for r in rays[i] if r[0] >= 0][-1][0] == top)
    top_in = min(top_in, STACK_TOP_VALLEY * top_out)
    rc = min(rc, STACK_ROSE * outline)  # the rose stands out of the photo: its foot is the center
    ccx, ccy = round(fr.x(p.head[0]), 3), round(fr.y(cy), 3)  # on the axis, as the designer
    rot = round((phase - 90.0) % step, 3)  # parts angles run from +x
    star = {"cx": ccx, "cy": ccy, "kind": "star", "count": int(n), "width_ratio": 0.5, "tips": 1}
    back_out = round(outline, 3)
    back_in = round(STACK_BACK_VALLEY * outline, 3)
    out = [q for q in parts if q.role not in ("points", "ring", "unknown", "center")]
    if edge >= 0.2 * tip.sum():  # the back star's edge in the top star's color
        out.append(Part("points-back", "points", dict(star, r_inner=back_in, r_outer=back_out, rotation_deg=rot),
                        int(top), 18, 0.5))
        out.append(Part("points-inset", "points", dict(star, r_inner=round(back_in - 3.0, 3),
                                                       r_outer=round(back_out - STACK_EDGE, 3), rotation_deg=rot),
                        inset, 19, 0.5))
    else:
        out.append(Part("points-inset", "points", dict(star, r_inner=back_in, r_outer=back_out, rotation_deg=rot),
                        inset, 19, 0.5))
    out.append(Part("points-between", "points", dict(star, r_inner=round(STACK_BETWEEN_VALLEY * btw_out, 3),
                                                     r_outer=round(btw_out, 3), rotation_deg=round(rot + step / 2, 3)),
                    between, 20, 0.5))
    out.append(Part("points-top", "points", dict(star, r_inner=round(top_in, 3), r_outer=round(top_out, 3),
                                                 rotation_deg=rot), int(top), 21, 0.5))
    if n == 8:  # the top star is three layers of the top color, each drawn with its own edge
        four = dict(star, count=4)
        out.append(Part("points-top-diag", "points", dict(four, r_inner=round(STACK_DIAG_VALLEY * top_out, 3),
                                                          r_outer=round(top_out, 3),
                                                          rotation_deg=round((rot + 45.0) % 90.0, 3)),
                        int(top), 22, 0.5))
        out.append(Part("points-top-small", "points", dict(four, r_inner=round(rc, 3),
                                                           r_outer=round(STACK_SMALL_TIP * top_out, 3),
                                                           rotation_deg=round(rot % 90.0, 3)),
                        int(top), 23, 0.5))
    out.append(Part("center", "center", {"cx": ccx, "cy": ccy, "r": round(rc, 3)}, g, 50, c.confidence))
    for q in out:  # built whole here: the app's layer classifier keeps their kind and count (S17-15)
        if q.role == "points" and q.id.startswith("points-"):
            q.params[KEEP_PARAM] = 1
    return out


# R46: a star between loops on a head disc (S26-15: a pink star edged in wine between six glitter
# loops on pink pleats). The edge color comes and goes around the head half a step from the loops;
# the star is drawn the designer's way, a star of the edge color under a star of the head's color,
# both standing on the ring around the center, the loops under them. Radii are the designer's
# (S26-15, in head radii): the edge star 0.43-0.92, the top star 0.40-0.84.
EDGE_STAR_SWING = 0.3
EDGE_STAR_PHASE_TOL = 8.0  # degrees off half a step from the loops
EDGE_STAR = (0.43, 0.92)
TOP_STAR = (0.40, 0.84)
EDGE_BOW = (0.43, 0.86, 0.40, 0.2)  # the loops: inner and outer radius, width (head radii), edge share; they end inside the pinking


def edged_star_between_bows(p, labels: np.ndarray, parts: list) -> list | None:
    head = next((q for q in parts if q.role == "head"), None)
    c = next((q for q in parts if q.role == "center"), None)
    bows = [q for q in parts if q.role == "points" and q.params.get("kind") == "bow"]
    if head is None or c is None or len(bows) != 1 or any(
            q.role == "points" and q.params.get("kind") != "bow" for q in parts):
        return None
    bow = bows[0]
    n = int(bow.params["count"])
    fr = Frame.of(p)
    h, w = labels.shape
    head_u = float(head.params["r"])
    th = np.arange(720) * 2 * np.pi / 720  # from straight up, clockwise
    radii = np.arange(0.45 * head_u, 0.95 * head_u, 1.0)
    X = np.clip(np.rint(p.head[0] + np.outer(np.sin(th), radii * fr.scale)).astype(int), 0, w - 1)
    Y = np.clip(np.rint(p.head[1] - np.outer(np.cos(th), radii * fr.scale)).astype(int), 0, h - 1)
    L = labels[Y, X]
    vis = np.abs(np.degrees(th) - 180.0) > STACK_VIS_DEG
    A = np.stack([np.ones(vis.sum()), np.cos(n * th[vis]), np.sin(n * th[vis])], axis=1)

    def fit(col):
        coef, *_ = np.linalg.lstsq(A, (L == col).mean(axis=1)[vis], rcond=None)
        return 2 * float(np.hypot(coef[1], coef[2])), (np.degrees(np.arctan2(coef[2], coef[1])) / n) % (360.0 / n)

    step = 360.0 / n
    _, bow_peak = fit(bow.color_group)
    best = None
    for col in np.unique(L[vis]):
        if col < 0 or col in (head.color_group, bow.color_group):
            continue
        amp, peak = fit(col)
        off = (peak - bow_peak) % step
        if amp >= EDGE_STAR_SWING and abs(off - step / 2) <= EDGE_STAR_PHASE_TOL and (best is None or amp > best[0]):
            best = (amp, int(col))
    if best is None:
        return None
    edge = best[1]
    rot = round(((bow_peak + step / 2) - 90.0) % step, 3)  # tips between the loops; parts angles from +x
    star = {"cx": head.params["cx"], "cy": head.params["cy"], "kind": "star", "count": n, "width_ratio": 0.5,
            "rotation_deg": rot, "tips": 1}
    out = list(parts)
    # the loops fill the room between the tips, under the star (designer S26-15: 0.43-0.97, 0.5 wide)
    ri, ro = EDGE_BOW[0] * head_u, EDGE_BOW[1] * head_u
    bow.params.update(r_inner=round(ri, 3), r_outer=round(ro, 3),
                      width_ratio=round(EDGE_BOW[2] * head_u / (ro - ri), 4))
    # each loop edged left and right in the head's color (designer S26-15)
    out.append(Part(f"edge-{bow.id}", "points", dict(bow.params, edge_width=round(EDGE_BOW[3] * EDGE_BOW[2] * head_u, 3)),
                    head.color_group, bow.z, 0.5))
    out.append(Part("points-edge", "points", dict(star, r_inner=round(EDGE_STAR[0] * head_u, 3),
                                                  r_outer=round(EDGE_STAR[1] * head_u, 3)), edge, bow.z + 1, 0.5))
    out.append(Part("points-top", "points", dict(star, r_inner=round(TOP_STAR[0] * head_u, 3),
                                                 r_outer=round(TOP_STAR[1] * head_u, 3)),
                    head.color_group, bow.z + 2, 0.5))
    return out


# R48: four loops on a head disc with a star of a third color half a step away (S28-15: glitter
# loops on the diagonals of blue pleats, light blue points on the axes). The designer lays the
# star on top of the loops, and a 4-point star of the loops' color on the loops' axes (a square
# with drawn-in sides around the center), the loops edged in the head's color. Owner's check of
# S28-15: bottom to top the head, the edged loops, the star, the square; the loops run in to the
# center under the star's points and the square, as the glitter does in the photo.
# Radii are the designer's (S28-15, head radii) except the loops' inner end.
SQUARE_LOOPS = (0.45, 0.9, 0.42, 0.2)  # loops: inner, outer, width, edge share
SQUARE_STAR_TIP = 0.85  # the star's points
SQUARE_TIP = 0.69  # the square's corners (on the loops' axes)
SQUARE_SIDE = 0.33  # the middle of its drawn-in sides


def star_on_square(p, labels: np.ndarray, parts: list) -> list | None:
    head = next((q for q in parts if q.role == "head"), None)
    c = next((q for q in parts if q.role == "center"), None)
    pts = [q for q in parts if q.role == "points"]
    if head is None or c is None or len(pts) != 2 or any(q.role == "ring" for q in parts):
        return None
    bow = next((q for q in pts if q.params.get("kind") == "bow"), None)
    star = next((q for q in pts if q.params.get("kind") == "star"), None)
    if bow is None or star is None or int(bow.params["count"]) != 4 or int(star.params["count"]) != 4:
        return None
    if len({head.color_group, bow.color_group, star.color_group}) != 3:
        return None
    off = (float(star.params["rotation_deg"]) - float(bow.params["rotation_deg"])) % 90.0
    if abs(off - 45.0) > 10.0:
        return None
    hu, rc = float(head.params["r"]), float(c.params["r"])
    ri, ro = rc, SQUARE_LOOPS[1] * hu  # in to the center, under the star
    bow.params.update(r_inner=round(ri, 3), r_outer=round(ro, 3),
                      width_ratio=round(SQUARE_LOOPS[2] * hu / (ro - ri), 4))
    bow.z = 20
    star.params.update(r_inner=round(rc, 3), r_outer=round(SQUARE_STAR_TIP * hu, 3),
                       rotation_deg=round((float(bow.params["rotation_deg"]) + 45.0) % 90.0, 3), tips=1)
    star.z = 22  # over the loops and their edges, under the square (23)
    out = list(parts)
    out.append(Part(f"edge-{bow.id}", "points", dict(bow.params, edge_width=round(SQUARE_LOOPS[3] * SQUARE_LOOPS[2] * hu, 3)),
                    head.color_group, 21, 0.5))
    out.append(Part("points-square", "points", {"cx": bow.params["cx"], "cy": bow.params["cy"], "kind": "star", "count": 4,
                                                "r_inner": round(max(rc, SQUARE_SIDE * hu), 3),
                                                "r_outer": round(SQUARE_TIP * hu, 3),
                                                "rotation_deg": bow.params["rotation_deg"], "width_ratio": 0.5, "tips": 1},
                    bow.color_group, 23, 0.5))
    for q in (bow, star, out[-2], out[-1]):  # built whole here: the layer classifier keeps them
        q.params[KEEP_PARAM] = 1
    return out
