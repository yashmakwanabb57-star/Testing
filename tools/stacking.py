"""Stacking order of head shapes: which kind of shape is painted above which (H5).

Usage: .venv/Scripts/python.exe tools/stacking.py [IDs...]   (mines the reference SVGs)

Reference side (`reference_pairs`): every solid color element is rendered alone
(`rosette.roles.element_masks`, document order = paint order: Layer_1, then the Layer_3 /
Layer_2_1 groups); its classified pieces get a kind (`piece_kind`); every pair of pieces
that overlap by at least MIN_OVERLAP_UNITS2 records (kind below, kind above).

Our side (`model_pairs`): the build's color shapes before the occlusion cuts; an
overlapping pair is ordered the way the written file shows it: higher z on top
(`geometry.clip` cuts the lower one), equal z by the shop's layer order (higher Color
later), then document order.

Kinds: head, ring, center, tail, loop, kite, petal, star, edge (edge strip), backing (a
points piece of another color behind a points piece on the same axis that reaches further
out), unknown.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# Pieces overlapping by less than this (canvas units^2) do not state a stacking.
MIN_OVERLAP_UNITS2 = 20.0
# Polar sampling of a points piece around the head center (radial rows, angles).
N_R, N_T = 60, 720
# Kind rules on a single-lobe piece's lateral width profile p (t = 0..1 along the lobe,
# H4 mining, `docs/pipeline.md`): edge strip when width/length < STRIP_ASPECT and the
# profile stays >= 0.75 between t = 0.2 and 0.8; loop when p(0.9) >= 0.8; star tip
# (triangle) when the widest point is at t <= 0.2; petal when p(0.1) >= 0.55; else kite.
STRIP_ASPECT = 0.35
# Touching pieces: the reference files draw most stackings already cut (the lower piece's
# outline follows the upper one), so pieces that touch along >= MIN_CONTACT_UNITS (pixels
# within CONTACT_PX of both) without overlapping are ordered by which one was cut: inside
# a window of WINDOW_PX around the contact, the cut piece's convex hull takes in more of
# the other piece (it wraps around it) than the other's hull takes in of it, by HULL_MARGIN.
MIN_CONTACT_UNITS, CONTACT_PX, WINDOW_PX, HULL_MARGIN = 6.0, 2, 10, 0.1
# A points piece is a backing of another points piece of another color when their axes
# lie within BACK_AXIS_DEG and it reaches BACK_REACH_R head radii further out.
BACK_AXIS_DEG, BACK_REACH_R = 4.0, 0.02

ROLE_KIND = {"head": "head", "ring": "ring", "center": "center", "tail_stripe": "tail", "unknown": "unknown"}


def _polar(mask: np.ndarray, cx: float, cy: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linspace(0, 1.2 * r, N_R)
    th = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    xs = np.rint(cx + radii[:, None] * np.cos(th)[None]).astype(int)
    ys = np.rint(cy + radii[:, None] * np.sin(th)[None]).astype(int)
    h, w = mask.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    pol = np.zeros(xs.shape, bool)
    pol[ok] = mask[ys[ok], xs[ok]]
    return pol, radii


def _runs(row: np.ndarray) -> int:
    return 1 if row.all() else int(np.count_nonzero(row & ~np.roll(row, 1)))


def lobe_axis(mask: np.ndarray, head: tuple[float, float, float]) -> tuple[float, float]:
    """(axis angle in degrees, outer radius in head radii) of a piece."""
    cx, cy, r = head
    ys, xs = np.nonzero(mask)
    ang = float(np.degrees(np.angle(np.exp(1j * np.arctan2(ys - cy, xs - cx)).mean())) % 360)
    return ang, float(np.hypot(xs - cx, ys - cy).max() / r)


def piece_kind(mask: np.ndarray, role: str, head: tuple[float, float, float]) -> str:
    """Kind of one classified piece (see the module doc); backings are found pairwise."""
    if role != "points":
        return ROLE_KIND.get(role, "unknown")
    cx, cy, r = head
    pol, radii = _polar(mask, cx, cy, r)
    rows = np.flatnonzero(pol.any(axis=1))
    if len(rows) < 3:
        return "unknown"
    mid = rows[len(rows) // 4: 3 * len(rows) // 4 + 1]
    runs = np.median([_runs(pol[i]) for i in mid])
    if runs >= 3:
        return "star"  # several lobes in one piece: a star polygon
    if runs == 2:  # a hollow frame (or two strips): fill the envelope
        sh = N_T // 2 - int(np.flatnonzero(pol.any(axis=0)).mean())
        pol = np.roll(pol, sh, axis=1)
        for i in rows:
            f = np.flatnonzero(pol[i])
            if len(f):
                pol[i, f[0]:f[-1] + 1] = True
    lat = pol.sum(axis=1) * (2 * np.pi / N_T) * radii
    r_in, r_out = radii[rows[0]], radii[rows[-1]]
    t = np.linspace(0, 1, 11)
    p = np.interp(r_in + t * (r_out - r_in), radii, lat) / max(lat.max(), 1e-9)
    if lat.max() / max(r_out - r_in, 1e-9) < STRIP_ASPECT and p[2:9].min() >= 0.75:
        return "edge"
    if p[9] >= 0.8:
        return "loop"
    if t[int(np.argmax(p[1:10])) + 1] <= 0.2:
        return "star"
    if p[1] >= 0.55:
        return "petal"
    return "kite"


def mark_backings(items: list[tuple[int, str, np.ndarray]], head) -> list[str]:
    """Kinds of `items` [(category, kind, mask)] with backings marked (BACK_*)."""
    pts = {"loop", "kite", "petal", "star", "edge"}
    from scipy import ndimage
    axes = [lobe_axis(m, head) if k in pts else None for _, k, m in items]
    # touching counts: a backing drawn after its host is drawn cut around it (a frame)
    grown = [ndimage.binary_dilation(m, iterations=CONTACT_PX) if a is not None else None
             for (_, _, m), a in zip(items, axes)]
    out = []
    for i, (cat, kind, m) in enumerate(items):
        back = False
        if axes[i] is not None:
            a, ro = axes[i]
            for j, (cat2, _, m2) in enumerate(items):
                if j == i or cat2 == cat or axes[j] is None:
                    continue
                d = abs((a - axes[j][0] + 180) % 360 - 180)
                if d <= BACK_AXIS_DEG and ro >= axes[j][1] + BACK_REACH_R and (grown[i] & m2).any():
                    back = True
                    break
        out.append("backing" if back else kind)
    return out


def overlap_pairs(items: list[tuple[str, np.ndarray]], min_px: int) -> list[tuple[str, str]]:
    """[(kind below, kind above)] of every pair of masks overlapping by >= min_px;
    `items` [(kind, mask)] are in paint order (later is above)."""
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if int((items[i][1] & items[j][1]).sum()) >= min_px:
                out.append((items[i][0], items[j][0]))
    return out


def _hull_share(a: np.ndarray, b: np.ndarray) -> float:
    """Share of b's pixels inside a's convex hull."""
    import cv2
    ys, xs = np.nonzero(a)
    if len(ys) < 3 or not b.any():
        return 0.0
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    m = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return float((m.astype(bool) & b).sum() / b.sum())


def cut_order(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> int:
    """For two touching, non-overlapping masks: 1 when b lies above a (a was cut along
    b), -1 when a lies above b, 0 when they do not touch or it is unclear."""
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    da = ndimage.binary_dilation(a, st, iterations=CONTACT_PX)
    db = ndimage.binary_dilation(b, st, iterations=CONTACT_PX)
    contact = da & db
    if contact.sum() < MIN_CONTACT_UNITS * scale * CONTACT_PX:
        return 0
    ys, xs = np.nonzero(contact)
    w = (slice(max(0, ys.min() - WINDOW_PX), ys.max() + WINDOW_PX + 1),
         slice(max(0, xs.min() - WINDOW_PX), xs.max() + WINDOW_PX + 1))
    sa, sb = _hull_share(a[w], b[w]), _hull_share(b[w], a[w])
    if sa >= sb + HULL_MARGIN:
        return 1
    if sb >= sa + HULL_MARGIN:
        return -1
    return 0


def _bbox(m: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    return ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad


def stacking_pairs(pieces: list[tuple[int, str, np.ndarray]], scale: float) -> list[tuple[str, str, bool, str]]:
    """[(kind below, kind above, same Color, how)] over `pieces` [(category, kind, mask)]
    in paint order: overlapping pairs by paint order (how 'overlap'), touching pairs by
    which one was cut (how 'cut')."""
    min_px = max(1, int(MIN_OVERLAP_UNITS2 * scale ** 2))
    boxes = [_bbox(m, CONTACT_PX + 1) for _, _, m in pieces]
    out = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            bi, bj = boxes[i], boxes[j]
            if bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]:
                continue
            (ci, ki, mi), (cj, kj, mj) = pieces[i], pieces[j]
            win = (slice(max(0, min(bi[0], bj[0])), max(bi[1], bj[1]) + 1),
                   slice(max(0, min(bi[2], bj[2])), max(bi[3], bj[3]) + 1))
            a, b = mi[win], mj[win]
            ov = int((a & b).sum())
            if ov >= min_px:
                out.append((ki, kj, ci == cj, "overlap"))
                continue
            o = cut_order(a & ~b, b & ~a, scale)
            if o == 1:
                out.append((ki, kj, ci == cj, "cut"))
            elif o == -1:
                out.append((kj, ki, ci == cj, "cut"))
    return out


def reference_pairs(svg: str) -> list[tuple[str, str, bool, str]]:
    """`stacking_pairs` of one reference SVG's classified pieces."""
    from rosette.roles import category, classify_masks, element_masks, frame_for
    f = frame_for(svg)
    pieces = [(category(el), role, m) for el, role, m in classify_masks(element_masks(svg), f)]
    kinds = mark_backings([(c, piece_kind(m, role, f.head), m) for c, role, m in pieces], f.head)
    return stacking_pairs([(c, k, m) for (c, _, m), k in zip(pieces, kinds)], f.scale)


def model_kind(part) -> str:
    q = part.params
    if part.role != "points":
        return ROLE_KIND.get(part.role, "unknown")
    if q.get("edge_width"):
        return "edge"
    return {"bow": "loop", "diamond": "kite", "petal": "petal", "star": "star"}.get(q.get("kind"), "unknown")


def model_pairs(model) -> list[tuple[str, str]]:
    """[(kind below, kind above)] of the overlapping color shapes our build writes for
    `model` (ordered as the written file shows them, see the module doc)."""
    from rosette.build import color_shapes
    from rosette.geometry.clip import to_geom
    shapes = color_shapes(model)  # (shape, part) in z order, before the cuts
    kinds = [model_kind(p) for _, p in shapes]
    pts = [i for i, k in enumerate(kinds) if k in ("loop", "kite", "petal", "star")]
    for i in pts:  # a points layer of another color on the same axes reaching further out
        p = shapes[i][1].params
        for j in pts:
            q = shapes[j][1].params
            if (shapes[i][1].color_group != shapes[j][1].color_group
                    and int(p["count"]) == int(q["count"])
                    and abs((float(p["rotation_deg"]) - float(q["rotation_deg"]) + 180) % 360 - 180) <= BACK_AXIS_DEG
                    and float(p["r_outer"]) > float(q["r_outer"])):
                kinds[i] = "backing"
    geoms = [to_geom(s) for s, _ in shapes]
    order = sorted(range(len(shapes)), key=lambda i: (shapes[i][0].z, shapes[i][0].category or 0, i))
    out = []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            if geoms[i].intersection(geoms[j]).area >= MIN_OVERLAP_UNITS2:
                out.append((kinds[i], kinds[j]))
    return out


def table(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[int, int]]:
    """{(x, y) with x < y: (count x above y, count y above x)}."""
    c = Counter((b, a) for a, b in pairs)  # (above, below)
    out = {}
    for x, y in {tuple(sorted(k)) for k in c}:
        if x != y:
            out[(x, y)] = (c[(x, y)], c[(y, x)])
    return out


def consensus(tab: dict[tuple[str, str], tuple[int, int]], share: float, min_n: int = 5) -> dict:
    """{(above, below): share} for the kind pairs where one direction holds >= share of
    at least min_n overlapping pairs."""
    out = {}
    for (x, y), (xy, yx) in tab.items():
        n = xy + yx
        if n >= min_n and max(xy, yx) >= share * n:
            out[(x, y) if xy >= yx else (y, x)] = max(xy, yx) / n
    return out


def disagreements(ref: list[tuple[str, str]], ours: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Kind pairs (above, below) the reference file states (majority) whose order ours reverses."""
    rt, ot = table(ref), table(ours)
    out = []
    for (x, y), (rxy, ryx) in rt.items():
        if (x, y) not in ot or rxy == ryx:
            continue
        oxy, oyx = ot[(x, y)]
        if oxy == oyx:
            continue
        if (rxy > ryx) != (oxy > oyx):
            out.append((x, y) if rxy > ryx else (y, x))
    return out


def main(argv: list[str]) -> int:
    ids = argv or sorted(p.stem for p in (ROOT / "reference" / "svg").glob("*.svg"))
    allp, same, files = [], [], Counter()
    for sid in ids:
        ps = reference_pairs((ROOT / "reference" / "svg" / f"{sid}.svg").read_text(encoding="utf-8"))
        allp += [(a, b) for a, b, _ in ps]
        same += [(a, b) for a, b, s in ps if s]
        for k in {tuple(sorted((a, b))) for a, b, _ in ps}:
            files[k] += 1
        print(sid, len(ps), flush=True)
    ts = table(same)
    print(f"{'above':8} {'below':8} {'n':>5} {'share':>6} {'files':>5} {'same-Color n':>12} {'share':>6}")
    for (x, y), (xy, yx) in sorted(table(allp).items(), key=lambda kv: -sum(kv[1])):
        a, b, n = (x, y, xy) if xy >= yx else (y, x, yx)
        s = ts.get((x, y), (0, 0))
        sn = s[0] + s[1]
        sa = s[0] if a == x else s[1]
        print(f"{a:8} {b:8} {xy + yx:5d} {n / (xy + yx):6.2f} {files[(x, y)]:5d} {sn:12d} "
              f"{(sa / sn if sn else float('nan')):6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
