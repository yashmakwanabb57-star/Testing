"""ML-4 piece groups: the SAM 2 head pieces the reader named, gathered into repeat rings, each counted
from the signal.

Pieces of one kind whose radial spans (the PCT_LO and PCT_HI percentile radii of their pixels from
the head centre, canvas units) agree within `MIN_ZONE_UNITS` form a group (greedy, largest piece
first, each piece compared with a group's first member). A piece that two groups would take joins the
one whose span holds its centroid radius (the nearest span when both or neither hold it), so a piece
straddling two bands is counted once. Disc, ring, badge and rose pieces are single groups with count
1; the disc group (the largest piece named pleated disc or covering DISC_COVER of the head) reaches
the middle (r_inner 0). Pieces the reader left unanswered are grouped by span alone as "other" with
confidence UNANSWERED_CONF.

A group's count comes from the signal (spec ruling 6): the number of its pieces when their angular
steps (`pieces._fit_steps` on their interior points) fit one equal step with no empty step, else the
strongest harmonic of the union's outline over the full circle (`parts_head.harmonic_count`), which
holds a ring with hidden members (8 arms, 2 behind the badge: 6 pieces, 8 lobes). The harmonic's
second peak at 2n or n/2 and the model's repeats (when they differ and lie in REPEATS_MIN..REPEATS_MAX)
are kept as alternatives for the judge, at most MAX_ALTERNATIVES.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rosette.perception.pieces import Piece, _centroid, _fit_steps, _interior_point
from rosette.perception.reader import Reading
from rosette.vision.parts_common import polar_sample
from rosette.vision.parts_head import MIN_ZONE_UNITS, RING_THETA, _harmonics, harmonic_count

PCT_LO, PCT_HI = 2.0, 98.0  # radial span of a piece: these percentiles of its pixels' radii
SINGLE_KINDS = ("pleated_disc", "pleated_ring", "ring", "badge", "rose")  # one piece, one part
DISC_COVER = 0.8  # a piece covering this share of the head disc is the disc whatever its name
UNANSWERED = "other"  # the kind of a piece the reader left unanswered
UNANSWERED_CONF = 0.3
SINGLE_CONF = 0.9  # a single part named by the reader
HARMONIC_CONF_CAP = 0.9  # a count read from the harmonic is never surer than this
REPEATS_MIN, REPEATS_MAX = 3, 16  # the model's repeats kept as an alternative only in this range
MAX_ALTERNATIVES = 2  # with the count itself: at most three options per group


@dataclass
class Group:
    kind: str
    texture: str
    color_group: int | None
    pieces: list[int]  # indices into the pieces list
    r_inner: float  # canvas units
    r_outer: float
    count: int
    count_source: str  # "pieces" | "harmonic" | "single"
    alternatives: list[int] = field(default_factory=list)
    rotation_deg: float = 0.0
    confidence: float = 0.0


def _span(pc: Piece, head: tuple[float, float, float], scale: float) -> tuple[float, float]:
    cx, cy, _ = head
    ys, xs = np.nonzero(pc.mask)
    rr = np.hypot(xs - cx, ys - cy) / scale
    return float(np.percentile(rr, PCT_LO)), float(np.percentile(rr, PCT_HI))


def _angle(pc: Piece, head: tuple[float, float, float]) -> float:
    """Angle (deg, from +x, clockwise on screen) of the piece's interior point."""
    x, y = _interior_point(pc)
    return float(np.degrees(np.arctan2(y - head[1], x - head[0])))


def _mode(values: list, default=None):
    """The most common value (the smallest on a tie), or default when there is none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return default
    counts = Counter(vals)
    top = max(counts.values())
    return min(v for v, c in counts.items() if c == top)


def _outline(group: Group, pieces: list[Piece], head: tuple[float, float, float]) -> np.ndarray | None:
    """Outermost radius (px) of the group's union per angle over the full circle; angles where it
    shows nothing take its innermost outline radius (a hidden member reads as a valley). None when
    the union shows on fewer than 3 angles."""
    cx, cy, r = head
    mask = np.logical_or.reduce([pieces[i].mask for i in group.pieces])
    radii = np.arange(0.0, 1.1 * r, 1.0)
    ind = polar_sample(mask.astype(np.uint8), cx, cy, radii, RING_THETA, fill=0) > 0
    on = ind.any(axis=0)
    if on.sum() < 3:
        return None
    last = ind.shape[0] - 1 - np.argmax(ind[::-1], axis=0)
    out = radii[last].astype(np.float64)
    out[~on] = out[on].min()
    return out


def _alternatives(count: int, n2: int | None, repeats: int | None) -> list[int]:
    alts: list[int] = []
    if n2 is not None and (n2 == 2 * count or 2 * n2 == count):
        alts.append(n2)
    if repeats is not None and repeats != count and REPEATS_MIN <= repeats <= REPEATS_MAX and repeats not in alts:
        alts.append(repeats)
    return alts[:MAX_ALTERNATIVES]


def _steps_fit(group: Group, pieces: list[Piece], head) -> tuple[int, float] | None:
    """(n, inlier share) when the pieces' angles fit one equal step of n with every step taken by
    exactly one inlier, else None."""
    angles = [_angle(pieces[i], head) for i in group.pieces]
    fit = _fit_steps(angles)
    if fit is None:
        return None
    n, off, inl = fit
    step = 360.0 / n
    steps = [int(round((a - off) / step)) % n for a, keep in zip(angles, inl) if keep]
    if len(steps) != n or len(set(steps)) != n:
        return None
    return n, float(inl.mean())


def _count(group: Group, pieces: list[Piece], head, repeats: int | None) -> tuple[int, str, list[int], float]:
    """(count, source, alternatives, confidence) of a repeat group."""
    fit = _steps_fit(group, pieces, head)
    if fit is not None:
        n, share = fit
        return n, "pieces", _alternatives(n, None, repeats), share
    out = _outline(group, pieces, head)
    if out is None:
        k = len(group.pieces)
        return k, "pieces", _alternatives(k, None, repeats), UNANSWERED_CONF
    n, amp, n2 = harmonic_count(out)
    rest = sorted((v for k, v in _harmonics(out).items() if k != n), reverse=True)
    second = rest[0] if rest else 0.0
    conf = amp / (amp + second) if amp + second > 0 else 0.0
    return n, "harmonic", _alternatives(n, n2, repeats), min(HARMONIC_CONF_CAP, conf)


def group_count(group: Group, pieces: list[Piece], head: tuple[float, float, float],
                repeats: int | None = None) -> tuple[int, str, list[int]]:
    """(count, source, alternatives) of a group: (pieces on one equal step with none empty, "pieces")
    or (strongest harmonic of the union, "harmonic"); alternatives the harmonic's second peak at 2n
    or n/2 and the model's `repeats` when it differs and lies in REPEATS_MIN..REPEATS_MAX. A single
    group (disc, ring, badge, rose) is (1, "single", [])."""
    if group.count_source == "single":
        return 1, "single", []
    n, source, alts, _conf = _count(group, pieces, head, repeats)
    return n, source, alts


def _rotation(group: Group, pieces: list[Piece], head) -> float:
    """Circular mean of the pieces' interior-point angles modulo one step (360 / count), degrees."""
    if group.count <= 1:
        return 0.0
    k = group.count
    a = np.radians([_angle(pieces[i], head) for i in group.pieces]) * k
    mean = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))) / k
    return round(mean % (360.0 / k), 3)


def piece_groups(reading: Reading, pieces: list[Piece], head: tuple[float, float, float],
                 scale: float) -> list[Group]:
    """The reader's marked head pieces gathered into groups (module docstring), each counted.
    `scale` is px per canvas unit."""
    cx, cy, r = head
    named: dict[int, tuple[str, str, int | None]] = {}  # piece index -> (kind key, texture, repeats)
    for m in reading.marks:
        i = reading.piece_index.get(m)
        if i is None or not (0 <= i < len(pieces)) or i in named or not pieces[i].in_head or pieces[i].area <= 0:
            continue
        ans = reading.answers.get(m)
        # "" = unanswered: grouped by span alone
        named[i] = ("", "plain", None) if ans is None else (ans.kind, ans.texture, ans.repeats)
    if not named:
        return []
    head_area = np.pi * r * r
    discs = [i for i, t in named.items() if t[0] == "pleated_disc" or pieces[i].area >= DISC_COVER * head_area]
    disc = max(discs, key=lambda i: (pieces[i].area, -i)) if discs else None
    spans = {i: _span(pieces[i], head, scale) for i in named}

    # (group, kind key, the members' repeats, the seed's span)
    built: list[tuple[Group, str, list[int | None], tuple[float, float]]] = []
    for i in sorted(named, key=lambda i: (-pieces[i].area, i)):
        key, texture, repeats = named[i]
        r0, r1 = spans[i]
        single = i == disc or key in SINGLE_KINDS
        if not single:
            gx, gy = _centroid(pieces[i])
            rc = float(np.hypot(gx - cx, gy - cy)) / scale
            fits = [b for b in built
                    if b[0].count_source != "single" and b[1] == key
                    and abs(r0 - b[3][0]) <= MIN_ZONE_UNITS and abs(r1 - b[3][1]) <= MIN_ZONE_UNITS]
            if fits:
                b = min(fits, key=lambda b: (not (b[3][0] <= rc <= b[3][1]), abs(r0 - b[3][0]) + abs(r1 - b[3][1])))
                b[0].pieces.append(i)
                b[2].append(repeats)
                continue
        grp = Group(kind=key or UNANSWERED, texture=texture, color_group=pieces[i].color_group, pieces=[i],
                    r_inner=0.0 if i == disc else r0, r_outer=r1, count=1,
                    count_source="single" if single else "")
        built.append((grp, key, [repeats], (r0, r1)))

    out: list[Group] = []
    for grp, key, reps, _seed in built:
        if grp.count_source == "single":
            grp.confidence = SINGLE_CONF if key else UNANSWERED_CONF
            out.append(grp)
            continue
        grp.texture = _mode([named[i][1] for i in grp.pieces], grp.texture)
        grp.color_group = _mode([pieces[i].color_group for i in grp.pieces])
        grp.r_inner = min(spans[i][0] for i in grp.pieces)
        grp.r_outer = max(spans[i][1] for i in grp.pieces)
        n, source, alts, conf = _count(grp, pieces, head, _mode(reps))
        grp.count, grp.count_source, grp.alternatives = n, source, alts
        grp.confidence = conf if key else UNANSWERED_CONF
        grp.rotation_deg = _rotation(grp, pieces, head)
        out.append(grp)
    return out
