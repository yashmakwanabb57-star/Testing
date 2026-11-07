"""Head shape library (H7): the reference heads as templates, and the matcher.

The 79 reference SVGs are the space of head shapes the shop's files use (they were made
by copying shapes from similar files, reference/generation-prompt.md; a library does the
same deterministically). Each product file (TEST and D1 left out) gives a template: every
non-tail color piece (`rosette.roles.classify_components`: head, ring, points, center,
unknown) with its category, in the file's paint order, as polygons normalized to the head
circle `prepare` finds on the rendered file (u, v in head radii from the head center, v
down). Pieces are the solid configuration's elements (outside the shop's hidden pattern
layer `Layer_3`, whose elements are the pattern twins of `Layer_2_1` ones; the writer
emits the Color 2 twins again). The library is cached in `rosette/data/head_templates.json`
(`build_library` writes it; `load_library` reads it).

Matching a photo, on a common GRID x GRID raster in head-radius units (u, v in [-EXTENT,
EXTENT]) restricted to v <= WINDOW_V (the tails cover the lower head in photos and are not
part of a template):
  partition  one-to-one mapping template category -> photo color group by overlap
             (Hungarian); share of the foreground union where the mapped group agrees
  iou        silhouette IoU
  profile    mean |outer radius difference| along rays in the window (head radii)
  lobes      lobe count of the outer radius profile (peaks over the arc, per 360 deg)
score = w_partition * partition + w_iou * iou + w_profile * (1 - min(1, profile / PROFILE_SCALE))
        + w_lobes * exp(-|lobes_photo - lobes_template| / LOBE_SCALE)
The weights (WEIGHTS) were chosen on the 76 photos leave-one-out (docs/pipeline.md H7).
The chosen template, its score and the category -> group mapping are the output.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Common raster: GRID x GRID samples over [-EXTENT, EXTENT] head radii.
GRID, EXTENT = 128, 1.9
# Only rows above this many head radii below the head center are compared.
WINDOW_V = 0.55
# Roles of reference pieces that belong to the head (everything but tail stripes).
HEAD_ROLES = ("head", "ring", "points", "center", "unknown")
# Feature weights (partition, iou, profile, lobes, colors) and scales of the template
# score (H7 fix rounds 1-2: chosen on own-design retrieval over the 76 photos with the
# partition scored under the tail-bound mapping, docs/pipeline.md). The lobe count acts
# through LOBE_GATE only: as a score term it put S30-15 over S35-19's own template.
WEIGHTS = (0.4, 0.3, 0.2, 0.0, 0.1)
PROFILE_SCALE, LOBE_SCALE = 0.4, 2.0
# Lobe gate: a template whose outer-profile lobe count (per 360 deg) differs from the
# photo's by more than LOBE_GATE is not considered (a disc for a star photo). The count
# moves in steps of about 1.68 (one peak over the window's arc). H10 (`band_lobes`): 5.1,
# three steps; with the head-band count S57-17's own design differs by three (5.05 photo,
# 10.09 template), and at 3.4 it is gated out and S58-17 drops out of its own top 3.
LOBE_GATE = 5.1
# The lobe count reads the outer profile within this band of head radii (H10): only the
# outline out to the upper bound counts (side flags and bunting loops beyond the head
# circle do not), and radii under the lower bound read as the lower bound.
LOBE_BAND = (0.6, 1.05)
# A peak of that profile is a lobe when it stands at least this many head radii above
# the valleys on both sides (its prominence): a bunting loop or a flag edge bumping the
# outline by a few hundredths is not a lobe, a star tip is. On the 76 photos: 0.1 keeps
# every own design in the top 1 and top 3; 0.08 drops S58-17 from its top 3, 0.12 S13-15.
LOBE_PROMINENCE = 0.1
# Round-template guard (H10): a round template (no head-band lobes and a whole-outline
# symmetry under 3) is refused for a photo whose head-band count reaches
# ROUND_GUARD_LOBES, whatever the gate. Photos of round heads count 0-3.36 (S11a-15 3.36;
# A2-15, 5.05, is the one exception), a 6-point star shows three tips, 5.05; on the 76 no
# own design is lost (A2-15 and S51-19 lose A1-15 as their pick). The files' points
# counts do not tell round heads (A12-15's disc: 5) and symmetry alone does not either
# (S57-17's own template: symmetry 0, ten band lobes).
ROUND_GUARD_LOBES = 4.2
# A color group (photo) or category (template) counts as a head color when it holds at
# least COLOR_MIN_SHARE of the window's foreground.
COLOR_MIN_SHARE = 0.05
# Rays sampled for the outer radius profile, and radial samples per ray.
N_RAYS, N_RHO = 180, 96
# Library files that are not products (unknown provenance / layout test).
NOT_PRODUCTS = ("TEST", "D1")
# Render height of the reference files for extraction (about 2.1 px per canvas unit) and
# the polygon simplification tolerance (render px).
EXTRACT_H, SIMPLIFY_PX = 1800, 0.7
# Decimals of the stored polygon coordinates (head radii; 1e-3 R is about 0.15 units).
DECIMALS = 3

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "data" / "head_templates.json"


def design_key(sid: str) -> str:
    """Same design across variants and sizes: S36-15B, S36-15, S36-19 -> 'S36';
    S11a-15 -> 'S11'."""
    m = re.match(r"^([A-Za-z])(\d+)", sid)
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else sid.upper()


@dataclass
class Piece:
    category: int  # the reference file's Color 1-4
    role: str  # rosette.roles classifier role
    rings: list[np.ndarray]  # (n, 2) polygons in head radii, even-odd (outer first, then holes)


@dataclass
class Template:
    sid: str
    key: str
    pieces: list[Piece]  # paint order
    grid: np.ndarray = field(default=None, repr=False)  # (GRID, GRID) uint8 category 0..4
    tail_categories: tuple[int, ...] = ()  # the reference's tail stripe Colors
    points: int = 0  # the reference's points count (tools/truth), 0 without points

    def __post_init__(self):
        if self.grid is None:
            self.grid = raster(self.pieces, (GRID, GRID), (GRID - 1) / 2, (GRID - 1) / 2,
                               (GRID - 1) / (2 * EXTENT))

    @property
    def categories(self) -> list[int]:
        return sorted({p.category for p in self.pieces})


# ---------------------------------------------------------------- rasters


def raster(pieces: list[Piece], shape: tuple[int, int], cx: float, cy: float, r: float,
           rotation_deg: float = 0.0, lut=None) -> np.ndarray:
    """uint8 image of the pieces painted in order (category, or lut[category]) placed on
    the head circle (cx, cy, r) in pixels, turned by rotation_deg (clockwise on screen)."""
    out = np.zeros(shape, np.uint8)
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    for p in pieces:
        polys = []
        for ring in p.rings:
            u, v = ring[:, 0], ring[:, 1]
            xs, ys = cx + r * (u * c - v * s), cy + r * (u * s + v * c)
            polys.append(np.rint(np.stack([xs, ys], axis=1) * 16).astype(np.int32))
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, polys, 1, lineType=cv2.LINE_8, shift=4)  # even-odd over the rings
        out[m > 0] = p.category if lut is None else lut[p.category]
    return out


def sample(img: np.ndarray, cx: float, cy: float, r: float, fill=0) -> np.ndarray:
    """Nearest-neighbour samples of img on the head grid (fill outside the image)."""
    t = np.linspace(-EXTENT, EXTENT, GRID)
    xi = np.broadcast_to(np.rint(cx + t[None, :] * r).astype(int), (GRID, GRID))
    yi = np.broadcast_to(np.rint(cy + t[:, None] * r).astype(int), (GRID, GRID))
    h, w = img.shape[:2]
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full((GRID, GRID), fill, dtype=img.dtype)
    out[ok] = img[yi[ok], xi[ok]]
    return out


def _window() -> np.ndarray:
    v = np.linspace(-EXTENT, EXTENT, GRID)
    return np.broadcast_to((v <= WINDOW_V)[:, None], (GRID, GRID))


_WIN = _window()


# ---------------------------------------------------------------- extraction


def mask_rings(mask: np.ndarray, cx: float, cy: float, r: float) -> list[np.ndarray]:
    """Polygons (head radii) of a bool mask: outer contours and holes, even-odd."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings = []
    for cnt in sorted(contours, key=lambda q: -cv2.contourArea(q)):
        q = cv2.approxPolyDP(cnt, SIMPLIFY_PX, True)[:, 0, :].astype(float)
        if len(q) < 3:
            continue
        rings.append(np.round(np.stack([(q[:, 0] - cx) / r, (q[:, 1] - cy) / r], axis=1), DECIMALS))
    return rings


def extract_template(sid: str, svg: str, height: int = EXTRACT_H) -> Template:
    """The head pieces of one reference file (document order is paint order)."""
    from rosette.roles import category, classify_components

    f, items = classify_components(svg, height)
    cx, cy, r = f.head
    pieces = []
    tails = set()
    for el, role, m in items:
        if role in HEAD_ROLES:
            rings = mask_rings(m, cx, cy, r)
            if rings:
                pieces.append(Piece(category(el), role, rings))
        elif role == "tail_stripe":
            tails.add(category(el))
    return Template(sid, design_key(sid), pieces, tail_categories=tuple(sorted(tails)))


def to_json(templates: list[Template]) -> dict:
    return {"grid": GRID, "extent": EXTENT, "templates": [
        {"id": t.sid, "tail_categories": list(t.tail_categories), "points": t.points, "pieces": [{"category": p.category, "role": p.role,
                                  "rings": [ring.reshape(-1).tolist() for ring in p.rings]}
                                 for p in t.pieces]} for t in templates]}


def from_json(d: dict) -> list[Template]:
    return [Template(t["id"], design_key(t["id"]),
                     [Piece(int(p["category"]), p["role"],
                            [np.asarray(ring, float).reshape(-1, 2) for ring in p["rings"]])
                      for p in t["pieces"]], tail_categories=tuple(t.get("tail_categories", ())),
                     points=int(t.get("points", 0))) for t in d["templates"]]


def build_library(svg_dir: Path, path: Path = LIBRARY_PATH) -> list[Template]:
    """Extract every product reference file of svg_dir and write the library cache."""
    # tools.truth is a development module (it reads the reference files and needs the repo
    # checkout); build_library is a development step too, the app only reads the cache.
    from tools.truth import truth_from_svg  # the points count of each file

    lib = []
    for p in sorted(svg_dir.glob("*.svg")):
        if p.stem not in NOT_PRODUCTS:
            t = extract_template(p.stem, p.read_text(encoding="utf-8"))
            t.points = int(truth_from_svg(p.stem)["points"])
            lib.append(t)
    path.write_text(json.dumps(to_json(lib), separators=(",", ":")), encoding="utf-8", newline="\n")
    load_library.cache_clear()
    return lib


@lru_cache(maxsize=4)
def load_library(path: Path = LIBRARY_PATH) -> tuple[Template, ...]:
    if not path.exists():
        return ()
    return tuple(from_json(json.loads(path.read_text(encoding="utf-8"))))


def template_by_id(tid: str, path: Path = LIBRARY_PATH) -> Template | None:
    return next((t for t in load_library(path) if t.sid == tid), None)


# ---------------------------------------------------------------- features


def _outer_profile(fg: np.ndarray, reach: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(angles, outer radius in head radii) of the rays that stay inside the window;
    reach: only foreground out to this many head radii counts."""
    th = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
    rho = np.linspace(0, EXTENT, N_RHO)
    th = th[np.sin(th) * EXTENT <= WINDOW_V]
    c, k = (GRID - 1) / 2, (GRID - 1) / (2 * EXTENT)
    xs = np.rint(c + np.cos(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
    ys = np.rint(c + np.sin(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
    hit = fg[ys, xs]
    if reach is not None:
        hit = hit & (rho <= reach + 1e-9)[None, :]
    idx = np.where(hit.any(axis=1), N_RHO - 1 - np.argmax(hit[:, ::-1], axis=1), 0)
    return th, rho[idx]


def lobes(profile: np.ndarray, arc_deg: float) -> float:
    """Peaks of the (circularly smoothed) outer profile, scaled to a full turn."""
    if len(profile) < 5:
        return 0.0
    p = np.convolve(np.r_[profile[-2:], profile, profile[:2]], np.ones(5) / 5, mode="valid")
    base = float(np.median(p))
    peaks = sum(1 for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > base + 0.04)
    return peaks * 360.0 / max(arc_deg, 1.0)


def band_lobes(fg: np.ndarray, arc_deg: float) -> float:
    """Lobes of the (circularly smoothed) outer profile read within LOBE_BAND: peaks of
    prominence LOBE_PROMINENCE or more, scaled to a full turn (H10)."""
    from scipy.signal import find_peaks

    lo, hi = LOBE_BAND
    _, prof = _outer_profile(fg, hi)
    n = len(prof)
    if n < 5:
        return 0.0
    prof = np.clip(prof, lo, hi)
    p = np.convolve(np.r_[prof[-2:], prof, prof[:2]], np.ones(5) / 5, mode="valid")
    peaks, _ = find_peaks(np.r_[p, p, p], prominence=LOBE_PROMINENCE)  # three turns: no edge effects
    return int(((peaks >= n) & (peaks < 2 * n)).sum()) * 360.0 / max(arc_deg, 1.0)


def partition(labels: np.ndarray, tgrid: np.ndarray) -> tuple[float, dict[int, int]]:
    """Map template categories onto photo labels one-to-one by overlap (Hungarian; extra
    template categories then go to their most-overlapping label) and return the share of
    the window's foreground union where the mapped label agrees, and the map.
    labels: photo group id per cell, -1 background."""
    win = _WIN
    pf, tf = (labels >= 0) & win, (tgrid > 0) & win
    union = pf | tf
    cats = [int(c) for c in np.unique(tgrid[tf])]
    groups = [int(g) for g in np.unique(labels[pf])]
    if not union.any() or not cats or not groups:
        return 0.0, {}
    ov = np.array([[float(((tgrid == c) & (labels == g) & win).sum()) for g in groups] for c in cats])
    rows, cols = linear_sum_assignment(-ov)
    mapping = {cats[i]: groups[j] for i, j in zip(rows, cols)}
    for i, c in enumerate(cats):
        if c not in mapping:
            mapping[c] = groups[int(ov[i].argmax())]
    mapped = np.full(tgrid.shape, -2, np.int32)
    for c, g in mapping.items():
        mapped[tgrid == c] = g
    return float(((mapped == labels) & union).sum() / union.sum()), mapping


def head_colors(values: np.ndarray, background) -> int:
    """Distinct labels holding at least COLOR_MIN_SHARE of the window's foreground."""
    sel = _WIN & (values != background) if background == 0 else _WIN & (values > background)
    if not sel.any():
        return 0
    _, counts = np.unique(values[sel], return_counts=True)
    return int((counts >= COLOR_MIN_SHARE * sel.sum()).sum())


def partition_under(labels: np.ndarray, tgrid: np.ndarray, mapping: dict[int, int]) -> float:
    """Share of the window's foreground union where the template category, turned into a
    photo label by mapping, equals the photo label."""
    pf, tf = (labels >= 0) & _WIN, (tgrid > 0) & _WIN
    union = pf | tf
    if not union.any():
        return 0.0
    mapped = np.full(tgrid.shape, -2, np.int32)
    for c, g in mapping.items():
        mapped[tgrid == c] = g
    return float(((mapped == labels) & union).sum() / union.sum())


def features(labels: np.ndarray, tgrid: np.ndarray, weights=None, mapping: dict[int, int] | None = None) -> dict:
    """Feature values, the weighted score and the category -> label mapping. mapping:
    score the partition under this mapping (the one the head will be drawn with)
    instead of the free one-to-one overlap mapping."""
    w = WEIGHTS if weights is None else weights
    wp, wi, wf, wl = w[:4]
    wc = w[4] if len(w) > 4 else 0.0
    pf, tf = (labels >= 0) & _WIN, (tgrid > 0) & _WIN
    union = (pf | tf).sum()
    iou = float((pf & tf).sum() / union) if union else 0.0
    th, rp = _outer_profile(pf)
    _, rt = _outer_profile(tf)
    arc = math.degrees(len(th) * 2 * math.pi / N_RAYS)
    if mapping is None:
        part, mapping = partition(labels, tgrid)
    else:
        part = partition_under(labels, tgrid, mapping)
    lp, lt = band_lobes(pf, arc), band_lobes(tf, arc)
    prof = float(np.abs(rp - rt).mean())
    cp, ct = head_colors(labels, -1), head_colors(tgrid, 0)
    total = (wp * part + wi * iou + wf * (1 - min(1.0, prof / PROFILE_SCALE))
             + wl * math.exp(-abs(lp - lt) / LOBE_SCALE) + wc * math.exp(-abs(cp - ct)))
    return dict(score=total, partition=part, iou=iou, profile=prof, lobes_photo=lp, lobes_template=lt,
                colors_photo=cp, colors_template=ct, mapping=mapping)


def choose(labels: np.ndarray, lib, exclude_key: str | None = None, weights=None,
           gate: float | None = LOBE_GATE, tail_groups: dict[int, int] | None = None) -> tuple[Template, dict] | None:
    """Best-scoring template (and its features) for the photo grid labels among those
    passing the lobe gate (LOBE_GATE; None: no gate), or None; templates of exclude_key
    (a design_key) are skipped (leave-one-out evaluation). tail_groups (our tail Color ->
    group): each template is scored under the tail-bound mapping it would be drawn with
    (tail_consistent_mapping), and one that cannot be mapped is skipped."""
    best = rank(labels, lib, exclude_key, weights, gate, tail_groups, n=1)
    return best[0] if best else None


# Pose search (H9): the head circle prepare finds can be off (S29-15: a 6-fold star whose
# lower arms pass for the tails top, so the circle sits too high and too small), and the
# references draw their heads at their own proportions. Each template is placed at the
# pose (center offset dx, dy and scale in head radii of the photo circle, rotation in
# degrees) whose silhouette fits the photo best: offset x scale on a coarse grid unturned,
# then the turns of its lobe symmetry at that offset, then the offset x scale grid again
# at the best turn (a staged search, not a finer grid); the full score is taken at that
# pose and at the unmoved one, and the better wins.
POSE_DY = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
POSE_DX = (-0.1, 0.0, 0.1)
POSE_SCALE = (0.8, 0.9, 1.0, 1.12, 1.25)
# Rotations tried within one symmetry period 360/k of a k-lobed template (k >= 3).
POSE_TURNS = 4


@dataclass(frozen=True)
class Pose:
    dx: float = 0.0  # head radii, + right
    dy: float = 0.0  # head radii, + down
    scale: float = 1.0
    rotation: float = 0.0  # degrees, clockwise on screen (rotation_deg)

    @property
    def unmoved(self) -> bool:
        return self == Pose()


def pose_grid(t: Template, pose: Pose) -> np.ndarray:
    """The template's category grid at a pose (nearest-neighbour warp of its grid)."""
    if pose.unmoved:
        return t.grid
    c, k = (GRID - 1) / 2, (GRID - 1) / (2 * EXTENT)
    m = cv2.getRotationMatrix2D((c, c), -pose.rotation, pose.scale)
    m[0, 2] += pose.dx * k
    m[1, 2] += pose.dy * k
    return cv2.warpAffine(t.grid, m, (GRID, GRID), flags=cv2.INTER_NEAREST, borderValue=0)


def _iou(fg: np.ndarray, g: np.ndarray) -> float:
    tf = (g > 0) & _WIN
    u = (fg | tf).sum()
    return float((fg & tf).sum() / u) if u else 0.0


_SYMMETRY: dict[tuple[str, int], int] = {}


def symmetry(t: Template) -> int:
    """Lobe count k of the template's whole outline (all around, not only the window),
    cached per template shape (id and grid hash)."""
    key = (t.sid, hash(t.grid.tobytes()))
    if key not in _SYMMETRY:
        fg = t.grid > 0
        th = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
        rho = np.linspace(0, EXTENT, N_RHO)
        c, k = (GRID - 1) / 2, (GRID - 1) / (2 * EXTENT)
        xs = np.rint(c + np.cos(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
        ys = np.rint(c + np.sin(th)[:, None] * rho[None] * k).astype(int).clip(0, GRID - 1)
        hit = fg[ys, xs]
        prof = rho[np.where(hit.any(axis=1), N_RHO - 1 - np.argmax(hit[:, ::-1], axis=1), 0)]
        _SYMMETRY[key] = int(round(lobes(prof, 360.0)))
    return _SYMMETRY[key]


def best_pose(labels: np.ndarray, t: Template) -> Pose:
    """The pose whose silhouette fits the photo best, coarse to fine: center offset and
    scale unturned, then the turns within the template's symmetry period, then offset
    and scale again at the best turn (see POSE_*)."""
    fg = (labels >= 0) & _WIN

    def place(rot: float) -> Pose:
        poses = [Pose(dx, dy, sc, rot) for dy in POSE_DY for dx in POSE_DX for sc in POSE_SCALE]
        # ties: the smaller move, then the grid order (never compare Pose objects)
        keys = [(_iou(fg, pose_grid(t, q)), -abs(q.dx) - abs(q.dy) - abs(q.scale - 1), -i)
                for i, q in enumerate(poses)]
        return poses[max(range(len(poses)), key=keys.__getitem__)]

    pose = place(0.0)
    k = symmetry(t)
    if k >= 3:
        period = 360.0 / k
        turns = [Pose(pose.dx, pose.dy, pose.scale, round(period * j / POSE_TURNS, 3)) for j in range(POSE_TURNS)]
        tk = [(_iou(fg, pose_grid(t, q)), -q.rotation) for q in turns]
        turned = turns[max(range(len(turns)), key=tk.__getitem__)]
        if turned.rotation:
            pose = place(turned.rotation)
    return pose


def _at_pose(t: Template, pose: Pose) -> Template:
    if pose.unmoved:
        return t
    return Template(t.sid, t.key, t.pieces, grid=pose_grid(t, pose), tail_categories=t.tail_categories,
                    points=t.points)


def posed_features(labels: np.ndarray, t: Template, weights=None, tail_groups: dict[int, int] | None = None,
                   free_ok: bool = False) -> dict | None:
    """Features of a template at its best pose or unmoved, whichever is better, with the
    pose ("pose") and whether its colors follow the tails ("anchored"). A pose whose
    colors follow the tails beats one that does not (free_ok: the free overlap mapping)
    whatever the scores; among equals the higher score wins, the unmoved one on a tie.
    None when neither can be colored under the tails and free_ok is False."""
    found = []
    for pose in (Pose(), best_pose(labels, t)):
        if found and pose.unmoved:
            break
        tp = _at_pose(t, pose)
        mapping = tail_consistent_mapping(labels, tp, tail_groups) if tail_groups else None
        anchored = mapping is not None or not tail_groups
        if tail_groups and mapping is None and not free_ok:
            continue
        f = features(labels, tp.grid, weights, mapping)
        f.update(pose=pose, anchored=anchored)
        found.append(f)
    if not found:
        return None
    out = max(found, key=lambda f: (f["anchored"], f["score"] - (1e-9 if not f["pose"].unmoved else 0.0)))
    # the lobe gate is judged on the unmoved template (its shape, not its pose)
    out["lobes_template"] = features(labels, t.grid)["lobes_template"] if found[0]["pose"] != Pose() \
        else found[0]["lobes_template"]
    return out


def _round(t: Template, band_count: float) -> bool:
    """A round template: no head-band lobes and a whole-outline symmetry under 3."""
    return band_count == 0 and symmetry(t) < 3


def rank(labels: np.ndarray, lib, exclude_key: str | None = None, weights=None,
         gate: float | None = LOBE_GATE, tail_groups: dict[int, int] | None = None,
         n: int = 3) -> list[tuple[Template, dict]]:
    """The n best-scoring templates (and their features, at their best pose), best first,
    under the rules of choose (lobe gate and round-template guard, exclude_key,
    tail-bound mapping); ties keep
    library order."""
    scored = []
    for t in lib:
        if exclude_key is not None and t.key == exclude_key:
            continue
        f = posed_features(labels, t, weights, tail_groups)
        if f is None:
            continue
        if gate is not None and abs(f["lobes_photo"] - f["lobes_template"]) > gate + 1e-6:
            continue
        if gate is not None and f["lobes_photo"] >= ROUND_GUARD_LOBES and _round(t, f["lobes_template"]):
            continue
        scored.append((t, f))
    scored.sort(key=lambda tf: -tf[1]["score"])  # stable: ties keep library order
    return scored[:n]


def score_template(labels: np.ndarray, t: Template, tail_groups: dict[int, int] | None = None) -> dict:
    """One template scored for the head card's library chooser: no lobe gate, the
    tail-bound mapping where the template has one (tail_consistent_mapping), else the free
    overlap mapping, so any template can be drawn. "anchored" is False when the free
    mapping was used: the head's colors then do not follow the tails."""
    return posed_features(labels, t, None, tail_groups, free_ok=True)


@lru_cache(maxsize=4)
def library_index(path: Path = LIBRARY_PATH) -> tuple[tuple[str, str], ...]:
    """(template id, description) of every library template, by id (the chooser's list)."""
    return tuple(sorted((t.sid, describe(t)) for t in load_library(path)))


# ---------------------------------------------------------------- description

# Words of describe(), in the order they are listed, with their singular forms.
_WORDS = (("loops", "loop"), ("points", "point"), ("kites", "kite"), ("strips", "strip"), ("trims", "trim"))
# A piece away from the center: an outline STRIP_ASPECT times longer than wide is a strip,
# one filling under TRIM_SOLIDITY of its convex hull a trim (a hollow edge piece); a piece
# as wide at its outer end (the outer OUTER_END of its radial reach) as LOOP_END of its
# full angular width is a loop, a tapered one with at most four corners a kite, else a point.
STRIP_ASPECT, TRIM_SOLIDITY, OUTER_END, LOOP_END = 4.0, 0.6, 0.15, 0.5
# A piece around the center reaching past DISC_REACH head radii is a disc, else a center;
# with STAR_MIN_LOBES or more lobes deeper than STAR_DEPTH (outer / inner radius) a star.
# (Three or fewer runs come from the notch the tails cut into the lower head.)
DISC_REACH, STAR_MIN_LOBES, STAR_DEPTH = 0.65, 4, 1.3


def _center_lobes(outer: np.ndarray, bins: int = 180) -> tuple[int, float]:
    """(lobe count, max / min radius) of an outline around the head center: runs of the
    angular radius profile above the middle radius."""
    nxt = np.roll(outer, -1, axis=0)
    t = np.linspace(0, 1, 8, endpoint=False)[None, :, None]
    dense = (outer[:, None, :] * (1 - t) + nxt[:, None, :] * t).reshape(-1, 2)
    b = np.minimum(((np.arctan2(dense[:, 1], dense[:, 0]) + np.pi) / (2 * np.pi) * bins).astype(int), bins - 1)
    prof = np.full(bins, -np.inf)
    np.maximum.at(prof, b, np.hypot(dense[:, 0], dense[:, 1]))
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        return 0, 1.0
    idx = np.arange(bins)
    prof = np.interp(idx, idx[ok], prof[ok], period=bins)
    lo, hi = float(prof.min()), float(prof.max())
    above = prof > (lo + hi) / 2
    runs = 0 if above.all() else int(np.sum(above & ~np.roll(above, 1)))
    return runs, hi / max(lo, 1e-9)


def _piece_word(p: Piece) -> str | None:
    outer = p.rings[0].astype(np.float32)
    area = abs(cv2.contourArea(outer))
    if area < 1e-4:
        return None
    rad = np.hypot(outer[:, 0], outer[:, 1])
    if cv2.pointPolygonTest(outer.reshape(-1, 1, 2), (0.0, 0.0), False) >= 0:
        lobes, depth = _center_lobes(p.rings[0])
        if lobes >= STAR_MIN_LOBES and depth >= STAR_DEPTH:
            return f"{lobes}-point star"
        if len(p.rings) > 1:
            return "ring"
        return "center" if rad.max() <= DISC_REACH else "disc"
    w, h = cv2.minAreaRect(outer)[1]
    if max(w, h) >= STRIP_ASPECT * max(min(w, h), 1e-9):
        return "strips"
    if area / max(cv2.contourArea(cv2.convexHull(outer)), 1e-9) < TRIM_SOLIDITY:
        return "trims"
    ang = np.arctan2(outer[:, 1], outer[:, 0])
    off = np.angle(np.exp(1j * (ang - np.angle(np.exp(1j * ang).mean()))))
    lo, hi = rad.min(), rad.max()
    end = rad >= hi - OUTER_END * (hi - lo)
    if np.ptp(off[end]) >= LOOP_END * np.ptp(off):
        return "loops"
    corners = len(cv2.approxPolyDP(outer, 0.04 * max(w, h), True))
    return "kites" if corners <= 4 else "points"


def describe(t: Template) -> str:
    """A one-line description of a template's head pieces for the head card, for example
    "6 loops, 6 kites, ring, center" (a piece repeated with the same outline counts once)."""
    seen: set = set()
    counts: dict[str, int] = {}
    for p in t.pieces:
        key = (p.category, p.rings[0].round(3).tobytes())
        if key in seen:
            continue
        seen.add(key)
        w = _piece_word(p)
        if w:
            counts[w] = counts.get(w, 0) + 1
    out = []
    for plural, single in _WORDS:
        if counts.get(plural):
            out.append(f"{counts[plural]} {single if counts[plural] == 1 else plural}")
    for w in sorted(k for k in counts if k.endswith("star")) + ["disc", "ring", "center"]:
        n = counts.get(w, 0)
        if n:
            out.append(w if n == 1 else f"{n} {w}s")
    return ", ".join(out)


def photo_grid(mask: np.ndarray, labels: np.ndarray, head: tuple[float, float, float],
               relabel: dict[int, int] | None = None) -> np.ndarray:
    """Photo color-group labels (-1 background) on the head grid; relabel maps label ids
    onto the model group that stands for them."""
    lab = np.where(mask, labels, -1).astype(np.int32)
    if relabel:
        lut_keys = np.array(sorted(relabel), np.int32)
        for a in lut_keys:
            lab[lab == a] = relabel[int(a)]
    cx, cy, r = head
    return sample(lab, cx, cy, r, fill=-1)


def relabel_map(model, groups, merged: dict[int, int] | None = None) -> dict[int, int]:
    """Photo label id -> model group id for labels the model has no group for: labels
    assign merged away go to the group that absorbed them, labels no part uses to the
    model group nearest in color."""
    have = {g.id: g for g in model.groups}
    out = {int(a): int(b) for a, b in (merged or {}).items() if b in have}
    if not have:
        return out
    for g in groups:
        if g.id in have or g.id in out:
            continue
        near = min(have.values(), key=lambda h: sum((int(a) - int(b)) ** 2 for a, b in zip(g.rgb, h.rgb)))
        out[int(g.id)] = int(near.id)
    return out


def tail_consistent_mapping(labels: np.ndarray, t: Template, tail_groups: dict[int, int]) -> dict[int, int] | None:
    """Template category -> photo group that agrees with the tails (H7 fix round 1). A
    category the reference also uses on its tails (its tail Color k) goes to the group
    our tail stripes carry as Color k (tail_groups: Color -> group); only head-only
    categories are mapped by overlap (Hungarian over the groups not taken by the tail
    categories, then the most-overlapping group). None when a tail category of the head
    has no tail group of that Color (the template cannot be colored consistently)."""
    cats = [c for c in t.categories]
    out: dict[int, int] = {}
    for c in cats:
        if c in t.tail_categories:
            if c not in tail_groups:
                return None
            out[c] = int(tail_groups[c])
    rest = [c for c in cats if c not in out]
    if not rest:
        return out
    pf = (labels >= 0) & _WIN
    groups = [int(g) for g in np.unique(labels[pf])]
    if not groups:
        groups = sorted(set(tail_groups.values()))
        if not groups:
            return None
    ov = np.array([[float(((t.grid == c) & (labels == g) & _WIN).sum()) for g in groups] for c in rest])
    free = [j for j, g in enumerate(groups) if g not in out.values()]
    if free:
        rows, cols = linear_sum_assignment(-ov[:, free])
        for i, j in zip(rows, cols):
            if ov[i, free[j]] > 0:
                out[rest[i]] = groups[free[j]]
    for i, c in enumerate(rest):
        if c not in out:
            out[c] = groups[int(ov[i].argmax())]
    return out
