from pathlib import Path

import numpy as np
import pytest

from rosette.build import build
from rosette.model import Part, RosetteModel
from rosette.validate import validate
from rosette.vision.colors import pixel_groups
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared, prepare
from rosette.writer import write_svg
from tests.helpers import fan_rosette_png, synthetic_photo, truth_from_svg

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"
IDS = ["S35-19", "S46-15", "S13-15", "S28-15", "S41-15"]
Z = {"head": (10, 10), "points": (20, 29), "ring": (30, 49), "center": (50, 50),
     "tail_stripe": (0, 9), "unknown": (0, 99)}
KEYS = {
    "head": {"cx", "cy", "r", "pinked"},
    "ring": {"cx", "cy", "r_outer", "r_inner"},
    "points": {"cx", "cy", "kind", "count", "r_inner", "r_outer", "rotation_deg", "width_ratio"},
    "center": {"cx", "cy", "r"},
    "tail_stripe": {"tail_index", "stripe_index", "tails_count", "top_y", "bottom_y", "top_width",
                    "bottom_width", "spread_deg", "stripes", "stripes_bottom"},
    "unknown": {"cx", "cy", "r"},
}

_cache: dict = {}


def detected(sid):
    if sid not in _cache:
        p = prepare(synthetic_photo(sid))
        labels, groups = pixel_groups(p)
        _cache[sid] = (p, labels, groups, detect_parts(p, labels))
    return _cache[sid]


def counts(parts: list[Part]) -> dict:
    pts = [q for q in parts if q.role == "points"]
    tails = {int(q.params["tails_count"]) for q in parts if q.role == "tail_stripe"}
    heads = [q for q in parts if q.role == "head"]
    assert len(tails) <= 1 and len(heads) <= 1
    return dict(points=int(pts[0].params["count"]) if pts else 0,
                tails=tails.pop() if tails else 0,
                rings=sum(q.role == "ring" for q in parts),
                pinked=bool(int(heads[0].params["pinked"])) if heads else False)


@pytest.mark.parametrize("sid", IDS)
def test_counts_match_designer_svg(sid):
    truth = truth_from_svg(sid)
    got = counts(detected(sid)[3])
    assert got["points"] == truth["points"], (got, truth)
    assert got["tails"] == truth["tails"], (got, truth)
    assert got["rings"] == truth["rings"], (got, truth)
    assert got["pinked"] == truth["pinked"], (got, truth)


@pytest.mark.parametrize("sid", IDS)
def test_parts_well_formed(sid):
    p, labels, groups, parts = detected(sid)
    ids = {g.id for g in groups}
    assert len({q.id for q in parts}) == len(parts)
    for q in parts:
        # edge strips of two-tone loops carry their strip width (H4)
        extra = {"edge_width"} if q.role == "points" else set()
        assert KEYS[q.role] <= set(q.params) <= KEYS[q.role] | extra, q
        assert q.color_group in ids
        assert 0.0 <= q.confidence <= 1.0
        lo, hi = Z[q.role]
        assert lo <= q.z <= hi, q
    heads = [q for q in parts if q.role in ("head", "points", "ring", "center")]
    for q in heads:  # canvas units: on the axis, inside the 600x850 frame
        assert q.params["cx"] == pytest.approx(300, abs=1e-6)
        assert 25 < q.params["cy"] < 425
    for q in parts:  # points rings span a real radial range
        if q.role == "points":
            assert q.params["r_outer"] - q.params["r_inner"] >= 5, q
            assert 0 < q.params["width_ratio"] < 10, q
    tails = [q for q in parts if q.role == "tail_stripe"]
    assert tails
    for q in tails:
        assert q.params["bottom_y"] == pytest.approx(825, abs=1)
        assert q.params["top_y"] < 425
        fr = [float(x) for x in str(q.params["stripes"]).split(",")]
        assert sum(fr) == pytest.approx(1, abs=1e-3)


def test_s35_points_and_tails_measurements():
    # Supplementary: the expected numbers below were measured by hand on the S35-19
    # designer file (not derived from it in code); the truth-derived checks are above.
    parts = detected("S35-19")[3]
    pts = [q for q in parts if q.role == "points"]
    # dominant layer: 6 blue bows, two of them horizontal (0 and 180 deg) in the designer file
    top = pts[0].params
    assert top["kind"] == "bow"
    rot = float(top["rotation_deg"]) % 60
    assert min(rot, 60 - rot) < 6  # bows at k*60
    # center disc ~0.3 head radii, ring 0.3-0.5 head radii (designer file)
    head_r = 142.0
    ctr = [q for q in parts if q.role == "center"]
    assert len(ctr) == 1 and ctr[0].params["r"] == pytest.approx(0.30 * head_r, rel=0.2)
    t = [q for q in parts if q.role == "tail_stripe"]
    # one fanned ribbon, 5 stripes red|blue|yellow|blue|red, mirrored
    assert len(t) == 5
    fr = [float(x) for x in str(t[0].params["stripes"]).split(",")]
    assert fr == pytest.approx(fr[::-1], abs=1e-6)
    order = sorted(t, key=lambda q: int(q.params["stripe_index"]))
    cg = [q.color_group for q in order]
    assert cg == cg[::-1] and len(set(cg)) == 3
    # tails 210-410 wide at the bottom in the designer file (x 145..455)
    assert t[0].params["bottom_width"] == pytest.approx(310, rel=0.1)


def test_detected_model_builds_and_validates():
    p, labels, groups, parts = detected("S46-15")
    for i, g in enumerate(groups):
        g.category = 1 + i % 3
    svg = write_svg(build(RosetteModel(parts=parts, groups=groups)))
    assert validate(svg) == []


def test_unclassifiable_region_becomes_unknown():
    # a plain disc head with a stray blob beside it (above the tails): never dropped
    h, w = 1200, 800
    yy, xx = np.mgrid[:h, :w]
    labels = np.full((h, w), -1, np.int32)
    head = (xx - 400) ** 2 + (yy - 300) ** 2 <= 250 ** 2
    tail = (yy > 550) & (yy < 1150) & (np.abs(xx - 400) < 100 + (yy - 550) * 0.1)
    blob = (xx - 720) ** 2 + (yy - 120) ** 2 <= 40 ** 2
    labels[head | tail] = 0
    labels[blob] = 1
    mask = labels >= 0
    p = Prepared(rgb=np.zeros((h, w, 3), np.uint8), mask=mask, axis_x=400.0, head=(400.0, 300.0, 250.0),
                 tails_top_y=551.0, bottom_y=1149.0, scale=(1149 - 50) / 800)
    parts = detect_parts(p, labels)
    unk = [q for q in parts if q.role == "unknown"]
    assert len(unk) == 1 and unk[0].color_group == 1
    assert unk[0].confidence < 0.5
    heads = [q for q in parts if q.role == "head"]
    assert len(heads) == 1 and heads[0].params["pinked"] == 0


FAN = dict(top_y=300.0, top_width=60.0, bottom_width=110.0, stripes=(0.3, 0.4, 0.3))


@pytest.mark.parametrize("spread", [60.0, 40.0, 28.0])  # 28: tails part only below ~80 % height
def test_three_fanned_tails_constructed(spread):
    # head disc + 3 tails drawn by geometry.tails with known params, through the
    # normal photo path
    p = prepare(fan_rosette_png(3, spread, **FAN))
    labels, groups = pixel_groups(p)
    parts = detect_parts(p, labels)
    t = [q for q in parts if q.role == "tail_stripe"]
    assert {int(q.params["tails_count"]) for q in t} == {3}
    q = t[0].params
    assert float(q["spread_deg"]) == pytest.approx(spread, abs=5)
    assert float(q["top_width"]) == pytest.approx(FAN["top_width"], rel=0.1)
    assert float(q["bottom_width"]) == pytest.approx(FAN["bottom_width"], rel=0.1)
    for k in range(3):
        stripes = [x for x in t if int(x.params["tail_index"]) == k]
        assert len(stripes) == 3, k
        fr = [float(v) for v in str(stripes[0].params["stripes"]).split(",")]
        assert fr == pytest.approx(list(FAN["stripes"]), abs=0.05)
        # left to right: three different colors, the same on every tail
        cg = [x.color_group for x in sorted(stripes, key=lambda x: int(x.params["stripe_index"]))]
        assert len(set(cg)) == 3
    assert not [x for x in parts if x.role == "unknown"]


def _stripe_width_at(q, y):
    """Canvas width of tail stripe q at canvas row y (top and bottom fractions)."""
    p = q.params
    i = int(p["stripe_index"])
    top = [float(x) for x in str(p["stripes"]).split(",")]
    bottom = [float(x) for x in str(p.get("stripes_bottom", p["stripes"])).split(",")]
    t = (y - p["top_y"]) / (p["bottom_y"] - p["top_y"])
    return (1 - t) * top[i] / sum(top) * p["top_width"] + t * bottom[i] / sum(bottom) * p["bottom_width"]


def test_constant_width_centre_ribbon_on_widening_tail():
    # S26-15 photo: the pink centre ribbon stays 88 px wide down the tail while the
    # tail widens (326 -> 428 px between 50 % and 95 % of the tails' height); one
    # set of stripe fractions for top and bottom made it 30 % too wide at the bottom
    p = prepare((PHOTOS / "S26-15.png").read_bytes())
    labels, _ = pixel_groups(p)
    parts = detect_parts(p, labels)
    t = [q for q in parts if q.role == "tail_stripe"]
    centre = next(q for q in t if int(q.params["stripe_index"]) == 3)
    y_of = lambda f: 25 + (p.tails_top_y + f * (p.bottom_y - p.tails_top_y)  # noqa: E731
                           - np.flatnonzero(p.mask.any(axis=1))[0]) / p.scale
    w50, w95 = _stripe_width_at(centre, y_of(0.5)), _stripe_width_at(centre, y_of(0.95))
    assert w95 == pytest.approx(88 / p.scale, rel=0.08)
    assert w95 / w50 == pytest.approx(1.0, abs=0.08)


def test_star_tips_found_from_their_blobs_when_pixel_symmetry_is_weak():
    # S40-15: six black star tips peek out between gold points under an inner
    # pleated ring. They are clearly six evenly spread blobs, but the hand-made
    # layout and a head center a few units off keep the pixel-level rotational
    # self-correlation far below SYM_T, so the layer was an 'unknown' disc.
    p = prepare((PHOTOS / "S40-15.png").read_bytes())
    labels, groups = pixel_groups(p)
    dark = min(groups, key=lambda g: sum(g.rgb)).id
    pts = [q for q in detect_parts(p, labels) if q.role == "points" and q.color_group == dark]
    assert len(pts) == 1 and pts[0].params["count"] == 6
    # H11: the tips continue under the gold points drawn above them, down to the ring
    assert 40 < pts[0].params["r_inner"] < 50 < pts[0].params["r_outer"] < 110



def test_pleated_ring_drawn_past_its_full_coverage_radii():
    # S30-15: the light purple pleated ring around the badge reaches ~82 units, but
    # its pleats keep its coverage between 0.6 and 0.8 beyond 70 units, so the ring
    # was drawn only from 58 to 70 and white showed between it and the loops.
    p = prepare((PHOTOS / "S30-15.png").read_bytes())
    labels, groups = pixel_groups(p)
    rings = [q for q in detect_parts(p, labels) if q.role == "ring"]
    assert len(rings) == 1 and rings[0].params["r_outer"] >= 76


def test_blob_read_points_layer_is_flagged_without_pixel_symmetry():
    # A layer read from its blobs has only their evenness behind it (>= 0.8 by
    # construction); without the pixel symmetry at the same count agreeing, its kind
    # and count are unverified and must be confirmed (confidence below FLAG_BELOW).
    from rosette.assign import FLAG_BELOW
    p = prepare((PHOTOS / "S40-15.png").read_bytes())
    labels, groups = pixel_groups(p)
    dark = min(groups, key=lambda g: sum(g.rgb)).id
    pts = [q for q in detect_parts(p, labels) if q.role == "points" and q.color_group == dark]
    assert len(pts) == 1 and pts[0].confidence < FLAG_BELOW


def _grow(label_cov, other_cov=None, bands_extra=(), layers=()):
    """_ring_outer on a constructed profile: radii 0..59 (1 unit each), a ring band of
    label 0 over 10..20, label 0's coverage `label_cov` beyond it."""
    from rosette.vision.parts_common import Frame
    from rosette.vision.parts_head import _Band, _ring_outer
    n = 60
    cov = np.zeros((n, 2))
    cov[10:20, 0] = 1.0
    cov[20:, 0] = label_cov
    cov[20:, 1] = other_cov if other_cov is not None else 1 - label_cov
    dom = cov.argmax(axis=1)
    bd = _Band(10, 20, 0, 1.0)
    bands = [bd] + [_Band(i0, i1, 1, 1.0) for i0, i1 in bands_extra]
    return _ring_outer(bd, bands, dom, cov, np.arange(n, dtype=float), Frame(0.0, 0.0, 1.0), list(layers))


def test_ring_growth_stops_at_another_band():
    assert _grow(0.7, bands_extra=[(23, 30)]) == 23


def test_ring_growth_stops_at_another_label_points_layer_that_shows_there():
    # a points layer of label 1 starting at 24 units and covering 0.45 (>= 0.4) of the
    # circle there: the ring must not be drawn over its lobes
    assert _grow(0.65, other_cov=0.45, layers=[(24.0, 1)]) == 24
    # the same layer barely showing (under 0.4) does not stop the ring
    assert _grow(0.65, other_cov=0.2, layers=[(24.0, 1)]) > 24


def test_ring_growth_is_capped():
    # coverage that stays high far out is not a pleated edge any more
    assert _grow(0.7) <= 20 + 8


def _ragged(seed: int, amp: float, n: int = 720) -> np.ndarray:
    """Smooth irregular radial offsets (px) around the circle: hand-pleated edges."""
    from scipy import ndimage
    v = ndimage.gaussian_filter1d(np.random.default_rng(seed).normal(0, 1, n), 3, mode="wrap")
    return amp * v / np.abs(v).max()


def test_pleated_inner_layer_without_symmetry_is_a_ring():
    # A7-15: a light pleated layer between the badge and the outer pleats. Its middle
    # covers the full circle, its edges are irregular (no rotational symmetry): it is a
    # concentric layer, drawn by the designer as an annulus, not an 'unknown' disc.
    from rosette.assign import FLAG_BELOW
    h, w = 1200, 800
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - 400, yy - 300)
    th = (np.degrees(np.arctan2(yy - 300, xx - 400)) % 360 * 2).astype(int) % 720
    labels = np.full((h, w), -1, np.int32)
    labels[(d <= 250) | ((yy > 550) & (yy < 1150) & (np.abs(xx - 400) < 100 + (yy - 550) * 0.1))] = 0
    labels[(d >= 100 + _ragged(1, 30)[th]) & (d < 165 + _ragged(2, 30)[th])] = 1
    p = Prepared(rgb=np.zeros((h, w, 3), np.uint8), mask=labels >= 0, axis_x=400.0, head=(400.0, 300.0, 250.0),
                 tails_top_y=551.0, bottom_y=1149.0, scale=(1149 - 50) / 800)
    parts = detect_parts(p, labels)
    assert not [q for q in parts if q.role == "unknown"]
    rings = [q for q in parts if q.role == "ring" and q.color_group == 1]
    assert len(rings) == 1
    q = rings[0]
    # true edges: 100 +- 30 px and 165 +- 30 px, i.e. about 73 and 120 units
    assert 66 < q.params["r_inner"] < 80 and 110 < q.params["r_outer"] < 126
    assert FLAG_BELOW <= q.confidence < 1.0


def _pleated(profile, band):
    """_pleated_ring on a constructed coverage profile of label 0 (radii 30, 31, ... units,
    1 unit each, zone = the whole profile) with a full band `band` dropped as a bridge."""
    from rosette.vision.parts_common import Frame
    from rosette.vision.parts_head import _Band, _pleated_ring
    cov = np.zeros((len(profile), 1))
    cov[:, 0] = profile
    radii = np.arange(len(profile), dtype=float) + 30
    return _pleated_ring(0, 0, len(profile), [_Band(band[0], band[1], 0, 1.0)], cov, radii,
                         Frame(0.0, 0.0, 1.0))


def test_pleated_annulus_profile_is_a_confident_ring():
    from rosette.assign import FLAG_BELOW
    prof = np.zeros(70)
    prof[20:26] = np.linspace(0.1, 0.8, 6)  # pleat teeth: a few units of ramp each side
    prof[26:44] = 1.0
    prof[44:50] = np.linspace(0.8, 0.1, 6)
    i0, i1, g, conf = _pleated(prof, (26, 44))
    assert conf >= FLAG_BELOW and 20 <= i0 < 26 and 44 < i1 <= 50


def test_poor_ring_fit_stays_under_flag_below():
    # a full band holding little of the layer: most of it lies loose far outside
    from rosette.assign import FLAG_BELOW
    prof = np.zeros(90)
    prof[20:30] = 1.0
    prof[30:90] = 0.3
    assert _pleated(prof, (20, 30))[3] < FLAG_BELOW


def test_star_with_solid_body_and_tips_is_not_a_confident_ring():
    # the solid body covers the circle, the tips reach 25 units beyond it (pleat teeth
    # reach <= 12): a star, so the ring drawn for it must be flagged
    from rosette.assign import FLAG_BELOW
    prof = np.zeros(80)
    prof[18:20] = 0.7
    prof[20:40] = 1.0
    prof[40:65] = np.linspace(0.5, 0.03, 25)
    assert _pleated(prof, (20, 40))[3] < FLAG_BELOW


def test_loops_covering_the_circle_only_at_their_base_are_not_a_confident_ring():
    # wide loops overlap (full coverage) only over a narrow band of their length
    from rosette.assign import FLAG_BELOW
    prof = np.zeros(80)
    prof[10:60] = 0.7
    prof[30:42] = 1.0
    assert _pleated(prof, (30, 42))[3] < FLAG_BELOW
