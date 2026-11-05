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

IDS = ["S35-19", "S46-15", "S13-15", "S28-15", "S41-15"]
Z = {"head": (10, 10), "points": (20, 29), "ring": (30, 49), "center": (50, 50),
     "tail_stripe": (0, 9), "unknown": (0, 99)}
KEYS = {
    "head": {"cx", "cy", "r", "pinked"},
    "ring": {"cx", "cy", "r_outer", "r_inner"},
    "points": {"cx", "cy", "kind", "count", "r_inner", "r_outer", "rotation_deg", "width_ratio"},
    "center": {"cx", "cy", "r"},
    "tail_stripe": {"tail_index", "stripe_index", "tails_count", "top_y", "bottom_y", "top_width",
                    "bottom_width", "spread_deg", "stripes"},
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
        assert set(q.params) == KEYS[q.role], q
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
