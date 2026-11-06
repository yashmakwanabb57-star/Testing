"""Head-fit check: the photo's head segmentation against our own rendered head."""
import numpy as np

from rosette import pipeline as P
from rosette.assign import FLAG_BELOW
from rosette.model import ColorGroup, Part, RosetteModel
from rosette.vision.prepare import Prepared

# A synthetic 'photo' on the canvas grid itself: 1 px per canvas unit, axis at x=300,
# the head's top (the mask's first row) at y=25, so photo pixels are canvas units.
H, W = 850, 600
CX, CY, R = 300.0, 175.0, 150.0


def _disc(r, cx=CX, cy=CY):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def photo(layers):
    """labels: head disc group 0, then (mask, group) layers painted in order."""
    labels = np.full((H, W), -1, np.int32)
    labels[_disc(R)] = 0
    for mask, g in layers:
        labels[mask] = g
    p = Prepared(rgb=np.zeros((H, W, 3), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                 tails_top_y=CY + R, bottom_y=CY + R, scale=1.0)
    return p, labels


def model(ring_outer, ring_inner):
    groups = [ColorGroup(0, (200, 30, 30), False, 1.0, 0.5, 1, 0.9),
              ColorGroup(1, (30, 30, 200), False, 1.0, 0.5, 2, 0.9)]
    parts = [Part("head", "head", {"cx": CX, "cy": CY, "r": R}, 0, 10, 0.9),
             Part("ring-0", "ring", {"cx": CX, "cy": CY, "r_outer": ring_outer, "r_inner": ring_inner}, 1, 30, 0.9)]
    return RosetteModel(parts=parts, groups=groups)


RING = _disc(90) & ~_disc(60)


def test_matching_head_fits():
    p, labels = photo([(RING, 1)])
    fit = P.head_fit(p, labels, model(90, 60))
    assert fit is not None
    assert fit.agreement > 0.93 and fit.silhouette > 0.93  # the drawn rim is pinked


def test_mismatched_head_is_flagged():
    # the photo has its blue ring near the center, we drew it near the edge
    p, labels = photo([(RING, 1)])
    m = model(140, 115)
    fit = P.head_fit(p, labels, m)
    assert fit.agreement < P.HEAD_FIT_MIN
    out = P.check_head(p, labels, m)
    head = next(q for q in out.parts if q.role == "head")
    assert head.confidence < FLAG_BELOW
    w = [x for x in out.warnings if "head" in x.lower()]
    assert w and "check" in w[0]
    assert all("—" not in x and "–" not in x for x in out.warnings)
    assert P.is_flagged(out, [])
    assert m.parts[0].confidence == 0.9 and not m.warnings  # input untouched


def test_matching_head_is_not_flagged():
    p, labels = photo([(RING, 1)])
    out = P.check_head(p, labels, model(90, 60))
    assert not out.warnings
    assert all(q.confidence == 0.9 for q in out.parts)


def test_photo_colors_merged_by_assign_count_as_their_group():
    # group 5 was merged into group 1 by assign: its pixels are group 1's
    p, labels = photo([(RING, 1), (_disc(70) & ~_disc(60), 5)])
    fit = P.head_fit(p, labels, model(90, 60), merged={5: 1})
    assert fit.agreement > 0.93


def test_no_head_parts_no_fit():
    p, labels = photo([])
    m = RosetteModel(parts=[], groups=[])
    assert P.head_fit(p, labels, m) is None
