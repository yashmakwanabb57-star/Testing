"""H4: points layers drawn the way the designer draws them. Synthetic label images only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

# Lobe width profile of the S35-19 photo's loops (inner -> outer, share of the widest):
# the pleated ring hides the base, the folded loop end narrows the tip.
PHOTO_LOOP = [0.26, 0.56, 0.77, 0.92, 1.0, 0.99, 0.92, 0.86, 0.65, 0.50, 0.05]


def _profile(values, n=80):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)


def test_wide_lobe_keeping_its_width_is_a_loop_block():
    w = 90 * _profile(PHOTO_LOOP)
    assert H._kind(w) == "petal"  # by its width profile alone a pointed oval
    # covering 0.6 of the angular spacing and >= 0.85 of its widest over most radii
    assert H._kind(w, np.full(len(w), 150.0)) == "bow"
    # the same profile on a narrow lobe (0.3 of the spacing) stays a petal
    assert H._kind(w, np.full(len(w), 300.0)) == "petal"
    # a sine lobe has no plateau: a petal however wide
    sine = 90 * np.sin(np.pi * np.linspace(0.02, 0.98, 80))
    assert H._kind(sine, np.full(80, 150.0)) == "petal"


def loops_photo(n=6, profile=PHOTO_LOOP, edge_px=0, backing_px=0):
    """Labels of a head: center (2) r < 0.3 R, ring (0) to 0.5 R, n loops (1) from
    0.47 R (just under the ring, whose pleat teeth reach 0.53 R) to 0.9 R with the lateral `profile`, and a tail (3) below.
    edge_px: a band of label 0 that wide along both loop sides. backing_px: n diamonds
    (4) between the loops, 0.5-0.85 R, on a label-0 copy reaching backing_px further."""
    h, w = 1200, 900
    cx, cy, R = 450.0, 330.0, 300.0
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    labels = np.full((h, w), -1, np.int32)
    labels[(yy > cy + 0.5 * R) & (yy < 1150) & (np.abs(xx - cx) < 120 + (yy - cy) * 0.1)] = 3
    prof = np.asarray(profile, float)

    def lobes(rot, r0, r1, half, prof, pad=0.0, tip_pad=0.0):
        out = np.zeros_like(r, bool)
        for k in range(n):
            a = np.radians(rot + k * 360 / n)
            ax = dx * np.cos(a) + dy * np.sin(a)
            lat = -dx * np.sin(a) + dy * np.cos(a)
            t = np.clip((ax - r0) / (r1 - r0), 0, 1)
            hw = half * np.interp(t, np.linspace(0, 1, len(prof)), prof) + pad
            out |= (ax >= r0) & (ax <= r1 + tip_pad) & (np.abs(lat) <= hw)
        return out
    if backing_px:
        kite = [0, 0.4, 0.7, 0.97, 1, 0.97, 0.8, 0.6, 0.4, 0.2, 0]
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite, backing_px / 3, backing_px)] = 0
        labels[lobes(180 / n, 0.5 * R, 0.85 * R, 0.06 * R, kite)] = 4
    if edge_px:
        labels[lobes(0, 0.47 * R, 0.9 * R, 0.2 * R, prof, edge_px)] = 0
    labels[lobes(0, 0.47 * R, 0.9 * R, 0.2 * R, prof)] = 1
    labels[r <= 0.5 * R] = 0
    bins = ((np.degrees(np.arctan2(dy, dx)) % 360) * 2).astype(int) % 720
    reach = np.random.default_rng(1).uniform(0.49, 0.555, 720)  # hand-pleated teeth
    labels[r <= reach[bins] * R] = 0
    labels[r <= 0.3 * R] = 2
    p = Prepared(rgb=np.zeros((h, w, 3), np.uint8), mask=labels >= 0, axis_x=cx, head=(cx, cy, R),
                 tails_top_y=cy + 0.5 * R, bottom_y=1149.0, scale=2.0)
    return p, labels


def test_photo_loops_are_loop_blocks_starting_at_the_ring():
    p, labels = loops_photo()
    parts = detect_parts(p, labels)
    loops = [q for q in parts if q.role == "points" and q.color_group == 1]
    rings = [q for q in parts if q.role == "ring"]
    assert len(loops) == 1 and len(rings) == 1
    q = loops[0].params
    assert q["kind"] == "bow" and q["count"] == 6
    # the loops run under the ring (its pleat teeth): they start where the ring ends
    assert rings[0].params["r_outer"] > 77
    assert q["r_inner"] == rings[0].params["r_outer"]
    # drawn with the designer's loop profile, not a photo-measured inner width
    assert "inner_ratio" not in q

