"""H15: a points layer in the base disc's own color (S70-19: a red 6-point star on a white star
over blue loops on a red pleated disc, the white showing only as chevron borders along the red
arms) is read from the base color's bordered wedges; the border is its backing. Synthetic
labels only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.parts_common import Frame
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0
SCALE = 2.0  # px per canvas unit: R is 150 units
UNIT = R / SCALE


def _grid(h=1200, w=900):
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    return yy, xx, dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def _photo(labels):
    return Prepared(rgb=np.zeros(labels.shape + (3,), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                    tails_top_y=CY + 0.5 * R, bottom_y=float(labels.shape[0] - 51), scale=SCALE)


def _star(th, n, rot, r_valley, r_tip):
    """Radius of an n-point star's outline (straight sides) at angle th."""
    a = np.pi / n
    phi = np.abs(((np.degrees(th) - rot) % (360 / n)) - 180 / n)
    phi = np.radians(180 / n - phi)
    return r_tip * r_valley * np.sin(a) / (r_valley * np.sin(a - phi) + r_tip * np.sin(phi))


def _blocks(dx, dy, n, rot, r0, r1, half):
    out = np.zeros(dx.shape, bool)
    for k in range(n):
        a = np.radians(rot + k * 360 / n)
        ax = dx * np.cos(a) + dy * np.sin(a)
        lat = -dx * np.sin(a) + dy * np.cos(a)
        out |= (ax >= r0) & (ax <= r1) & (np.abs(lat) <= half)
    return out


def base_head(white=True, red_star=True, white_tip=0.8, inner_tip=None, blocks_over=False):
    """Tails (3); a red disc (0) to 0.95 R; six blue loops (2) on 0 + 60 k from 0.3 R to 0.75 R;
    with `white` a white (1) 6-point star on 30 + 60 k (tips 0.8 R, valleys 0.36 R) over them;
    with `red_star` a red 6-point star on it (tips 0.72 R, valleys 0.33 R: white shows as
    chevrons along its arms); with `inner_tip` an inner white star on the red one (tips inner_tip R,
    valleys 0.25 R, under the ring: red shows as chevrons between the two white stars, S70-19); a red ring (0)
    0.22-0.3 R and a white center (1) to 0.22 R. blocks_over: the loops lie over the white and red stars,
    under the inner star (S70-19's zoomed photo)."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    labels[r <= 0.95 * R] = 0
    labels[_blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R)] = 2
    if white:
        labels[r <= _star(th, 6, 30, 0.36 * R, white_tip * R)] = 1
    if red_star:
        labels[r <= _star(th, 6, 30, 0.33 * R, 0.72 * R)] = 0
    if blocks_over:
        labels[_blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R)] = 2
    if inner_tip:
        labels[r <= _star(th, 6, 30, 0.25 * R, inner_tip * R)] = 1
    labels[r <= 0.3 * R] = 0
    labels[r <= 0.22 * R] = 1
    return labels


def _head(labels):
    return [q for q in detect_parts(_photo(labels), labels) if q.role != "tail_stripe"]


def test_a_star_of_the_disc_color_on_a_white_star_is_a_front_layer_with_its_backing():
    parts = _head(base_head())
    head = [q for q in parts if q.role == "head"]
    assert len(head) == 1 and head[0].color_group == 0
    front = [q for q in parts if q.role == "points" and q.color_group == 0]
    white = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(front) == 1 and len(white) == 1
    f, w = front[0], white[0]
    assert f.params["kind"] == "star" and f.params["count"] == 6
    assert abs(f.params["rotation_deg"] % 60 - 30) < 3  # on the loops' half-axes, not their axes
    assert abs(f.params["r_outer"] - 0.72 * UNIT) < 7
    assert w.params["count"] == 6 and abs(w.params["r_outer"] - 0.8 * UNIT) < 7
    assert abs((w.params["rotation_deg"] - f.params["rotation_deg"] + 30) % 60 - 30) < 2
    assert head[0].z < w.z < f.z  # disc, white backing, red front star
    loops = [q for q in parts if q.role == "points" and q.color_group == 2]
    assert len(loops) == 1 and loops[0].params["count"] == 6
    assert loops[0].z < w.z  # the stars are drawn over the loops


def test_a_plain_disc_stays_a_disc():
    parts = _head(base_head(white=False, red_star=False))
    assert not [q for q in parts if q.role == "points" and q.color_group == 0]


def test_disc_showing_between_loops_without_a_border_is_not_a_layer():
    parts = _head(base_head(white=False, red_star=True))
    assert not [q for q in parts if q.role == "points" and q.color_group == 0]


def test_a_border_that_ends_at_the_wedge_tips_is_not_a_backing():
    # a backing shows beyond the front star's tips (S70-19: 101.5 -> 111.5); a band that stops
    # with the wedges is the edge of something else (S68-19's chevrons, S5-15)
    parts = _head(base_head(white_tip=0.72))
    assert not [q for q in parts if q.role == "points" and q.color_group == 0]



def test_a_border_color_that_is_not_an_n_fold_layer_of_its_own_gives_no_front_layer():
    # S68-19: the lilac along the purple "wedges" is irregular stacked chevrons, read as unknown
    # parts, not a backing star; here a white patch over one side breaks the white star's symmetry
    labels = base_head()
    yy, xx, dx, dy, r, th = _grid()
    d = np.degrees(th) % 360
    labels[(d > 190) & (d < 260) & (r > 0.45 * R) & (r < 0.7 * R) & (labels != 0)] = 1
    parts = _head(labels)
    assert not [q for q in parts if q.role == "points" and q.color_group == 0]


def test_an_inner_star_of_the_border_color_on_the_front_star_is_its_own_layer():
    # H16, S70-19: white under red under white on the same axes; the inner white star shows from
    # the ring out to its tips, the red front star only as chevrons between the two white stars
    parts = _head(base_head(inner_tip=0.55))
    front = [q for q in parts if q.role == "points" and q.color_group == 0]
    white = sorted((q for q in parts if q.role == "points" and q.color_group == 1),
                   key=lambda q: q.params["r_outer"])
    assert len(front) == 1 and len(white) == 2
    f, (inner, back) = front[0], white
    assert inner.params["kind"] == "star" and inner.params["count"] == 6
    assert abs((inner.params["rotation_deg"] - f.params["rotation_deg"] + 30) % 60 - 30) < 2
    assert abs(inner.params["r_outer"] - 0.55 * UNIT) < 7
    assert inner.params["r_inner"] == f.params["r_inner"]
    assert abs(back.params["r_outer"] - 0.8 * UNIT) < 7
    assert back.z < f.z < inner.z  # white under red under white
    # (the loops' place against the stars is the photo's vote: tests/test_front_stacking.py)


def test_the_backing_width_is_read_from_its_visible_tips_not_the_merged_white():
    # straight-sided white star (valleys 0.36 R): its width is the straight star's, not the white
    # of the inner star and the border read together (S70-19 width_ratio 1.326)
    parts = _head(base_head(inner_tip=0.55))
    back = max((q for q in parts if q.role == "points" and q.color_group == 1), key=lambda q: q.params["r_outer"])
    a = back.params
    straight = 2 * np.pi * 0.36 * UNIT / 6 / (a["r_outer"] - a["r_inner"])
    assert abs(a["width_ratio"] - straight) < 0.15 * straight
    f = next(q.params for q in parts if q.role == "points" and q.color_group == 0)
    # never narrower than its front (the fit's valley inside the front's keeps the H15 width)
    assert a["width_ratio"] * (a["r_outer"] - a["r_inner"]) >= f["width_ratio"] * (f["r_outer"] - f["r_inner"])


def test_the_backing_width_fit_needs_the_axes_and_a_radial_span():
    # the backing's own tips read on one axis of six (or over one row) are no evidence
    t, n = 360, 6
    radii = np.arange(100.0, 140.0, 2.0)
    fr = Frame(axis_x=CX, top=0.0, scale=SCALE)
    pol = np.zeros((len(radii), t), np.int32)
    params = {"count": n, "rotation_deg": 0.0}
    win = np.ones(t, bool)
    rows = range(len(radii))
    cols = [int(c) for c in (np.arange(n) * t // n)]

    def lobe(c, i, half):
        pol[i, [(c + d) % t for d in range(-half, half + 1)]] = 1

    for i in rows:
        lobe(cols[0], i, 12 - i // 2)
    assert H._straight_valley(pol, radii, rows, 1, params, win) is not None  # H16 as before
    assert H._straight_valley(pol, radii, rows, 1, params, win, fr=fr) is None  # 1 of 6 axes
    for c in cols:
        for i in rows:
            lobe(c, i, 12 - i // 2)
    v = H._straight_valley(pol, radii, rows, 1, params, win, fr=fr)
    assert v is not None
    assert H._straight_valley(pol, radii, range(0, 2), 1, params, win, fr=fr) is None  # 2 px: 1 unit
    lobe(cols[1], 5, 25)  # a jump on one axis is dropped by the median over the axes
    assert H._straight_valley(pol, radii, rows, 1, params, win, fr=fr) == v
    assert H._straight_valley(pol, radii, rows, 1, params, win) != H._straight_valley(
        np.where(np.arange(t) == -1, 0, pol), radii, rows, 1, params, win, fr=fr)


def test_the_star_halfangle_falls_with_the_radius_on_a_convex_star():
    # r_tip 100, r_valley 90, 6 points: the side dips under 90 before the valley
    r = np.linspace(85.0, 100.0, 31)
    phi = H._star_halfangle(r, 100.0, 90.0, 6)
    assert np.all(np.diff(phi) <= 1e-12) and abs(phi[0] - np.pi / 6) < 1e-9 and abs(phi[-1]) < 1e-9
    concave = H._star_halfangle(np.array([60.0, 80.0]), 100.0, 40.0, 6)
    assert np.allclose(H._star_side(concave, 100.0, 40.0, 6), [60.0, 80.0], atol=0.2)


def test_no_inner_star_where_the_front_star_holds_its_axes_from_the_ring():
    parts = _head(base_head())
    assert len([q for q in parts if q.role == "points" and q.color_group == 1]) == 1


# ML-2: SAM 2 pieces. The front layer in the disc's color is read from its own piece's outline,
# and the stacking vote counts visible pieces, so the order of the blocks against the stars comes
# from which masks are whole (S70-19's zoomed photo: the blocks lie over the white backing and the
# red star, the inner white star over the blocks' inner ends).

def _piece(mask, g):
    from rosette.perception.pieces import Piece
    ys, xs = np.nonzero(mask)
    return Piece(mask=mask, score=0.95, stability=0.95, area=int(mask.sum()),
                 bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), in_head=True, color_group=g)


def _star_pieces(blocks_on_top):
    """Pieces of base_head(inner_tip=0.55): the disc, ring and center, six blocks, the white backing's
    six chevrons, the red star and the inner white star. blocks_on_top: the blocks are whole and cut
    the backing and the red star (the inner star stays whole); else every star is whole and cuts
    the blocks."""
    yy, xx, dx, dy, r, th = _grid()
    white = r <= _star(th, 6, 30, 0.36 * R, 0.8 * R)
    red = r <= _star(th, 6, 30, 0.33 * R, 0.72 * R)
    inner = (r <= _star(th, 6, 30, 0.25 * R, 0.55 * R)) & (r > 0.3 * R)
    core = r <= 0.3 * R
    blocks = _blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R) & ~core
    red_vis = red & ~inner & ~core
    back_vis = white & ~red
    if blocks_on_top:
        red_vis, back_vis = red_vis & ~blocks, back_vis & ~blocks
        blocks = blocks & ~inner
    else:
        blocks = blocks & ~white
    d = np.degrees(th) % 360
    out = [_piece((r <= 0.95 * R) & ~white & ~blocks & ~core, 0), _piece(core & (r > 0.22 * R), 0),
           _piece(r <= 0.22 * R, 1), _piece(red_vis, 0), _piece(inner, 1)]
    for k in range(6):
        out.append(_piece(blocks & (np.abs((d - 60 * k + 180) % 360 - 180) <= 30), 2))
        out.append(_piece(back_vis & (np.abs((d - 30 - 60 * k + 180) % 360 - 180) <= 30), 1))
    return out


def _with_pieces(blocks_on_top):
    labels = base_head(inner_tip=0.55, blocks_over=True)  # the labels: S70-19's order
    return [q for q in detect_parts(_photo(labels), labels, pieces=_star_pieces(blocks_on_top))
            if q.role != "tail_stripe"]


def _stars_and_blocks(parts):
    front = [q for q in parts if q.role == "points" and q.color_group == 0]
    white = sorted((q for q in parts if q.role == "points" and q.color_group == 1), key=lambda q: q.params["r_outer"])
    blocks = [q for q in parts if q.role == "points" and q.color_group == 2]
    assert len(front) == 1 and len(white) == 2 and len(blocks) == 1
    return front[0], white[0], white[1], blocks[0]


def test_pieces_give_the_front_star_and_its_inner_star_on_one_set_of_axes():
    f, inner, back, _ = _stars_and_blocks(_with_pieces(blocks_on_top=True))
    assert f.params["source"] == "piece" and inner.params["source"] == "piece"
    assert "source" not in back.params  # the backing has no piece of its own: the rules read it
    assert f.params["kind"] == "star" and f.params["count"] == 6 and inner.params["count"] == 6
    assert abs(f.params["r_outer"] - 0.72 * UNIT) < 7 and abs(inner.params["r_outer"] - 0.55 * UNIT) < 7
    assert abs(back.params["r_outer"] - 0.8 * UNIT) < 7
    for q in (inner, back):
        assert abs((q.params["rotation_deg"] - f.params["rotation_deg"] + 30) % 60 - 30) < 3
    assert abs(f.params["rotation_deg"] % 60 - 30) < 3


def test_pieces_stack_the_blocks_over_the_outer_stars_and_under_the_inner_star(monkeypatch):
    monkeypatch.setattr(H, "PIECE_STACKING", True)  # off in the pipeline (the color gate): the mechanism
    # S70-19: the block masks are whole over the backing's and the red star's arms; the inner
    # star's mask is whole over the blocks' inner ends
    f, inner, back, blocks = _stars_and_blocks(_with_pieces(blocks_on_top=True))
    assert back.z < f.z < blocks.z < inner.z


def test_pieces_stack_whole_stars_over_cut_blocks(monkeypatch):
    # the masks say the opposite of the labels: the pieces decide
    monkeypatch.setattr(H, "PIECE_STACKING", True)
    f, inner, back, blocks = _stars_and_blocks(_with_pieces(blocks_on_top=False))
    assert blocks.z < back.z < f.z < inner.z


def test_with_piece_stacking_off_the_labels_order_the_layers():
    # PIECE_STACKING off: the masks that cut the blocks do not move them; the labels (blocks over the
    # outer stars) and H16 (the front just on its backing, the inner star on top) do
    f, inner, back, blocks = _stars_and_blocks(_with_pieces(blocks_on_top=False))
    assert not H.PIECE_STACKING
    assert back.z < f.z < blocks.z < inner.z


def test_without_pieces_the_rules_layers_stand():
    labels = base_head(inner_tip=0.55)
    a = [(q.id, q.params, q.z) for q in detect_parts(_photo(labels), labels) if q.role != "tail_stripe"]
    b = [(q.id, q.params, q.z) for q in detect_parts(_photo(labels), labels, pieces=None) if q.role != "tail_stripe"]
    assert a == b and not any("source" in p for _, p, _ in a)


def test_a_piece_tip_in_the_tails_sector_still_counts():
    # A16-17: a piece inside the head is read over the full circle (the tails window is for labels);
    # an 8-point star with a tip straight down reads 8, not 9
    yy, xx, dx, dy, r, th = _grid()
    labels = base_head(white=False, red_star=False)
    star = (r <= _star(th, 8, 90, 0.4 * R, 0.7 * R)) & (r > 0.3 * R)
    parts = [q for q in detect_parts(_photo(labels), labels, pieces=[_piece(star, 0)]) if q.role != "tail_stripe"]
    front = [q for q in parts if q.role == "points" and q.color_group == 0]
    assert len(front) == 1 and front[0].params["count"] == 8 and front[0].params["source"] == "piece"
    assert abs((front[0].params["rotation_deg"] + 22.5) % 45 - 22.5) < 3  # a tip on 90 deg
    assert abs(front[0].params["r_outer"] - 0.7 * UNIT) < 5 and abs(front[0].params["r_inner"] - 0.4 * UNIT) < 5


def test_an_inner_piece_lies_on_its_front_piece_without_a_backing():
    # no backing (no white border beyond the red tips): with the vote on labels the front is void
    # and is drawn just under its inner layer
    yy, xx, dx, dy, r, th = _grid()
    labels = base_head(white=False, red_star=False)
    inner = (r <= _star(th, 6, 30, 0.35 * R, 0.55 * R)) & (r > 0.3 * R)
    labels[inner] = 1
    red = (r <= _star(th, 6, 30, 0.4 * R, 0.75 * R)) & (r > 0.3 * R) & ~inner
    parts = [q for q in detect_parts(_photo(labels), labels, pieces=[_piece(red, 0), _piece(inner, 1)])
             if q.role != "tail_stripe"]
    f = [q for q in parts if q.role == "points" and q.color_group == 0]
    w = [q for q in parts if q.role == "points" and q.color_group == 1]
    assert len(f) == 1 and len(w) == 1 and w[0].params.get("source") == "piece"
    blocks = [q for q in parts if q.role == "points" and q.color_group == 2]
    assert f[0].z < w[0].z and all(b.z < f[0].z for b in blocks)
