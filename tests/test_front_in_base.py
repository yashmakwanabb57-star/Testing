"""H15: a points layer in the base disc's own color (S70-19: a red 6-point star on a white star
over blue loops on a red pleated disc, the white showing only as chevron borders along the red
arms) is read from the base color's bordered wedges; the border is its backing. Synthetic
labels only."""
import numpy as np

from rosette.vision.parts import detect_parts
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


def base_head(white=True, red_star=True, white_tip=0.8):
    """Tails (3); a red disc (0) to 0.95 R; six blue loops (2) on 0 + 60 k from 0.3 R to 0.75 R;
    with `white` a white (1) 6-point star on 30 + 60 k (tips 0.8 R, valleys 0.36 R) over them;
    with `red_star` a red 6-point star on it (tips 0.72 R, valleys 0.33 R: white shows as
    chevrons along its arms); a red ring (0) 0.22-0.3 R and a white center (1) to 0.22 R."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    labels[r <= 0.95 * R] = 0
    labels[_blocks(dx, dy, 6, 0, 0.3 * R, 0.75 * R, 0.16 * R)] = 2
    if white:
        labels[r <= _star(th, 6, 30, 0.36 * R, white_tip * R)] = 1
    if red_star:
        labels[r <= _star(th, 6, 30, 0.33 * R, 0.72 * R)] = 0
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
