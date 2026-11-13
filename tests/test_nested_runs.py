"""R19: nested same-axis stars read from the color runs along a drawn star's axes (s44-15: a
black front star on gold, black and gold chevron stars nested on one set of axes, a white
4-point star between their arms). Synthetic labels only."""
import numpy as np

from rosette.assign import FLAG_BELOW
from rosette.model import Part
from rosette.vision import parts_head as H
from rosette.vision.parts_common import Frame, polar_sample
from tests.test_front_in_base import CX, CY, R, UNIT, _grid, _head, _photo, _star

TIPS = (0.77, 0.67, 0.59, 0.48)  # gold, black, gold, black: the shorter on top


def nested_head(tips=TIPS, colors=(2, 0, 2, 0), white=True, band=None, notch=0.0):
    """Tails (3); a black disc (0) to 0.95 R; a white (1) 4-point star on 45 + 90 k (tips 0.85 R,
    valleys 0.3 R) with `white`; 4-point stars on 0 + 90 k (valleys 0.3 R) of `colors` with
    `tips` (R), outermost first, each over the last; `band` (r0, r1, color): a full band painted
    over them, less a notch of `notch` deg at each valley ray (45 + 90 k); a black ring 0.28-0.34 R
    and a white center to 0.28 R."""
    yy, xx, dx, dy, r, th = _grid()
    labels = np.full(r.shape, -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    labels[r <= 0.95 * R] = 0
    if white:
        labels[r <= _star(th, 4, 45, 0.3 * R, 0.85 * R)] = 1
    for tip, g in zip(tips, colors):
        labels[r <= _star(th, 4, 0, 0.3 * R, tip * R)] = g
    if band is not None:
        off = np.abs((np.degrees(th) - 45 + 45) % 90 - 45)
        labels[(r >= band[0] * R) & (r <= band[1] * R) & (off >= notch / 2)] = band[2]
    labels[r <= 0.34 * R] = 0
    labels[r <= 0.28 * R] = 1
    return labels


def _on_axes(parts):
    return sorted((q for q in parts if q.role == "points" and q.params["count"] == 4
                   and min(float(q.params["rotation_deg"]) % 90, 90 - float(q.params["rotation_deg"]) % 90) < 5),
                  key=lambda q: -float(q.params["r_outer"]))


def test_each_nested_run_on_the_axes_is_a_star_and_the_shorter_lies_on_top():
    parts = _head(nested_head())
    stars = _on_axes(parts)
    got = [(q.color_group, round(float(q.params["r_outer"]) / UNIT, 2)) for q in stars]
    assert [g for g, _ in got] == [2, 0, 2, 0], got
    for (g, tip), want in zip(got, TIPS):
        assert abs(tip - want) < 0.04, got
    assert all(q.params["kind"] == "star" for q in stars)
    assert [q.z for q in stars] == sorted(q.z for q in stars)  # outermost lowest
    assert len({q.z for q in stars}) == 4
    assert max(q.z for q in parts if q.role == "points") < 30  # under the rings


def test_two_nested_stars_add_nothing():
    # the front and its backing (H15) are drawn already: no run is left on their axes
    parts = _head(nested_head(tips=TIPS[:2], colors=(2, 0)))
    assert len(_on_axes(parts)) == 2


def test_a_band_the_ring_reader_draws_is_no_star():
    # a band of one color around the circle drawn as a ring (a pleated ring: S18-15, S53-19) is no
    # nested star, even with stars of another color outside it
    parts = _head(nested_head(tips=TIPS[:2], colors=(2, 0), band=(0.4, 0.5, 2)))
    stars = _on_axes(parts)
    assert not any(q.color_group == 2 and abs(float(q.params["r_outer"]) / UNIT - 0.5) < 0.04 for q in stars)


def _polar(labels):
    """`detect_head`'s polar labels, radii and window for `labels`, with its Frame."""
    p = _photo(labels)
    fr = Frame.of(p)
    radii = np.arange(fr.scale / 2, R, fr.scale)
    pol = polar_sample(labels.astype(np.int32), CX, CY, radii, H.RING_THETA, fill=-1)
    theta = np.degrees(np.linspace(0, 2 * np.pi, H.RING_THETA, endpoint=False))
    return pol, radii, np.abs(theta - 90.0) > H.TAIL_SECTOR_DEG, fr


def _hosts(z=(20, 21)):
    """The front (black, tips 0.67 R) and its gold backing (tips 0.77 R) as H15 draws them."""
    def star(pid, g, tip, z):
        return Part(pid, "points", {"cx": 300.0, "cy": 167.5, "kind": "star", "count": 4, "r_inner": 0.34 * UNIT,
                                    "r_outer": tip * UNIT, "rotation_deg": 0.0, "width_ratio": 1.0}, g, z, 0.9)
    return [star("front-1", 2, 0.77, z[0]), star("front-2", 0, 0.67, z[1])]


def test_a_full_band_of_one_color_on_the_axes_is_no_star():
    # a gold band inside the front holding all the circle but four 6-deg notches on the valley rays
    # (coverage 0.92, n-fold: a star's body, S80-22 0.94) and no ring drawn for it: its coverage is
    # FULL_COV or more, so it is no nested star (without that guard it is one)
    pol, radii, win, fr = _polar(nested_head(tips=TIPS[:2], colors=(2, 0), band=(0.4, 0.5, 2), notch=6.0))
    parts = _hosts()
    H._nested_runs(parts, pol, radii, win, fr)
    assert len(parts) == 2


def test_the_nested_stars_are_flagged_their_z_assumed():
    pol, radii, win, fr = _polar(nested_head())
    parts = _hosts()
    H._nested_runs(parts, pol, radii, win, fr)
    new = parts[2:]
    assert [q.color_group for q in new] == [2, 0]
    assert all(q.confidence < FLAG_BELOW for q in new)


def test_no_z_left_under_the_rings_flags_the_stars_on_those_axes_instead():
    pol, radii, win, fr = _polar(nested_head())
    parts = _hosts(z=(28, 29))
    H._nested_runs(parts, pol, radii, win, fr)
    assert len(parts) == 2 and max(q.z for q in parts) == 29
    assert all(q.confidence < FLAG_BELOW for q in parts)
