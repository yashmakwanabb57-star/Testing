"""A16 (C4): star valleys at the wrong radius. A star of 4 or fewer points whose valleys lie
beyond STAR_CONVEX_FEW (0.84) of its convex limit has them hidden under the layers inside it;
a pleated band of its own color under those valleys stays a pinked ring drawn over the star
(S32-15, S3-15). Synthetic labels only."""
import numpy as np

from rosette.vision import parts_head as H
from rosette.vision.parts import detect_parts
from rosette.vision.prepare import Prepared

R = 300.0
CX, CY = 450.0, 330.0


def test_convex_bar_catches_near_square_stars():
    lim = 160.0 * np.cos(np.pi / 4)
    # the drawn 4-point stars of the diagnosis measured 0.85-0.93 of their convex limit
    assert H._convex_star({"kind": "star", "count": 4, "r_inner": 0.85 * lim, "r_outer": 160.0})
    assert H._convex_star({"kind": "star", "count": 4, "r_inner": 0.93 * lim, "r_outer": 160.0})
    assert not H._convex_star({"kind": "star", "count": 4, "r_inner": 0.8 * lim, "r_outer": 160.0})
    # stars of more points keep the H11 bar (0.95): S10-19's and S9-19's 6-point stars stay
    lim6 = 160.0 * np.cos(np.pi / 6)
    assert not H._convex_star({"kind": "star", "count": 6, "r_inner": 0.9 * lim6, "r_outer": 160.0})
    assert H._convex_star({"kind": "star", "count": 6, "r_inner": 0.96 * lim6, "r_outer": 160.0})


def _photo(pleats: bool):
    """Tails (3); a white badge (2) to 0.4 R; a red band (0) 0.4-0.62 R, pleated (brightness
    stripes every 6 degrees) or plain; a red 4-point star (0) whose tips reach 0.98 R; a white
    4-point star (2) turned by 45 degrees between its tips."""
    h, w = 1200, 900
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    r, th = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx)) % 360
    labels = np.full((h, w), -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3

    def star(rot, rv, rt):
        a = np.pi / 4
        phi = np.radians(np.abs(((th - rot) % 90) - 45))
        phi = a - phi
        return rt * rv * np.sin(a) / (rv * np.sin(a - phi) + rt * np.sin(phi))
    labels[r <= star(45, 0.55 * R, 0.92 * R)] = 2
    labels[r <= star(0, 0.55 * R, 0.98 * R)] = 0
    labels[r <= 0.62 * R] = 0
    labels[r <= 0.4 * R] = 2
    rgb = np.full((h, w, 3), 200, np.uint8)
    if pleats:
        band = (r > 0.4 * R) & (r <= 0.62 * R)
        rgb[band & ((th // 3) % 2 == 0)] = 120
    p = Prepared(rgb=rgb, mask=labels >= 0, axis_x=CX, head=(CX, CY, R), tails_top_y=CY + 0.5 * R,
                 bottom_y=1149.0, scale=2.0)
    return p, labels


def _red(parts):
    return [q for q in parts if q.color_group == 0 and q.role in ("points", "ring")]


def test_a_pleated_band_over_a_stars_valleys_is_a_pinked_ring_on_top():
    p, labels = _photo(pleats=True)
    red = _red(detect_parts(p, labels))
    stars = [q for q in red if q.role == "points"]
    rings = [q for q in red if q.role == "ring"]
    assert len(stars) == 1 and stars[0].params["kind"] == "star" and stars[0].params["count"] == 4
    assert len(rings) == 1 and rings[0].params.get("pinked") == 1
    assert rings[0].z > stars[0].z  # drawn over the star
    # the valleys lie under the ring, with notches
    assert float(stars[0].params["r_inner"]) < float(rings[0].params["r_outer"])
    assert not H._convex_star(stars[0].params)


def test_a_plain_band_stays_the_stars_body():
    p, labels = _photo(pleats=False)
    red = _red(detect_parts(p, labels))
    assert [q.role for q in red] == ["points"]  # the star covers its body band (H11)


def _star_and_trims(monkeypatch, trims):
    """A convex 4-point star (label 1, read beyond a disc at 0.6 R) and trim layers added as
    the trims rule would (`trims`: (label, count) each, a copy of the star's params)."""
    h, w = 1200, 900
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx, dy = xx - CX, yy - CY
    r, th = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx)) % 360
    labels = np.full((h, w), -1, np.int32)
    labels[(yy > CY + 0.5 * R) & (yy < 1150) & (np.abs(xx - CX) < 110)] = 3
    a = np.pi / 4
    phi = a - np.radians(np.abs((th % 90) - 45))
    labels[r <= 0.95 * R * 0.3 * np.sin(a) / (0.3 * np.sin(a - phi) + 0.95 * np.sin(phi))] = 1
    labels[r <= 0.6 * R] = 2
    labels[r <= 0.2 * R] = 0
    p = Prepared(rgb=np.zeros((h, w, 3), np.uint8), mask=labels >= 0, axis_x=CX, head=(CX, CY, R),
                 tails_top_y=CY + 0.5 * R, bottom_y=1149.0, scale=2.0)
    orig = H._split_and_explain

    def with_trims(layers, meta, *args):
        layers, meta = orig(layers, meta, *args)
        k = next(i for i, lay in enumerate(layers) if lay[0] == "points" and lay[1] == 1)
        for g, n in trims:
            q = dict(layers[k][3], count=n, r_outer=float(layers[k][3]["r_outer"]) + 5.0)
            layers.append(("points", g, layers[k][2], q, 0.0, layers[k][5]))
            meta.append(dict(meta[k], trim=True))
        return layers, meta
    monkeypatch.setattr(H, "_split_and_explain", with_trims)
    return {q.color_group: q for q in detect_parts(p, labels) if q.role == "points"}


def test_a_backing_follows_its_hosts_hidden_valleys(monkeypatch):
    pts = _star_and_trims(monkeypatch, [(5, 4), (6, 8)])
    host, back, other = pts[1], pts[5], pts[6]
    assert not H._convex_star(host.params)  # its valleys moved under the disc
    assert back.params["r_inner"] == host.params["r_inner"]  # the backing copy moved with it
    assert back.params["width_ratio"] == host.params["width_ratio"]
    # a trim that is no copy of the host (another count) keeps its own valleys
    assert other.params["r_inner"] != host.params["r_inner"]
