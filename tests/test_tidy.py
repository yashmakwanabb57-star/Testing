import math

from rosette import tidy
from rosette.model import ColorGroup, Part, RosetteModel


def group(i, pattern=False):
    return ColorGroup(i, (10 * i, 20, 30), pattern, 10.0 if pattern else 1.0, 0.3, None, 0.0)


def head(r=160.0):
    return Part("head", "head", {"cx": 300, "cy": 190, "r": r, "pinked": 1}, 0, 10, 0.9)


def unknown(pid, r, cx=430.0, cy=290.0, g=0):
    return Part(pid, "unknown", {"cx": cx, "cy": cy, "r": r}, g, 60, 0.2)


def test_tiny_unknown_is_dropped_with_a_warning_naming_it():
    # a speck of 0.4 % of the head on the rim (A13-15B: r 9.6 and 10.2 on a head of r 163.6)
    m = RosetteModel(parts=[head(), unknown("unknown-0", 10.0)], groups=[group(0)])
    out = tidy.drop_specks(m)
    assert [q.id for q in out.parts] == ["head"]
    assert len(out.warnings) == 1 and "unknown-0" in out.warnings[0]
    assert [q.id for q in m.parts] == ["head", "unknown-0"]  # input untouched


def test_tiny_unknown_is_dropped_wherever_it_lies():
    inside = unknown("unknown-0", 8.0, cx=300.0, cy=150.0)  # inside the head disc
    below = unknown("unknown-1", 8.0, cx=300.0, cy=600.0)  # down on the tails
    out = tidy.drop_specks(RosetteModel(parts=[head(), inside, below], groups=[group(0)]))
    assert [q.id for q in out.parts] == ["head"]
    assert len(out.warnings) == 2


def test_larger_unknown_is_kept():
    r = 160.0 * math.sqrt(2 * tidy.SPECK_MAX_HEAD_SHARE)  # twice the bar
    m = RosetteModel(parts=[head(), unknown("unknown-0", r)], groups=[group(0)])
    out = tidy.drop_specks(m)
    assert [q.id for q in out.parts] == ["head", "unknown-0"]
    assert out.warnings == []


def test_without_a_head_nothing_is_dropped():
    m = RosetteModel(parts=[unknown("unknown-0", 2.0)], groups=[group(0)])
    assert [q.id for q in tidy.drop_specks(m).parts] == ["unknown-0"]


def test_real_small_parts_are_never_dropped():
    c = Part("center", "center", {"cx": 300, "cy": 190, "r": 5.0}, 0, 50, 0.9)
    out = tidy.drop_specks(RosetteModel(parts=[head(), c], groups=[group(0)]))
    assert [q.id for q in out.parts] == ["head", "center"]
