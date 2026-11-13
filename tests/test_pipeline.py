from pathlib import Path

import pytest

from rosette.model import ColorGroup, Part, RosetteModel
from rosette.pipeline import RenderResult, analyze, is_flagged, render

PHOTOS = Path(__file__).resolve().parent.parent / "reference" / "photos"


def _photo(sid: str) -> bytes:
    return (PHOTOS / f"{sid}.png").read_bytes()


def test_end_to_end_s46_valid():
    result = render(analyze(_photo("S46-15")))
    assert isinstance(result, RenderResult)
    assert result.issues == []
    assert result.svg.startswith("<svg")


def test_end_to_end_s35_valid():
    assert render(analyze(_photo("S35-19"))).issues == []


def test_analyze_assigns_every_used_group_and_drops_unused():
    model = analyze(_photo("S46-15"))
    used = {q.color_group for q in model.parts}
    assert {g.id for g in model.groups} == used
    assert all(g.category in (1, 2, 3, 4) for g in model.groups)
    assert model.rose == any(q.role == "center" for q in model.parts)


def test_render_is_deterministic():
    data = _photo("S35-19")
    a = render(analyze(data))
    b = render(analyze(data))
    assert a.svg == b.svg


def _model(role="head", warnings=()):
    g = ColorGroup(0, (200, 0, 0), False, 0.0, 1.0, 1, 0.9)
    part = Part("p", role, {"cx": 300.0, "cy": 175.0, "r": 150.0}, 0, 0, 0.9)
    return RosetteModel(parts=[part], groups=[g], warnings=list(warnings))


def test_flagged_rules():
    assert not is_flagged(_model(), [])
    assert is_flagged(_model(role="unknown"), [])
    assert is_flagged(_model(warnings=["part p: unrecognised shape, drawn as a plain disc"]), [])
    low = _model()
    low.groups[0].confidence = 0.5
    assert is_flagged(low, [])


def test_low_confidence_points_ring_is_flagged():
    # S41-15: a points ring whose rotational symmetry is weak (detector confidence
    # below FLAG_BELOW) is a misread layer as often as not - ask the user to check
    m = _model()
    pts = Part("points-1", "points", {"cx": 300.0, "cy": 175.0, "kind": "star", "count": 6, "r_inner": 50.0,
                                      "r_outer": 90.0, "rotation_deg": 0.0, "width_ratio": 0.5}, 0, 20, 0.9)
    m.parts.append(pts)
    assert not is_flagged(m, [])
    pts.confidence = 0.55
    assert is_flagged(m, [])


def test_merged_print_and_satin_is_flagged():
    # S2-15B: two colors found for a three-color ribbon (print merged with satin)
    model = analyze(_photo("S2-15B"))
    assert any("partly patterned" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)


def test_low_confidence_part_of_any_role_is_flagged():
    # head, ring, center and tail stripe confidences count, not just points
    for role in ("head", "ring", "center", "tail_stripe"):
        m = _model(role=role)
        assert not is_flagged(m, []), role
        m.parts[0].confidence = 0.55
        assert is_flagged(m, []), role


def test_s31_low_confidence_tails_are_flagged():
    # S31-15: every tail stripe at 0.545 used to come back unflagged
    model = analyze(_photo("S31-15"))
    stripes = [q for q in model.parts if q.role == "tail_stripe"]
    assert stripes and all(q.confidence < 0.6 for q in stripes)
    assert is_flagged(model, render(model).issues)


@pytest.mark.slow
def test_every_reference_analysis_passes_check_params():
    # the pipeline must never produce a model that /api/render rejects
    for f in sorted(PHOTOS.iterdir()):
        model = analyze(f.read_bytes())
        RosetteModel.from_json(model.to_json())  # runs check_params on every part


def test_unsure_badge_print_is_warned(monkeypatch):
    # a group that may be print on the center badge but is not clearly enclosed is
    # kept as a color and the user is asked to check it
    import rosette.pipeline as pl
    monkeypatch.setattr(pl, "badge_print_unsure", lambda p, labels, groups: [groups[0].id])
    model = analyze(_photo("S46-15"))
    assert any("center badge" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)


def test_merged_badge_print_is_warned(monkeypatch):
    # merging print into the badge color changes the center: never silently
    import rosette.pipeline as pl
    real = pl.pixel_groups

    def fake(p, merged=None):
        labels, groups = real(p, merged=merged)
        merged.append((217, 201, 155))
        return labels, groups
    monkeypatch.setattr(pl, "pixel_groups", fake)
    model = analyze(_photo("S46-15"))
    assert any("Center badge print was merged" in w for w in model.warnings)
    assert is_flagged(model, render(model).issues)


def test_unknown_speck_is_left_out_and_warned(monkeypatch):
    # an unknown speck the detector reports is not drawn; the model says so
    import rosette.pipeline as pl
    real = pl.detect_parts

    def fake(p, labels, *args, **kwargs):
        parts = real(p, labels, *args, **kwargs)
        g = parts[0].color_group
        return parts + [Part("unknown-9", "unknown", {"cx": 430.0, "cy": 290.0, "r": 3.0}, g, 60, 0.2)]
    monkeypatch.setattr(pl, "detect_parts", fake)
    model = analyze(_photo("S46-15"))
    assert not any(q.id == "unknown-9" for q in model.parts)
    assert any("unknown-9" in w for w in model.warnings)
    assert render(model).issues == []


def test_uncertain_head_circle_bands():
    # the fit and the symmetry circle disagree without a takeover (ratio 1.1-1.25), or the
    # takeover hung on a coverage within 0.1 of its bar: the head circle is uncertain
    from types import SimpleNamespace

    from rosette.pipeline import CIRCLE_UNCERTAIN, circle_warnings
    from rosette.vision.prepare import FIT_MIN_COVERAGE, FIT_TAKEOVER

    def p(ratio, coverage):
        return SimpleNamespace(fit_ratio=ratio, fit_coverage=coverage)
    assert circle_warnings(p(1.0, 0.9)) == []
    assert circle_warnings(p(1.0, FIT_MIN_COVERAGE)) == []  # the fit agrees: coverage is moot
    assert circle_warnings(p(1.1, 0.9)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(FIT_TAKEOVER - 0.01, 0.2)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(FIT_TAKEOVER, FIT_MIN_COVERAGE + 0.15)) == []
    assert circle_warnings(p(FIT_TAKEOVER, FIT_MIN_COVERAGE + 0.05)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(1.4, FIT_MIN_COVERAGE - 0.05)) == [CIRCLE_UNCERTAIN]
    assert circle_warnings(p(1.4, FIT_MIN_COVERAGE - 0.2)) == []
    assert "–" not in CIRCLE_UNCERTAIN and "—" not in CIRCLE_UNCERTAIN


def test_uncertain_head_circle_is_warned_and_flagged(monkeypatch):
    import rosette.pipeline as pl
    monkeypatch.setattr(pl, "circle_warnings", lambda p: [pl.CIRCLE_UNCERTAIN])
    model = analyze(_photo("S46-15"))
    assert pl.CIRCLE_UNCERTAIN in model.warnings
    assert is_flagged(model, render(model).issues)


# ------------------------------------------------------------ ML-4 reader step

import types  # noqa: E402

import rosette.perception.reader as R  # noqa: E402
import rosette.pipeline as pl  # noqa: E402
import rosette.vision.candidates as C  # noqa: E402
import rosette.vision.judge as J  # noqa: E402

READER_KEYS = {"status", "message", "seconds", "cached", "images", "decisions", "winner", "margin", "note", "tie",
               "removed"}


@pytest.fixture(scope="module")
def s46():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ROSETTE_SAM2", "off")
        mp.setenv("ROSETTE_READER", "off")
        return pl.detect(_photo("S46-15"))


def _reader_on(monkeypatch):
    monkeypatch.setenv("ROSETTE_READER", "on")
    monkeypatch.setattr(pl, "_resolve_reader", lambda settings: R.Resolved(Path("x"), Path("y"), Path("z"), ""))


def _reading(marks=()):
    marks = list(marks)
    return R.Reading(marks=marks, piece_index={m: m - 1 for m in marks},
                     answers={m: R.MarkAnswer(m, "badge", "plain", 1) for m in marks}, pairs=[], seconds=12.5,
                     cached=False, images=1 if marks else 0)


def _one_decision(monkeypatch, decision):
    monkeypatch.setattr(R, "read", lambda *a, **k: _reading([1]))
    monkeypatch.setattr(pl.vgroups, "piece_groups", lambda *a, **k: [])
    monkeypatch.setattr(C, "open_decisions", lambda rules, groups, reading: [decision] if decision else [])


def test_reader_off_leaves_detect_unchanged(monkeypatch, s46):
    assert s46.reader is not None and set(s46.reader) == READER_KEYS
    assert s46.reader["status"] == "off"
    monkeypatch.setattr(pl, "reader_step", lambda p, labels, pieces, model, env=None, merged=None: (model, None))
    plain = pl.detect(_photo("S46-15"))
    assert plain.reader is None  # the step really was left out of this run
    assert s46.model.to_json() == plain.model.to_json()
    # off: the step hands the very model back, untouched
    m = s46.model
    monkeypatch.undo()
    monkeypatch.setenv("ROSETTE_READER", "off")
    assert pl.reader_step(s46.prepared, s46.labels, [], m)[0] is m


def test_the_rules_alone_never_run_the_reader(monkeypatch):
    def no_reader(*a, **k):
        raise AssertionError("the reader ran for the classifier's training data")
    monkeypatch.setattr(pl, "reader_step", no_reader)
    assert pl.detect(_photo("S46-15"), classify=False).reader is None


def test_reader_on_without_pieces_is_skipped(monkeypatch, s46):
    _reader_on(monkeypatch)
    out, st = pl.reader_step(s46.prepared, s46.labels, None, s46.model)
    assert out is s46.model and st["status"] == "skipped" and st["message"] == "no pieces"


def test_reader_auto_without_the_files_is_off(monkeypatch, tmp_path, s46):
    monkeypatch.setenv("ROSETTE_READER", "auto")
    monkeypatch.setenv("ROSETTE_MODELS_DIR", str(tmp_path))
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert out is s46.model and st["status"] == "off"


def test_reader_on_without_the_files_is_missing_with_the_install_line(monkeypatch, tmp_path, s46):
    monkeypatch.setenv("ROSETTE_READER", "on")
    monkeypatch.setenv("ROSETTE_MODELS_DIR", str(tmp_path))
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert out is s46.model and st["status"] == "missing" and "b12342" in st["message"]


def test_empty_reading_keeps_the_rules_head(monkeypatch, s46):
    _reader_on(monkeypatch)
    monkeypatch.setattr(R, "read", lambda *a, **k: _reading())
    before = s46.model.to_json()
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert st["status"] == "skipped" and "no head pieces" in st["message"]
    assert out.to_json() == before and s46.model.to_json() == before


def test_reader_failure_is_reported_and_the_rules_head_stays(monkeypatch, s46):
    _reader_on(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("llama-server exited with 3")
    monkeypatch.setattr(R, "read", boom)
    before = s46.model.to_json()
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert st["status"] == "failed" and "llama-server exited with 3" in st["message"]
    assert out.to_json() == before
    # a fault later in the step (the judge) is reported the same way
    _one_decision(monkeypatch, C.Decision("center", ("c",), ("badge", "rose"), "reader"))

    def bad_judge(*a, **k):
        raise ValueError("render failed")
    monkeypatch.setattr(J, "judge", bad_judge)
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert st["status"] == "failed" and "render failed" in st["message"] and out.to_json() == before


def test_a_judged_candidate_replaces_the_head_with_flags(monkeypatch, s46):
    _reader_on(monkeypatch)
    state = "rose" if s46.model.rose else "badge"
    other = "badge" if state == "rose" else "rose"
    center = next(q for q in s46.model.parts if q.role == "center")
    _one_decision(monkeypatch, C.Decision("center", (center.id,), (state, other), "reader"))
    seen = {}

    def fake_judge(cands, p, labels, pieces, decisions=None, merged=None):
        seen.update(n=len(cands), decisions=decisions, merged=merged)
        return J.Verdict(winner=1, scores=[0.70, 0.75], terms=[{}, {}], margin=0.05, labels=decisions,
                         changed=[center.id])
    monkeypatch.setattr(J, "judge", fake_judge)
    before = s46.model.to_json()
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model, merged=s46.merged)
    assert seen["n"] == 2 and seen["merged"] == s46.merged
    assert seen["decisions"][0] == "rules head" and other in seen["decisions"][1]
    assert out.rose == (other == "rose")
    assert any(line.startswith("judge: rules head -> ") for line in out.info)
    assert next(q for q in out.parts if q.id == center.id).confidence <= 0.5 + 10 * 0.05
    assert s46.model.to_json() == before  # the rules head itself is not touched
    assert st["status"] == "on" and st["decisions"] == 1 and st["winner"] == 1 and st["margin"] == 0.05
    assert st["images"] == 1 and st["cached"] is False and set(st) == READER_KEYS
    assert st["message"].startswith("judge: rules head -> ")


def test_no_open_decisions_keep_the_rules_head_without_judging(monkeypatch, s46):
    _reader_on(monkeypatch)
    _one_decision(monkeypatch, None)

    def no_judge(*a, **k):
        raise AssertionError("judged without decisions")
    monkeypatch.setattr(J, "judge", no_judge)
    out, st = pl.reader_step(s46.prepared, s46.labels, [], s46.model)
    assert out.to_json() == s46.model.to_json()
    assert st["status"] == "on" and st["decisions"] == 0 and st["winner"] == 0


def test_the_reader_server_starts_once_per_process_and_stops_at_exit(monkeypatch):
    started, stopped, registered = [], [], []

    class FakeServer:
        def __init__(self, settings):
            self.settings, self.proc = settings, None

        @property
        def url(self):
            return "http://127.0.0.1:1"

        def start(self):
            started.append(self)
            self.proc = types.SimpleNamespace(poll=lambda: None)
            return self

        def stop(self):
            stopped.append(self)
            self.proc = None
    monkeypatch.setattr(pl, "_READER_SERVER", None)
    monkeypatch.setattr(pl, "_READER_ATEXIT", False)
    monkeypatch.setattr(R, "ReaderServer", FakeServer)
    monkeypatch.setattr(R, "_http_post", lambda url, body: {"url": url})
    monkeypatch.setattr(pl.atexit, "register", lambda fn: registered.append(fn))
    s = pl.perception_settings.load({})
    post = pl._reader_post(s)
    assert started == []  # lazy: a warm cache never starts it
    assert post("ignored", {})["url"] == "http://127.0.0.1:1/v1/chat/completions"
    pl._reader_post(s)("ignored", {})
    assert len(started) == 1 and len(registered) == 1
    registered[0]()
    assert stopped == started and pl._READER_SERVER is None
