"""ML-4 reader: numbered pieces asked per mark and per touching pair. No test starts the model:
the HTTP call is a fake that answers from the recorded S43-15 pair (tests/fixtures/reader)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from rosette.perception import reader
from rosette.perception.pieces import Piece
from rosette.perception.reader import MissingReader, ReaderServer, read
from rosette.perception.settings import load
from rosette.vision.prepare import Prepared

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "reader" / "S43-15.json").read_text())
H = W = 600
HEAD = (300.0, 300.0, 250.0)
YY, XX = np.mgrid[:H, :W]


def _prepared(seed: int = 0) -> Prepared:
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[0, 0] = (seed, 1, 2)  # a distinct cache key per photo
    mask = (XX - 300) ** 2 + (YY - 300) ** 2 <= 250 ** 2
    return Prepared(rgb=rgb, mask=mask, axis_x=300.0, head=HEAD, tails_top_y=560.0, bottom_y=590, scale=1.0)


def _piece(mask: np.ndarray, in_head: bool = True) -> Piece:
    ys, xs = np.nonzero(mask)
    return Piece(mask=mask, score=0.95, stability=0.95, area=int(mask.sum()),
                 bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), in_head=in_head)


def _disc(cx: float, cy: float, r: float, in_head: bool = True) -> Piece:
    return _piece((XX - cx) ** 2 + (YY - cy) ** 2 <= r * r, in_head)


def _ring_head() -> list[Piece]:
    """A centre disc and 11 touching round pieces around it: 12 marks."""
    out = [_disc(300, 300, 60)]
    for k in range(11):
        a = 2 * np.pi * k / 11
        out.append(_disc(300 + 130 * np.cos(a), 300 + 130 * np.sin(a), 35))
    return out


class FakePost:
    """Answers Q1 from the recorded S43-15 answers (any mark beyond them is a block) and Q2 from its recorded pairs
    (unclear for a pair it never saw). `drop` removes marks from the Q1 answer."""

    def __init__(self, drop=()):
        self.calls: list[dict] = []
        self.drop = set(drop)
        self.recorded = {r["mark"]: r for r in json.loads(FIXTURE["q1"])}
        self.tops = {(r["a"], r["b"]): r["on_top"] for r in json.loads(FIXTURE["q2"])}

    def __call__(self, url, body):
        self.calls.append(body)
        text = body["messages"][0]["content"][1]["text"]
        if text.startswith("Same image."):
            pairs = [(int(a), int(b)) for a, b in re.findall(r"\((\d+),(\d+)\)", text)]
            out = [{"a": a, "b": b, "on_top": self.tops.get((a, b), "unclear")} for a, b in pairs]
        else:
            lo, hi = map(int, re.search(r"from (\d+) to (\d+)", text).groups())
            out = [self.recorded.get(k, {"mark": k, "kind": "block", "texture": "plain", "repeats": 1})
                   for k in range(lo, hi + 1) if k not in self.drop]
        return {"choices": [{"message": {"content": json.dumps(out)}}]}


def test_choose_pieces_takes_head_pieces_over_the_area_floor_largest_first():
    p = _prepared()
    floor = 0.004 * np.pi * 250 ** 2  # 0.4 % of the head area, about 785 px
    pieces = [_disc(100, 100, 18), _disc(300, 300, 50), _disc(500, 500, 40, in_head=False),
              _disc(200, 200, 30), _disc(400, 200, 10), _disc(250, 400, 20)]
    assert pieces[0].area > floor > pieces[4].area
    # radii 50, 30, 20, 18: the piece outside the head and the one under the floor are left out
    assert reader.choose_pieces(p, pieces) == [1, 3, 5, 0]


def test_marks_image_numbers_every_chosen_piece_once():
    p = _prepared()
    pieces = _ring_head()
    chosen = reader.choose_pieces(p, pieces)
    im, meta = reader.marks_image(p, pieces, chosen)
    assert im.size == (640, 640) and im.mode == "RGB"
    assert [m["mark"] for m in meta] == list(range(1, 13))
    assert meta[0]["r_centroid"] < 0.05 and 0.4 < meta[1]["r_centroid"] < 0.7
    assert meta[0]["area_share"] == pytest.approx(pieces[chosen[0]].area / (np.pi * 250 ** 2), abs=1e-4)
    plain = reader.marks_image(p, pieces, [])[0]  # the numbers and outlines are drawn on the crop
    assert np.asarray(im).astype(int).std() != np.asarray(plain).astype(int).std()


def test_touching_pairs_need_overlap_after_dilation_and_are_capped():
    a, b = _disc(100, 100, 20), _disc(144, 100, 20)  # 4 px apart: touch after a 3 px dilation of each
    c = _disc(300, 100, 20)  # far from both
    d = _disc(100, 180, 20)  # a 40 px gap below a: no touch
    assert reader.touching_pairs([a, b, c, d], [0, 1, 2, 3]) == [(1, 2)]
    # a wide contact comes before a narrow one, and the list is capped
    big = _piece((XX >= 300) & (XX < 340) & (YY >= 100) & (YY < 200))
    left = _piece((XX >= 256) & (XX < 296) & (YY >= 100) & (YY < 200))  # 4 px gap, 100 px tall contact
    low = _piece((XX >= 300) & (XX < 340) & (YY >= 204) & (YY < 224))   # 4 px gap, 40 px wide contact
    pieces = [big, low, left]
    assert reader.touching_pairs(pieces, [0, 1, 2]) == [(1, 3), (1, 2)]
    assert reader.touching_pairs(pieces, [0, 1, 2], limit=1) == [(1, 3)]


def test_read_returns_answers_per_mark_and_pairs(tmp_path):
    p, pieces, post = _prepared(1), _ring_head(), FakePost()
    r = read(p, pieces, load({}), cache_dir=tmp_path, post=post)
    assert r.marks == list(range(1, 13)) and r.images == 1 and r.cached is False
    assert r.answers[1].kind == "rose" and r.answers[2].kind == "block"
    assert (r.answers[2].texture, r.answers[2].repeats) == ("glitter", 4)
    assert isinstance(r.answers[1], reader.MarkAnswer)
    assert r.piece_index[1] == 0 and sorted(r.piece_index.values()) == list(range(12))
    assert r.pairs and all(isinstance(x, reader.PairAnswer) for x in r.pairs)
    assert all(x.on_top in ("a", "b", "unclear") for x in r.pairs)
    # per image: one closed question on the marks, one on the pairs, thinking off, temperature 0, schema bound to 12
    assert len(post.calls) == 2
    body = post.calls[0]
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["minItems"] == schema["maxItems"] == 12
    assert set(schema["items"]["properties"]["kind"]["enum"]) == set(reader.KINDS)


def test_read_uses_the_cache_on_the_second_call(tmp_path):
    p, pieces, post = _prepared(2), _ring_head(), FakePost()
    first = read(p, pieces, load({}), cache_dir=tmp_path, post=post)
    n = len(post.calls)
    second = read(p, pieces, load({}), cache_dir=tmp_path, post=post)
    assert len(post.calls) == n and first.cached is False and second.cached is True
    assert second.answers == first.answers and second.pairs == first.pairs and second.marks == first.marks
    assert second.piece_index == first.piece_index
    assert len(list(tmp_path.glob(f"*-{reader.PROMPT_VERSION}.json"))) == 1


def test_short_answers_leave_marks_unanswered(tmp_path):
    p, pieces = _prepared(3), _ring_head()[:6]
    r = read(p, pieces, load({}), cache_dir=tmp_path, post=FakePost(drop=(3, 5)))
    assert sorted(r.answers) == [1, 2, 4, 6] and r.marks == [1, 2, 3, 4, 5, 6]
    assert r.answers[4].kind == "star"  # mark 4 stays mark 4: ids are never shifted
    assert r.answers[6].kind == "block"


def test_a_malformed_or_out_of_range_answer_is_dropped_not_raised(tmp_path):
    def post(url, body):
        text = body["messages"][0]["content"][1]["text"]
        out = ([] if text.startswith("Same image.") else
               [{"mark": 1, "kind": "star", "texture": "plain", "repeats": 6},
                {"mark": 2, "kind": "unicorn", "texture": "plain", "repeats": 2},
                {"mark": 9, "kind": "star", "texture": "plain", "repeats": 2},
                {"mark": 1, "kind": "petal", "texture": "plain", "repeats": 2}])
        return {"choices": [{"message": {"content": json.dumps(out)}}]}
    r = read(_prepared(4), _ring_head()[:3], load({}), cache_dir=tmp_path, post=post)
    assert list(r.answers) == [1] and r.answers[1].kind == "star"  # the first answer of a mark stands


def test_no_head_pieces_gives_an_empty_reading(tmp_path):
    def never(url, body):
        raise AssertionError("the model must not be asked")
    r = read(_prepared(5), [], load({}), cache_dir=tmp_path, post=never)
    assert (r.marks, r.piece_index, r.answers, r.pairs, r.images, r.cached) == ([], {}, {}, [], 0, False)
    outside = [_disc(550, 550, 30, in_head=False)]
    assert read(_prepared(5), outside, load({}), cache_dir=tmp_path, post=never).marks == []
    assert not list(tmp_path.glob("*.json"))


def test_missing_executable_raises_missing_reader_with_the_install_line(tmp_path):
    s = load({"ROSETTE_MODELS_DIR": str(tmp_path)})
    with pytest.raises(MissingReader) as e:
        ReaderServer(s).start()
    assert "b12342" in str(e.value) and "README" in str(e.value)
    # a cold cache with no injected post goes through the same check
    with pytest.raises(MissingReader):
        read(_prepared(6), _ring_head()[:3], s, cache_dir=tmp_path / "cache")


def _install(models, q8=False, q4=False, exe=True, proj=True):
    gg = models / "Qwen__Qwen3.5-4B-GGUF"
    gg.mkdir(parents=True)
    (models / "llama-cpp").mkdir()
    if exe:
        (models / "llama-cpp" / "llama-server.exe").write_bytes(b"x")
    if proj:
        (gg / "mmproj-Qwen3.5-4B-Q8_0.gguf").write_bytes(b"x")
    if q8:
        (gg / "Qwen3.5-4B-Q8_0.gguf").write_bytes(b"x")
    if q4:
        (gg / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x")


def test_the_q8_weights_are_preferred_and_q4_is_the_stated_fallback(tmp_path):
    _install(tmp_path / "both", q8=True, q4=True)
    _install(tmp_path / "q4", q4=True)
    _install(tmp_path / "none")
    both = ReaderServer(load({"ROSETTE_MODELS_DIR": str(tmp_path / "both")})).resolve()
    q4 = ReaderServer(load({"ROSETTE_MODELS_DIR": str(tmp_path / "q4")})).resolve()
    assert both.model.name == "Qwen3.5-4B-Q8_0.gguf" and "Q4" not in both.note
    assert q4.model.name == "Qwen3.5-4B-Q4_K_M.gguf" and "Q4_K_M" in q4.note and "Q8_0" in q4.note
    with pytest.raises(MissingReader):
        ReaderServer(load({"ROSETTE_MODELS_DIR": str(tmp_path / "none")})).resolve()


def test_the_server_command_has_the_fixed_flags(tmp_path):
    _install(tmp_path, q8=True)
    cmd = ReaderServer(load({"ROSETTE_MODELS_DIR": str(tmp_path), "ROSETTE_READER_PORT": "9123",
                             "ROSETTE_READER_THREADS": "5"})).command()
    flags = " ".join(cmd)
    for want in ("--host 127.0.0.1", "--port 9123", "-t 5", "-c 4096", "-np 1", "--image-max-tokens 768",
                 "--reasoning-budget 0", "--jinja", "--no-mmproj-offload"):
        assert want in flags, want


def test_more_than_twelve_pieces_use_a_second_image_with_global_ids(tmp_path):
    pieces = [_disc(100 + 44 * (k % 5), 150 + 44 * (k // 5), 20) for k in range(15)]  # a 5 x 3 grid, 4 px gaps
    post = FakePost()
    r = read(_prepared(7), pieces, load({}), cache_dir=tmp_path, post=post)
    assert r.images == 2 and r.marks == list(range(1, 16))
    assert sorted(r.answers) == list(range(1, 16)) and sorted(r.piece_index.values()) == list(range(15))
    q1 = [c for c in post.calls if not c["messages"][0]["content"][1]["text"].startswith("Same image.")]
    assert len(q1) == 2
    assert "from 1 to 12" in q1[0]["messages"][0]["content"][1]["text"]
    assert "from 13 to 15" in q1[1]["messages"][0]["content"][1]["text"]
    assert q1[1]["response_format"]["json_schema"]["schema"]["maxItems"] == 3
    # pairs keep global ids and stay inside their image
    assert r.pairs and all((x.a <= 12) == (x.b <= 12) for x in r.pairs)
    assert any(x.a > 12 for x in r.pairs)
