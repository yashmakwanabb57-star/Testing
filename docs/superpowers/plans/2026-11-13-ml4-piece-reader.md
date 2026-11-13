# ML-4 Piece Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name the SAM 2.1 head pieces with a local vision-language model, count repeat rings from the signal, and settle every open head decision by rendering the candidates and keeping the one closest to the photo, with the rules head as candidate 0 and the reader off byte-identical to today.

**Architecture:** A new `rosette/perception/reader.py` talks to `llama-server` (llama.cpp, 127.0.0.1) with schema-constrained questions per numbered piece and caches the answers per photo. Three new pure modules under `rosette/vision/` turn the answers into piece groups with signal counts (`groups.py`), into a short list of candidate heads (`candidates.py`), and into a verdict by rendering each candidate against the photo (`judge.py`). `pipeline.detect` wires them in behind `ROSETTE_READER`; `/api/analyze` and the workbench show one status line.

**Tech Stack:** Python 3.14 (`.venv`), numpy, scipy, OpenCV (already installed), Pillow, the existing `rosette.build` + `rosette.roles.render_categories` renderer, llama.cpp b12342 Windows CPU build under `models/llama-cpp/`, `ggml-org/Qwen3.5-4B-GGUF` Q8_0 + projector under `models/Qwen__Qwen3.5-4B-GGUF/`.

**Spec:** `docs/superpowers/specs/2026-11-13-ml4-piece-reader-design.md` (read its "Builder check: rulings" section: those eight rulings bind every task).

## Global Constraints

- No paid service, no cloud API, no network call at run time; the model runtime listens on 127.0.0.1 only and is started and stopped by the tool.
- Model files: `models/Qwen__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf` and `mmproj-Qwen3.5-4B-Q8_0.gguf`; executable `models/llama-cpp/llama-server.exe` (release b12342). Missing either: status `missing` with the install line, pipeline unchanged.
- Settings (new, in `rosette/perception/settings.py`): `ROSETTE_READER` = `on` | `off` | `auto` (default `auto`: on when the files exist), `ROSETTE_READER_PORT` (default `8089`), `ROSETTE_READER_THREADS` (default `14`). Server flags: `-c 4096`, `--image-max-tokens 768`, `--reasoning-budget 0`, `--jinja`, `-np 1`, `--no-mmproj-offload`. Request: `temperature 0`, `chat_template_kwargs {"enable_thinking": false}`, `response_format` json_schema.
- Marks image 640 px; pieces chosen: head pieces >= 0.4 % of the head area, largest first, at most 12 per image; more pieces go in a second image with global mark ids.
- Only one model process at a time; the reader never runs inside `tools/score.py` runs or page rebuilds unless its cache is warm.
- `ROSETTE_READER=off`: the 76 SVGs byte-identical to ad56d5a's output (the pre-ML-4 head). Tests run with the reader off (`tests/conftest.py`) and feed recorded answers as fixtures; no test starts the model.
- Gates with the reader on (34-ID scoreboard, new image): PASS >= 21/34, color >= 0.855, silently wrong <= 1; per-ID layer IoU changes listed, drops noted.
- Python only via `.venv/Scripts/python.exe`; exact pins in `requirements.txt` (no new packages expected). Plain hyphens in user-facing text; no AI attribution in commits.

## Review Focus

1. A photo where SAM finds one piece (the whole head, S4-15) or none: the reader must return an empty reading and the pipeline must emit candidate 0 unchanged, no exception (Task 1 test `test_no_head_pieces_gives_an_empty_reading`, Task 5 test `test_empty_reading_keeps_the_rules_head`).
2. The model names a piece "loop" where the rules drew blocks, with equal counts: no candidate may be generated from a kind disagreement alone; kinds only change the layer kind inside a candidate that the judge accepts (Task 3 test `test_kind_disagreement_alone_is_one_candidate_not_a_silent_change`).
3. A piece whose mask straddles two rules layers (a glitter block with its border): the group must take the piece once, in the group whose radial span holds its centroid (Task 2 test `test_a_piece_joins_one_group_only`).
4. The judge with two candidates that render identically inside the head mask (a z change under an opaque layer): margin 0, candidate 0 wins, part flagged "judge undecided" (Task 4 test `test_identical_renders_keep_candidate_zero`).
5. The server answers with valid JSON that violates the schema's count (fewer marks than asked): the reader must mark the missing marks unanswered, never shift ids (Task 1 test `test_short_answers_leave_marks_unanswered`).

---

### Task 1: Reader module, settings and cache

**Files:**
- Modify: `rosette/perception/settings.py` (add reader settings)
- Create: `rosette/perception/reader.py`
- Create: `tests/test_reader.py`
- Create: `tests/fixtures/reader/S43-15.json` (a recorded answer pair from the probe)
- Modify: `tests/conftest.py` (reader off by default)
- Modify: `README.md` (install section: llama.cpp b12342 and the GGUF files, with SHA-256 lines)

**Interfaces:**
- Consumes: `rosette.perception.pieces.Piece` (mask, area, bbox, in_head), `rosette.perception.pieces._interior_point(pc) -> (x, y)`, `rosette.vision.prepare.Prepared` (`rgb`, `head`).
- Produces:
  - `settings.load(env) -> Settings` gains fields `reader: str` (`on|off|auto`), `reader_port: int`, `reader_threads: int`.
  - `reader.PROMPT_VERSION: str` (`"ml4-1"`), `reader.KINDS: tuple[str, ...]` = `("pleated_disc", "pleated_ring", "star", "petal", "loop", "block", "chevron", "ring", "badge", "rose", "other")`, `reader.TEXTURES = ("plain", "glitter", "patterned")`.
  - `@dataclass MarkAnswer(mark: int, kind: str, texture: str, repeats: int)`, `@dataclass PairAnswer(a: int, b: int, on_top: str)` (`"a"|"b"|"unclear"`), `@dataclass Reading(marks: list[int], piece_index: dict[int, int], answers: dict[int, MarkAnswer], pairs: list[PairAnswer], seconds: float, cached: bool, images: int)`.
  - `reader.choose_pieces(p, pieces) -> list[int]` (indices into `pieces`, spec rule), `reader.marks_image(p, pieces, chosen, side=640) -> tuple[PIL.Image.Image, list[dict]]`, `reader.touching_pairs(pieces, chosen, dilate=3, limit=12) -> list[tuple[int, int]]`.
  - `reader.read(p, pieces, settings, cache_dir=DEFAULT_READER_CACHE, post=None) -> Reading` where `post(url, body) -> dict` is the HTTP call (injectable for tests); `reader.MissingReader(Exception)` carrying the install line; `reader.ReaderServer` context manager (`start()`, `stop()`, `url`).
  - `reader.DEFAULT_READER_CACHE = <repo>/.cache/reader`; cache file `<pieces key>-<PROMPT_VERSION>.json`.

- [ ] **Step 1: Write the failing settings test** in `tests/test_perception_settings.py`:

```python
def test_reader_settings_default_to_auto_8089_14():
    s = load({})
    assert (s.reader, s.reader_port, s.reader_threads) == ("auto", 8089, 14)

def test_reader_settings_reject_an_unknown_mode():
    with pytest.raises(ValueError):
        load({"ROSETTE_READER": "maybe"})
```

- [ ] **Step 2: Run them; expected FAIL** (`AttributeError: reader`). `.venv/Scripts/python.exe -m pytest tests/test_perception_settings.py -q -p no:cacheprovider`
- [ ] **Step 3: Add the three fields to `Settings` and `load`** (`_choice` for the mode; ints parsed with a ValueError on bad text).
- [ ] **Step 4: Run; expected PASS.**
- [ ] **Step 5: Write the failing reader tests** in `tests/test_reader.py` using synthetic pieces (a disc mask, six arm masks, a rose mask on a 600 x 600 frame, `Prepared` built by `tests/helpers.py`'s existing helper or a minimal stand-in) and a fake `post` that returns the fixture answers:

```python
def test_choose_pieces_takes_head_pieces_over_the_area_floor_largest_first(): ...   # 0.4 % floor, max 12, order
def test_marks_image_numbers_every_chosen_piece_once(): ...   # meta has marks 1..n with r_centroid and area_share
def test_touching_pairs_need_overlap_after_dilation_and_are_capped(): ...   # 3 px, limit 12, largest contact first
def test_read_returns_answers_per_mark_and_pairs(): ...   # fake post -> Reading.answers[k].kind etc.
def test_read_uses_the_cache_on_the_second_call(): ...   # post called once; cached True second time
def test_short_answers_leave_marks_unanswered(): ...   # fake post answers 4 of 6 marks: 2 missing, ids unshifted
def test_no_head_pieces_gives_an_empty_reading(): ...   # pieces=[] -> Reading(marks=[], ...), post never called
def test_missing_executable_raises_missing_reader_with_the_install_line(tmp_path): ...
def test_more_than_twelve_pieces_use_a_second_image_with_global_ids(): ...   # 15 pieces -> images 2, marks 1..15
```

- [ ] **Step 6: Run; expected FAIL** (`ModuleNotFoundError: rosette.perception.reader`).
- [ ] **Step 7: Implement `rosette/perception/reader.py`**: the prompts from the probe (`Q1`, `S1`, `Q2`, `S2`) with `PROMPT_VERSION`; `choose_pieces`, `marks_image` (Pillow drawing as in the probe), `touching_pairs` (scipy dilation), `ReaderServer` (subprocess, `/health` wait up to 240 s, `terminate` on stop), `read` (cache check, images, two `post` calls per image via `urllib.request` when `post` is None, parse with the dataclasses, unanswered marks absent from `answers`), `MissingReader`. Schema per image: `minItems = maxItems = len(marks in image)`.
- [ ] **Step 8: Run `tests/test_reader.py` and `tests/test_perception_settings.py`; expected PASS.**
- [ ] **Step 9: `tests/conftest.py`: set `ROSETTE_READER=off` beside the pieces switch; README install lines; commit.**

```bash
git add rosette/perception/settings.py rosette/perception/reader.py tests/test_reader.py tests/fixtures/reader tests/conftest.py tests/test_perception_settings.py README.md
git commit -m "feat: ML-4 reader - numbered SAM pieces asked per mark (kind, texture, repeats) and per touching pair (on top) through a local llama-server with schema-constrained answers, cached per photo; settings ROSETTE_READER/_PORT/_THREADS; off by default in tests"
```

### Task 2: Piece groups and signal counts

**Files:**
- Create: `rosette/vision/groups.py`
- Modify: `rosette/vision/parts_head.py` (extract `harmonic_count` from `_front_pieces`'s harmonic reading; no behaviour change)
- Create: `tests/test_groups.py`

**Interfaces:**
- Consumes: `Reading`, `Piece`, `pieces._fit_steps(angles) -> (n, offset, inliers) | None`, `parts_head._harmonics(outline) -> dict[int, float]`, `parts_head.MIN_ZONE_UNITS`, `parts_head.BACK_TOL_DEG`, the R18 n - 1 rule (`_maxima_on_axes`).
- Produces:
  - `parts_head.harmonic_count(polar_outline: np.ndarray) -> tuple[int, float, int | None]` = (n, amplitude, second peak n2 when its amplitude >= HARMONIC_SECOND * first, else None); `HARMONIC_SECOND = 0.6`.
  - `@dataclass Group(kind: str, texture: str, color_group: int | None, pieces: list[int], r_inner: float, r_outer: float, count: int, count_source: str, alternatives: list[int], rotation_deg: float, confidence: float)`.
  - `groups.piece_groups(reading: Reading, pieces: list[Piece], head: tuple[float, float, float], scale: float) -> list[Group]`: same kind, radial spans within `MIN_ZONE_UNITS` (2nd and 98th percentile radii in canvas units), a piece joins the group whose span holds its centroid radius when two qualify; disc, ring, badge, rose pieces form singleton groups with `count = 1`.
  - `groups.group_count(group, pieces, head) -> tuple[int, str, list[int]]`: (`len(pieces)`, `"pieces"`, []) when `_fit_steps` fits one step with no empty step; else (`harmonic_count` n, `"harmonic"`, alternatives = [n2] if n2 in (2n, n // 2) ... plus the model's repeats when it differs from n and lies in 3..16).

- [ ] **Step 1: Write failing tests** in `tests/test_groups.py` (synthetic pieces on a 600 px frame):

```python
def test_six_visible_arm_pieces_count_six_from_their_steps(): ...   # count 6, source "pieces", alternatives []
def test_eight_arms_with_two_hidden_count_eight_from_the_harmonic(): ...   # 6 visible on an 8-step lattice -> 8, "harmonic"
def test_the_models_repeats_become_an_alternative_when_they_differ(): ...   # repeats 6, signal 8 -> alternatives [6]
def test_a_piece_joins_one_group_only(): ...   # straddling piece -> in the group holding its centroid radius
def test_disc_ring_badge_rose_are_singleton_groups_with_count_one(): ...
def test_harmonic_count_reports_a_second_peak_at_twice_n(): ...   # 2k alternative when amplitude >= 0.6 of first
```

- [ ] **Step 2: Run; expected FAIL** (`ModuleNotFoundError: rosette.vision.groups`).
- [ ] **Step 3: Extract `harmonic_count` in `parts_head.py`** from the harmonic reading of `_front_pieces` and make `_front_pieces` call it; run `tests/test_front_in_base.py tests/test_stacking_pieces.py tests/test_tip_stars.py -q` and confirm green (no behaviour change).
- [ ] **Step 4: Implement `rosette/vision/groups.py`.**
- [ ] **Step 5: Run `tests/test_groups.py`; expected PASS. Commit.**

```bash
git add rosette/vision/groups.py rosette/vision/parts_head.py tests/test_groups.py
git commit -m "feat: ML-4 groups - pieces of one kind on one radial band form a repeat ring; count from the pieces' equal steps or the strongest harmonic with the R18 n - 1 rule, the model's repeats kept as an alternative; harmonic_count extracted from the piece front reader"
```

### Task 3: Open decisions and candidate heads

**Files:**
- Create: `rosette/vision/candidates.py`
- Create: `tests/test_candidates.py`

**Interfaces:**
- Consumes: `RosetteModel`, `Part`, `Group`, `Reading`, `parts_head.nested_pairs` / the H12 vote result exposed as `parts_head.photo_order(parts) -> dict[tuple[str, str], str]` if not already available (add a thin accessor: pair id -> `"a"|"b"|"undecided"` from the last `photo_stacking` run stored on the parts' params `vote_<id>`; if the vote is not recoverable from params, Task 3 adds `params["vote"]` writing in `photo_stacking`, one line), R19's `_nested_runs` evidence (`params["z_assumed"]`).
- Produces:
  - `@dataclass Decision(kind: str, part_ids: tuple[str, ...], options: tuple, source: str)` with `kind` in `("count", "order", "nested", "center", "layer_kind")`.
  - `candidates.open_decisions(rules: RosetteModel, groups: list[Group], reading: Reading) -> list[Decision]` (at most 4, ordered: count, order, nested, center, layer_kind; spec rulings 5 and 6).
  - `candidates.candidate_heads(rules: RosetteModel, decisions: list[Decision], limit: int = 24) -> list[RosetteModel]` where index 0 is `rules` unchanged (deep copy) and each other candidate differs from it by one option per decision; a count change rebuilds the points layer's `count`, keeping radii and rotation; an order change swaps z; a nested change adds or removes the R19 stack; a center change flips rose/badge; a layer_kind change sets the points kind.

- [ ] **Step 1: Write failing tests** in `tests/test_candidates.py` with hand-built models (the `tests/helpers.py` part factory):

```python
def test_candidate_zero_is_the_rules_head_unchanged(): ...
def test_a_count_disagreement_makes_count_candidates_only(): ...   # group 8 vs rules 6 with alt 6 -> candidates 6 and 8
def test_an_unclear_or_contradicted_order_makes_an_order_decision(): ...
def test_agreeing_order_makes_no_decision(): ...
def test_kind_disagreement_alone_is_one_candidate_not_a_silent_change(): ...
def test_decisions_are_capped_at_four_and_candidates_at_twenty_four(): ...
```

- [ ] **Step 2: Run; expected FAIL.**
- [ ] **Step 3: Implement `rosette/vision/candidates.py`** (and the one-line vote accessor if needed).
- [ ] **Step 4: Run; expected PASS. Commit.**

```bash
git add rosette/vision/candidates.py tests/test_candidates.py rosette/vision/parts_head.py
git commit -m "feat: ML-4 candidates - the open head decisions (count, order, nested stack, centre, layer kind) become at most 24 candidate heads with the rules head as candidate 0"
```

### Task 4: The judge

**Files:**
- Create: `rosette/vision/judge.py`
- Create: `tests/test_judge.py`

**Interfaces:**
- Consumes: `pipeline._head_render`-style rendering: `rosette.build.build(model)`, `rosette.writer.write_svg`, `rosette.roles.render_categories(svg, height, colors)`; the photo `labels`, `p.mask`, `p.head`; `Piece.mask`; OpenCV `Canny`.
- Produces:
  - `@dataclass Verdict(winner: int, scores: list[float], terms: list[dict], margin: float)`.
  - `judge.render_classes(model: RosetteModel, p, height: int = 400) -> np.ndarray` (class-id map in the photo's head frame, 0 = none).
  - `judge.score(model, p, labels, pieces, cache=None) -> tuple[float, dict]` = `W_COLOR * color_agreement + W_EDGE * edge_agreement + W_PIECE * piece_agreement` with `W_COLOR, W_EDGE, W_PIECE = 0.5, 0.3, 0.2` (Task 6 may re-mine them), each term in [0, 1], computed inside the head mask minus the tails sector.
  - `judge.judge(candidates: list[RosetteModel], p, labels, pieces) -> Verdict`; `JUDGE_MARGIN = 0.01`; margin = best minus runner-up; when margin < JUDGE_MARGIN the winner is 0.
  - `judge.flag(model: RosetteModel, verdict: Verdict) -> RosetteModel`: sets `confidence = min(confidence, 0.5 + 10 * margin)` on the parts the winning candidate changed and adds `info` lines `"judge: <decision> -> <option> (margin 0.0xx)"`; an undecided verdict adds the warning `"judge undecided: <decision> kept as the rules read it"`.

- [ ] **Step 1: Write failing tests** in `tests/test_judge.py` using a synthetic photo made by rendering a known model (the renderer is the photo: a 6-point star head rendered to labels) so that the right candidate is known:

```python
def test_the_candidate_matching_the_photo_wins(): ...   # photo = render of 6-point; candidates 6 and 8 -> winner 6
def test_identical_renders_keep_candidate_zero(): ...   # margin 0 -> winner 0, warning present
def test_terms_are_in_unit_range_and_ignore_the_tails(): ...
def test_flag_caps_confidence_by_the_margin_and_writes_the_info_line(): ...
```

- [ ] **Step 2: Run; expected FAIL.**
- [ ] **Step 3: Implement `rosette/vision/judge.py`** (render at 400 px once per candidate; edges by `cv2.Canny` on the photo's grey crop and on the class map's boundaries, agreement = share of photo edge pixels within 3 px of a render edge; piece agreement = mean over chosen pieces of IoU between the piece mask and the candidate's part that covers most of it).
- [ ] **Step 4: Run; expected PASS. Commit.**

```bash
git add rosette/vision/judge.py tests/test_judge.py
git commit -m "feat: ML-4 judge - every candidate head rendered at the photo's frame and scored on colour classes, edges and piece masks inside the head mask; the closest wins by a margin, else the rules head stays and the part is flagged"
```

### Task 5: Pipeline, API and UI wiring; off path byte-identical

**Files:**
- Modify: `rosette/pipeline.py` (`detect`: after `tidy`, the reader step; `Detection.reader: dict | None`; `PhotoContext.reader`)
- Modify: `app/main.py` (`/api/analyze` returns `reader`), `app/static/logic.js` (`readerText(r)`), `app/static/app.js` (one line under the perception line), `tests/js/*` (logic test for `readerText`)
- Modify: `tests/test_pipeline.py`, `tests/test_app.py`
- Modify: `docs/pipeline.md` (ML-4 section placeholder filled in Task 6)

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `pipeline.reader_step(p, labels, pieces, model, env=None) -> tuple[RosetteModel, dict]` returning the (possibly replaced) model and the status dict `{"status": "on"|"off"|"missing"|"failed"|"skipped", "message", "seconds", "cached", "images", "decisions", "winner", "margin"}`; `Detection.reader`; `/api/analyze` field `reader`.

- [ ] **Step 1: Write failing tests**:

```python
# tests/test_pipeline.py
def test_reader_off_leaves_detect_unchanged(monkeypatch): ...   # Detection.reader["status"] == "off"; model equal to a run without the step
def test_empty_reading_keeps_the_rules_head(monkeypatch): ...   # reader returns Reading(marks=[]) -> status "skipped", model identical
def test_reader_failure_is_reported_and_the_rules_head_stays(monkeypatch): ...
def test_a_judged_candidate_replaces_the_head_with_flags(monkeypatch): ...   # fake read + fake judge -> winner 1 applied, info line present
# tests/test_app.py
def test_analyze_returns_the_reader_status(): ...
# tests/js: readerText renders "Reader: on, 2 decisions, winner 1 (margin 0.03)" style lines and nothing when null
```

- [ ] **Step 2: Run; expected FAIL.**
- [ ] **Step 3: Implement**: `reader_step` wraps `reader.read` -> `groups.piece_groups` -> `candidates.open_decisions` / `candidate_heads` -> `judge.judge` -> `judge.flag`; every exception becomes status `failed` with the rules head kept; `detect` calls it only when `pieces` is not None and the reader setting resolves to on; the server is started once per process (`ReaderServer` kept on a module-level handle, stopped at exit with `atexit`).
- [ ] **Step 4: Run the fast suite; expected all PASS.**
- [ ] **Step 5: Byte-identical check**: export the 76 SVGs with `ROSETTE_READER=off` (and pieces on) and compare with the export of ad56d5a; expected 76 of 76 identical. Record the command and result in the report.
- [ ] **Step 6: Commit.**

```bash
git add rosette/pipeline.py app/main.py app/static/logic.js app/static/app.js tests/test_pipeline.py tests/test_app.py tests/js
git commit -m "feat: ML-4 wiring - the reader step after tidy (read, group, candidates, judge, flags) behind ROSETTE_READER, status in Detection and /api/analyze, one line in the workbench; off path byte-identical"
```

### Task 6: Calibration, gates and the 52 before/after

**Files:**
- Create: `tools/judge_mine.py` (weights and margin mined leave-one-out on the 37 known-good heads; prints a table)
- Modify: `rosette/vision/judge.py` (the mined constants, if they change), `docs/pipeline.md` (ML-4 section: what, why, the probe numbers, mining tables, before/after per ID, gates, constants row), `README.md` (settings)

**Interfaces:** consumes everything above; produces the documented constants and the report.

- [ ] **Step 1: Warm the reader cache** for the 76 photos once (`tools/reader_warm.py`, a 20-line script: `detect` per photo with the reader on; about 2 hours; one process; nothing else heavy running).
- [ ] **Step 2: Mine the judge** (`tools/judge_mine.py`): for each of the 37 known-good heads (the 24 OK rows plus the 13 matches: list the ids in the script from the ledger), candidate 0 must win leave-one-out over a grid of `W_COLOR, W_EDGE, W_PIECE` (steps of 0.1, summing to 1) and `JUDGE_MARGIN` in {0.005, 0.01, 0.02}; pick the setting with the most wins on the 37 and the most changed heads among the other 39 that improve layer IoU; write the table to the report and the constants into `judge.py`.
- [ ] **Step 3: Scoreboard with the reader on** (`tools/score.py`): PASS >= 21/34, color >= 0.855, silently wrong <= 1; per-ID layer IoU list with drops noted.
- [ ] **Step 4: Outline comparison**: rebuild the 52 head zooms from a clean export with the reader on; look at every changed ID; per-ID matches / does not match; count of match-or-near against the 13 baseline.
- [ ] **Step 5: Docs and commit.**

```bash
git add tools/judge_mine.py tools/reader_warm.py rosette/vision/judge.py docs/pipeline.md README.md
git commit -m "docs: ML-4 section - reader probes on the 52, groups and counts, candidates, the judge's mined weights and margin, before and after per ID, gates, constants"
```
