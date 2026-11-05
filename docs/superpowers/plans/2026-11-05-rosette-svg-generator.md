# Rosette Photo to Configurator SVG - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web tool that turns a rosette product photo into a clean, symmetric, configurator-ready SVG in the brroset.dk layer/class structure, with a hard validation gate and a human-correction UI.

**Architecture:** Classical-CV pipeline (prepare -> detect parts -> color groups -> assign Color 1-4) produces an editable `RosetteModel`; a parametric geometry builder turns the model into shapes; a writer emits the exact shop SVG structure; a validator + Python port of the shop's configurator logic gate the download. FastAPI serves a plain-JS UI (workbench + guided mode) that edits the model and re-renders instantly.

**Tech Stack:** Python 3.14 (installed), venv, FastAPI, uvicorn, opencv-python-headless, numpy, scikit-image, Pillow, lxml, resvg-py (rasterizing SVGs for tests/scoring), pytest. Frontend: plain HTML/CSS/JS, no build step.

**Spec:** `docs/superpowers/specs/2026-11-05-rosette-svg-generator-design.md` (read it; section 3 "contract" is binding).

## Global Constraints

- No AI/ML models, no paid APIs, no network calls at runtime. Free, commercial-friendly licenses only.
- All Python runs inside `.venv` (`.venv/Scripts/python.exe`); never the global interpreter or global pip.
- Pin dependency versions in `requirements.txt` to the latest stable releases at install time (verify on pypi.org; if a package has no Python 3.14 wheel, stop and report rather than downgrading Python silently).
- Output root: `<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 600 850" style="enable-background:new 0 0 600 850;" xml:space="preserve">`, no `id` on root, no comments/metadata.
- Top-level group order: `Layer_1`, `Layer_3` (`Layer_3_2`, `Layer_3_3`, [`Layer_3_4`]), `Layer_2_1` (`Layer_2_1_2`, `Layer_2_1_3`, [`Layer_2_1_4`]), `Layer_6`, `Layer_rose`. All non-`_4` groups always present, even empty. `_4` groups only when Color 4 used, and then both.
- Color class N in {0,1,2,3,5} only. `st4` = rose only (in `Layer_rose`). `st6` = outlines only (in `Layer_6`). `st_circel` = shown when rose off.
- Pattern def exactly: `<pattern id="image_stN_2_pattern" patternUnits="userSpaceOnUse" width="100" height="100"><image xlink:href="" x="0" y="0" width="100" height="100"/></pattern>` inside `<defs>` as first child of `Layer_3_2`, one per used `stN_2_pattern`, none unused.
- Style block: each used color class `{fill:none;}`; `.st4{fill:none;stroke:#000000;stroke-linecap:round;stroke-miterlimit:10;}`; `.st6{fill:none;stroke:#000000;stroke-miterlimit:10;}`.
- Pinked edge tooth: 5.383 units wide, 5.383 deep (matches live files).
- Numbers written with at most 3 decimals.
- Output filename: input stem lowercased + `.svg`; delivered as a browser download, never written next to the input.
- User-facing strings use plain hyphens, never em/en dashes.
- Commit messages: no Claude attribution trailer (user's global rule overrides harness).

## Review Focus

1. Transparent PNG input (alpha background, no white) - expected: alpha used as the foreground mask. Test in Task 7.
2. Very large or tiny photos (e.g. 4000x8000, 300x600) - expected: normalized to a working height of 1200 px, same result. Test in Task 7.
3. JPEG with EXIF rotation, grayscale or CMYK mode - expected: auto-rotated, converted to RGB, no crash. Test in Task 7.
4. A photo with no rosette (blank / random image) or an unsupported/corrupt file - expected: clear error message, HTTP 422, no SVG. Tests in Tasks 7 and 13.
5. Filenames with spaces, uppercase, unicode (`S46-15 kopi.WEBP`) - expected: download name `s46-15 kopi.svg`. Test in Task 13.

---

## File Structure

```
brroset/
  CLAUDE.md, VERSION, CHANGELOG.md, README.md, .gitignore, requirements.txt, pyproject.toml
  reference/
    svg/<ID>.svg            79 live designer SVGs
    photos/<ID>.png         76 product photos
    canonical.txt           IDs of the 35 full-structure files (one per line)
    shop-configurator.js    verbatim globalMethods block from the S46-15 page
    rose.svg                master rose: S46-15 Layer_rose (st4 path + 2 polylines)
  rosette/
    model.py                RosetteModel, Part, ColorGroup (+ JSON)
    shapes.py               Shape dataclass + svg element helpers
    writer.py               shapes -> SVG string (exact contract)
    validate.py             SVG string -> list[Issue]
    simulate.py             Python port of shop coloring/toggle logic
    geometry/
      common.py             pinked edges, mirroring, polar helpers
      head.py rings.py points.py center.py tails.py outline.py rose.py
    build.py                RosetteModel -> list[Shape]
    vision/
      prepare.py            load + normalize + mask + head/tails split
      parts.py              detect head/rings/points/center/tails
      colors.py             color groups + pattern texture
    assign.py               color groups -> Color 1-4 using mined rules
    data/assign_rules.json  output of tools/mine_rules.py
    pipeline.py             analyze(bytes) -> RosetteModel ; render(model) -> RenderResult
  tools/
    mine_rules.py           mines role->category stats from canonical refs
    score.py                photo vs designer scoreboard
    rasterize.py            category-colored rasterization (shared by tests/score)
  app/
    main.py                 FastAPI app
    static/index.html app.js simulator.js guided.js styles.css
  tests/ ... mirrors modules
  docs/pipeline.md
  output/                   generated SVGs for sample images (committed)
```

---

### Task 1: Project bootstrap and reference data

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pyproject.toml`, `CLAUDE.md`, `reference/**`, `tests/test_reference_data.py`

**Interfaces:**
- Produces: `reference/svg/*.svg`, `reference/photos/*.png`, `reference/canonical.txt`, `reference/shop-configurator.js`, `reference/rose.svg`; pytest configured with `testpaths=["tests"]` and marker `slow`.

- [ ] **Step 1:** `git init`, branch `main`. `.gitignore`: `.venv/`, `__pycache__/`, `.superpowers/`, `.claude/worktrees/`, `*.pyc`, `.pytest_cache/`.
- [ ] **Step 2:** `python -m venv .venv`; install fastapi, uvicorn, opencv-python-headless, numpy, scikit-image, pillow, lxml, resvg-py, python-multipart, httpx, pytest with `.venv/Scripts/python.exe -m pip`; write exact versions to `requirements.txt`.
- [ ] **Step 3:** Copy reference data from the brainstorming scratchpad `C:\Users\GIRIRA~1\AppData\Local\Temp\claude\fdbf5e502741217c\4b714a08-6a6b-43bf-ac36-4906f5b6303c\scratchpad\site\` (`svg_files\` -> `reference/svg/`, `photos\` -> `reference/photos/`). If the scratchpad is gone, re-fetch: product IDs from `https://www.brroset.dk/en/products/design-selv.br10.010` (hrefs `.../<slug>.br<ID>`), SVG JSON from `https://www.brroset.dk/h/data/product-configurators-svg.axd?productid=br<ID>` (field `svg`; empty for S77-22 S78-22 S79-22 S83-22 S96-23 S97-23 S100-23 S101-23), photo `https://www.brroset.dk/files/data/img/products/<ID>.png`. 0.4 s delay between requests.
- [ ] **Step 4:** `reference/shop-configurator.js`: the `vueApp.config.globalProperties.globalMethods = { ... };` block (setSvgImgColor, setSvgImgRoseVisibility, getColorByNode) copied verbatim from the S46-15 product page (scratchpad `site/page.html` lines 25953-26041, or re-fetch the page). `reference/rose.svg`: standalone SVG containing only the `Layer_rose` group of `reference/svg/S46-15.svg` (viewBox `0 0 150 400`). `reference/canonical.txt`: IDs whose top-level groups include `Layer_1`, `Layer_3` and `Layer_2_1` (35 IDs).
- [ ] **Step 5: Write test** `tests/test_reference_data.py`: `test_reference_counts` asserts 79 svgs, 76 photos, 35 canonical lines; `test_canonical_have_toggle_groups` parses each canonical file with lxml and asserts `#Layer_3` and `#Layer_2_1` exist; `test_rose_asset` asserts rose.svg has exactly 1 path + 2 polylines, all class `st4`.
- [ ] **Step 6:** Run `.venv/Scripts/python.exe -m pytest -q` -> 3 passed.
- [ ] **Step 7:** `CLAUDE.md` (short: purpose, venv rule, the contract pointer to spec section 3, test/score commands, solo project so no worktrees). Run `/bootstrap-project` for VERSION (`0.1.0.0`) and CHANGELOG.
- [ ] **Step 8: Commit** `chore: bootstrap project and reference data`.

### Task 2: Shapes and SVG writer

**Files:** Create `rosette/__init__.py`, `rosette/shapes.py`, `rosette/writer.py`; Test `tests/test_writer.py`

**Interfaces:**
- Produces:
  - `@dataclass Shape: tag: Literal["path","polygon","polyline","circle","ellipse"]; attrs: dict[str,str]; category: Literal[1,2,3,4] | None; pattern: bool = False; role: str = ""; z: int = 0; kind: Literal["color","outline","rose","circel"] = "color"`
  - `write_svg(shapes: list[Shape]) -> str`

- [ ] **Step 1: Write failing tests:**
  - `test_structure_always_has_required_groups`: `write_svg([])` parsed; group ids in document order == `["Layer_1","Layer_3","Layer_3_2","Layer_3_3","Layer_2_1","Layer_2_1_2","Layer_2_1_3","Layer_6","Layer_rose"]`; root viewBox `0 0 600 850`; root has no `id`.
  - `test_color2_pattern_duplicated`: one category-2 pattern shape -> one `stN_2_pattern` in Layer_3_2, one `stN_2` with identical `d` in Layer_2_1_2, and `<pattern id="image_stN_2_pattern">` inside `Layer_3_2/defs`.
  - `test_color2_solid_only_still_duplicated`: category 2, pattern=False -> same as above (Color 2 always gets both copies; pattern flag only affects confidence/UI).
  - `test_color3_in_both_toggles`, `test_color4_groups_only_when_used` (no `_4` groups for 1-3 only; both `Layer_3_4` and `Layer_2_1_4` when used).
  - `test_class_numbers_allowed`: across 10 distinct shapes of one category, every N in {0,1,2,3,5}.
  - `test_z_order_preserved`: shapes emitted within each group sorted by `z` ascending.
  - `test_style_block_matches_used_classes` and `test_no_metadata` (no comments, no `<metadata>`, no `data-` attrs).
- [ ] **Step 2:** Run `pytest tests/test_writer.py -v` -> fails (module missing).
- [ ] **Step 3:** Implement with lxml. N assignment: per category, distinct visual sub-groups get N cycling 0,1,2,3,5 by first appearance of `role` (same role -> same N). Pattern-capable shapes reuse the same N for `_2` and `_2_pattern`. Numbers formatted via a shared `fmt(x: float) -> str` (3 decimals, trailing zeros stripped) in `shapes.py`.
- [ ] **Step 4:** Run tests -> pass.
- [ ] **Step 5: Commit** `feat: SVG writer for configurator structure`.

### Task 3: Validator

**Files:** Create `rosette/validate.py`; Test `tests/test_validate.py`

**Interfaces:**
- Consumes: `write_svg`, `Shape`.
- Produces: `@dataclass Issue: code: str; message: str; element: str | None`; `validate(svg: str) -> list[Issue]` (empty = pass). Codes: `xml`, `root`, `missing_group`, `bad_parent`, `dup_id`, `class_layer`, `class_number`, `pattern_def_missing`, `pattern_def_unused`, `toggle_mismatch`, `dup_geometry`, `not_closed`, `style_mismatch`, `sim_unfilled`.

- [ ] **Step 1: Write failing tests:** `test_writer_output_passes` (a mixed 1/2/3/4 shape set -> `[]`); one mutation test per code, each starting from a valid writer output, applying one lxml edit, asserting that exact code appears (e.g. move a `st0_1` into `Layer_2_1_3` -> `class_layer`; rename class to `st4_1` -> `class_number`; delete the `<pattern>` -> `pattern_def_missing`; delete one path from `Layer_2_1_3` only -> `toggle_mismatch`; open path `d` without `z` and not polygon/circle -> `not_closed`). `test_s46_reference_structure`: `validate(reference/svg/S46-15.svg)` returns only codes in `{"root","style_mismatch","not_closed"}` (legacy file quirks) - proves the validator accepts real shop structure.
- [ ] **Step 2:** Run -> fail.
- [ ] **Step 3:** Implement. Geometry identity for toggle/dup checks = (tag, normalized attrs excluding `class`, whitespace-collapsed). `sim_unfilled` uses Task 4's `simulate` (add after Task 4; until then the check is skipped by a feature flag argument `simulate_check: bool = True` - Task 4 wires it).
- [ ] **Step 4:** Run -> pass.
- [ ] **Step 5: Commit** `feat: SVG structure validator`.

### Task 4: Configurator simulator (Python port)

**Files:** Create `rosette/simulate.py`; Modify `rosette/validate.py`; Test `tests/test_simulate.py`

**Interfaces:**
- Produces: `@dataclass SimConfig: colors: dict[int,str]  # color index -> css color or "url(...)"; pattern_index: int | None; rose: bool`; `simulate(svg: str, cfg: SimConfig) -> SimResult` with `SimResult.visible_color_elements: list[str]` (xpath), `SimResult.unfilled: list[str]`, `SimResult.svg: str` (with inline fills/display applied).

- [ ] **Step 1: Write failing tests:** port semantics from `reference/shop-configurator.js` exactly: classes loop `['st0','st1','st2','st3','st5']`; pattern for index i fills `.stN_i_pattern` with `url(#image_stN_i_pattern)` and shows `#Layer_3`, hides `#Layer_2_1`; solid shows `#Layer_2_1`; rose toggles `.st4` and inverse `.st_circel`. Tests: `test_solid_mode_fills_all_visible` (writer output, colors 1-3 set, pattern None -> `unfilled == []`, Layer_3 hidden); `test_pattern_mode_fills_all_visible`; `test_rose_off_hides_st4`; `test_st4_color_class_never_filled` (hand-made svg with `st4_1` -> appears in `unfilled`).
- [ ] **Step 2:** Run -> fail. **Step 3:** Implement on lxml (display via ancestor chain). Wire into `validate`: for every color index used, run both modes (pattern on index 2, and solid) x rose on/off; any `unfilled` -> Issue `sim_unfilled`. **Step 4:** Run all tests -> pass.
- [ ] **Step 5: Commit** `feat: configurator simulator and validation gate`.

### Task 5: Geometry generators

**Files:** Create `rosette/geometry/{__init__,common,head,rings,points,center,tails,outline,rose}.py`; Test `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Shape`, `fmt`.
- Produces (all return `list[Shape]` with `category=None`; build.py sets categories):
  - `pinked_circle(cx, cy, r, tooth=5.383) -> Shape` (polygon)
  - `ring(cx, cy, r_outer, r_inner) -> Shape` (path, even-odd, two subpaths)
  - `disc(cx, cy, r) -> Shape` (circle)
  - `points_ring(cx, cy, kind: Literal["star","diamond","bow","petal"], count: int, r_inner, r_outer, rotation_deg, width_ratio) -> list[Shape]` (one closed shape per point; `star` = one closed star polygon)
  - `tails(top_y, bottom_y, count: int, top_width, bottom_width, spread_deg, stripes: list[float], tooth=5.383) -> list[list[Shape]]` (per tail, per stripe a closed polygon; stripes are width fractions summing to 1; bottom edge pinked)
  - `outlines(shapes: list[Shape]) -> list[Shape]` (kind="outline", class st6, polyline/path tracing each shape boundary)
  - `place_rose(cx, cy, r) -> list[Shape]` (kind="rose": loads `reference/rose.svg`, computes the rose's own bbox center and half-diagonal at load, then scales/translates so that maps to (cx, cy, r))
- [ ] **Step 1: Write failing tests:** `test_pinked_circle_closed_and_tooth`; `test_points_ring_rotational_symmetry` (count 8 -> 8 shapes; each shape's centroid at angle `rotation + k*45` +/- 0.5 deg); `test_tails_mirror_symmetry` (count 5: tail k and tail 4-k mirror about x=300 within 0.01); `test_tails_stripes_cover_width` (stripe polygons tile the tail, area sum == tail area within 1%); `test_all_inside_canvas` (any generator, all coords in [0,600]x[0,850]); `test_place_rose_scaled` (bbox of rose radius ~ target r +/- 5%).
- [ ] **Step 2:** fail. **Step 3:** implement. **Step 4:** pass.
- [ ] **Step 5: Commit** `feat: parametric rosette geometry`.

### Task 6: RosetteModel and builder

**Files:** Create `rosette/model.py`, `rosette/build.py`; Test `tests/test_model.py`, `tests/test_build.py`

**Interfaces:**
- Produces:
  - `@dataclass ColorGroup: id: int; rgb: tuple[int,int,int]; pattern: bool; texture: float; area: float; category: int | None; confidence: float`
  - `@dataclass Part: id: str; role: Literal["head","ring","points","center","tail_stripe","unknown"]; params: dict[str, float|int|str]; color_group: int; z: int; confidence: float`
  - `@dataclass RosetteModel: parts: list[Part]; groups: list[ColorGroup]; rose: bool; warnings: list[str]`; `to_json() -> dict`, `RosetteModel.from_json(d) -> RosetteModel`
  - `build(model: RosetteModel) -> list[Shape]` (maps each part to generator via role, sets `category` and `pattern` from its group, z stacking: tail_stripe 0-9, head 10, points 20, ring 30+, center 50; adds outlines and rose when `model.rose`, plus a `st_circel` disc copy of the center when rose)
- [ ] **Step 1: Write failing tests:** `test_json_roundtrip`; `test_build_s35_like_model_validates` (hand-built model: pinked head, 12 star points, center disc, 3 tails with 1 stripe each, groups for categories 1,2(pattern),3 -> `validate(write_svg(build(m))) == []`); `test_unknown_part_becomes_flagged_disc` (role "unknown" -> one disc shape and a warning in model).
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5: Commit** `feat: rosette model and builder`.

### Task 7: Image preparation

**Files:** Create `rosette/vision/{__init__,prepare}.py`; Test `tests/test_prepare.py`

**Interfaces:**
- Produces: `class InputError(Exception)`; `@dataclass Prepared: rgb: np.ndarray (H,W,3 uint8); mask: np.ndarray (H,W bool); axis_x: float; head: tuple[float,float,float] (cx,cy,r); tails_top_y: float; bottom_y: float; scale: float  # px per canvas unit`; `prepare(data: bytes) -> Prepared`.
- [ ] **Step 1: Write failing tests** (use `reference/photos/S46-15.png` and synthetic images made with `tools/rasterize.py` from reference SVGs - create `tools/rasterize.py` here: `rasterize(svg: str, height: int, category_colors: dict[int,str] | None) -> np.ndarray` via resvg-py, injecting a style that fills `_1/_2/_2_pattern/_3/_4` classes and hides `#Layer_3`):
  - `test_mask_excludes_white_background` (S46 photo: mask covers 25-60% of pixels, corner pixels False)
  - `test_transparent_png_uses_alpha` (RGBA image with transparent bg)
  - `test_normalizes_size` (S46 upscaled x4 and downscaled to 300x600 -> head (cx,cy,r) within 3% of original after scale normalization; working height 1200)
  - `test_exif_and_modes` (EXIF-rotated JPEG, grayscale, CMYK -> no exception, rgb shape (H,W,3))
  - `test_head_on_synthetic` (rasterized S35-19 at height 1200: detected head center within 2% of the designer circle center (300,167.4) scaled)
  - `test_no_rosette_raises` (blank white image and uniform noise -> `InputError`); `test_corrupt_bytes_raises`.
- [ ] **Step 2-4:** fail, implement (Pillow `ImageOps.exif_transpose`, convert RGB/RGBA; mask = alpha or Lab distance from border-sampled background > threshold + morphology + largest component; axis = column-weighted mask centroid refined by mirror-correlation; head = Hough/min-enclosing circle on the upper mask where row width profile has its top maximum; tails_top_y = first row below head bottom), pass.
- [ ] **Step 5: Commit** `feat: image preparation and rosette localization`.

### Task 8: Part detection

**Files:** Create `rosette/vision/parts.py`, `rosette/roles.py`, `tests/helpers.py`; Test `tests/test_parts.py`, `tests/test_roles.py`

**Interfaces:**
- Consumes: `Prepared`.
- Produces: `detect_parts(p: Prepared, labels: np.ndarray) -> list[Part]` where `labels` (H,W int, -1 background) comes from Task 9's `pixel_groups`; params in **canvas units** (600x850 frame per spec section 9: centered x=300, rosette top y=25, bottom y=825).
- [ ] **Step 1: Write failing tests** on synthetic rasters (`tools/rasterize.py`, category colors) of canonical refs. Truth is computed from the reference SVG, never hard-coded, via `tests/helpers.py::truth_from_svg(id) -> dict(points=int, tails=int, rings=int, pinked=bool)` (reuses the role classifier from Task 10's `tools/mine_rules.py` - build that classifier here as `rosette/roles.py::classify_roles(svg: str) -> list[tuple[element, role]]` and have Task 10 import it). Parametrize over S35-19, S46-15, S13-15, S28-15, S41-15: points count exact, tails count exact, rings +/-1, pinked equal.
- [ ] **Step 2-4:** fail, implement (radial profile from head center over labels for rings/center; angular profile at mid radius between center and head edge, count via FFT dominant frequency for points; kind via profile shape at two radii (star = width decreasing linearly to tip, diamond = width peak mid, bow = constant width, petal = rounded); tails: horizontal label profile at 3 heights below tails_top_y, count/stripes by run-length, symmetric enforcement by mirroring about axis), pass.
- [ ] **Step 5: Commit** `feat: rosette part detection`.

### Task 9: Color groups and pattern detection

**Files:** Create `rosette/vision/colors.py`; Test `tests/test_colors.py`

**Interfaces:**
- Produces: `pixel_groups(p: Prepared, max_groups: int = 6) -> tuple[np.ndarray, list[ColorGroup]]` (labels H,W; groups with rgb, area, texture, pattern; category None).
- [ ] **Step 1: Write failing tests:** `test_shading_merges` (synthetic: one color rendered with a lightness gradient 30-80 L* plus a second hue -> exactly 2 groups); `test_glitter_is_pattern` (S46 photo: the group under the glitter stripes has `pattern=True`, satin green/gold `False`; locate stripes via a fixed canvas-space sample point mapped through `Prepared`); `test_hearts_motif_is_pattern` (S35-19 photo: dark purple heart ribbon group pattern=True); `test_four_plus_colors_reported` (synthetic with 5 hues -> 5 groups returned; merging happens in assign).
- [ ] **Step 2-4:** fail, implement (Lab; features a*, b*, and down-weighted L*; k chosen by silhouette over 2..max_groups on a pixel sample; motif/glitter pixels (small high-contrast blobs inside a region) absorbed into the surrounding region's group and that group's `texture` = local std of L* / high-frequency energy; `pattern = texture > T`, T tuned on S46/S35 and recorded as a module constant), pass.
- [ ] **Step 5: Commit** `feat: color grouping and pattern detection`.

### Task 10: Mined assignment rules and Color 1-4 assignment

**Files:** Create `tools/mine_rules.py`, `rosette/data/assign_rules.json`, `rosette/assign.py`; Test `tests/test_assign.py`

**Interfaces:**
- Produces: `assign(model: RosetteModel) -> RosetteModel` (sets each group's `category` and `confidence`; merges groups beyond 4 into nearest Lab group with a warning).
- [ ] **Step 1:** `tools/mine_rules.py`: for each canonical ref, get roles from `rosette/roles.py::classify_roles` (Task 8; rules: `circle` at head center = center; closed shape crossing the head radius band = points; path with ring subpaths = ring; centroid y below head bottom = tail_stripe, sub-role by x-position rank outer/middle/inner; largest pinked polygon = head) and record its category (from class suffix; `_2_pattern` -> 2 + pattern). Output `assign_rules.json`: `{role: {category: probability}}` with Laplace smoothing, plus `pattern_role_prob`. Print the table.
- [ ] **Step 2: Write failing tests:** `test_pattern_group_is_color2`; `test_shared_group_shares_category`; `test_merge_over_four` (5 groups -> 4 categories used, warning present); `test_low_margin_flags` (two equally plausible mappings -> confidence < 0.6 on the affected groups); `test_rules_file_covers_roles`.
- [ ] **Step 3:** Implement: enumerate all injective mappings of groups -> categories (<= 4! permutations after merging); score = sum over parts of `area * log P(category | role)` + hard constraint pattern group -> 2; best mapping wins; per-group confidence = softmax margin between best and best alternative that changes that group. Constant `FLAG_BELOW = 0.6`.
- [ ] **Step 4:** pass. **Step 5: Commit** `feat: Color 1-4 assignment from mined designer rules`.

### Task 11: Pipeline and scoreboard

**Files:** Create `rosette/pipeline.py`, `tools/score.py`; Test `tests/test_pipeline.py`

**Interfaces:**
- Produces: `analyze(data: bytes) -> RosetteModel`; `@dataclass RenderResult: svg: str; issues: list[Issue]`; `render(model: RosetteModel) -> RenderResult`; `score_one(id: str) -> dict(color_agreement: float, points_ok: bool, tails_ok: bool, iou: float, flagged: bool)`.
- [ ] **Step 1: Write failing tests:** `test_end_to_end_s46_valid` (`render(analyze(S46 photo bytes)).issues == []`); `test_end_to_end_s35_valid`; `test_render_is_deterministic`.
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5:** `tools/score.py`: for each canonical ID with a photo: generate, rasterize generated and designer SVG with category colors (`tools/rasterize.py`), align by mask bounding box, `color_agreement` = fraction of designer-mask pixels with matching category; success = `color_agreement >= 0.9` and no validation issues and not flagged-wrong. Print table + summary line `PASS x/N (target >= 80%)`. Add `tests/test_score.py::test_bar` marked `slow`, asserting >= 80% of IDs succeed OR the failing ones have `flagged=True`.
- [ ] **Step 6:** Run `python tools/score.py`; record baseline in `docs/pipeline.md`. **Step 7: Commit** `feat: pipeline and scoreboard`.

### Task 12: Tuning to the bar

**Files:** Modify `rosette/vision/*.py`, `rosette/assign.py` as needed; `docs/pipeline.md`

- [ ] **Step 1:** Run scoreboard; list failures by cause (localization / parts / colors / assignment).
- [ ] **Step 2:** Fix the largest cause; add a regression test for that ID in the owning module's test file before changing code.
- [ ] **Step 3:** Repeat until `pytest -m slow` passes (>= 80% unaided success, rest flagged). If a point kind is too rare to support, make it fall back to `unknown` + flag rather than guessing.
- [ ] **Step 4:** Update scoreboard results in `docs/pipeline.md`. **Commit** per fix: `fix: <cause> for <IDs>`.

### Task 13: Web API

**Files:** Create `app/__init__.py`, `app/main.py`; Test `tests/test_api.py`

**Interfaces:**
- Produces: `POST /api/analyze` (multipart `file`) -> `{model, svg, issues, filename}`; `POST /api/render` (json `{model}`) -> `{svg, issues}`; `GET /` serves `app/static/index.html`; static at `/static`. `download_name(upload_name: str) -> str` in `app/main.py`.
- [ ] **Step 1: Write failing tests** (FastAPI TestClient): `test_analyze_png`, `test_analyze_webp` (S46 webp from photos converted in-test), `test_reject_unsupported` (`.gif`, `.txt` -> 422 with message `Unsupported file type. Use PNG, WEBP, JPG or JPEG.`), `test_no_rosette_422` (blank image -> 422 `No rosette found in the image.`), `test_render_roundtrip` (edit model points count, re-render, svg changes, issues empty), `test_download_name` (`"S46-15 kopi.WEBP"` -> `"s46-15 kopi.svg"`, `"Ø.jpg"` -> `"ø.svg"`).
- [ ] **Step 2-4:** fail, implement (max upload 25 MB -> 413), pass.
- [ ] **Step 5:** Add `run.bat` / README command: `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8765` and auto-open browser. **Commit** `feat: web API`.

### Task 14: Workbench UI

**Files:** Create `app/static/{index.html,app.js,simulator.js,styles.css}`

**Interfaces:**
- Consumes: Task 13 endpoints.
- `simulator.js`: `applyConfig(svgEl, {colors: {1:'#hex',...}, patternIndex: 2|null, patternImage: dataUrl|null, rose: bool})` - logic copied from `reference/shop-configurator.js` (same class loop, same IDs), adapted only to take arguments instead of Vue nodes.

- [ ] **Step 1:** Layout per approved mockup A: left photo, middle SVG preview with simulator controls (4 color pickers, Color 2 Solid/Pattern toggle with a built-in sample pattern image, Rose on/off), right panel: parts list (editable role params: counts, kinds, radii) and color groups (swatch, category dropdown 1-4, flagged items with amber border and "please confirm" text). Bottom of right panel: validation checklist (each Issue code as pass/fail line) and Download button disabled while issues exist or flags unconfirmed. Toggles: before/after overlay slider, "SVG source" view, error banner.
- [ ] **Step 2:** Every edit -> `POST /api/render` (debounced 200 ms) -> replace preview.
- [ ] **Step 3: Verify:** run server, use `/qa` (gstack browser) on the S46 webp and S35 webp: upload, change Color 2 to pattern, toggle rose, change a group category, download; confirm downloaded name and that Download is disabled while a flag is unconfirmed.
- [ ] **Step 4: Commit** `feat: workbench UI`.

### Task 15: Guided mode

**Files:** Create `app/static/guided.js`; Modify `index.html`, `app.js`

- [ ] **Step 1:** Header switch Workbench / Guided. Guided steps: 1 Upload, 2 Parts (confirm each detected part, edit inline), 3 Colors (one question per color group, flagged first, cannot advance with unconfirmed flags), 4 Test (simulator, must toggle pattern and rose once each), 5 Download (validation list). Shares the same model/state object as workbench; switching modes keeps edits.
- [ ] **Step 2: Verify** with `/qa`: complete guided flow for S46; confirm Next is blocked on an unconfirmed flag. **Commit** `feat: guided mode`.

### Task 16: Docs and sample outputs

**Files:** Create `docs/pipeline.md` (finalize), `README.md`, `output/s35-19.svg`, `output/s46-15.svg`

- [ ] **Step 1:** Generate SVGs for `C:\Users\Giriraj-Yash\Downloads\S35-19.webp` and `S46-15.webp` through the UI (after human confirmation of flags) into `output/`. Do not write into Downloads.
- [ ] **Step 2:** `docs/pipeline.md`: each stage, its module, its tuning constants, the scoreboard results, the shop contract (link spec section 3), known limitations. README: install, run, test, score.
- [ ] **Step 3:** `validate()` on both outputs returns `[]`. **Commit** `docs: pipeline documentation and sample outputs`.
- [ ] **Step 4:** Hand the user the website test checklist (spec `task.md` section 11 items 3-8) - those run on the live shop and need the user.
