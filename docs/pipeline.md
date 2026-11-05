# Pipeline

Photo in, configurator SVG out (`rosette/pipeline.py`):

1. `prepare` (rosette/vision/prepare.py) - decode, normalize to 1200 px, foreground mask, symmetry axis, head circle, head/tails split.
2. `pixel_groups` (rosette/vision/colors.py) - per-pixel color groups with pattern texture; a cluster hiding two similar stripe colors is split (`_split_stripes`).
3. `detect_parts` (rosette/vision/parts.py) - head, rings, points, center, tail stripes (stripe shares fitted separately at the top and bottom edge: `stripes`, `stripes_bottom`), leftovers as `unknown`.
4. Model - groups no part uses are dropped (ids kept); rose on iff a center part exists.
5. `assign` (rosette/assign.py) - Color 1-4 per group from mined designer rules, confidence, warnings (including a coloring no more plausible than a random guess).
6. `build` -> `write_svg` -> `validate` - shapes, SVG string, contract issues (`render` returns `RenderResult(svg, issues)`). `build` cuts every color shape where a part meant to lie above it would be hidden by the fixed layer order (Color 1 is always painted first; `rosette/geometry/clip.py`, shapely).

A result is **flagged** (`pipeline.is_flagged`) when a color group has confidence below `FLAG_BELOW` (0.6), a points ring has detector confidence (rotational symmetry) below `FLAG_BELOW`, a part is `unknown` (drawn as a placeholder), the model has any warning, or validation reports an issue.

## Scoreboard

`.venv/Scripts/python.exe tools/score.py [IDs...] [--sheet DIR]` scores every ID in `reference/canonical.txt` that has a photo (34; `TEST` has none). Generated and designer SVGs are rasterized at 850 px with category colors (solid configuration, outlines hidden), cropped to their non-white bounding boxes, the generated crop resized onto the designer crop (nearest neighbor).

- **agree** (primary, `color_agreement`) - per part region: each designer color element in `Layer_1` / `Layer_2_1_2/3/4` (visible part under 0.2 % of the rosette area skipped) is rendered alone in the same frame minus every element painted after it (its visible region); the generated category with the most pixels in that region decides correct or wrong; area-weighted share of correct elements. `--debug` prints each element with its winner and winning share.
- **pixel** (secondary) - share of designer-mask pixels with the same category (geometry mismatches count against it).
- **success** - agree >= 0.9 and no validation issues. **silently wrong** - not success and not flagged.
- **points / tails** - detected / designer count (`tools/truth.truth_from_svg`).
- Bar test: `pytest -m slow` (`tests/test_score.py::test_bar`): success >= 80 % AND every non-success flagged; slow tests are skipped by default.

## Baseline (Task 11, before tuning)

| ID | agree (region) | pixel | IoU | flagged | success | points | tails | points ok | tails ok |
|---|---|---|---|---|---|---|---|---|---|
| A10-15B | 0.851 | 0.771 | 0.87 | yes | no | 0/1 | 1/1 | no | yes |
| A13-15B | 0.000 | 0.173 | 0.92 | yes | no | 0/4 | 1/1 | no | yes |
| A14-17 | 0.407 | 0.404 | 0.96 | no | no | 6/6 | 1/1 | yes | yes |
| A15-17 | 0.082 | 0.119 | 0.96 | yes | no | 6/6 | 1/1 | yes | yes |
| A16-17 | 0.534 | 0.328 | 0.90 | yes | no | 0/8 | 1/1 | no | yes |
| A9-15B | 1.000 | 0.828 | 0.94 | no | yes | 19/1 | 1/1 | no | yes |
| S10-19 | 0.227 | 0.239 | 0.86 | yes | no | 0/4 | 1/1 | no | yes |
| S13-15 | 0.759 | 0.676 | 0.96 | no | no | 6/6 | 1/1 | yes | yes |
| S17-15 | 0.624 | 0.497 | 0.83 | yes | no | 0/8 | 1/1 | no | yes |
| S2-15B | 0.000 | 0.047 | 0.90 | no | no | 4/4 | 1/1 | yes | yes |
| S26-15 | 0.892 | 0.751 | 0.93 | yes | no | 6/6 | 1/1 | yes | yes |
| S28-15 | 0.888 | 0.755 | 0.98 | no | no | 4/4 | 1/1 | yes | yes |
| S29-15 | 0.211 | 0.263 | 0.88 | yes | no | 0/6 | 1/1 | no | yes |
| S30-15 | 0.872 | 0.693 | 0.84 | yes | no | 6/6 | 1/1 | yes | yes |
| S31-15 | 0.929 | 0.759 | 0.93 | no | yes | 6/6 | 1/1 | yes | yes |
| S33-15 | 0.802 | 0.736 | 0.92 | no | no | 6/6 | 1/1 | yes | yes |
| S35-19 | 0.933 | 0.811 | 0.87 | yes | yes | 6/6 | 1/1 | yes | yes |
| S37-15 | 1.000 | 0.692 | 0.91 | yes | yes | 8/8 | 1/1 | yes | yes |
| S40-15 | 0.585 | 0.454 | 0.86 | yes | no | 0/6 | 1/1 | no | yes |
| S41-15 | 0.650 | 0.590 | 0.91 | no | no | 6/6 | 1/1 | yes | yes |
| S43-15 | 0.943 | 0.785 | 0.95 | no | yes | 6/6 | 1/1 | yes | yes |
| S45-15 | 0.902 | 0.606 | 0.93 | yes | yes | 7/8 | 1/1 | no | yes |
| S46-15 | 0.715 | 0.508 | 0.83 | no | no | 6/6 | 1/1 | yes | yes |
| S5-15 | 0.773 | 0.555 | 0.85 | yes | no | 4/4 | 1/1 | yes | yes |
| S53-19 | 0.476 | 0.398 | 0.77 | yes | no | 22/12 | 3/1 | no | no |
| S56-17B | 0.693 | 0.602 | 0.88 | no | no | 6/6 | 1/1 | yes | yes |
| S58-17 | 1.000 | 0.762 | 0.90 | yes | yes | 6/6 | 1/1 | yes | yes |
| S67-19 | 0.943 | 0.797 | 0.95 | no | yes | 6/6 | 1/1 | yes | yes |
| S68-19 | 0.600 | 0.556 | 0.88 | yes | no | 0/8 | 1/1 | no | yes |
| S69-19 | 0.973 | 0.781 | 0.90 | no | yes | 6/8 | 1/1 | no | yes |
| S70-19 | 0.960 | 0.790 | 0.96 | yes | yes | 6/6 | 1/1 | yes | yes |
| S8-15 | 0.637 | 0.592 | 0.93 | yes | no | 0/8 | 1/1 | no | yes |
| S80-22 | 0.598 | 0.547 | 0.93 | no | no | 6/6 | 1/1 | yes | yes |
| s34-15 | 0.897 | 0.746 | 0.94 | yes | no | 6/6 | 1/1 | yes | yes |

**Summary:** success 10, flagged-not-success 15, silently-wrong 9, errors 0 - `PASS 10/34 (target >= 80%)`.

Runtime: 1.4-2.2 s per photo for analyze + render; the scoreboard over all 34 IDs (with per-element designer renders) takes a few minutes.

## Task 12 tuning

**Split** (deterministic, fixed before tuning): the 34 scored IDs sorted with Python's `sorted()`; index `i % 10 in (2, 5, 8)` goes to holdout. Holdout (10): A14-17, A9-15B, S17-15, S29-15, S33-15, S40-15, S46-15, S56-17B, S68-19, S80-22. Tune (24): the rest. Only the tune IDs' `--debug` output and side-by-side sheets were inspected while tuning.

**What was changed** (one commit per fix, regression test first):

1. `colors._split_stripes` - a k-means cluster hiding two similar ribbon colors (navy beside navy-blue, dark beside light orange) is split by 2-means when the halves are >= 10 feature units apart, each holds >= 17 % of the flat pixels and >= 30 % of the cluster's tail pixels, and on the tails the halves alternate >= 2.5x more often along rows than along columns (stripe edges run down the tails; shading and folds do not). Fixed A14-17 (holdout), A15-17, S10-19. Over all 76 reference photos it splits 12, all of whose designer files use 3 colors.
2. `geometry/clip.clip_occluded` (new dependency `shapely==2.2.0`) - Color 1 is always painted first (Layer_1), so a Color 1 center over a Color 3 star, or the Color 1 head over the tails' tops, was hidden. Lower shapes are now cut by the higher-z shapes the layer order paints beneath them, as the designer files do. Fixed A10-15B, S28-15, s34-15.
3. `assign` plausibility warning - when even the best coloring's area-weighted mean log P(Color | role) is below log(1/k) (no better than a random guess) the model is warned. Flags S2-15B (2 colors found for a 3-color ribbon).
4. `pipeline.is_flagged` - a points ring with detector confidence below `FLAG_BELOW` is flagged (S41-15's misread star layers).
5. `parts_tails._edge_fracs` + `geometry.tails(stripes_bottom=...)` - stripe boundaries are fitted as straight lines over the measured rows and read at the tail's top and bottom edges, so a ribbon of constant width on a widening tail keeps its width. S41-15 0.650 -> 0.925 (success); S26-15 0.892 -> 0.800 (its glitter stripes really narrow toward the head in the photo, the designer drew them wider; still flagged).

**Scoreboard after tuning** (`tools/score.py`):

| ID | set | agree (region) | pixel | IoU | flagged | success | points | tails |
|---|---|---|---|---|---|---|---|---|
| A10-15B | tune | 0.906 | 0.750 | 0.87 | yes | yes | 0/1 | 1/1 |
| A13-15B | tune | 0.000 | 0.172 | 0.92 | yes | no | 0/4 | 1/1 |
| A14-17 | holdout | 0.968 | 0.860 | 0.96 | yes | yes | 6/6 | 1/1 |
| A15-17 | tune | 0.949 | 0.856 | 0.96 | yes | yes | 6/6 | 1/1 |
| A16-17 | tune | 0.837 | 0.507 | 0.90 | yes | no | 0/8 | 1/1 |
| A9-15B | holdout | 1.000 | 0.809 | 0.94 | yes | yes | 19/1 | 1/1 |
| S10-19 | tune | 0.917 | 0.696 | 0.88 | yes | yes | 0/4 | 1/1 |
| S13-15 | tune | 0.815 | 0.689 | 0.96 | yes | no | 6/6 | 1/1 |
| S17-15 | holdout | 0.633 | 0.532 | 0.83 | yes | no | 0/8 | 1/1 |
| S2-15B | tune | 0.000 | 0.035 | 0.90 | yes | no | 4/4 | 1/1 |
| S26-15 | tune | 0.800 | 0.762 | 0.93 | yes | no | 6/6 | 1/1 |
| S28-15 | tune | 0.927 | 0.821 | 0.98 | no | yes | 4/4 | 1/1 |
| S29-15 | holdout | 0.247 | 0.254 | 0.88 | yes | no | 0/6 | 1/1 |
| S30-15 | tune | 0.872 | 0.702 | 0.84 | yes | no | 6/6 | 1/1 |
| S31-15 | tune | 0.929 | 0.762 | 0.93 | no | yes | 6/6 | 1/1 |
| S33-15 | holdout | 0.876 | 0.808 | 0.92 | yes | no | 6/6 | 1/1 |
| S35-19 | tune | 0.933 | 0.811 | 0.87 | yes | yes | 6/6 | 1/1 |
| S37-15 | tune | 1.000 | 0.725 | 0.91 | yes | yes | 8/8 | 1/1 |
| S40-15 | holdout | 0.585 | 0.476 | 0.86 | yes | no | 0/6 | 1/1 |
| S41-15 | tune | 0.925 | 0.612 | 0.91 | yes | yes | 6/6 | 1/1 |
| S43-15 | tune | 0.943 | 0.828 | 0.95 | no | yes | 6/6 | 1/1 |
| S45-15 | tune | 0.925 | 0.651 | 0.93 | yes | yes | 7/8 | 1/1 |
| S46-15 | holdout | 0.741 | 0.540 | 0.83 | no | no | 6/6 | 1/1 |
| S5-15 | tune | 0.847 | 0.652 | 0.85 | yes | no | 4/4 | 1/1 |
| S53-19 | tune | 0.509 | 0.511 | 0.76 | yes | no | 22/12 | 3/1 |
| S56-17B | holdout | 0.696 | 0.574 | 0.88 | yes | no | 6/6 | 1/1 |
| S58-17 | tune | 1.000 | 0.810 | 0.90 | yes | yes | 6/6 | 1/1 |
| S67-19 | tune | 0.964 | 0.841 | 0.95 | yes | yes | 6/6 | 1/1 |
| S68-19 | holdout | 0.600 | 0.562 | 0.88 | yes | no | 0/8 | 1/1 |
| S69-19 | tune | 0.940 | 0.782 | 0.90 | yes | yes | 6/8 | 1/1 |
| S70-19 | tune | 0.960 | 0.856 | 0.96 | yes | yes | 6/6 | 1/1 |
| S8-15 | tune | 0.626 | 0.589 | 0.93 | yes | no | 0/8 | 1/1 |
| S80-22 | holdout | 0.598 | 0.526 | 0.93 | no | no | 6/6 | 1/1 |
| s34-15 | tune | 0.951 | 0.814 | 0.94 | yes | yes | 6/6 | 1/1 |

| set | n | success | flagged-not-success | silently-wrong | flagged although correct |
|---|---|---|---|---|---|
| all | 34 | 17 | 15 | 2 (S46-15, S80-22) | 14 |
| tune | 24 | 15 | 9 | 0 | 12 |
| holdout | 10 | 2 | 6 | 2 (S46-15, S80-22) | 2 |

Baseline (Task 11): success 10, flagged-not-success 15, silently-wrong 9 (A14-17, S13-15, S2-15B, S28-15, S33-15, S41-15, S46-15, S56-17B, S80-22). The bar (`pytest -m slow`: >= 80 % success and no silent case) is **not met**: `PASS 17/34`.

Flagged although correct: 14 of the 17 successes are flagged. 9 carry `unknown` placeholder discs in the head (A10-15B, A15-17, S10-19, S35-19, S37-15, S45-15, S58-17, S70-19, s34-15) - the colors are right by the region metric but the drawing has shapes to fix, so the flag is warranted. 5 are flagged only for a low-confidence points ring (A14-17, A9-15B, S41-15, S67-19, S69-19); before Task 12 the flagged-although-correct count was 5 of 10.

**Remaining failures by cause:**

- Head structure not modeled (bows/loops instead of a round head, concentric pleated rings, layered stars read as unknown or wrong kind): A16-17, S5-15, S30-15, S53-19, S8-15 head, S17-15, S29-15, S40-15, S68-19 (0 points detected on several).
- Similar hues merged or split into the wrong regions: S13-15 (split moved the shiny navy head ruffle into the blue group; small star tips absorbed), A13-15B (glitter navy merged with royal blue on an asymmetric tail), S2-15B (snowflake print merged with white satin, pattern lost).
- Designer differs from the photo (same color drawn as another category, or other proportions): S8-15 (centre stripe same red as the outer, designer Color 3), S26-15 (glitter stripe widths).
- Holdout silent: S46-15, S80-22 - not diagnosed (holdout debug output not inspected).
