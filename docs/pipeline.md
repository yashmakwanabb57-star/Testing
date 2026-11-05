# Pipeline

Photo in, configurator SVG out (`rosette/pipeline.py`):

1. `prepare` (rosette/vision/prepare.py) - decode, normalize to 1200 px, foreground mask, symmetry axis, head circle, head/tails split.
2. `pixel_groups` (rosette/vision/colors.py) - per-pixel color groups with pattern texture; a cluster hiding two similar stripe colors is split (`_split_stripes`; stripe edges are counted across a third stripe between them); small blobs are absorbed as motifs unless they form an even ring around the head center (star tips, `parts_common.blob_rings`).
3. `detect_parts` (rosette/vision/parts.py) - head, rings (grown outward while their label still covers >= 60 % of the circle), points (from rotational symmetry, or from an even ring of blobs when that is weak; bows are loop blocks widening from `inner_ratio`), center, tail stripes (stripe shares fitted separately at the top and bottom edge: `stripes`, `stripes_bottom`), leftovers as `unknown`.
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
2. `geometry/clip.clip_occluded` (new dependency `shapely==2.2.0`) - Color 1 is always painted first (Layer_1), so a Color 1 center over a Color 3 star, or the Color 1 head over the tails' tops, was hidden. Lower shapes are now cut by the strictly-higher-z shapes the layer order paints beneath them, as the designer files do; a color shape it cannot read raises ValueError. Fixed A10-15B, S28-15, s34-15.
3. `assign` plausibility warning - P(Color | role) is renormalized over the Colors 1..k in use; when even the best coloring's area-weighted mean log of it is below log(1/k) (no better than a random guess) the model is warned. (Review fix: without the renormalization 17 of 23 real 2-color photos warned; now 12 of 23, none of A12-15, S4-15, S6-15, S55-15, S65-18.)
3b. `colors.mixed_texture_groups` + `analyze` warning - a plain group whose tail interior has local texture above `PATTERN_T` on >= 42 % of its pixels may hide a merged patterned ribbon. Flags S2-15B (snowflake print merged with white satin; 0.45). Thin margin: set from that one case, next plain groups S70-19 0.39, S46-15 0.36; S58-17 (0.56, white text on black) is also warned. None of the 23 two-color photos warn.
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
| S10-19 | tune | 0.904 | 0.696 | 0.88 | yes | yes | 0/4 | 1/1 |
| S13-15 | tune | 0.840 | 0.689 | 0.96 | yes | no | 6/6 | 1/1 |
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

## Task 12 phase B

S46-15 and S80-22 were inspected and leave the holdout. Holdout-remaining (8): A14-17, A9-15B, S17-15, S29-15, S33-15, S40-15, S56-17B, S68-19. **It is no longer clean:** S40-15 and S56-17B were inspected and used as regression cases (fixes 6-8), and the side-by-side sheets of S17-15, S29-15 and S68-19 were viewed. Only A14-17, A9-15B and S33-15 were never looked at beyond their score rows.

**What was changed** (one commit each, regression test first, seen failing):

6. `colors._split_stripes` - stripe-edge changes are counted between consecutive pixels of the cluster along a row or column, skipping other groups' pixels. Two satins kept apart by a glitter stripe never touch, so the old neighbour count saw no stripe edges (S80-22: light pink and burgundy merged into one group, a confident 2-color coloring, hence silent). Over the 76 photos this adds exactly three splits (S80-22, S56-17B, S40-15), all 3-color designer files; no other cluster's decision changes.
7. `colors._absorb` keeps star tips - same-label blobs at one distance from the head center (within 0.15 head radii), of similar area (within 2.5x of their median) and evenly spread (`|mean exp(i N theta)| >= 0.8`, N = count .. count+2) are not motifs (`parts_common.blob_rings`). Motif-sized blobs are kept on 11 of 76 photos, all star tips, loop ends or pleat teeth around the head. The evenness margin is thin on S40-15 (0.85).
8. `parts_head._blob_layer` - when a layer's rotational self-correlation is below LOW_SYM, its blobs in the zone are tested as an even ring (same helper): count, rotation and radial extent come from the blobs, confidence is the evenness. S40-15 0.882 -> 0.903, S8-15 points 0/8 -> 8/8; S30-15 0.872 -> 0.800 (its unknown placeholder disc had covered the inner ring; fixed by 10).
9. `geometry.points` bow = ribbon loop - a block from `inner_ratio` x the width at its inner end to the full width at its outer end (new optional points param `inner_ratio`, default 1; detected as the inner quarter's width over the widest). The designer's bows are loops; the old bow-tie pinched to a quarter of the width in the middle. S5-15 0.847 -> 0.910, S30-15 0.800 -> 0.878, S43-15 0.943 -> 1.000, S28-15 and S69-19 up; nothing down.
10. `parts_head._ring_outer` - a ring band grows outward while its label is dominant with >= `RING_EXTEND_COV` (0.6) of the circle, never into another band or a same-label points layer. 0.5 / 0.6 / 0.7 compared on the 34 IDs: 0.6 is the only value gaining S30-15 (0.878 -> 0.972) without losing S40-15; A15-17 0.976 -> 0.949 and S43-15 1.000 -> 0.957 (both still success).

Tried and not kept: growing rings inward too (S40-15's head ring then covers its gold points).

**Scoreboard after phase B** (`tools/score.py`):

| ID | set | agree (region) | pixel | IoU | flagged | success | points | tails |
|---|---|---|---|---|---|---|---|---|
| A10-15B | tune | 0.906 | 0.757 | 0.87 | yes | yes | 0/1 | 1/1 |
| A13-15B | tune | 0.000 | 0.172 | 0.92 | yes | no | 0/4 | 1/1 |
| A14-17 | holdout | 0.968 | 0.860 | 0.96 | yes | yes | 6/6 | 1/1 |
| A15-17 | tune | 0.949 | 0.858 | 0.96 | yes | yes | 6/6 | 1/1 |
| A16-17 | tune | 0.837 | 0.507 | 0.90 | yes | no | 0/8 | 1/1 |
| A9-15B | holdout | 1.000 | 0.809 | 0.94 | yes | yes | 19/1 | 1/1 |
| S10-19 | tune | 0.904 | 0.735 | 0.88 | yes | yes | 0/4 | 1/1 |
| S13-15 | tune | 0.840 | 0.699 | 0.96 | yes | no | 6/6 | 1/1 |
| S17-15 | holdout | 0.633 | 0.539 | 0.84 | yes | no | 0/8 | 1/1 |
| S2-15B | tune | 0.000 | 0.035 | 0.90 | yes | no | 4/4 | 1/1 |
| S26-15 | tune | 0.800 | 0.764 | 0.93 | yes | no | 6/6 | 1/1 |
| S28-15 | tune | 0.967 | 0.842 | 0.98 | no | yes | 4/4 | 1/1 |
| S29-15 | holdout | 0.247 | 0.242 | 0.88 | yes | no | 0/6 | 1/1 |
| S30-15 | tune | 0.972 | 0.759 | 0.86 | yes | yes | 6/6 | 1/1 |
| S31-15 | tune | 0.929 | 0.762 | 0.93 | no | yes | 6/6 | 1/1 |
| S33-15 | holdout | 0.905 | 0.837 | 0.94 | no | yes | 6/6 | 1/1 |
| S35-19 | tune | 0.958 | 0.833 | 0.90 | yes | yes | 6/6 | 1/1 |
| S37-15 | tune | 1.000 | 0.728 | 0.91 | yes | yes | 8/8 | 1/1 |
| S40-15 | holdout | 0.903 | 0.715 | 0.89 | yes | yes | 6/6 | 1/1 |
| S41-15 | tune | 0.925 | 0.613 | 0.91 | yes | yes | 6/6 | 1/1 |
| S43-15 | tune | 0.957 | 0.832 | 0.95 | no | yes | 6/6 | 1/1 |
| S45-15 | tune | 0.933 | 0.650 | 0.93 | yes | yes | 8/8 | 1/1 |
| S46-15 | inspected | 0.741 | 0.540 | 0.83 | no | no | 6/6 | 1/1 |
| S5-15 | tune | 0.910 | 0.661 | 0.85 | yes | yes | 4/4 | 1/1 |
| S53-19 | tune | 0.513 | 0.514 | 0.76 | yes | no | 24/12 | 3/1 |
| S56-17B | holdout | 0.972 | 0.837 | 0.87 | no | yes | 6/6 | 1/1 |
| S58-17 | tune | 1.000 | 0.830 | 0.93 | yes | yes | 6/6 | 1/1 |
| S67-19 | tune | 0.964 | 0.841 | 0.95 | yes | yes | 6/6 | 1/1 |
| S68-19 | holdout | 0.600 | 0.562 | 0.88 | yes | no | 0/8 | 1/1 |
| S69-19 | tune | 0.980 | 0.793 | 0.91 | yes | yes | 6/8 | 1/1 |
| S70-19 | tune | 0.956 | 0.857 | 0.96 | yes | yes | 6/6 | 1/1 |
| S8-15 | tune | 0.636 | 0.568 | 0.93 | yes | no | 8/8 | 1/1 |
| S80-22 | inspected | 0.979 | 0.875 | 0.93 | yes | yes | 6/6 | 1/1 |
| s34-15 | tune | 0.951 | 0.804 | 0.94 | yes | yes | 6/6 | 1/1 |

| set | n | success | flagged-not-success | silently-wrong | flagged although correct |
|---|---|---|---|---|---|
| all | 34 | 23 | 10 | 1 (S46-15) | 18 |
| tune | 24 | 17 | 7 | 0 | 14 |
| holdout-remaining (not clean, see above) | 8 | 5 | 3 | 0 | 3 |
| inspected (S46-15, S80-22) | 2 | 1 | 0 | 1 | 1 |

The bar (`pytest -m slow`) is **not met**: `PASS 23/34` (28 needed) and S46-15 is silent.

**S46-15 (silent) is a scorer alignment effect, not a coloring error.** Every color is in the right layer (head and outer and centre stripes green = Color 1, glitter = Color 2, gold = Color 3, as in the designer file). The designer drew this rosette with a larger head and narrower tails than the photo shows (head width / height 0.36 vs 0.32, bottom width / height 0.34 vs 0.45). The scorer stretches the generated crop onto the designer's bounding box on each axis separately, so the generated head is squeezed to 84 % of its width and the designer's head ring lands on white. Scored with one uniform scale (height match, centered on the axis; a diagnostic only, `tools/score.py` is unchanged) S46-15 reaches 0.926; on that diagnostic 25 of 34 succeed (also S26-15 0.985) and none lose. No photo-based signal separates it: its head reconstruction agreement (model rasterized back onto the photo's own labels) is 0.81, the same as successful IDs (0.80-0.82). Its gold layer also has six small secondary points between the six big ones that the model does not draw (about 2 % of the area).

Flagged although correct (18): unknown placeholder parts (A10-15B, A15-17, S10-19, S35-19, S37-15, S40-15, S45-15, S5-15, S58-17, s34-15); only a points ring with rotational symmetry below 0.6 (A14-17, A9-15B, S30-15, S41-15, S67-19, S69-19, S70-19, S80-22).

**Remaining failures by cause:**

- Head structure still not modeled: S17-15 (three layered stars, no head disc), S29-15 (glitter loops over a star, badge), S68-19 (stacked chevron points), A16-17 (an irregular 8-point star in the head's own black: the head label is never read as a points layer, and its symmetry is 0.2-0.3), S13-15 (the green loops are drawn as a star because the navy ruffle merged into the blue group).
- Mask: S53-19 - its white stripes are pure white on a pure white background (distance 0), so the mask cuts them and the single tail reads as three.
- Color grouping: A13-15B (glitter navy merged with royal blue), S2-15B (snowflake print merged with white satin; flagged by the mixed-texture warning).
- Designer differs from the photo: S8-15 (centre stripe the same red as the outer, designer Color 3), S26-15 (glitter stripe widths; 0.985 with uniform alignment), S46-15 (proportions, above).
