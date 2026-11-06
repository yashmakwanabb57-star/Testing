# Pipeline

Photo in, configurator SVG out (`rosette/pipeline.py`):

1. `prepare` (rosette/vision/prepare.py) - decode, normalize to 1200 px, foreground mask, symmetry axis, head circle, head/tails split.
2. `pixel_groups` (rosette/vision/colors.py) - per-pixel color groups with pattern texture; a cluster hiding two similar stripe colors is split (`_split_stripes`; stripe edges are counted across a third stripe between them); small blobs are absorbed as motifs unless they form an even ring around the head center (star tips, `parts_common.blob_rings`); print on a center badge whose shape is not a disc or ring joins the badge's group (H3, below).
3. `detect_parts` (rosette/vision/parts.py) - head, rings (grown outward while their label still covers >= 60 % of the circle), points (from rotational symmetry, or from an even ring of blobs when that is weak; bows are loop blocks widening from `inner_ratio`), center, tail stripes (stripe shares fitted separately at the top and bottom edge: `stripes`, `stripes_bottom`), pleated inner layers without rotational symmetry as rings (H3, below), leftovers as `unknown`.
4. Model - groups no part uses are dropped (ids kept); rose on iff a center part exists.
5. `assign` (rosette/assign.py) - Color 1-4 per group, confidence, warnings (including a coloring no more plausible than a random guess). Primary rule (the designer's, exact on 73 of 79 designer SVGs): with mirror-symmetric tails the Colors are numbered by the tail stripes from the outer tail edge inward (sliver stripes under `SLIVER_SHARE` of the median width skipped; head-only colors next, larger area first). The mined P(Color | role) rules decide only for asymmetric or missing tails, and otherwise act as a check: a strong disagreement with the tail order lowers the group's confidence under `FLAG_BELOW`. A pattern group that is not Color 2 by the tail order is warned, not moved.
5b. Head fit (`head_fit` / `check_head` in rosette/pipeline.py) - the model's head parts (everything but tail stripes) are rendered and compared with the photo's color labels over the head circle (1.05 head radii, rows down to 0.55 radii below the center): layer agreement and silhouette IoU. It is a measurement first: `analyze_with_fit` returns it, `/api/analyze` sends it as `head_fit`, and the UI shows it as information next to the fixed checklist item "Compare the head with the photo" (Workbench and Guided; it must be ticked before download but is not a flag). The head is flagged only on a gross failure: agreement below `HEAD_FIT_MIN` = 0.6 (the heads the H1 analysis reads as right score 0.60-0.96) or silhouette below `HEAD_SILHOUETTE_MIN` = 0.78 (right heads 0.80-0.96; the bar sits in the gap 0.764-0.786 under them). That flags 23 of the 76 reference heads (24 before H3). Agreement cannot separate shape errors on heads whose layers overlap the photo well: right heads and wrong heads overlap from 0.60 to 0.88, a synthetic perfect head scores 0.95 (pinked rim) and a ring 5 units off 0.88. **S42-15, S43-15 and S28-15 (petals, loops or star layers drawn as blocks) are not caught by the flag** and stay unflagged until shape features (lobe count per color layer, outline profile) land; S61-19 and S56-17B are flagged (outline 0.76 and 0.69). Cost: one extra head render per analyze, about 0.15 s.
6. `build` -> `write_svg` -> `validate` - shapes, SVG string, contract issues (`render` returns `RenderResult(svg, issues)`). `build` cuts every color shape where a part meant to lie above it would be hidden by the fixed layer order (Color 1 is always painted first; `rosette/geometry/clip.py`, shapely).

A result is **flagged** (`pipeline.is_flagged`, mirrored by `flagItems` in `app/static/logic.js`) when a color group has confidence below `FLAG_BELOW` (0.6), any part (head, ring, points, center, tail stripe) has detector confidence below `FLAG_BELOW`, a part is `unknown` (drawn as a placeholder), the model has any warning, or validation reports an issue.

## Scoreboard

`.venv/Scripts/python.exe tools/score.py [IDs...] [--sheet DIR]` scores every ID in `reference/canonical.txt` that has a photo (34; `TEST` has none). Generated and designer SVGs are rasterized at 850 px with category colors (solid configuration, outlines hidden), cropped to their non-white bounding boxes, the generated crop resized onto the designer crop (nearest neighbor).

- **agree** (primary, `color_agreement`) - per part region: each designer color element in `Layer_1` / `Layer_2_1_2/3/4` (visible part under 0.2 % of the rosette area skipped) is rendered alone in the same frame minus every element painted after it (its visible region); the generated category with the most pixels in that region decides correct or wrong; area-weighted share of correct elements. `--debug` prints each element with its winner and winning share.
- **pixel** (secondary) - share of designer-mask pixels with the same category (geometry mismatches count against it).
- **success** - agree >= 0.9 and no validation issues. **silently wrong** - not success and not flagged.
- **points / tails** - detected / designer count (`tools/truth.truth_from_svg`).
- **shape / headS / tailS / layer / sok** (reported next to agree, not the bar; H3) - see the H3 section.
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
7. `colors._absorb` keeps star tips - same-label blobs at one distance from the head center (within 0.15 head radii), of similar area (within 2.5x of their median) and evenly spread (`|mean exp(i N theta)| >= 0.8`, N = count .. count+2) are not motifs (`parts_common.blob_rings`). Motif-sized blobs are kept on 10 of 76 photos (rings found on 33), all in even rings inside the head circle; S35-19 and S40-15 were checked to be star tips, the other eight not one by one. The evenness margin is thin on S40-15 (0.85).
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

At the end of phase B the bar (`pytest -m slow`) was **not met**: `PASS 23/34` (28 needed) and S46-15 was silent. (The current count is in the scoreboard under Final state.)

**S46-15 (silent) is a scorer alignment effect, not a coloring error.** Every color is in the right layer (head and outer and centre stripes green = Color 1, glitter = Color 2, gold = Color 3, as in the designer file). The designer drew this rosette with a larger head and narrower tails than the photo shows (head width / height 0.36 vs 0.32, bottom width / height 0.34 vs 0.45). The scorer stretches the generated crop onto the designer's bounding box on each axis separately, so the generated head is squeezed to 84 % of its width and the designer's head ring lands on white. Scored with one uniform scale (height match, centered on the axis; a diagnostic only, `tools/score.py` is unchanged) S46-15 reaches 0.926; on that diagnostic 25 of 34 succeed (also S26-15 0.985) and none lose. No photo-based signal separates it: its head reconstruction agreement (model rasterized back onto the photo's own labels) is 0.81, the same as successful IDs (0.80-0.82). Its gold layer also has six small secondary points between the six big ones that the model does not draw (about 2 % of the area).

Flagged although correct (18): unknown placeholder parts (A10-15B, A15-17, S10-19, S35-19, S37-15, S40-15, S45-15, S5-15, S58-17, s34-15); only a points ring with rotational symmetry below 0.6 (A14-17, A9-15B, S30-15, S41-15, S67-19, S69-19, S70-19, S80-22).

**Remaining failures by cause:**

- Head structure still not modeled: S17-15 (three layered stars, no head disc), S29-15 (glitter loops over a star, badge), S68-19 (stacked chevron points), A16-17 (an irregular 8-point star in the head's own black: the head label is never read as a points layer, and its symmetry is 0.2-0.3), S13-15 (the green loops are drawn as a star because the navy ruffle merged into the blue group).
- Mask: S53-19 - its white stripes are pure white on a pure white background (distance 0), so the mask cuts them and the single tail reads as three.
- Color grouping: A13-15B (glitter navy merged with royal blue), S2-15B (snowflake print merged with white satin; flagged by the mixed-texture warning).
- Designer differs from the photo: S8-15 (centre stripe the same red as the outer, designer Color 3), S26-15 (glitter stripe widths; 0.985 with uniform alignment), S46-15 (proportions, above).

### Phase B review fixes

- `blob_rings` also requires the blobs to go round the head: the largest angular gap between neighbours is at most 3.3 steps of 360/N (4 blobs within 24 degrees scored 0.82 on evenness alone).
- A points layer read from its blobs gets confidence 0.5 (flagged) unless the layer's zone pixels are rotationally symmetric at the blob count (>= LOW_SYM) and no blob is assumed hidden; since the blob reading is only tried when that symmetry is weak, every blob-read layer is flagged. Over the 76 photos `_blob_layer` returns a layer on 15 (in either detection pass).
- Ring growth is capped at 8 units beyond the band and stops at any points layer whose label covers >= 0.4 of the circle there (it grew on 36 photos by up to 23 units).
- Stripe-edge changes are not counted across background between tails (the same 15 photos split as before).
- `part_area` estimates a bow as its block; points parts carry `inner_ratio` only when they are bows.
- `tools/score.py` prints a reported-only uniform-alignment column (`uni`, `uok`): the generated crop scaled by the height ratio and centered, not stretched per axis. The per-axis agreement stays the metric and the bar.

| set | n | success | flagged-not-success | silently-wrong | flagged although correct |
|---|---|---|---|---|---|
| all | 34 | 23 | 10 | 1 (S46-15) | 19 |
| tune | 24 | 17 | 7 | 0 | 14 |
| holdout-remaining (not clean) | 8 | 5 | 3 | 0 | 4 |
| inspected (S46-15, S80-22) | 2 | 1 | 0 | 1 | 1 |

Uniform-alignment column (reported only): 25/34 (adds S26-15 0.985, S46-15 0.926). A15-17 0.949 -> 0.954; S33-15 is now flagged (blob-read layer).

## H3: shape score, center badge print, pleated inner layers, stripe fractions

Started at 476d07e, approved at 112b021. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h3-report.md`.

**Shape score** (`tools/score.py shape_compare`, reported next to agree, not the bar). Both SVGs are rendered as silhouettes (every color class one fill, the shop's hidden pattern layer hidden) and aligned like `compare()`. Reported: `shape` whole-silhouette IoU, `headS` the rows above the designer's tail top, `tailS` the rows below, and `layer`: each designer head piece (role by `rosette.roles.classify_components`) against the union of our pieces of the same role, area-weighted. Known weakness of `layer`: the classifier calls the designer's pinked inner annuli "head" or "points", never a plain "ring", so a right plain ring of ours can lower it; read it as a coarse interior signal. `sok` is `shape >= SHAPE_OK_IOU` = 0.90. That bar is a coarse label: whole IoU over the 76 photos runs 0.762-0.980 with no gap, median 0.909, and the grossly wrong outlines sit below 0.85. It gives 21/34 on the scoreboard. The silhouette cannot see layers inside the head disc, so H3's head fixes barely move it.

**Center badge print** (`colors.py`, step 6 of `pixel_groups`). A small group lying wholly in the head is print on a badge (a logo), not a ribbon color, when all of these hold:
- the structural conditions: >= `BADGE_IN_HEAD` 0.95 of its pixels inside the head, <= `BADGE_MAX_AREA` 0.09 of the mask, one other group holding >= `BADGE_ENCLOSED` 0.9 of the circles just outside it, and that group's disc ending before `BADGE_EDGE` 0.8 head radii (a badge, not the head disc);
- its shape is neither a disc nor a ring: less than `BADGE_SHAPE_MIN` 0.7 of its pixels lie on circles around its own centroid that it covers to >= 0.8, measured with its holes filled so print on a ribbon-colored center does not count against it.

Real values: the A7-15 logo 0.549, merged; the A1-15 gold badge, the only photographed satin center on a badge, 0.978 (0.647 before hole filling), kept. Synthetic solid discs and rings measure >= 0.855. Texture was tried first and rejected: head-only groups are measured on the pleats, and no texture or tone signal separates the logo from the 66 center discs of the photos.

A merge gives its pixels to the badge group, so the center takes the badge color and radius and the badge rim ring goes away. Every merge adds the warning "Center badge print was merged into the badge color ... - check the center against the photo". Candidates that are kept are reported by `badge_print_unsure` with a warning. Residual risk: one real positive (A7-15) and one real disc (A1-15); a satin center with an irregular outline, enclosed >= 0.9, would be merged (with the warning).

**Pleated inner layer -> ring** (`parts_head._pleated_ring`). A head-zone layer has no rotational symmetry, but its label covers the full circle over part of the zone (a band that `_drop_bridges` took for touching lobes, because pleated edges ramp the coverage slowly). Such a layer becomes a ring instead of an `unknown` disc: grown both ways while its label holds `RING_EXTEND_COV` of the circle, with confidence = share of the layer inside the ring x mean coverage. The designer draws these layers as annuli (43 pinked inner annuli in 36 files; no plain ring piece in 79 files).

The confidence is capped at 0.5 (`BLOB_UNCONFIRMED_CONF`, flagged) unless the layer is annulus-shaped:
- the label covers the full circle on >= `PLEATED_FULL_MIN` 0.4 of the ring's radii (annuli 0.478-0.828; the S55-15 loops 0.338, the S11-15 star 0.206);
- its fringe beyond the ring reaches <= `PLEATED_REACH_UNITS` 15 units (annuli <= 12; the S11-15 and S40-15 star layers 18).

16 IDs change from `unknown` to ring. Every one converted at >= 0.6 is a designer annulus, or for A1-15 the badge, which the designer draws as the center. Color: S18-15B +0.148 (now OK), S11-15 +0.061, S13-15 +0.054, S15-15B +0.042, S51-19 +0.039. S40-15 0.903 -> 0.893 (loses success): the flagged ring covers star tips that showed before. This was accepted by ruling.

**Tail stripe fractions (Task 2, no code change).** Measured on the 79 designer files (rendered rows down the visible tails): in 73 of 74 multi-stripe files one stripe keeps a constant width (bottom/top 0.94-1.07) while the tail union widens 1.40-2.61. This is the center tail, a constant-width ribbon over the fanned outer tails (the exception is S80-22). The other stripes widen 1.07-5.29, and a constant-fraction model misses the boundaries by 18-44 units. The existing top/bottom stripe shares (`stripes`, `stripes_bottom`, straight boundaries) already represent this, so the geometry is unchanged.

**76-photo numbers** (scratch `compare_all.py`): OK 45 -> 45, mean color agreement 0.832 -> 0.837, mean whole shape IoU 0.902 -> 0.902 (head 0.806, tail 0.947), layer 0.217 -> 0.221. No ID loses more than 0.02 color or whole-shape IoU. Scoreboard: `PASS 24/34` -> `PASS 23/34` (S40-15).

## H4: points layers drawn the way the designer draws them

Started at 1bba1f2. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h4-report.md`. The reference SVGs are generating-model output (`reference/generation-prompt.md`); "designer" below means those files.

**Mined vocabulary.** Every single-lobe head element of the 79 reference SVGs was measured around the head center and grouped into layers (same Color, radii, width). Kinds: loop block (21 layers, 20 files: near-constant width, outer end flat in polar terms), kite (41 layers, 22 files: straight sides, widest at 0.4 of the length, quartiles 0.3-0.5), petal (14 layers, 13 files: wide inner end, straight taper to a pointed tip), triangle (54 layers, 28 files: the tips of star polygons), edge strip (18 layers, 16 files: about 4.3 degrees, 0.108 of the loop width), wedge (5 layers). Backings are frames or tips behind each point (11 files), never a disc under everything.

**Geometry** (`rosette/geometry/points.py`): `bow` uses `LOOP_PROFILE` (0.85 of the width at the base, full width by 0.7 of the length) with a shallow convex outer end (`END_SAG` 0.6 of a straight cut's sag; r_outer stays the axis end, as before H4), `diamond` the straight-sided kite (`KITE_PEAK` 0.4), `petal` `PETAL_PROFILE`; stars are unchanged. `edge_width` draws a loop's two edge strips (`EDGE_OUTSIDE` 0.19 strip widths outside the edge). An explicit `inner_ratio` still makes a linear bow; the detector no longer emits it.

**Detector** (`rosette/vision/parts_head.py`): a petal that keeps 0.85 of its width over at least 5 of 9 radii and fills at least 0.45 of the spacing is a loop block (`LOOP_*`; 7 designer loops converted, one designer kite, s39-15). Loops start where the ring or center over them ends; kites and petals that start within one pleat depth outside a ring start at it (`COVER_GAP_UNITS`). A low-confidence layer that runs as a narrow band along both sides of most loops becomes edge strips; one that shows beyond the tips of most points of another layer becomes a backing copy of them reaching the measured tip (`EDGE_*`, `BACK_*`, `EXPLAIN_*`). An unknown in a seam between two bands (at most `SEAM_UNITS` 16 wide, one of the bands' colors, under `SPECK_MAX_UNITS2` 6000) is folded into the band and the seam is closed.

**76-photo numbers** (scratch `compare_all.py`): OK 45 -> 45, color 0.8365 -> 0.8387, whole shape 0.9020 -> 0.9046, head 0.8061 -> 0.8138, layer 0.2210 -> 0.2210; head-zone unknown discs 68 -> 53. No ID loses more than 0.02 (A14-17 color -0.019, s34-15 shape -0.018). S35-19: loops with edge strips, kites with backing frames, no specks; color 0.958 -> 0.959, head 0.757 -> 0.752, layer 0.027 -> 0.027 (see the report).

## Final state (Task 16)

### Stages, modules and tuning constants

Constants are module-level names; read their comments in the source before changing one.

| Stage | Module | Tuning constants |
|---|---|---|
| prepare: decode, 1200 px, mask, axis, head circle | `rosette/vision/prepare.py` | `WORK_HEIGHT`, `CANVAS_SPAN`, `ALPHA_CUTOFF`, `MIN_DELTA_E`, `NOISE_FACTOR`, `MAX_BG_NOISE`, `OPEN_PX`, `CLOSE_PX`, `MIN_AREA_FRAC`, `MAX_AREA_FRAC`, `AXIS_SEARCH_FRAC`, `HEAD_PROFILE_FRAC`, `HEAD_RING_MARGIN` |
| color groups, stripe split, motif absorption, mixed-texture warning | `rosette/vision/colors.py` | `L_WEIGHT`, `SMOOTH_UNITS`, `FLAT_PCT`, `GRAIN_WEIGHT`, `SAMPLE_N`, `MIN_CENTER_SEP`, `PRINT_MAX_FRAC`, `TEXTURE_WIN_UNITS`, `PATTERN_T`, `MIXED_PATTERN_SHARE`, `BADGE_IN_HEAD`, `BADGE_MAX_AREA`, `BADGE_EDGE`, `BADGE_UNSURE`, `BADGE_ENCLOSED`, `BADGE_SHAPE_MIN`, `SEED` |
| part detection driver, leftovers as `unknown` | `rosette/vision/parts.py` | `LEFTOVER_OPEN_UNITS`, `TAIL_TOUCH_PX`, `HEAD_MARGIN`, `UNKNOWN_CONFIDENCE` |
| head: rings, points, bows, center, blob rings | `rosette/vision/parts_head.py`, `parts_common.py` | `RING_EXTEND_COV`, `FULL_COV`, `MIN_BAND_UNITS`, `MIN_LAYER_COV`, `MAX_COUNT`, `BOW_ENDS`, `PETAL_FULLNESS`, `BLOB_UNCONFIRMED_CONF`, `MIN_INNER_RATIO`, `HEAD_ROUND`, `PINKED_DEPTH`, `RING_MAX_GAP`, `PLEATED_FULL_MIN`, `PLEATED_REACH_UNITS` |
| tail stripes | `rosette/vision/parts_tails.py` | `ROW_FRACS`, `COUNT_ROWS`, `DENSE_SPAN`, `GAP_UNITS`, `ROW_BAND_PX`, `TOP_INSET` |
| Color 1-4 assignment, confidence, warnings | `rosette/assign.py`, `rosette/roles.py`, rules in `rosette/data/assign_rules.json` (mined by `tools/mine_rules.py`) | `FLAG_BELOW`, `MAX_COLORS`, `PATTERN_MIN_RIBBON_SHARE`, `TAIL_POS_TOL`, `FLAGGED_CONF`, `PATTERN_MERGE_PENALTY`, `SLIVER_SHARE`, `RULE_LOGODDS` |
| geometry, shapes, occlusion cuts | `rosette/geometry/`, `rosette/shapes.py`, `rosette/build.py`, `rosette/geometry/clip.py` | `QUAD_SEGS`, `SIMPLIFY`, `MIN_PIECE` |
| SVG writer and contract validator | `rosette/writer.py`, `rosette/validate.py` | none (the contract is fixed) |
| orchestration, flagging, head fit | `rosette/pipeline.py` | `FLAG_BELOW` (imported from assign), `HEAD_FIT_MIN`, `HEAD_SILHOUETTE_MIN`, `HEAD_FIT_EXTENT`, `HEAD_FIT_WINDOW`, `HEAD_FIT_GRID` |
| shop simulation (solid and pattern previews) | `rosette/simulate.py` | none |

### Shop contract

Every generated file must satisfy section 3 of the [design spec](superpowers/specs/2026-11-05-rosette-svg-generator-design.md): fixed top-level layer order (`Layer_1`, `Layer_3`, `Layer_2_1`, `Layer_6`, `Layer_rose`; issue `group_order` when it is not kept), category classes for Color 1-4, Color 2 usable in solid and pattern mode, and Color 3 correct in both. `rosette/validate.py` checks it and `render` returns the issues; the UI does not offer a download while an issue is open. The shop's own coloring logic is in `reference/shop-configurator.js` (verbatim); `rosette/simulate.py` mirrors it for the UI previews.

### UI modes

`app/main.py` (FastAPI, started with `run.bat` or uvicorn on 127.0.0.1:8765) serves `app/static`. **Workbench** shows the photo, the detected model, the solid and pattern previews, edit controls and the flagged items on one screen. **Guided** walks through the same state one question at a time (`guided.js`); switching modes keeps the state. Edits go back through `render`, so the validator runs after every change.

### Current scoreboard (`tools/score.py`, 34 IDs, re-recorded for H3 at 112b021)

Re-recorded after the H1 scorer fix (every solid color element is scored; old-layout files render as in the shop), which changed two rows at the old HEAD c8d35c1: A13-15B 0.000 -> 0.187, S37-15 1.000 -> 0.949 (counts unchanged: 23/34). The H2 changes are the out-to-in Color numbering (S2-15B 0.000 -> 0.687, S68-19 0.600 -> 0.966 success) and the head-fit check (gross failures only after fix round 1). The H3 changes on these IDs come from the pleated-layer rings: S13-15 0.840 -> 0.894, S40-15 0.903 -> 0.893 (success lost, accepted by ruling, see H3), uniform column A10-15B 0.906 -> 0.942 and A16-17 0.837 -> 0.926. The last column is the H3 start (476d07e); `shape` is the whole-silhouette IoU (H3).

| ID | agree (per-axis, official) | uni (uniform scale, reported only) | success | uni success | shape | before H3 (476d07e) |
|---|---|---|---|---|---|---|
| A10-15B | 0.906 | 0.942 | yes | yes | 0.870 | 0.906 yes |
| A13-15B | 0.187 | 0.187 | no | no | 0.924 | 0.187 no |
| A14-17 | 0.968 | 0.968 | yes | yes | 0.957 | 0.968 yes |
| A15-17 | 0.954 | 0.959 | yes | yes | 0.965 | 0.954 yes |
| A16-17 | 0.837 | 0.926 | no | yes | 0.895 | 0.837 no |
| A9-15B | 1.000 | 1.000 | yes | yes | 0.939 | 1.000 yes |
| S10-19 | 0.904 | 0.904 | yes | yes | 0.881 | 0.904 yes |
| S13-15 | 0.894 | 0.894 | no | no | 0.961 | 0.840 no |
| S17-15 | 0.633 | 0.743 | no | no | 0.839 | 0.633 no |
| S2-15B | 0.687 | 0.687 | no | no | 0.903 | 0.687 no |
| S26-15 | 0.800 | 0.985 | no | yes | 0.933 | 0.800 no |
| S28-15 | 0.967 | 0.967 | yes | yes | 0.980 | 0.967 yes |
| S29-15 | 0.247 | 0.229 | no | no | 0.883 | 0.247 no |
| S30-15 | 0.972 | 0.972 | yes | yes | 0.863 | 0.972 yes |
| S31-15 | 0.929 | 0.966 | yes | yes | 0.929 | 0.929 yes |
| S33-15 | 0.905 | 0.905 | yes | yes | 0.936 | 0.905 yes |
| S35-19 | 0.958 | 0.960 | yes | yes | 0.897 | 0.958 yes |
| S37-15 | 0.949 | 0.936 | yes | yes | 0.905 | 0.949 yes |
| S40-15 | 0.893 | 0.882 | no | no | 0.885 | 0.903 yes |
| S41-15 | 0.925 | 0.939 | yes | yes | 0.906 | 0.925 yes |
| S43-15 | 0.957 | 0.971 | yes | yes | 0.953 | 0.957 yes |
| S45-15 | 0.933 | 0.935 | yes | yes | 0.926 | 0.933 yes |
| S46-15 | 0.741 | 0.926 | no | yes | 0.835 | 0.741 no |
| S5-15 | 0.910 | 0.988 | yes | yes | 0.850 | 0.910 yes |
| S53-19 | 0.513 | 0.513 | no | no | 0.762 | 0.513 no |
| S56-17B | 0.972 | 0.975 | yes | yes | 0.874 | 0.972 yes |
| S58-17 | 1.000 | 1.000 | yes | yes | 0.928 | 1.000 yes |
| S67-19 | 0.964 | 0.964 | yes | yes | 0.947 | 0.964 yes |
| S68-19 | 0.966 | 0.620 | yes | no | 0.885 | 0.966 yes |
| S69-19 | 0.980 | 0.988 | yes | yes | 0.912 | 0.980 yes |
| S70-19 | 0.956 | 0.956 | yes | yes | 0.957 | 0.956 yes |
| S8-15 | 0.636 | 0.636 | no | no | 0.936 | 0.636 no |
| S80-22 | 0.979 | 0.989 | yes | yes | 0.927 | 0.979 yes |
| s34-15 | 0.951 | 0.941 | yes | yes | 0.936 | 0.951 yes |

Official (per-axis): **success 23, flagged-not-success 10, silently wrong 1 (S46-15), errors 0 - `PASS 23/34`** against the bar of 28 (80 %). Reported only (uniform-scale alignment): 25/34; shape: mean whole IoU 0.908 (head 0.805, tail 0.955), shape_ok 21/34. The per-axis column is the metric and the bar; the uniform column only explains misses.

Flagged although correct: 21 of the 23 successes; S28-15 and S43-15 are unflagged successes with visibly wrong heads (see 5b). S46-15 is silent again: its head fit (0.82 agreement, 0.94 outline) is not a gross failure; its miss is the scorer alignment effect described below.

Over all 76 reference photos (scratch `compare_all.py`, match >= 0.9 and no issue): 36 -> 45 OK, mean color agreement 0.689 -> 0.832 (H2), then 0.832 -> 0.837 with OK still 45 (H3); mean whole shape IoU 0.902. Silently wrong (not OK, not flagged): A13-15 (designer differs from photo), S46-15 (scorer alignment), S48-17 (0.672, up from 0.382; the remaining miss is not re-diagnosed) and s44-15 (designer differs: much narrower outer tails). The every-head reminder is the UI checklist item, not a flag.

### Sample outputs

`output/s35-19.svg` and `output/s46-15.svg` were regenerated for H2 (and regenerated again for H3: byte-identical, so unchanged) from `reference/photos/S35-19.png` and `reference/photos/S46-15.png` with `rosette.pipeline.analyze` + `render`, the same code path the UI uses, and checked with `rosette.validate.validate` (both return `[]`). They are the tool's unedited output, not confirmed by a person.

| Output | Flagged | Flagged items | Warnings |
|---|---|---|---|
| `output/s35-19.svg` | yes | two points layers with detector confidence 0.46 and 0.47 (below 0.6); three `unknown` parts (unknown-1, unknown-3, unknown-4) drawn as plain discs in the head; head fit 0.58, outline 0.69 (gross failure). No color group below 0.6. | 2 tail slivers not counted for numbering; head fit (layers and outline); unknown-1, unknown-3 and unknown-4: unrecognised shape, drawn as a plain disc |
| `output/s46-15.svg` | no | none (head fit 0.82 / outline 0.94 is information only) | none |

Before uploading either file to the shop, the user should open each photo in the UI, confirm the flags and compare the drawing with the photo. That confirmation has not been done; it is the user's step.

### Known limitations

- **The success bar is not met: 23/34 succeed against 28 needed (80 %).** `pytest -m slow` fails accordingly. Tuning stopped; it is not finished.
- **S46-15 counts as silently wrong under per-axis scoring** (0.741, not flagged). It is a scorer alignment effect (the designer drew a larger head and narrower tails; uniform scale gives 0.926); its head fit (0.82) is not a gross failure.
- **Head shape errors are mostly not flagged.** The head flag fires on gross failures only (23 of 76 reference heads). S42-15, S43-15 and S28-15 have wrong head shapes that pixel overlap cannot tell from right heads; the fixed UI checklist item "Compare the head with the photo" is the safeguard until shape features land.
- The holdout is no longer clean (see Task 12 phase B), so the numbers on the tuned IDs overstate how the tool will do on new photos.
- Most successes are still flagged (21 of 23: placeholder `unknown` discs, or a part read below 0.6, mostly points rings). The flag is warranted but means most results need a human look.
- Remaining failure clusters:
  - Head structure not modeled: S17-15 (three layered stars, no head disc), S29-15 (glitter loops over a star, badge), S68-19 (stacked chevron points), A16-17 (irregular 8-point star in the head's own black), S13-15 (loops drawn as a star because the navy ruffle merged into the blue group).
  - Mask: S53-19 (pure white stripes on a pure white background are cut out; one tail reads as three).
  - Color grouping: A13-15B (glitter navy merged with royal blue), S2-15B (snowflake print merged with white satin).
  - Designer differs from the photo: S8-15 (centre stripe the same red as the outer), S26-15 (glitter stripe widths), S46-15 (proportions).
- Photos need a single rosette on a plain background; a rosette whose colors match the background can fail the mask. The messages for no rosette and unreadable files come from `prepare`.
- Classical CV only: unusual head constructions fall back to `unknown` placeholder discs that the user must fix in the UI.
