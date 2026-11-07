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

**Geometry** (`rosette/geometry/points.py`): `bow` uses `LOOP_PROFILE` (0.85 of the width at the base, full width by 0.7 of the length) with a shallow convex outer end (`END_SAG` 0.6 of a straight cut's sag; r_outer stays the axis end, as before H4), `diamond` the straight-sided kite (`KITE_PEAK` 0.4), `petal` `PETAL_PROFILE`; stars are unchanged. `edge_width` draws a loop's two edge strips (`EDGE_OUTSIDE` 0.19 strip widths outside the edge). An explicit `inner_ratio` sets only the loop's base width (the workbench shows 0.85 when none is set); the detector no longer emits it.

**Detector** (`rosette/vision/parts_head.py`): a petal that keeps 0.85 of its width over at least 5 of 9 radii and fills at least 0.45 of the spacing is a loop block (`LOOP_*`; 7 designer loops converted, one designer kite, s39-15). Loops start where the ring or center over them ends; kites and petals that start within one pleat depth outside a ring start at it (`COVER_GAP_UNITS`). A low-confidence layer that runs as a narrow band along both sides of most loops becomes edge strips; one that shows beyond the tips of most points of another layer becomes a backing copy of them reaching the measured tip (`EDGE_*`, `BACK_*`, `EXPLAIN_*`; the backing is a copy of the points with the same width ratio over its longer length, so it is wider in proportion). An unknown in a seam between two bands (at most `SEAM_PLEATS` 2 pleat depths, 16 units, wide, one of the bands' colors, under `SPECK_MAX_UNITS2` 6000) is folded into the band and the seam is closed. The trims and the seam folding live in `rosette/vision/parts_trims.py`; a layer hosting another's trims is never replaced in the same pass.

**76-photo numbers** (scratch `compare_all.py`): OK 45 -> 45, color 0.8365 -> 0.8387, whole shape 0.9020 -> 0.9046, head 0.8061 -> 0.8138, layer 0.2210 -> 0.2210; head-zone unknown discs 68 -> 53. No ID loses more than 0.02 (A14-17 color -0.019, s34-15 shape -0.018). S35-19: loops with edge strips, kites with backing frames, no specks; color 0.958 -> 0.959, head 0.757 -> 0.752, layer 0.027 -> 0.027 (see the report; after the scorer frame fix e9ac42d, which takes our head center from the drawn rose circle or rings, S35-19's layer reads 0.222).

## H5: tail tops under the head, stacking order, edge-plus-tip trims

Started at e9ac42d. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h5-report.md`.

**Tail tops** (`rosette/build.py` `_under_head`). Measured on the 511 tail stripe pieces of the 78 reference files with tails: no reference draws a free tail top (top row under 0.06 R wide in every stripe of 60 files, median 0.015 R); 298 tops follow the lower outline of the head shapes over them (the high end on the outer side in 224) and 213 end in a point under one. Our stripes are drawn at full width from `top_y` down (its meaning is unchanged), run on up their side lines into every gap between head shapes that opens onto the stripe's top and is closed above by the head, and are cut to the outside of every head shape whatever the layers, as the reference tails are. Every top edge then lies on a head shape's outline, in the shop and on outline pages that draw hidden edges. Where the head does not close a stripe's top edge it stays flat at `top_y` and the model warns. (The first H5 version closed each tail in an apex 0.88 R below the head center; on S35-19 its taper lines showed beside the bottom kite, so fix round 1 replaced it.) Free top edges above `top_y` (near-horizontal tail outline segments off the head outline): A7-15, S35-19, S30-15 all 0.

**Line art** (`rosette/geometry/outline.py`). The shop draws Layer_6 over everything, so st6 outlines now trace only the edges no later-painted shape hides (Layer_1 first, then Layer_2_1, each in document order). Before, the full polygon of a Color 1 tail stripe under a Color 1 head showed its top inside the head (A7-15). Shop-style render at 1700 px: tail pixels inside the head A7-15 0, S35-19 1, S30-15 4 (antialiasing; 601 before on S30-15).

**Stacking order** (`tools/stacking.py`, reported by `tools/score.py`). Every pair of head pieces of the 79 references that overlap (paint order decides) or are cut along each other (the cut piece lies under; local convex hulls) was ordered by kind. Among points pieces with distinct ends the one ending further in lies on top in 92 % (1292 of 1400), which is our z rule; the one consistent kind exception is loop over star, 89 % (63 of 71 pairs, 10 files), so stars now rank under every loop (`parts_head.points_z`). Also consistent and already ours: center over ring 0.95, loop/kite/star/petal/backing over head 0.96-1.00, ring over loop 1.00, backing under its host 0.92 (by radius). Inconsistent, ours kept: kite vs star 0.55 (explained by radius), star vs ring 0.56, center vs star 0.71, head vs tail 0.54. The scoreboard prints, per ID, the kind pairs our file stacks the other way round: 4 pairs in 4 of 34 IDs after fix round 1 (8 in 7 before the tails were cut to the head's outside).

**Edge-plus-tip trims** (`rosette/vision/parts_trims.py`). The brief's premise for S30-15 did not hold: its reference draws 12 pink edge strips along the loops, not a larger kite behind each kite. Same-kind backings exist elsewhere (17 files; kite behind kite in 7, out 0.10 R and sides 0.035 R median), so a low-confidence points layer of the host's count and axes running along both sides of a kite, petal or star layer and ending beyond its tips becomes a backing of the host's kind, wider and longer by the measured margins (`EDGETIP_AXIS_DEG` 4; fires on none of the 76 photos). S30-15 itself is fixed through the edge-strip rule: a band may lie up to `EDGE_SEAM_UNITS` 3 off the loop side (an antialiased lilac seam), a band layer of the loops' count counts 2/3 of the loop sides instead of both sides of 2/3 of the loops (6 of 8 sides; an unknown band still needs both sides, S5-15 lost 0.04 color otherwise), and `EXPLAIN_MIN_SHARE` is 0.45 (S30-15 0.49). Only S30-15 and S56-17 change: S30-15 head 0.743 -> 0.748 (0.731 at the H5 start), layer 0.138 -> 0.146, color 0.974 -> 0.973 (a loss of 0.001); S56-17 (not on the scoreboard) color 0.951 -> 0.984.

**Scoreboard** (34 IDs): `PASS 23/34` unchanged; shape 0.909 -> 0.913, head 0.809 -> 0.821 (after fix round 1); no ID loses more than 0.004 color or shape. Color gains: S28-15 0.965 -> 0.998 and S69-19 0.980 -> 0.993 (loops over stars).

## H6: specks, tail slivers, glitter read as two tones, diagonal tails as a fan

Started at db5d68f. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h6-report.md`. Seen on A13-15B (two rim specks drawn as discs, a navy glitter ribbon read as thin light/dark stripes, three diagonal ribbons drawn as one striped tail).

**Specks** (`rosette/tidy.py` `drop_specks`, called by `analyze` after detection). An `unknown` part under `SPECK_MAX_HEAD_SHARE` = 0.005 of the photo's head circle (pi r^2 over pi R^2) is not drawn, wherever it lies, with one warning naming it. Over the 76 photos (67 unknowns) the rim specks measure 0.0022-0.0039 (S1-15, S13-15, S55-15, S51-19, A13-15B twice); the next unknowns are real structure drawn as placeholders: S9-19 bow loop tips 0.0059 and 0.0111, S58-17 badge print 0.0087, S29-15 loop tips 0.0100. The bar sits in that gap (x1.28 the largest speck, x0.85 the smallest kept piece: a thin margin). Real parts can be smaller (A1-15 center 0.0024), but only unknown parts are dropped. The H4 seam folding (`SPECK_MAX_UNITS2`) runs before and is unchanged.

**Tail slivers** (`tidy.merge_tail_slivers`). A tail stripe under `SLIVER_SHARE` (0.35) of the median stripe width is merged into a neighbour: one of its own group, else both neighbours when they share a group (they join), else the wider one; stripes are renumbered and one warning names the merged stripes. Over the 76 photos 14 slivers on 7 photos (S31-15, S52-19, S15-15, S35-19, S30-15, S13-15 and, before the glitter rule, A13-15B) measure 0.05-0.21 of the median; the narrowest kept stripes 0.33-0.35 were A13-15B glitter fragments (gone with the glitter rule), then 0.50 and up. The numbering rule in `assign` still skips slivers in edited models.

**Glitter read as two plain tones** (`rosette/vision/parts_tails.py` `_glitter`). The brief's premise (a pattern group split into two tones) does not occur in the data: every textured tail stretch of a pattern group is >= 0.77 that group, with 1.4-4 units of spill at its ends. What occurs is A13-15B: no pattern group at all, the navy glitter labelled as alternating light-blue and royal stripes. On each measured row the texture is the local L* spread (window `GLITTER_WIN_UNITS` 3.5, over `PATTERN_T` 5 after a `GLITTER_SMOOTH_UNITS` 2 median filter, stretches of `GLITTER_MIN_UNITS` 10 or more). A stretch whose labels, `GLITTER_EDGE_UNITS` 2 in from both ends, make two or more runs of `GLITTER_RUN_UNITS` 3 or more of two or more plain groups is glitter: with the color groups given (`detect_parts(p, labels, groups)`, as `analyze` does) it becomes a new pattern group (median color of those pixels) and one stripe of its full width. Satin stripes 10+ units wide read <= 1.8 on A13-15B, its glitter 9.3-21.6; the rule fires on A13-15B only (S2-15B's print is plain-tone spill at one end, never two interior runs). A13-15B rows at 25 % and 50 % read the glitter as 95 % royal (one tone), so the glitter stripe is measured on 5 of 11 rows and its shares fall back to the mean (no top/bottom fit).

**Diagonal stripes as a fan** (`parts_tails.fan_asymmetry`, `_fan`, `fan_z`). Per stripe boundary of a single tail a straight line is fitted over the measured rows (rms residual in units, not R^2: a vertical boundary has no x variance), angles relative to the tail axis. The brief's divergence premise does not separate anything: every multi-stripe photo diverges (7-17 degrees between inner boundaries: outer tails fanning over a constant-width center ribbon), which the top and bottom stripe shares already draw. What separates A13-15B is that its fan is not mirror-symmetric: a boundary and its mirror partner differ by 8.0 degrees (fits within 0.4 units rms); every other measured tail reads <= 4.9 except S31-15 6.1 (a sliver at one side shifts the pairing; its colors repeat); 9 tails are unmeasurable (fewer than 4 rows or 3 stripes). Rule: asymmetry >= `FAN_ASYM_DEG` 7 with fits within `FAN_RMS_UNITS` 2 flags the tail (confidence <= 0.5); with `FAN_MIN_STRIPES` 3+ stripes of distinct colors it is drawn as that many separate tails (`spread_deg`), each one constant-width ribbon as wide as the axis spacing at the bottom, fanning from an apex set so the outer extent widens as measured, stacked as the boundaries show (a boundary is the side of the tail on top). Fires on A13-15B only. `geometry.tails` draws the fan with each tail's own sides and pinked bottom on one line, adjacent tails meeting at the bottom (no gap), and `validate` passes (synthetic tests).

**A13-15B** (outline, photo | ours | reference): before, one tail with six stripes (two light slivers) and two rim discs; after, no discs, three tails: light left, glitter (its own pattern Color) in the middle, royal right, with a vertical and a diagonal inner line as in the reference. Shape 0.924 -> 0.926, head 0.962 -> 0.967, tail 0.898 -> 0.898; color 0.187 -> 0.187 (1.000 after the glitter rule alone, with one striped tail; as three flag tails the mined rules number royal Color 1 where the reference has light; the reference uses another color scheme).

**Scoreboard** (34 IDs): `PASS 23/34` unchanged (24/34 between the glitter and fan commits); only A13-15B changes beyond 0.003 (S31-15 shape 0.932 -> 0.929, slivers merged). No ID loses more than 0.003 color or shape.

## H7: head shape library (templates from the reference files)

Started at 1339a24. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h7-report.md`. Seen on S29-15 (glitter loops behind a pink star behind two dark stars around a pleated ring): no head part, three placeholder blobs. Per-kind rules cannot build that; the references were made by copying shapes from similar files (`reference/generation-prompt.md`), and a library does the same deterministically.

**Library** (`rosette/vision/templates.py`, cache `rosette/data/head_templates.json`, 1.4 MB). Every product reference file (77; TEST and D1 left out) gives a template: each non-tail color piece of the solid configuration (`rosette.roles.classify_components`, so the Layer_3 pattern twins are left out; the writer emits Color 2 twins itself) with its reference Color and classifier role, in document (paint) order, as polygons in head radii around the head circle `prepare` finds on the rendered file (rendered at 1800 px, contours simplified to 0.7 px, 3 decimals). `build_library(Path("reference/svg"))` rebuilds the cache (about 13 minutes). The prototype `tools/head_templates.py` is now the evaluation tool over this module.

**Matcher.** Photo color groups (merged groups sent to their absorbing group, groups no part uses to the nearest model group in color) and each template are compared on a 128 x 128 grid of head radii, rows above 0.55 R below the center (the tails hide the rest). Features as in the prototype: partition agreement (Hungarian category -> group map), silhouette IoU, outer profile, lobe count. Weight sets, 76 photos (head agreement geo of the prototype, mean; LOO excludes the photo's design_key):

| weights (partition, iou, profile, lobes) | rules geo | LOO geo | full geo | full picks own design |
|---|---|---|---|---|
| 0.4, 0.3, 0.2, 0.1 (prototype) | 0.814 | 0.776 | 0.889 | 41/76 |
| **1, 0, 0, 0 (chosen)** | 0.814 | 0.819 | 0.893 | 43/76 |
| 0.6, 0.2, 0.1, 0.1 | 0.814 | 0.786 | 0.892 | 44/76 |
| 0.7, 0.1, 0.2, 0 | 0.814 | 0.802 | 0.895 | 44/76 |
| 0.5, 0, 0.3, 0.2 | 0.814 | 0.777 | 0.882 | 40/76 |

The rules improved since H1 (geo 0.758 -> 0.814), so the template alone barely beats them on this measure. A scale search (0.8-1.2 R) did not help (own design 39/76). On the gate measure (score.py head IoU, below) the prototype weights give LOO 0.8511 vs 0.8494 for partition-only, but lose layer IoU (-0.001 per photo) and S35-19 color (0.959 -> 0.867, it picks S30-15); partition-only was kept.

**Template part** (`role: template`, id `template`): params `template_id`, `cx`, `cy`, `r` (the photo head circle in canvas units), `rotation_deg` (0: no rotation is measured), `score`, `group_1`..`group_4` (reference Color -> color group). Confidence = score, so a match under `FLAG_BELOW` 0.6 is flagged ("library head shape is a weak match"). `build` places piece i at z `TEMPLATE_Z` 10 + i, so the occlusion cuts stack the pieces exactly as the reference paints them whatever Colors they map to; tails stay ours and are cut to the outside of the template head (`_under_head`); a piece repeated later with the same outline is drawn once (the shop validator's `dup_geometry`). The rose sits on the template's center pieces (their reach), else on the detected center moved onto the template. The detected head parts stay in `RosetteModel.alternatives`; since fix round 4 the head card's picker (`pipeline.use_head`, below) chooses between them and the library candidates.

**Decision** (`pipeline.prefer_template`, `TEMPLATE_MIN_GAIN` 0.02). Measured per photo with `tools/head_templates.decision_row` (head IoU = `tools/score.py` head_iou, through `with_template`, `build` and the writer). The brief's rule, a threshold on the match score, does not work: the score does not predict which head is better (score bins 0.45-0.95 show gains of -0.02 to +0.14 with no trend; low scores are the hard photos where the rules fail too), and every threshold leaves the LOO mean between 0.830 and 0.841. What predicts it is the photo's own head fit: the template's `head_fit` silhouette IoU minus the rule-based head's (correlation 0.63 with the head IoU gain; the layer agreement difference only 0.35). Margin sweep (LOO mean head IoU / IDs losing more than 0.02 head IoU): -0.02 0.8460 / 13, 0 0.8491 / 4, **0.02 0.8494 / 1**, 0.04 0.8490 / 1, 0.10 0.8458 / 1. No score floor (`TEMPLATE_MIN_SCORE` 0).

| 76 photos, head IoU (score.py) | rules | template always | decision rule |
|---|---|---|---|
| leave-one-out (honest for new designs) | 0.8299 | 0.8405 | **0.8494** (18 photos use the template) |
| full library (production) | 0.8299 | 0.8542 | **0.8575** (24 photos) |
| S29-15 | 0.699 | 0.865 (A1-15) | 0.865 (LOO and full) |

Full-library effect on the 24 photos that switch (score.py compare and shape_compare, rules -> template), per 76 photos: color +0.0041, whole shape +0.0078, head +0.0276, layer +0.0074; color >= 0.9 on these 24: 14 -> 16. Losses over 0.02: color S17-15 0.687 -> 0.618, S36-15B 0.782 -> 0.713, S52-19 0.908 -> 0.879, S58-17 1.000 -> 0.972 (S29-15 and S11a-15 -0.020); whole shape S52-19 0.952 -> 0.924; head S52-19 0.850 -> 0.804, S56-17 0.883 -> 0.857.

**Limitations.** The matcher picks the photo's own design for only 43 of 76 photos with the full library; S29-15 gets A1-15, a plain pinked disc around a center, not its layered stars (its own template scores 0.329 partition against 0.462: the photo's glitter and two pinks do not split the way the reference's Colors do). The head IoU rises because the disc covers the star outline better than the blobs did; the picture is a cleaner but simpler head. Template Colors reach the model only through the one-to-one category -> group map (a per-piece majority map was tried: color +0.003 instead of +0.004, not kept). Rotation and mirroring are not measured. The library is a 1.4 MB data file regenerated by hand when the references change.

**Fix round 1** (33799d0, 8d8aa03; full notes in the H7 report). The library is rebuilt with `build_library(Path("reference/svg"))` (about 13 minutes; it also stores each file's tail Colors and points count, `tools/truth`).
- Matcher judged by own-design retrieval (full library, design_key of the photo top-1 / top-3 of 76). Added features: head color count (labels / categories holding >= 5 % of the window, `COLOR_MIN_SHARE`; score term exp(-|difference|)) and a lobe gate (|lobe count difference| <= `LOBE_GATE` 3.4: the photo's own design is within one lobe step for 61 of 76 photos and within two for 68). A pattern feature could not be built: every template has a Color 2 head piece and the files do not mark glitter. A three-class gate (round / 4-8 / 10+) excluded the photo's own design on 4-18 photos for every boundary tried, so the gate is a tolerance.

| weights (partition, iou, profile, lobes, colors), gate | top-1 | top-3 | S29-15 full / LOO pick | S35-19 full pick |
|---|---|---|---|---|
| A 1, 0, 0, 0, 0 (H7) | 43 | 57 | A1-15 / A1-15 (disc) | S35-19 |
| B 0.4, 0.3, 0.2, 0.1, 0 | 41 | 54 | S57-17 / S57-17 | S30-15 |
| C 0.4, 0.3, 0.2, 0.1, 0.1 | 49 | 60 | S57-17 / S57-17 | S30-15 |
| D B + gate 3.4 | 41 | 51 | S57-17 / S57-17 | S30-15 |
| **E C + gate 3.4 (kept)** | 46 | 56 | S57-17 / S57-17 | S30-15 |
| F 0.6, 0.2, 0.1, 0.1, 0.1 + gate | 48 | 56 | S57-17 / S57-17 | S30-15 |

S29-15's own template is not reachable with these features: it is worse than S57-17 on every one (partition 0.329 vs 0.385, IoU 0.717 vs 0.909; a rotation and scale search lifts its IoU only to 0.775): the reference draws the side loops shorter than the photo shows. S35-19's own template ranks second (0.839 vs S30-15 0.864) whenever IoU, profile or colors count.
- Template colors and the tails (`TEMPLATE_TAIL_RULE`). "map" (tail Color k forced onto our tail Color k group, head-only categories by overlap) and "check" (the overlap map is kept only when it already does that; otherwise the rule-based head, with a warning) on the 76 photos:

| full library | head IoU (rules 0.8299) | color mean delta | color losses > 0.02 | color OK (rules 47) | layer delta |
|---|---|---|---|---|---|
| map | 0.8536 (22 photos) | -0.0027 | A16-17, S17-15, S35-19, S40-15, S51-19, S52-19, S58-17 | 47 | +0.0043 |
| **check (kept)** | 0.8445 (13 photos) | +0.0027 | S35-19, S58-17 | 48 | +0.0048 |

LOO: map 0.8481, check 0.8381. Under "map" the template heads' color agreement with the photo falls to 0.1-0.2 (S40-15 0.153, S52-19 0.143): a different design's head Color k is not the photo's tail Color k. Under "check" S29-15 keeps its rule-based head (S57-17's colors disagree with the tails).
- The template head goes through the same gross-failure check as the rule-based head (`_check_head`); when the two heads overlap less than `TEMPLATE_DIFFERS_IOU` (0.7 in fix round 1, 0.8 since fix round 2; silhouette IoU) a warning asks the user to compare and use the toggle (4 of the 13 switching photos). Color groups only the hidden head uses move to `RosetteModel.alternative_groups`; the template part carries the reference's `points` count (head card, scoreboard).
- Scoreboard (34): PASS 23 -> 22 (S35-19 0.959 -> 0.867 with S30-15's head), shape 0.913 -> 0.916, head 0.821 -> 0.830, silently wrong 1 (S46-15).

**Fix round 2** (a98cfbe). The matcher now scores the partition under the mapping the head is drawn with: a reference category that is also one of the reference's tail Colors k goes to the group our tails carry as Color k, head-only categories follow the overlap (`templates.tail_consistent_mapping`); a template that cannot be mapped (a tail Color of its head that our tails do not have) is not considered. Own-design retrieval, full library, 76 photos:

| weights (partition, iou, profile, lobes, colors) + lobe gate 3.4, tail-bound partition | top-1 | top-3 | S29-15 full / LOO | S35-19 full / LOO |
|---|---|---|---|---|
| P 1, 0, 0, 0, 0 | 49 | 55 | S58-17 / S58-17 | own / S30-15 |
| Q 0.4, 0.3, 0.2, 0.1, 0.1 | 47 | 55 | S57-17 / S57-17 | S30-15 / S30-15 |
| R 0.6, 0.2, 0.1, 0.1, 0.1 | 48 | 56 | S57-17 / S57-17 | S30-15 / S30-15 |
| S 0.5, 0.3, 0.1, 0.1, 0 | 46 | 56 | S57-17 / S57-17 | S30-15 / S30-15 |
| **T 0.4, 0.3, 0.2, 0, 0.1 (kept; a fifth set)** | 48 | 58 | S57-17 star / S57-17 | own / S30-15 |

With the soft lobe term S30-15 beat S35-19's own template only through it (lobes 1.0 vs 0.19; own wins partition 0.68 vs 0.57, IoU 0.907 vs 0.882, profile); the gate already enforces the lobe count, so T drops the term. S29-15's own template stays out of reach (tail-bound partition 0.173 vs S57-17 0.231, IoU 0.717 vs 0.909).

Decision: the outline gain (silhouette +0.02, as before) and `TEMPLATE_MIN_PARTITION` 0.70 on the tail-bound partition. Photos where the outline wins (full library 23, LOO 20), sorted by partition, color change vs the rules head: full S29-15 0.231 0.000, S52-19 0.282 -0.032, S10-19 0.348 +0.045, S9-19 0.359 +0.074, S15-15 0.440 +0.050, S17-15 0.483 -0.029, S36-15B 0.516 +0.048, S51-19 0.526 -0.059, S2-15B 0.661 0.000, S30-15 0.665 +0.008, S57-17 0.673 +0.118, S35-19 0.680 -0.014, S58-17 0.697 -0.028, then 0.706-0.844 (10 photos) -0.014..+0.156. LOO losses reach up to 0.697 as well (S58-17); none at or above 0.70.

| floor | LOO head IoU / used / color losses > 0.02 | full head IoU / used / color losses > 0.02 | color OK (rules 47) LOO / full | S29-15 | S35-19 |
|---|---|---|---|---|---|
| 0 | 0.8523 / 20 / 11 | 0.8597 / 23 / 4 | 44 / 48 | S57-17 star | own 0.944 |
| 0.30 | 0.8473 / 16 / 8 | 0.8586 / 21 / 3 | 45 / 49 | rules | own 0.944 |
| 0.54 | 0.8388 / 9 / 5 | 0.8479 / 15 / 1 (S58-17) | 45 / 49 | rules | own 0.944 |
| **0.70 (kept)** | 0.8343 / 3 / 0 | 0.8400 / 10 / 0 | 47 / 48 | rules (S57-17 on the toggle) | rules 0.959 (own on the toggle) |

No shape losses over 0.02 at 0.30 and above. Below the floor the rule-based head is drawn, the library head waits in `alternatives` (one click on the head card) and a warning says that it fits the outline better but its colors do not agree.

**Fix round 3** (628c53f, ruling: shape first). `TEMPLATE_MIN_PARTITION` 0.54, and when the rule-based head fails the gross head check (`gross_failure`: agreement under 0.6 or silhouette under 0.78) the library head is drawn whatever its partition (still only when its outline wins by 0.02), flagged (confidence capped at 0.5) with the "heads differ" warning. Measured on the 76 photos (cached detections; new color gate: no ID from >= 0.9 to under 0.9, no loss over 0.05, whole shape losses at most 0.02):

| | head IoU (rules 0.8299) | library heads (override) | color >= 0.9 (rules 47) | color gate losers | losses 0.02-0.05 | shape losers > 0.02 | S29-15 | S35-19 |
|---|---|---|---|---|---|---|---|---|
| LOO | 0.8511 | 18 (9) | 44 | S30-15 -0.068, S35-19 -0.092, S61-19 -0.081 (floor); S51-19 -0.059, S52-19 -0.032 (to 0.876), S57-17 -0.127 (override) | S15-15, S17-15, S56-17, S58-17 | S52-19 -0.023 | S57-17, 0.257 | S30-15, 0.867 |
| full library | 0.8597 | 23 (8) | 48 | S51-19 -0.059, S52-19 0.908 -> 0.876 (both override) | S17-15 -0.029, S58-17 -0.028 | S52-19 -0.023 | S57-17, 0.257 | own, 0.944 |

S57-17's head is loop blocks around a pinked ring (outline lobes like the photo's), not a star: S29-15's own star template ranks 8th among the mappable, gated templates (score 0.506 vs 0.624).

**Fix round 4** (af9dbc1 override, 9d71440 picker). The override (rule-based head a gross failure, library head drawn below the color floor) now also needs the library outline to win by `TEMPLATE_OVERRIDE_GAIN` 0.10 (head_fit silhouette) and the tail-bound partition to reach `TEMPLATE_OVERRIDE_PARTITION` 0.30 (`pipeline.override_allowed`, `library_head_path`: "floor", "override", or "colors" = the rule-based head with the colors warning and the library heads on the head card). The floor path is unchanged (partition >= 0.54, gain >= 0.02). Measured on the 76 photos (cached detections, same rows as fix round 3); the photos where the rule-based head fails grossly and the library outline wins by 0.02 but the partition is under 0.54:

| ID | library head (full) | silhouette gain | partition | color rules -> library | whole shape | path now |
|---|---|---|---|---|---|---|
| S10-19 | A13-15B | 0.240 | 0.348 | 0.904 -> 0.950 | +0.055 | override |
| S15-15 | S15-15B | 0.278 | 0.440 | 0.544 -> 0.593 | +0.069 | override |
| S17-15 | own | 0.187 | 0.483 | 0.687 -> 0.658 | +0.027 | override |
| S36-15B | S36-15 | 0.327 | 0.516 | 0.782 -> 0.830 | +0.043 | override |
| S51-19 | A1-15 | 0.265 | 0.526 | 0.919 -> 0.860 | +0.008 | override |
| S9-19 | S1-15B | 0.151 | 0.359 | 0.896 -> 0.971 | +0.032 | override |
| S29-15 | S57-17 | 0.285 | 0.231 | 0.257 -> 0.257 | +0.031 | override (no head layer) |
| S52-19 | S31-15 | 0.163 | 0.282 | 0.908 -> 0.876 | -0.023 | rules (partition) |

| 76 photos | head IoU (rules 0.8299) | library heads (override) | color >= 0.9 (rules 47) | color gate losers | shape losers > 0.02 |
|---|---|---|---|---|---|
| fix round 3, full | 0.8597 | 23 (8) | 48 | S51-19 -0.059, S52-19 -0.032 | S52-19 -0.023 |
| **fix round 4, full** | 0.8602 | 22 (7) | 49 | S51-19 -0.059 | none |
| fix round 4, LOO | 0.8476 | 15 (6) | 45 | S30-15, S35-19, S61-19 (floor); S51-19 (override) | none |

S52-19 is back on its rule-based head (it has points parts, so the partition bar applies). Ruling: the partition bar does not apply when the rule-based path drew no head layer at all (no head, ring or points part: only a center and/or unrecognised blobs; `has_head_layer`, `HEAD_LAYER_ROLES`): a missing head is worse than any mappable library head, whose colors the user can fix on the head card. The gain bar still applies and the library head is flagged (confidence cap 0.5) with the heads-differ warning. Among the photos above S10-19, S29-15, S36-15B and S9-19 have no head layer; only S29-15 (partition 0.231) changes, it draws S57-17 by default again. S51-19 keeps its override (gain 0.265, partition 0.526, flagged; color -0.059 accepted by ruling, the picker is the way out).

**Fix round 5** (review fixes). `pipeline.choose_head(rules, lib, fit_of, overlap_of)` is the decision glue of `analyze_with_fit`, testable with stub fits. (1) The library candidates stay in `alternatives` on every photo, also when no library head wins (no warning then), so the head card's picker shows wherever the matcher found a mappable template. (2) Every head on the card is checked at analyze time as if it were drawn: its head fit, the FLAGGED_CONF cap for a library head drawn without color agreement (partition under 0.54), the gross head check (`_check_head`) and the heads-differ warning; the flagged confidences are stored on the parts, the head-dependent warnings and the head fit in `RosetteModel.head_states` (keyed by template_id or "rules"). `use_head` and `logic.js switchHead` swap those warnings with the parts, the UI shows the picked head's fit, and each library head keeps its own pose (since the H9 review: cx, cy, r and rotation from its own search, changed only by a user edit of that head; switching never copies one head's placement onto another). Going back to the rules head after an override shows the rules head's own gross-failure warning and cap, plus the colors warning. (3) A model with nothing drawn in the head (no head fit) counts as a failed rules head for the override, and with no head layer the partition bar is skipped. A test checks that the library holds one template per product reference file (a stale cache fails it).

Head card picker. `templates.rank` lists the matcher's best templates (the decision rule's choice first) and `with_template` keeps the next `TEMPLATE_CANDIDATES` - 1 in `alternatives` (part ids `template_2`, `template_3`; params `rank` and `description`), next to the detected head parts. `templates.describe` mines the one-line description from the template pieces: a piece around the center is a ring (with a hole), a center or a disc (reach over 0.65 R), or an N-point star (4 or more lobes deeper than 1.3x); a piece away from it is a strip (4x longer than wide), a trim (under 0.6 of its hull), a loop (as wide at its outer end as half its angular width), a kite (tapered, at most 4 corners) or a point; a piece repeated with the same outline counts once (S35-19: "6 loops, 6 kites, 12 strips, 6 trims, ring, center"). The head card (Workbench and Guided share it) lists "Rule-based head (the detected parts)" and each candidate as "Library template <id> - match NN%, N points - <description>" with radio buttons; picking one draws it (`logic.js switchHead`, mirror of `pipeline.use_head`; the render API takes the same choice as `head` and returns the switched model), unconfirms the head cards and the head check, and the picked head is flagged like any library head (a candidate's confidence is its own match score). The default stays the decision rule's choice. Candidates for the IDs above: S29-15 S57-17 / S69-19 / S58-17 (its own star template is not among them); S51-19 A1-15 / A13-15B / S63-17; S52-19 S31-15 / own / S69-19; S35-19 own / S30-15 / S31-15.

## H8: any library head by id; matcher normalization

Started at d812567. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h8-report.md`.

**Choose from the library (H8; since H9 not in the UI).** H8 added a head card control to pick any library head by id. H9 removed it and the top-3 radios from the UI (the tool converts a new image automatically); the server side stays for scripted use and tests: every library template by id with its mined description (`GET /api/library`, `templates.library_index`). Analyze still places, fits and flags only the top 3. A head outside them is requested from the server with `/api/render` and `head`: `/api/analyze` keeps the prepared photo, labels, groups and detected model (`pipeline.PhotoContext`) in a small in-memory cache (`app/main.py PhotoCache`, the last `PHOTO_CACHE_SIZE` 8 photos, least recently used out) under the model's `photo_token`; `pipeline.add_library_head` places the template at its own pose on the photo head circle (pose search, H9), colors it under the tail-bound mapping (else the free overlap mapping, no lobe gate: `templates.score_template`) and checks it exactly like a candidate (`_library_head_state`: head fit, FLAGGED_CONF cap without color agreement, gross head check, heads-differ warning), adds it to the candidates (id `library_<id>`, rank 4 and up) and returns the switched model. Each cache entry holds the photo arrays and the detected model, about 5-15 MB, so the cache holds at most about 120 MB. A template whose tail Colors have no tail group on the photo is colored by the free overlap mapping (`free_colors`): it is flagged (FLAGGED_CONF) with "Library head colors are not anchored to the tails - check every Color on the head card." An unknown id gives 422 "There is no library head with this id", a template none of whose Colors maps onto the photo's groups 422 "This library head cannot be colored with the photo's colors". An unknown token (server restarted, or more than 8 newer photos) gives 422 "The photo for this model is no longer in memory - upload the photo again to choose a head from the library." The part card reads "Head shape: chosen from the library: <id> - <description>". Timings (this machine): analyze A7-15 2.1 s, S29-15 2.6 s (H7 at d812567: 2.9 s, 3.4 s; an eager version that checked all 77 templates at analyze took 14.4 s and 15.5 s with 48-62 KB models), model JSON 3 KB and 6 KB, an on-demand pick 0.5 s including the render. Analyze got faster than H7 because the detected head's render is shared by its fit and overlap and computed once.

**Normalization (evaluated, not kept).** Templates and photo heads are normalized to the outer head radius. S29-15's reference draws its loops much shorter than the photo, so its own star template is too large when normalized that way (own rank 8). Tried, on the 76 cached photos with the full library (own-design top-1 / top-3), then leave-one-out head IoU through the decision rule (`tools/head_templates.decision_row`): inner normalization by the radius holding 35 % or 50 % of the head foreground in the window (`m35`, `m50`), by the center badge's radius (`c`), each alone or as a 5-step scale search between 1 and that ratio (clamped 0.6-1.6).

| normalization | top-1 | top-3 | S29-15 pick (own rank) | S35-19 pick | LOO head IoU | full head IoU | color >= 0.9 (full) |
|---|---|---|---|---|---|---|---|
| **outer (kept)** | 48 | 58 | S57-17 (8) | own | 0.8476 | 0.8602 | 49 |
| m50 only | 47 | 54 | S69-19 (-) | S30-15 | | | |
| m35 only | 48 | 54 | S69-19 (15) | S30-15 | | | |
| center only | 31 | 39 | S2-15 (-) | S30-15 | | | |
| m50 search | 49 | 56 | S69-19 (19) | own | | | |
| m35 search | 50 | 56 | S69-19 (17) | own | 0.8481 | 0.8599 | 48 |
| center search | 48 | 57 | S69-19 (18) | own | | | |

The m35 search gains 2 top-1 (A12-15, S42-15, S49-17 in; S48-17 out) but loses 2 top-3 (S1-15, S10-19, s34-15 out; S49-17 in); its LOO head IoU is +0.0005, its full-library head IoU -0.0003, and it sends S52-19 back to a library head (color 0.908 -> 0.827, whole shape loss over 0.02, against the fix round 4 ruling) and S56-17 loses shape (LOO). Not kept: outer normalization stays. S29-15's own star stays out of reach with every variant (rank 15-19 or not mappable): its reference is not faithful to the photo, which is why the user can now choose any head by id.

## H9: the new-image case is the headline

Started at 79db842. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h9-report.md`.

**Why leave-one-out is the headline.** The tool must convert any new image automatically; picking an existing design is not the workflow. For a photo of a new design the head library holds no "own design", so the honest estimate of production is the photo scored with its own design (design_key: the file and its variants) left out of the library. `tools/score.py` now reports that as the headline ("new image") and the full library as a second, labelled column and summary ("known design", `--known` makes it the headline). The scratch 76-photo comparison scripts take the same `--known` switch (default new image). The pipeline itself does not exclude anything.

**Head card.** The head card shows the automatic head only: "Automatic result: library template <id>, match NN%" (or "rule-based head") as its heading, then the template part with its "Template Color k drawn as" selectors, cx/cy/r/rotation, flag and confirm. The head radios and the library chooser are gone from both modes; `use_head`, `/api/render` `head`, `/api/library` and the photo cache stay server-side with their tests.

**Head circle on star heads (attempt, not kept).** On S29-15 `prepare`'s head circle is too high and too small (center y 162 px, r 158 px; the badge sits at about y 212 px, the head reaches about 190-210 px): the lower star arms pass for the tails top (321 px) and the 90-degree symmetry score does not fit a 6-fold star. Rules tried against the reference files' own head circles (`drawn_frame`, 76 photos, units of the rosette height 800; the reference circles are a noisy truth: S35-19 +16.2 and A7-15 +10.1 although right on the photo):

| rule | mean / worst abs error | S29-15 | worsened (change in abs error) |
|---|---|---|---|
| current (90-degree symmetry) | 9.2 / 29.2 | -27.5 | - |
| best of 60/72/90-degree symmetry | 10.4 / 40.4 | +12.5 | S36-15B +36, S57-17 +32, S59-17 +32, S36-15 +30, S17-15 +19 |
| best of 45/60/72/90 | 9.7 / 40.4 | +12.5 | S36-15B +36, S57-17 +32, S36-15 +30 |
| 90-degree + half-width penalty 0.2 | 9.3 / 30.6 | -17.4 | S10-19 +13 |
| 90-degree + half-width penalty 0.3-0.5 | 10.4-10.7 / 54-55 | +12 to -16 | S57-17 +49, S33-15 +27, S36-15B +22 |

No rule improved S29-15 without making another photo worse by more than 13 units, so `prepare` is unchanged; the head circle is a later round.

**Pose search (kept).** `templates.rank` places each template at its best pose on the photo circle: center offset dy -0.2..+0.3 and dx -0.1..+0.1 head radii, scale 0.8-1.25 (`POSE_*`), then the turns within its symmetry period 360/k (k = `templates.symmetry`, the lobes of its whole outline; `POSE_TURNS` 4), then offset and scale again at the best turn; the silhouette IoU picks the pose (`best_pose`, nearest-neighbour warps of the template grid, `pose_grid`), and the full score at that pose and unmoved decides (`posed_features`). The pose is stored on the template part (cx, cy, r, rotation_deg, and pose_dx / pose_dy / pose_scale); a drawn head moved by `RECENTER_MIN` 0.1 or more leaves an information line in `RosetteModel.info` (not a flag) for the head-circle round; `use_head` recomputes it on a switch and the UI lists it under Warnings as plain text. Ties in the pose search keep the smaller move, then the grid order; a pose whose colors follow the tails beats one colored by the free mapping, and the unmoved pose is kept when the searched one cannot be colored under the tails.

| 76 photos (cached detections) | known top-1 / top-3 | LOO head IoU | full head IoU | LOO color >= 0.9 | LOO color losers | S29-15 pick (own rank) | analyze S29-15 / A7-15 |
|---|---|---|---|---|---|---|---|
| unmoved (H8) | 48 / 58 | 0.8476 | 0.8602 | 45 | S30-15, S35-19, S51-19, S61-19 | S57-17 (8) | 2.6 / 2.1 s |
| pose search | 49 / 57 | 0.8489 | 0.8585 | 46 | S17-15, S30-15, S51-19, S61-19 | S58-17 at dy +0.1, scale 1.25 (9) | 3.3 / 2.3 s |

S29-15 is not fixed by it: its true head is about 0.32 R lower and 1.3x larger than the circle, at the edge of the search range.

**Decision constants on the new-image case (grid, kept as they are).** Grid on the leave-one-out rows at the new poses: TEMPLATE_MIN_PARTITION 0.40-0.70 x TEMPLATE_MIN_GAIN 0-0.04 x TEMPLATE_OVERRIDE_GAIN 0.05-0.20 x TEMPLATE_OVERRIDE_PARTITION 0.20-0.40 (unflagged = template confidence >= 0.6, heads overlap >= 0.8, head fit not gross).

| MIN_PARTITION, MIN_GAIN, OVERRIDE_GAIN, OVERRIDE_PARTITION | LOO head IoU | library heads | color >= 0.9 | color losers (> 0.05 or below 0.9) | shape losers > 0.02 | new unflagged library heads under 0.9 |
|---|---|---|---|---|---|---|
| **0.54, 0.02, 0.10, 0.30 (kept)** | 0.8489 | 17 | 46 | S17-15, S30-15, S51-19, S61-19 | - | - |
| 0.40, 0.02, 0.10, 0.30 | 0.8502 | 19 | 46 | same | - | - |
| 0.45 or 0.50, 0.02, 0.10, 0.30 | 0.8496 | 18 | 46 | same | - | - |
| 0.54, 0.00, 0.10, 0.30 | 0.8460 | 30 | 46 | + S11-15, S27-15, S28-15, S53-19 | S27-15, S56-17 | A13-15B, A9-15, S28-15, S8-15 |
| 0.54, 0.04, 0.10, 0.30 | 0.8486 | 16 | 46 | same | - | - |
| 0.54, 0.02, 0.10, 0.20 | 0.8512 | 19 | 45 | + S52-19, S57-17 | S52-19 | - |
| 0.60, 0.02, 0.10, 0.30 | 0.8467 | 14 | 47 | S17-15, S30-15, S51-19 | - | - |
| 0.40, 0.02, 0.05, 0.20 | 0.8525 | 21 | 45 | + S52-19, S57-17 | S52-19 | - |

The only gain without new losers (partition 0.40) comes from two photos, S1-15B and S69-19, and costs S69-19 0.048 color; ruled out. The override gain changes nothing at the 0.30 partition bar. Every library head drawn for a new image keeps its confidence from the match, the caps and the warnings (a weak match is flagged).

**Scoreboard (34 IDs, `tools/score.py`, after H9).** New image (headline): PASS 23/34, color agreement 0.854, shape 0.920, head 0.844. Known design (full library): PASS 23/34, color 0.859, shape 0.921, head 0.847. Before H9 (new image / known): PASS 22/34 / 23/34, color 0.852 / 0.857, shape 0.920 / 0.922, head 0.846 / 0.851.

## Final state (Task 16)

### Stages, modules and tuning constants

Constants are module-level names; read their comments in the source before changing one.

| Stage | Module | Tuning constants |
|---|---|---|
| prepare: decode, 1200 px, mask, axis, head circle | `rosette/vision/prepare.py` | `WORK_HEIGHT`, `CANVAS_SPAN`, `ALPHA_CUTOFF`, `MIN_DELTA_E`, `NOISE_FACTOR`, `MAX_BG_NOISE`, `OPEN_PX`, `CLOSE_PX`, `MIN_AREA_FRAC`, `MAX_AREA_FRAC`, `AXIS_SEARCH_FRAC`, `HEAD_PROFILE_FRAC`, `HEAD_RING_MARGIN` |
| color groups, stripe split, motif absorption, mixed-texture warning | `rosette/vision/colors.py` | `L_WEIGHT`, `SMOOTH_UNITS`, `FLAT_PCT`, `GRAIN_WEIGHT`, `SAMPLE_N`, `MIN_CENTER_SEP`, `PRINT_MAX_FRAC`, `TEXTURE_WIN_UNITS`, `PATTERN_T`, `MIXED_PATTERN_SHARE`, `BADGE_IN_HEAD`, `BADGE_MAX_AREA`, `BADGE_EDGE`, `BADGE_UNSURE`, `BADGE_ENCLOSED`, `BADGE_SHAPE_MIN`, `SEED` |
| part detection driver, leftovers as `unknown` | `rosette/vision/parts.py` | `LEFTOVER_OPEN_UNITS`, `TAIL_TOUCH_PX`, `HEAD_MARGIN`, `UNKNOWN_CONFIDENCE` |
| head: rings, points, bows, center, blob rings | `rosette/vision/parts_head.py`, `parts_common.py` | `RING_EXTEND_COV`, `FULL_COV`, `MIN_BAND_UNITS`, `MIN_LAYER_COV`, `MAX_COUNT`, `BOW_ENDS`, `PETAL_FULLNESS`, `BLOB_UNCONFIRMED_CONF`, `MIN_INNER_RATIO`, `HEAD_ROUND`, `PINKED_DEPTH`, `RING_MAX_GAP`, `PLEATED_FULL_MIN`, `PLEATED_REACH_UNITS` |
| tail stripes, glitter read as two tones, fans (H6) | `rosette/vision/parts_tails.py` | `ROW_FRACS`, `COUNT_ROWS`, `DENSE_SPAN`, `GAP_UNITS`, `ROW_BAND_PX`, `TOP_INSET`, `GLITTER_*`, `FAN_*` |
| head shape library and matcher (H7) | `rosette/vision/templates.py`, cache `rosette/data/head_templates.json`; decision in `rosette/pipeline.py` | `WEIGHTS`, `PROFILE_SCALE`, `LOBE_SCALE`, `WINDOW_V`, `GRID`, `EXTENT`, `TEMPLATE_MIN_GAIN`, `TEMPLATE_MIN_SCORE`, `TEMPLATE_MIN_PARTITION`, `TEMPLATE_OVERRIDE_GAIN`, `TEMPLATE_OVERRIDE_PARTITION`, `HEAD_LAYER_ROLES`, `TEMPLATE_CANDIDATES`, describe: `STRIP_ASPECT`, `TRIM_SOLIDITY`, `LOOP_END`, `STAR_DEPTH` |
| tidy: unknown specks, tail slivers (H6) | `rosette/tidy.py` | `SPECK_MAX_HEAD_SHARE` (and `SLIVER_SHARE` from assign) |
| Color 1-4 assignment, confidence, warnings | `rosette/assign.py`, `rosette/roles.py`, rules in `rosette/data/assign_rules.json` (mined by `tools/mine_rules.py`) | `FLAG_BELOW`, `MAX_COLORS`, `PATTERN_MIN_RIBBON_SHARE`, `TAIL_POS_TOL`, `FLAGGED_CONF`, `PATTERN_MERGE_PENALTY`, `SLIVER_SHARE`, `RULE_LOGODDS` |
| geometry, shapes, occlusion cuts | `rosette/geometry/`, `rosette/shapes.py`, `rosette/build.py`, `rosette/geometry/clip.py` | `QUAD_SEGS`, `SIMPLIFY`, `MIN_PIECE` |
| SVG writer and contract validator | `rosette/writer.py`, `rosette/validate.py` | none (the contract is fixed) |
| orchestration, flagging, head fit | `rosette/pipeline.py` | `FLAG_BELOW` (imported from assign), `HEAD_FIT_MIN`, `HEAD_SILHOUETTE_MIN`, `HEAD_FIT_EXTENT`, `HEAD_FIT_WINDOW`, `HEAD_FIT_GRID` |
| shop simulation (solid and pattern previews) | `rosette/simulate.py` | none |

### Shop contract

Every generated file must satisfy section 3 of the [design spec](superpowers/specs/2026-11-05-rosette-svg-generator-design.md): fixed top-level layer order (`Layer_1`, `Layer_3`, `Layer_2_1`, `Layer_6`, `Layer_rose`; issue `group_order` when it is not kept), category classes for Color 1-4, Color 2 usable in solid and pattern mode, and Color 3 correct in both. `rosette/validate.py` checks it and `render` returns the issues; the UI does not offer a download while an issue is open. The shop's own coloring logic is in `reference/shop-configurator.js` (verbatim); `rosette/simulate.py` mirrors it for the UI previews.

### UI modes

`app/main.py` (FastAPI, started with `run.bat` or uvicorn on 127.0.0.1:8765) serves `app/static`. **Workbench** shows the photo, the detected model, the solid and pattern previews, edit controls and the flagged items on one screen. **Guided** walks through the same state one question at a time (`guided.js`); switching modes keeps the state. Edits go back through `render`, so the validator runs after every change.

### Current scoreboard (`tools/score.py`, 34 IDs, re-recorded for H9)

Re-recorded after H9 (09e0555): the columns are the new-image case (each photo's own design left out of the head library, the headline); the last column is the known design (full library) agree / success / headS. `stack reversed` counts the kind pairs our file stacks the other way round from the designer file (reported only).

| ID | agree | uni | success | uni success | shape | headS | stack reversed | known design agree / success / headS |
|---|---|---|---|---|---|---|---|---|
| A10-15B | 0.906 | 0.942 | yes | yes | 0.870 | 0.684 | 0 | 0.906 / yes / 0.684 |
| A13-15B | 0.187 | 0.187 | no | no | 0.929 | 0.967 | 1 (head>center) | 0.187 / no / 0.967 |
| A14-17 | 0.949 | 0.949 | yes | yes | 0.957 | 0.963 | 0 | 0.949 / yes / 0.963 |
| A15-17 | 0.968 | 0.963 | yes | yes | 0.965 | 0.957 | 0 | 0.968 / yes / 0.957 |
| A16-17 | 0.837 | 0.926 | no | yes | 0.895 | 0.830 | 0 | 0.837 / no / 0.830 |
| A9-15B | 1.000 | 1.000 | yes | yes | 0.939 | 0.828 | 0 | 1.000 / yes / 0.828 |
| S10-19 | 0.950 | 0.950 | yes | yes | 0.902 | 0.780 | 0 | 0.987 / yes / 0.812 |
| S13-15 | 0.894 | 0.894 | no | no | 0.962 | 0.928 | 0 | 0.894 / no / 0.928 |
| S17-15 | 0.599 | 0.843 | no | no | 0.886 | 0.730 | 0 | 0.669 / no / 0.738 |
| S2-15B | 0.687 | 0.687 | no | no | 0.945 | 0.853 | 0 | 0.687 / no / 0.920 |
| S26-15 | 0.800 | 0.985 | no | yes | 0.933 | 0.856 | 0 | 0.800 / no / 0.856 |
| S28-15 | 0.998 | 0.995 | yes | yes | 0.980 | 0.964 | 1 (star>loop) | 0.998 / yes / 0.964 |
| S29-15 | 0.265 | 0.254 | no | no | 0.931 | 0.815 | 0 | 0.265 / no / 0.815 |
| S30-15 | 0.907 | 0.907 | yes | yes | 0.963 | 0.934 | 0 | 0.983 / yes / 0.944 |
| S31-15 | 0.929 | 0.966 | yes | yes | 0.929 | 0.838 | 0 | 0.929 / yes / 0.838 |
| S33-15 | 0.905 | 0.905 | yes | yes | 0.955 | 0.890 | 0 | 0.905 / yes / 0.890 |
| S35-19 | 0.963 | 0.963 | yes | yes | 0.927 | 0.845 | 0 | 0.942 / yes / 0.805 |
| S37-15 | 0.949 | 0.936 | yes | yes | 0.905 | 0.825 | 0 | 0.949 / yes / 0.825 |
| S40-15 | 0.893 | 0.889 | no | no | 0.927 | 0.879 | 0 | 0.893 / no / 0.879 |
| S41-15 | 0.925 | 0.932 | yes | yes | 0.906 | 0.914 | 0 | 0.925 / yes / 0.914 |
| S43-15 | 1.000 | 1.000 | yes | yes | 0.953 | 0.926 | 0 | 1.000 / yes / 0.926 |
| S45-15 | 0.935 | 0.935 | yes | yes | 0.926 | 0.809 | 0 | 0.935 / yes / 0.809 |
| S46-15 | 0.751 | 0.936 | no | yes | 0.835 | 0.660 | 0 | 0.751 / no / 0.660 |
| S5-15 | 0.910 | 0.988 | yes | yes | 0.852 | 0.734 | 0 | 0.910 / yes / 0.734 |
| S53-19 | 0.513 | 0.513 | no | no | 0.763 | 0.814 | 0 | 0.513 / no / 0.814 |
| S56-17B | 0.969 | 0.963 | yes | yes | 0.902 | 0.817 | 0 | 0.988 / yes / 0.848 |
| S58-17 | 0.972 | 0.986 | yes | yes | 0.940 | 0.868 | 0 | 0.972 / yes / 0.868 |
| S67-19 | 0.964 | 0.964 | yes | yes | 0.947 | 0.861 | 0 | 0.964 / yes / 0.861 |
| S68-19 | 0.966 | 0.620 | yes | no | 0.885 | 0.676 | 0 | 0.966 / yes / 0.676 |
| S69-19 | 0.993 | 0.988 | yes | yes | 0.920 | 0.759 | 0 | 0.993 / yes / 0.759 |
| S70-19 | 0.956 | 0.956 | yes | yes | 0.957 | 0.939 | 0 | 0.956 / yes / 0.939 |
| S8-15 | 0.640 | 0.640 | no | no | 0.936 | 0.841 | 0 | 0.640 / no / 0.841 |
| S80-22 | 0.989 | 1.000 | yes | yes | 0.934 | 0.878 | 0 | 0.989 / yes / 0.878 |
| s34-15 | 0.956 | 0.951 | yes | yes | 0.928 | 0.844 | 1 (backing>star) | 0.956 / yes / 0.844 |

Official (per-axis), tools/score.py summary: new image `success 23  flagged-not-success 10  silently-wrong 1  errors 0` / `PASS 23/34 (target >= 80%)` / `color agreement mean 0.854  shape: mean IoU 0.920  head 0.844  tail 0.955  shape_ok 27/34`; known design (reported only) `PASS 23/34`, color 0.859, shape 0.921, head 0.847, shape_ok 27/34; `stacking: 3 kind pairs reversed vs the designer in 3/34 IDs (reported only)` / `success under uniform-scale alignment 25/34` (reported only).

Flagged although correct: 21 of the 23 successes; S28-15 and S43-15 are unflagged successes with visibly wrong heads (see 5b). S46-15 is silent again: its head fit (0.82 agreement, 0.94 outline) is not a gross failure; its miss is the scorer alignment effect described below.

Over all 76 reference photos (scratch `compare_all.py`, match >= 0.9 and no issue): 36 -> 45 OK, mean color agreement 0.689 -> 0.832 (H2), then 0.832 -> 0.837 with OK still 45 (H3); mean whole shape IoU 0.902. Silently wrong (not OK, not flagged): A13-15 (designer differs from photo), S46-15 (scorer alignment), S48-17 (0.672, up from 0.382; the remaining miss is not re-diagnosed) and s44-15 (designer differs: much narrower outer tails). The every-head reminder is the UI checklist item, not a flag.

### Sample outputs

`output/s35-19.svg` and `output/s46-15.svg` were regenerated for H2 (and regenerated again for H3: byte-identical, so unchanged) from `reference/photos/S35-19.png` and `reference/photos/S46-15.png` with `rosette.pipeline.analyze` + `render`, the same code path the UI uses, and checked with `rosette.validate.validate` (both return `[]`). Regenerated again for H5 (pointed tail tops, visible-edge line art; both validate to `[]`), for H6 (`s35-19.svg` changed: its two tail slivers are merged; `s46-15.svg` byte-identical; both validate to `[]`) and for H7 (`s35-19.svg` changed: its head is the S35-19 library template, match 0.68; `s46-15.svg` byte-identical; both validate to `[]`), for H7 fix round 2 (`s35-19.svg` back to the rule-based head) and for fix round 3 (`s35-19.svg` drawn with its own S35-19 library template again, partition 0.68 over the 0.54 floor; `s46-15.svg` byte-identical; both validate to `[]`); for fix round 4 both are byte-identical and validate to `[]`. They are the tool's unedited output, not confirmed by a person.

| Output | Flagged | Flagged items | Warnings |
|---|---|---|---|
| `output/s35-19.svg` | yes | the warnings only: the S35-19 library head (match 0.82) is above 0.6; no `unknown` parts; no color group below 0.6. | 2 tail slivers merged into their neighbours (H6); the library and rule-based heads overlap 79 % (heads differ; the head card offers S30-15 and S31-15 as well) |
| `output/s46-15.svg` | no | none (head fit 0.82 / outline 0.94 is information only) | none |

Before uploading either file to the shop, the user should open each photo in the UI, confirm the flags and compare the drawing with the photo. That confirmation has not been done; it is the user's step.

### Known limitations

- **Library heads (H7, after fix round 4) pick the photo's own design for 48 of 76 photos (top-3 58, offered on the head card).** S29-15 draws S57-17's head (loop blocks around a pinked ring, flagged), not its own star, which is not among its 3 candidates; S51-19 loses 0.059 color through the override (flagged; accepted). See the H7 section.

- **The success bar is not met: 23/34 succeed against 28 needed (80 %).** `pytest -m slow` fails accordingly. Tuning stopped; it is not finished.
- **S46-15 counts as silently wrong under per-axis scoring** (0.741, not flagged). It is a scorer alignment effect (the designer drew a larger head and narrower tails; uniform scale gives 0.926); its head fit (0.82) is not a gross failure.
- **Head shape errors are mostly not flagged.** The head flag fires on gross failures only (23 of 76 reference heads). S42-15, S43-15 and S28-15 have wrong head shapes that pixel overlap cannot tell from right heads; the fixed UI checklist item "Compare the head with the photo" is the safeguard until shape features land.
- The holdout is no longer clean (see Task 12 phase B), so the numbers on the tuned IDs overstate how the tool will do on new photos.
- Most successes are still flagged (21 of 23: placeholder `unknown` discs, or a part read below 0.6, mostly points rings). The flag is warranted but means most results need a human look.
- Remaining failure clusters:
  - Head structure not modeled: S17-15 (three layered stars, no head disc), S29-15 (glitter loops over a star, badge), S68-19 (stacked chevron points), A16-17 (irregular 8-point star in the head's own black), S13-15 (loops drawn as a star because the navy ruffle merged into the blue group).
  - Mask: S53-19 (pure white stripes on a pure white background are cut out; one tail reads as three).
  - Color grouping: S2-15B (snowflake print merged with white satin). A13-15B's glitter is its own group since H6; its colors now differ from the reference's other scheme (royal Color 1 by the mined rules).
  - Designer differs from the photo: S8-15 (centre stripe the same red as the outer), S26-15 (glitter stripe widths), S46-15 (proportions).
- Photos need a single rosette on a plain background; a rosette whose colors match the background can fail the mask. The messages for no rosette and unreadable files come from `prepare`.
- Classical CV only: unusual head constructions fall back to `unknown` placeholder discs that the user must fix in the UI.
