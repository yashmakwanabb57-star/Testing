# Pipeline

Photo in, configurator SVG out (`rosette/pipeline.py`):

1. `prepare` (rosette/vision/prepare.py) - decode, normalize to 1200 px, foreground mask, symmetry axis, head circle (90-degree symmetry, replaced by a fit to the silhouette when that is much larger: H10), head/tails split.
2. `pixel_groups` (rosette/vision/colors.py) - per-pixel color groups with pattern texture; a cluster hiding two similar stripe colors is split (`_split_stripes`; stripe edges are counted across a third stripe between them); small blobs are absorbed as motifs unless they form an even ring around the head center (star tips, `parts_common.blob_rings`) or are ribbon pieces on the head outline (H13); print on a center badge whose shape is not a disc or ring joins the badge's group (H3, below).
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

**Line art** (`rosette/geometry/outline.py`). The shop draws Layer_6 over everything, so st6 outlines now trace only the edges no later-painted shape hides (Layer_1 first, then Layer_2_1, each in document order). Since H14 every color shape is already its visible piece (see H14), so no later shape hides an edge any more; the rule now only draws an edge two abutting pieces share once and leaves out the hairline crossings of overlaps under `MIN_PIECE`, which the cut keeps whole. Before, the full polygon of a Color 1 tail stripe under a Color 1 head showed its top inside the head (A7-15). Shop-style render at 1700 px: tail pixels inside the head A7-15 0, S35-19 1, S30-15 4 (antialiasing; 601 before on S30-15).

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

**Fix round 5** (review fixes). `pipeline.choose_head(rules, lib, fit_of, overlap_of)` is the decision glue of `analyze_with_fit`, testable with stub fits. (1) The library candidates stay in `alternatives` on every photo, also when no library head wins (no warning then), so the head card's picker shows wherever the matcher found a mappable template. (2) Every head on the card is checked at analyze time as if it were drawn: its head fit, the FLAGGED_CONF cap for a library head drawn without color agreement (partition under 0.54), the gross head check (`_check_head`) and the heads-differ warning; the flagged confidences are stored on the parts, the head-dependent warnings and the head fit in `RosetteModel.head_states` (keyed by template_id or "rules"). `use_head` (server side; the head picker was removed from the UI in H9) swaps those warnings with the parts, the UI shows the picked head's fit, and each library head keeps its own pose (since the H9 review: cx, cy, r and rotation from its own search, changed only by a user edit of that head; switching never copies one head's placement onto another). Going back to the rules head after an override shows the rules head's own gross-failure warning and cap, plus the colors warning. (3) A model with nothing drawn in the head (no head fit) counts as a failed rules head for the override, and with no head layer the partition bar is skipped. A test checks that the library holds one template per product reference file (a stale cache fails it).

Head card picker. `templates.rank` lists the matcher's best templates (the decision rule's choice first) and `with_template` keeps the next `TEMPLATE_CANDIDATES` - 1 in `alternatives` (part ids `template_2`, `template_3`; params `rank` and `description`), next to the detected head parts. `templates.describe` mines the one-line description from the template pieces: a piece around the center is a ring (with a hole), a center or a disc (reach over 0.65 R), or an N-point star (4 or more lobes deeper than 1.3x); a piece away from it is a strip (4x longer than wide), a trim (under 0.6 of its hull), a loop (as wide at its outer end as half its angular width), a kite (tapered, at most 4 corners) or a point; a piece repeated with the same outline counts once (S35-19: "6 loops, 6 kites, 12 strips, 6 trims, ring, center"). The head card (Workbench and Guided share it) lists "Rule-based head (the detected parts)" and each candidate as "Library template <id> - match NN%, N points - <description>" with radio buttons; (historical, H7: the picker is no longer in the UI since H9) picking one drew it (`pipeline.use_head`; the render API takes the same choice as `head` and returns the switched model), unconfirms the head cards and the head check, and the picked head is flagged like any library head (a candidate's confidence is its own match score). The default stays the decision rule's choice. Candidates for the IDs above: S29-15 S57-17 / S69-19 / S58-17 (its own star template is not among them); S51-19 A1-15 / A13-15B / S63-17; S52-19 S31-15 / own / S69-19; S35-19 own / S30-15 / S31-15.

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

## H10: the head circle fitted to the silhouette

Started at eefda96; commits bed7e0f (the fit), 63879fb (head library rebuilt), 704b6d8 (sample) and the lobe gate on the head band (the commit after ff51cd3). Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h10-report.md`.

**The fit (`prepare._tails_start`, `_fit_head`, `_coverage`).** (1) Tails start: the silhouette's outer left and right edges are followed upward from 0.7 of the mask height (`TAILS_TRACK_FROM`); each row is predicted by lines fitted to 60 rows starting 40 rows below it (`TAILS_FIT_ROWS`, `TAILS_GAP_ROWS`; on the 76 photos' tails such a prediction misses by at most 2.2 px, median 1.0, and the edges slope at most 0.27 px per row). The first row whose left or right edge is more than 6 px off on 3 consecutive rows (`TAILS_EDGE_TOL`, `TAILS_RUN_ROWS`) ends the tails, outward (head or star arms wider than the tails) or inward (tails wider than the head). (2) Head silhouette: the mask above that row. (3) Its convex hull bridges a star's concavities, so the hull vertices are the tips; vertices within 3 % of the head height of the cut are dropped (`FIT_CUT_MARGIN`). A circle centered on the axis is voted: each vertex weighs half the angle to its neighbours (each gap capped at 15 degrees, `FIT_GAP_CAP_DEG`, so a long loop is one weak vertex), each candidate center row takes the radius whose 4 % band (`FIT_TOL`) collects the most angle, then least squares on the inliers. (4) The radius is the fit's radius. Coverage is the angle spanned by consecutive on-circle vertices (gaps counted up to 60 degrees, `FIT_COVER_GAP_DEG`). (5) The fitted circle replaces the 90-degree symmetry circle only when its radius is at least 1.25 times the symmetry radius (`FIT_TAKEOVER`) and its coverage is at least 0.5 (`FIT_MIN_COVERAGE`): the symmetry score has locked onto an inner structure. On the 76 photos that is S29-15, S9-19, S10-19 and S57-17 (ratios 1.31-1.37, coverage 0.67-0.93); the next ratios are 1.19 (s34-15, a broken fit through a 6-point star's arms, coverage 0.31) and 1.12. The tails top stays the bottom of the circle.

**Tried and dropped.** (a) The fit on every photo (a full replacement): (every photo) had mean error 12.1 against the reference circles (old 9.2) and broke on 6-point stars and bows (s34-15 +44, S80-22 +16, S36-15 +37); hence the takeover rule. (b) Badge refinement: a concentric-edge center finder (radial gradient coverage on circles around candidate centers on the axis, radius at least 25 units, found without the head circle) lands on the badge for many photos but picks small spurious circles elsewhere (mean error 14.7, S63-17 +104, S52-19 -91), and no coverage or peak-margin threshold separated right from wrong.

**Error against the reference circles (76 photos, units of the rosette height 800).** T1 is the H9 truth: `drawn_frame` of each reference file (the old symmetry rule on its render, frozen before this change; no file has `st_circel` centers). T2 is the center of the reference file's own centered `<circle>` elements (61 files have one; center only).

| | cy mean / worst (T1) | r mean / worst (T1) | cy mean (T2, 61) |
|---|---|---|---|
| before (symmetry) | 9.23 / 29.2 (A3-15) | 9.23 / 29.2 | 8.67 |
| after (bed7e0f) | 9.64 / 42.0 (S57-17) | 9.70 / 43.5 | 8.67 |
| after, S57-17 left out | 9.21 / 29.2 | 9.25 / 29.2 | - |

| photo (all 4 that moved) | cy err T1 | r err T1 | cy err T2 | tails top |
|---|---|---|---|---|
| S57-17 | +5.7 -> +42.0 | +5.7 -> +43.5 | -27.4 -> +8.9 | 232 -> 306 |
| S29-15 | -27.5 -> +9.5 | -27.5 -> +11.8 | - | 215 -> 291 |
| S10-19 | -13.5 -> +21.0 | -13.5 -> +22.0 | -11.9 -> +22.6 | 228 -> 298 |
| S9-19 | -15.6 -> +21.6 | -15.6 -> +21.0 | -14.7 -> +22.5 | 222 -> 296 |

Every other photo, S35-19 and A7-15 included, keeps its circle exactly. On the overlays the four new circles sit on the badge and around the whole head, the old ones high and small (S9-19 and S10-19: the reference files draw the head smaller relative to the tails, so both truths read the right photo circle as low). S57-17's T1 is wrong: the old rule puts the reference's center at the top of its rose (110.0), the file's own circle is at 143.1, and the new prepare run on that reference render gives 143.1. **Gate:** S29-15 within 15 units of its badge and radius (center (292.9, 216.8) px against the badge at about (290, 212); r 216.1 px against 190-210); S35-19 and A7-15 unchanged; no photo worse by more than 10 against T1 except S57-17 (+36.3, a T1 error by T2 and by eye), and the T1 mean goes 9.23 -> 9.64 only through S57-17 (9.27 -> 9.21 without it). Ruling: accepted; S57-17 is judged by the overlay and its file's own circle.

**Reference frames.** `frame_for` runs `prepare` on reference renders, so the change reaches the scorer's role frames and a rebuilt head library: on the 76 renders only S57-17's frame moves (to its own circle, 110.0 -> 143.1); S80-22's render had a ratio 1.31 fit that the coverage bar (0.29) refuses. The head library was rebuilt with this prepare (63879fb): only the S57-17 template changes (normalized on its file circle now). `tests/test_head_templates.py::test_library_cache_is_not_stale_for_a_file_whose_head_circle_prepare_fits` re-extracts S57-17 and compares it with the cache (it fails on the library built before H10).

**Knock-on (S29-15, S35-19, A7-15 and the four that moved; rebuilt library).**

| photo | circle canvas (cx, cy, r) | tails top canvas | rules head layer | own rank (full library) | top-1 full / LOO (score, pose dy, scale) | drawn known / LOO |
|---|---|---|---|---|---|---|
| S29-15 | (300, 132.4, 107.4) -> (300, 169.5, 146.8) | 240.3 -> 316.3 | none -> points, ring | 9 -> 6 | S58-17 0.714 dy +0.1 x1.25 -> S58-17 0.751 dy -0.1 x1.0 (both) | S58-17 / S58-17 (unchanged) |
| S57-17 | (300, 140.7, 115.7) -> (300, 177.0, 153.5) | 257.0 -> 330.7 | yes -> yes | 1 -> not ranked | S57-17 0.810 / S29-15 0.624 -> S52-19 0.604 / S52-19 0.604 | S57-17 / rules -> rules / rules |
| S9-19 | (300, 135.8, 110.8) -> (300, 172.9, 147.4) | 247.1 -> 320.8 | none -> points | not ranked -> 1 | S1-15B 0.707 dy +0.3 x1.25 -> S9-19 0.869 / S4-15 0.831 | S1-15B / S1-15B -> rules / rules |
| S10-19 | (300, 139.0, 114.0) -> (300, 173.4, 149.4) | 253.5 -> 322.9 | none -> points | 1 -> not ranked | S10-19 0.734 dy +0.3 x1.25 / A13-15B 0.664 -> S58-17 0.624 / S58-17 0.624 | S10-19 / A13-15B -> rules / rules |
| S35-19, A7-15 | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |

The tails of S29-15 now start under the star and the pose search no longer runs to its edge. S57-17 and S10-19 lost their own templates to the lobe gate, not to the score: at the new circle their own templates score 0.839 and 0.719, but the photo's outer profile had 11.8 and 6.7 lobes (the bunting loops and side flags inside the larger window) against 0 for the templates. Fixed by the lobe gate change below (S57-17 then draws its own head, S10-19 ranks it first).

**Retrieval (76 photos; own design in the matcher's full-library top-1 / top-3; head IoU of the LOO and full-library picks through `tools/head_templates.decision_row`, every template drawn).**

| | top-1 | top-3 | LOO head IoU | full head IoU |
|---|---|---|---|---|
| before H10 (eefda96) | 49 | 57 | 0.8363 | 0.8517 |
| bed7e0f, old library | 48 | 56 | 0.8340 | 0.8492 |
| bed7e0f, rebuilt library | 48 | 56 | 0.8340 | 0.8492 |

Before the lobe gate change all four numbers were worse than before H10 (top-1 -1, top-3 -1, LOO -0.0022, full -0.0025), and the rebuild changes none of them. The changes come from the four photos that moved: S9-19 gains its own design (top-1), S57-17 and S10-19 lose theirs (lobe gate), and the head IoU against the reference files drops on all four (S29-15 0.815 -> 0.740, S57-17 0.825 -> 0.797, S10-19 0.780 -> 0.746, S9-19 0.790 -> 0.756 LOO) because the heads are now drawn at the photo's head size while the reference files draw them smaller relative to the tails; the rules heads of S57-17 (0.613 -> 0.786) and S10-19 (0.726 -> 0.768) improve.

**Lobe gate on the head band (`templates.band_lobes`, `LOBE_BAND` (0.6, 1.05), `LOBE_PROMINENCE` 0.1, `LOBE_GATE` 3.4 -> 5.1).** The gate's lobe count (photo and template; the lobe count has no weight in the score) now reads the outer profile only out to 1.05 head radii, clipped up from 0.6, and counts a peak only when it stands at least 0.1 head radii above the valleys on both sides (prominence). Tried first, the band alone with the old peak rule (peak over the median + 0.04): an upper bound of 1.05 or more changes nothing on the 76 (48 / 56; the flags and loops start at the head outline, inside the band); 1.0 gives 51 / 59 but drops S2-15's own design and leaves S57-17 out; 0.95 flattens every profile (all counts 0). Grid over the prominence (0.04-0.15) and the gate (3.4, 5.1, 6.8, none), top-1 / top-3 of own designs (full library, 76 photos):

| prominence, gate | top-1 | top-3 | own designs lost (top-1 / top-3) | S57-17 own |
|---|---|---|---|---|
| old count, 3.4 (H10 before this change) | 48 | 56 | - | no |
| 0.1, 3.4 | 53 | 62 | - / S58-17 | no |
| **0.1, 5.1 (kept)** | **53** | **64** | none | yes |
| 0.08, 5.1 | 54 | 63 | - / S58-17 | yes |
| 0.12, 5.1 | 52 | 64 | S13-15 / - | yes |
| 0.1, 6.8 | 51 | 64 | S13-15 / - | yes |
| 0.1, no gate | 51 | 64 | S13-15 / - | yes |

Gate counts: S57-17 photo 11.78 -> 5.05 (own template 0 -> 10.09: three steps, hence 5.1), S10-19 6.73 -> 5.05 (own 0 -> 5.05), S29-15 5.05 -> 5.05 (own 3.36 -> 5.05), S9-19 1.68 -> 5.05 (own 0 -> 3.36); largest other changes S35-19 6.73 -> 1.68, S69-19 8.41 -> 3.36, S52-19 11.78 -> 6.73, A13-15 5.05 -> 0, A1-15 5.05 -> 1.68, S27-15 5.05 -> 1.68. A 6-point star shows three tips in the window (5.05 < 5.1), so the count alone would let a disc template through; the round-template guard below closes that. Retrieval (same measure as above):

| | top-1 | top-3 | LOO head IoU | full head IoU |
|---|---|---|---|---|
| before H10 (eefda96) | 49 | 57 | 0.8363 | 0.8517 |
| H10, rebuilt library, old gate | 48 | 56 | 0.8340 | 0.8492 |
| H10 + lobe gate on the head band | 53 | 64 | 0.8400 | 0.8587 |

No own design that ranked first or in the top 3 before H10 is lost; gained top-1: A1-15, S1-15, S10-19, S27-15, S57-17. Picks: S57-17 draws its own head (known; LOO S56-17B); S10-19 ranks its own first (drawn: rules); S9-19 own first (drawn: rules); S29-15 now matches S10-19 (0.754, drawn in both columns; its own design ranks 33rd); S35-19 and A7-15 unchanged.

**Round-template guard (`templates._round`, `ROUND_GUARD_LOBES` 4.2).** A round template (no head-band lobes and a whole-outline symmetry under 3) is refused for a photo whose head-band count reaches 4.2, independent of the count difference. Photo counts on the 76 (head band): photos of round heads count 0 (most), 1.68 (A1-15, A10-15B, S27-15, S4-15) or 3.36 (S11a-15), with A2-15 (5.05) the one exception; a 6-point star shows 5.05. The bar sits between 3.36 and 5.05 (at 3.36 S11a-15 loses its own design). The files' points counts do not tell round heads (A12-15's disc file counts 5) and symmetry alone does not either (S57-17's own template: symmetry 0, ten band lobes), so the rule is shape-only. Retrieval, measured on the 31 photos the guard can touch (count >= 4.2), the rest unchanged:

| | top-1 | top-3 | LOO head IoU | full head IoU |
|---|---|---|---|---|
| lobe gate (45161f5) | 53 | 64 | 0.8400 | 0.8587 |
| + round-template guard | 54 | 64 | 0.8412 | 0.8572 |

No own design lost. Changes: S51-19 gains its own design (full pick A1-15 -> S51-19, head IoU 0.856 -> 0.773 against its reference; LOO A1-15 -> S63-17, 0.856 -> 0.825); A2-15 (a disc photo counting 5.05) loses A1-15 (0.956 -> 0.923, S65-18); LOO picks S10-19 A13-15B -> S51-19 (0.743 -> 0.847) and S56-17 A13-15 -> S4-15 (0.801 -> 0.856). `tools/score.py`: no row changes.

**Uncertain head circle (`pipeline.circle_warnings`, `Prepared.fit_ratio`, `Prepared.fit_coverage`).** When the fit and the symmetry circle disagree without a takeover (1.1 <= fit/symmetry radius < 1.25, `CIRCLE_RATIO_MIN`) or a takeover was decided by a coverage within 0.1 of the 0.5 bar (`CIRCLE_COVERAGE_MARGIN`), analyze adds "The head circle is uncertain - check the head size and position against the photo." (a warning, so the result is flagged). The coverage clause applies only where coverage decided (ratio >= 1.25): read on every photo it would warn 13 more photos whose fit agrees with the symmetry circle (ratio 0.91-1.08, S35-19 among them). On the 76 photos the warning fires on S51-19 (ratio 1.115, coverage 0.74) and s34-15 (1.191, 0.31); no takeover photo is near the coverage bar (0.67-0.93).

**Known limit: loop envelope.** The fit is the circle around the outermost hull vertices, so long loops or star arms that reach past the head proper inflate it: S29-15's radius is 216 px against about 190-210 px for its badge-centered head (its glitter loops), S57-17's encloses the bunting loops and side flags.

**Scoreboard (34 IDs, `tools/score.py`, after H10 with the rebuilt library, the lobe gate and the guard).** New image (headline): PASS 23/34, color 0.855, shape 0.919, head 0.842 (before H10 0.854 / 0.920 / 0.844). Known design: PASS 23/34, color 0.859, shape 0.921, head 0.847 (before H10 0.859 / 0.921 / 0.847). The lobe gate changes S29-15 (agree 0.221 -> 0.321, headS 0.740 -> 0.851: S10-19's head), S30-15 (0.907 -> 0.972, headS 0.934 -> 0.879) and S2-15B (0.687 -> 0.613, headS 0.853 -> 0.820). Before the gate change: new image PASS 23/34, color 0.852, shape 0.919, head 0.842; known 23/34, 0.856, 0.920, 0.844; against before H10 only S29-15 (agree 0.265 -> 0.221, headS 0.815 -> 0.740) and S10-19 (0.950 -> 0.942, headS 0.780 -> 0.768; known 0.987 -> 0.942) change; the rebuilt library changes no row. Ruling: the head drop is a reference-proportion artifact (the S29-15 head is now centered on the photo's head and sized to it).

## H11: star layers under loops, pleated edges, convex stars; layer agreement in the head decision

Started at 43589ba; commits b42f66a (detector), 8823344 and 0714d88 (layer agreement), 1d51263 (point-count warning). Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h11-report.md`. Seen on S29-15 (a pink 6-point star over a dark 6-point star over glitter cross loops), A10-15 (three concentric pleated rings read as 24 bows and 24 kites), S53-19 (two 4-point "stars" drawn as a square and a turned square) and s44-15 (two layered 6-point stars read as 4-petal layers, unflagged).

**Detector rules (`rosette/vision/parts_head.py`).**
- Start at the cover (`_cover_start`, `COVER_CUT_SHARE` 0.6, `COVER_PROBE_UNITS` 3, `COVER_MAX_UNITS` 80). A points layer that is not a star, whose lobe axes just inside its r_inner show (on >= 0.6 of the visible lobes) the label of a part drawn above it (a ring, the center, or a points layer of higher z), starts at the outer edge of the ring or center that cover reaches down to; the width stays. The H4 rule (loops under a ring, kites within a pleat depth of a ring) is unchanged. Stars keep their valleys: moving them inward too made stars skinny (mean rules silhouette over the 76 photos 0.875 -> 0.851, S9-19 0.937 -> 0.658), and a version without the "drawn above" and "reaches a ring" conditions ran loops to the middle (A14-17, S2-15, S56-17 r_inner 1.5); both dropped.
- A star outline under a star (`_star_from_outline`, `STAR_MIN_ORDER` 3, `STAR_INSIDE_TOL` 0.05). An unknown whose outline (its outermost radius per angle) has an n-fold turned self-correlation >= SYM_T, where n is a points layer's count or half of it and the outline's tips stay within 5 % of that layer's r_outer, becomes a points layer of n lobes, valleys at the outline's 10th percentile and tips at its 95th, flagged (confidence at most 0.5). "n divides the count" was tried first and turned S29-15's badge print into a 4-point kite layer.
- Convex stars (`STAR_CONVEX` 0.95, `STAR_HIDDEN_DEPTH` 0.85). A star whose valleys lie at or beyond 0.95 of the convex limit (r_outer cos(180/n)) would be drawn as a polygon (a square for 4 points). Its valleys are hidden under the layers inside its zone: they move to the zone's start, or to 0.85 of the convex limit when that still lies inside it; a star that cannot get notches either way becomes a flagged annulus (fires on none of the 76). Drawing the annulus directly cost S18-15, S3-15 and S18-15B 0.3-0.4 silhouette.
- Pleated edges (`PLEAT_MIN_COUNT` 16, `PLEAT_FILL` 0.8, `ALT_RADIUS_UNITS` 3, `ALT_OFFSET` 0.25). A points layer of 16 or more lobes that fill >= 0.8 of their angular spacing where widest, in a zone closed by bands on both sides, is a ring with a pinked outer edge (ring params `pinked` 1 and `pleats` n; `geometry.rings.pinked_ring`; the re-centering uses the pleat count). Two such layers of one count and radii whose lobes interleave are a two-tone pleated edge and stay points (S6-15, S69-19, 17 + 17: as rings one color hid the other, S69-19 agreement 0.80 -> 0.70). Rings now stack by their outer edge (a pleated annulus can enclose a narrower ring: A10-15).

Rules head fit over the 76 photos (head_fit of the detected model): mean silhouette 0.8750 -> 0.8848, mean agreement 0.7192 -> 0.7279; 17 photos change, 59 are identical.

| rule | layers (photos) | r_inner or result | head fit change (agreement / silhouette) |
|---|---|---|---|
| start at the cover | 8 (8) | S29-15 loops 74.5 -> 52.5, S40-15 petals 70.5 -> 42.5, S48-17 petals 63.5 -> 26.5, S58-17 loops 67.5 -> 58.5, S59-17 petals 81.5 -> 46.5, S61-19 petals 94.5 -> 42.5, S69-19 kites 83.5 -> 35.5, s34-15 kites 76.5 -> 41.5 | S40-15 +0.005 / +0.014, S48-17 -0.026 / -0.020, S58-17 -0.016 / -0.003, S59-17 +0.023 / +0.033, S61-19 +0.040 / +0.045, S69-19 -0.012 / -0.004, s34-15 -0.015 / -0.013 |
| star outline under a star | 2 (2) | S29-15 dark 6-point star 81.5-129.5; S15-15 8-point star 83.5-142.5 (an unknown disc before) | S15-15 +0.087 / +0.285 |
| convex star valleys | 10 (7) | S18-15 x2, S18-15B x2, S3-15, S32-15, S47-15, S53-19 x2, S58-17 (S18-15 110.5 / 116.5 -> 91.7, S53-19 99.5 / 95.5 -> 79.6 / 79.0) | S18-15 +0.106 / +0.042, S3-15 +0.069 / +0.022, S18-15B +0.076 / +0.050 |
| pleats -> pinked rings | 7 (3) | A10-15 (24 bows, 24 kites), A9-15B (22, 19), S53-19 (19, 18, 24) | A10-15 +0.124 / +0.006, S53-19 +0.145 / +0.066 |

**Layer agreement in the decision (`pipeline.prefer_template`, `TEMPLATE_AGREE_GAIN` 0.10, `TEMPLATE_AGREE_SIL_LOSS` 0.02).** Besides the outline gain (silhouette +0.02), a library head is preferred when its head_fit agreement beats the rule-based head's by 0.10 and its silhouette is at most 0.02 worse; the color floor and the override rules apply as before. Measured offline on the 76 photos after b42f66a (rule-based head and the matcher's top 3 plus best-IoU templates each drawn alone, the decision replayed; LOO = own design left out):

| agreement margin | LOO head IoU | full head IoU | LOO layer | full layer | color >= 0.9 LOO / full | library heads LOO / full | color losers |
|---|---|---|---|---|---|---|---|
| none | 0.8513 | 0.8588 | 0.2230 | 0.2425 | 47 / 49 | 16 / 20 | - |
| 0.05 | 0.8513 | 0.8581 | 0.2253 | 0.2523 | 48 / 51 | 17 / 25 | none |
| 0.08 | 0.8513 | 0.8584 | 0.2253 | 0.2508 | 48 / 51 | 17 / 23 | none |
| **0.10 (kept)** | 0.8513 | 0.8587 | 0.2253 | 0.2484 | 48 / 51 | 17 / 22 | none |
| 0.12-0.20 | 0.8513 | 0.8587 | 0.2230 | 0.2484 | 47 / 51 | 16 / 22 | none |

0.10 switches S3-15 (LOO S18-15, known its own head; color 0.844 -> 0.946 / 1.000) and S32-15 (known, 0.858 -> 1.000); 0.08 and below also draw S6-15's own head (head IoU -0.018) and, at 0.05, S47-15 and S55-15. It does not reach the two IDs that prompted it. S53-19's rule-based head after the detector rules fits its photo better than its own template on both measures (rules 0.791 / 0.931, own 0.707 / 0.877; against the reference headS 0.855 vs 0.871, color 0.501 vs 0.588). Ruling: accepted on its rule-based head, the photo is the truth. s44-15's wrong rule-based head has the higher agreement too (0.774 / 0.950 vs own 0.692 / 0.903) and is not worse against its reference (headS 0.654 vs 0.648); its own file has 4 points.

**Point-count warning (`pipeline.points_count_warnings`, `POINTS_COUNT_MIN` 3).** When the rule-based head is drawn, the closest library design (the first candidate) agrees in colors (partition >= 0.54, anchored to the tails), has 3 or more points, and no rule-based points layer has its count, the rule-based head's state gets "The head's point count differs from the closest library design - check the number of points against the photo." (a warning, so the result is flagged). On the 76 photos it fires with the full library on S10-19, S18-15, S53-19, S69-19 and S9-19 (all flagged already) and with the own design left out on S28-15 (an unflagged success whose head is wrong: a gain), S45-15, S53-19, S63-17, S80-22 (a false alarm: its neighbour S10-19's file has 4 points; accepted by ruling), S9-19 and s44-15 (flagged by it in the new-image case only; with the full library its own file's 4 points match).

**Shape-first pick (tried, left out).** A gated, mappable template whose silhouette IoU beats the best-scoring template's by SHAPE_FIRST_IOU_GAIN was ranked first and drawn flagged below the color floor ("Library head shape chosen over its colors"). The brief's rule kept it only if known top-1 rises without a LOO head IoU loss:

| SHAPE_FIRST_IOU_GAIN | known top-1 (without: 54) | LOO head IoU (0.8513) | full head IoU (0.8588) | color losers |
|---|---|---|---|---|
| 0.02 | 51 | 0.8513 | 0.8594 | LOO S36-15 1.000 -> 0.817, S51-19 0.905 -> 0.795; known S65-18 1.000 -> 0.746, S69-19 0.993 -> 0.944 |
| 0.03 | 53 | 0.8513 | 0.8596 | known S65-18, S69-19 |
| 0.04 | 53 | 0.8513 | 0.8597 | known S69-19 |
| 0.05 | 54 | 0.8513 | 0.8596 | none |
| 0.07 | 54 | 0.8513 | 0.8588 | none |

Top-1 never rises; the code was removed. It would not have drawn S29-15's own star either: that template is shape-first from 0.03 (IoU 0.930 vs S10-19's 0.899), but its head fit (0.188 / 0.867) loses to the new rule-based head (0.539 / 0.885).

**Before and after.**

| photo | rule-based head before (head fit) | after | drawn before -> after (both columns) |
|---|---|---|---|
| S29-15 | 6 glitter loops 74.5-151.5, 12 thin pink tips 106-150, a dark unknown disc r 70.7, ring, center (0.515 / 0.674) | 6 loops 52.5-151.5, dark 6-point star 81.5-129.5 (flagged), 12 pink petals 106-150, ring, center (0.539 / 0.885) | S10-19 -> rule-based head |
| A10-15 | disc, 24 bows, 24 kites, ring, center (0.733 / 0.940) | disc, pinked ring 71.5-129.5 (24 pleats) under a ring 98.5-122.5, ring, center (0.857 / 0.945) | rule-based head |
| S53-19 | 19 and 18 petals, 24 kites, two 4-point "stars" drawn as a square and a turned square, center (0.645 / 0.865) | three pinked rings, two 4-point stars with valleys at 79 (an 8-point outline in two colors), center (0.791 / 0.931); point-count warning | rule-based head (flagged) |
| s44-15 | pinked disc, two 4-petal layers, ring, center (0.774 / 0.950) | unchanged | rule-based head; flagged by the point-count warning in the new-image case only |

Scoreboard (`tools/score.py`, 34 IDs): new image PASS 23/34, color 0.855 -> 0.852, shape 0.919, head 0.842 (the gate: not below 0.842); known design PASS 23/34, color 0.859 -> 0.856, shape 0.921 -> 0.920, head 0.847. Changed rows (agree / shape / headS, both columns alike): S29-15 0.321 / 0.935 / 0.851 -> 0.237 / 0.908 / 0.765, S53-19 0.513 / 0.763 / 0.814 -> 0.501 / 0.773 / 0.855, S69-19 shape 0.920 -> 0.933, headS 0.759 -> 0.801, s34-15 0.956 / 0.928 / 0.844 -> 0.951 / 0.926 / 0.838. S28-15 is now flagged (the point-count warning); S53-19 adds a reversed pair (star > ring). The agreement margin changes no scoreboard row (S3-15 and S32-15 are not among the 34).

S29-15 against its reference: agree 0.321 -> 0.237, headS 0.851 -> 0.765 (the reference draws the loops shorter and the dark star pieces at the rim; the drawn head follows the photo). The H5 stacking rule (loops over stars, 89 % of the reference pairs) is kept, so S29-15's loops lie on its dark star where the photo has the stars on top: a known exception. The color grouping reads most of its pink star as the dark color, so only its tips are pink.

## H12: points layers split by radius, stacking from the photo, nested stars, loop width

Started at 03e7fb7; commits 6d3bbd0 (split), dd46e8e (stacking), d079c62 (loop width), fbcf1a8 (a split star's tips), a37827d (nested points layers), fcd50b6 (review fix round 1). Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h12-report.md`. Seen on S29-15: a pink 6-point star and the pink edge strips of six glitter loops were read as one 12-count layer at the rim (the star body lost), and the H5 order drew the loops over the stars where the photo has the stars on top.

**Split by radius (`parts_head.band_orders`, `radial_split`, `_split_by_radius`, `_split_and_explain`; `SPLIT_BANDS` 8, `SPLIT_FLOOR` 0.2, `SPLIT_MARGIN` 0.1, `SPLIT_MIN_INNER` 2).** Each points layer's radial extent (its label's rows in its zone) is cut into 8 bands. A band's count is the largest n from 3 to 24 whose turned self-correlation lies within 0.1 of the band's best (divisors of the true count score as high); a band whose best is under 0.2 is undecided. n = 2 is left out: on S29-15 the rim band read 2 (0.71, the half turn of a mirror-symmetric head) over 6 (0.57) and 12 (0.56). The layer splits when the outermost decided bands agree on a count c2, no decided band inside them holds c2 (without that guard S30-15 [6,3,3,6,...] and S63-17 [8,4,4,4,8] split), and at least 2 of them hold a divisor of c2; c1 is the largest such divisor (fix round 1: the bands' majority was noisy). The inner part is a layer of c1 lobes read off its rows, its lobe widths measured as the run of the label through each lobe axis (the mean coverage counted the strips and read S29-15's star as a loop); a star's tips (a star over its body band too) are the last row its label holds on most lobe axes (fbcf1a8: the rows end where the outer count starts, beyond the tips; S29-15's pink star 138.5 -> 109.5). For the H11 star-outline rule a split counts as the layer it came from. The outer part is a flagged layer of c2 lobes handed to the H4 trims rule. **The split stands only when the outer part becomes another layer's trims (edge strips); otherwise the layer is put back whole** (ruling: accepted): the outline and trims steps then run again on the layers as they were, so an undone split leaves no trace (fix round 1; before, trims hosted by the inner part were dropped with the layer they replaced). Mined on the 76 photos: 118 points layers on 56 photos are band-read, 22 show more than one count, 3 meet the rule: S29-15 pink [0,0,6,6,6,6,6,12], S56-17B glitter [6 x7, 12] and S57-17 red [18,6,6,6,6,12,12,12]. Only S29-15's stands: on S56-17B the 12 rim lobes are the loops' own notched ends, on S57-17 the red borders of striped loops (as a layer of their own they were drawn as 12 red tabs), and neither explains another layer's trims.

**Stacking from the photo (`parts_head.stack_votes`, `photo_stacking`; `STACK_SHARE` 0.65, `STACK_MIN_UNITS2` 200).** For every pair of drawn points layers of different colors (trims left out) the drawn overlap, less the rings, the center, unknown discs, other layers of either color and the tails sector, is sampled 1 canvas unit apart on the photo's labels. The color holding at least 0.65 of the two colors' samples is on top when they number at least 200 (one sample per canvas unit^2); otherwise the H5 order (`points_z`) stands. Decided pairs, strongest first, are kept unless they close a cycle; the layers are ordered by them with the H5 rank breaking ties, trims stay next to their host and the layers keep their set of z values (20-29), so the shop paint order and the clipping are unchanged. The H11 cover start (`_cover_start`, which reads z) runs after this order (fix round 1; `_finish_points`). Mined on the 76 photos: 50 overlapping pairs on 38 photos; 42 decided, 20 of them against the H5 prior; 7 near 50/50; 1 under the size bar; 18 photos get another order. Rule-based head agreement over the 76 photos: 0.7284 (H5 order) -> 0.7498, silhouette 0.885 both; every one of the 18 gains or holds (S3-15 0.52 -> 0.89, S32-15 0.53 -> 0.87, S33-15 0.48 -> 0.71, S18-15 0.55 -> 0.73, S15-15B 0.59 -> 0.75; the smallest S80-22 +0.0005). Shares 0.58, 0.60 and 0.65 give the same head fits on the borderline photos; the smallest decided overlap is 296 samples.

**Nested points layers (`parts_head.nested_pairs`; `NEST_AXIS_DEG` 6, `NEST_UNITS` 3).** Where the photo does not decide a pair, of two concentric points layers of one count and different colors on the same axes whose tips are nested, the one with the shorter tips lies on top (a rosette is built from layers each sitting on a larger one), after every photo decision; only z changes. (A first version also moved the valleys of a star behind under the front star, scaled to its tips: on S29-15 that put the dark star's valleys at 70.4, outside the pink star's at 59.5 on the same angles, so they showed and cost 0.042 agreement and 0.018 silhouette; dropped in fix round 1.) Mined on the 76 photos: 7 such pairs; the photo decides 6 and agrees with the rule on 5 (S47-15, S58-17, S59-17, S61-19, S69-19); A15-17 disagrees (orange kites over the black star, 0.75) and the photo wins; the rule decides S29-15 only. Tried first and dropped: a border test (the share of a layer's visible samples within 4 units of the other layer's outline, bar 0.6). Over the same 7 pairs it read S29-15's dark star 0.215 (its label also covers the pink satin body and the pleats: 0.68 only at 12 units) and fired only on A15-17 (0.689, the wrong way: the black star there is in front), so it could not tell a border.

**Loop width (`parts_head._block_width`; `BLOCK_MID` 0.6, `BLOCK_MIN_SHARE` 0.5).** A loop block's width is the widest per-row median (over the visible lobes) of the run of its label through each lobe axis, over the middle 0.6 of its radial extent (the base under the cover and the folded end left out); when that reads under 0.5 of the mean-coverage width (the axis falls on another color: S45-15 read 0.052) the mean-coverage width stays. The brief's premise for S29-15 did not hold: its loops were drawn 60.2 units wide against 59.5-62.5 visible (the 0.8 width ratio was the edge-strip copy's, over its own longer extent: the same 60 units). Widths in canvas units:

| photo | drawn before | drawn after | visible (median over 0.3-0.7 of the extent) |
|---|---|---|---|
| S29-15 | 60.2 | 62.8 | 59.5 |
| S35-19 | 64.0 | 66.7 | 65.8 |
| S30-15 | 66.9 | 72.2 | 71.6 |
| S56-17 | 89.5 | 90.0 | 83.3 (widening to 87.7 outward) |
| S1-15B | 111.6 / 113.4 | 109.9 / 112.2 | 100.8 / 104.2 |

Rule-based head fits: S30-15 0.537 -> 0.567, S28-15 0.844 -> 0.863, S57-17 +0.011, S35-19 +0.009; worst S2-15 -0.002 (silhouette -0.011) and S56-17B -0.008. A first version (the median at mid radius over the middle 0.2) cost S56-17 0.04 and S45-15 0.17 and was replaced before the commit.

**S29-15 after H12.** Before: glitter loops over a dark 6-point star (81.5-129.5) and 12 thin pink petals (106-150), ring, center. After: a pink 6-point star (59.5-109.5) on top of the dark 6-point star (81.5-129.5, showing between the pink arms and beyond the pink tips), both over the glitter loops (62.8 wide) with pink edge strips, then the dark ring and center. Rule-based head agreement 0.539 -> 0.636, silhouette 0.885 -> 0.876; scoreboard agree 0.237 -> 0.301. **Not resolved:** the photo itself cannot order the two stars (their overlap's labels split 49/51, pink 6554 vs dark 6346, because the pink satin body reads dark in the color groups), and the pink pleated ring reads as a dark ring 40.5-52.5. Both are the next color-grouping item, not this round.

**Gates and rulings.** Scoreboard (34 IDs): new image PASS 23/34 -> 21/34, color 0.852 (unchanged), shape 0.919, head 0.842 (held), layer IoU mean 0.2112 -> 0.2171 (held), shape_ok 27 -> 26 (S10-19 shape 0.900 rounds under the bar); known design PASS 23 -> 21, color 0.856 (unchanged), shape 0.920, head 0.847. Changed rows: S5-15 0.910 -> 0.868 (loop width: its loops measure 88.7 -> 83.4 units, the reference draws them wider; still flagged), S33-15 0.905 -> 0.886 (the photo puts the blue 6-point star over the purple glitter star, 0.89 of the overlap; the reference has it the other way; its head fit rises 0.48 -> 0.71, no longer a gross failure, so it is unflagged and counts as silently wrong), A15-17 0.968 -> 0.949 (orange kites over the black star, 0.75; still a success), S29-15 0.237 -> 0.301, S28-15 0.998 -> 1.000, S80-22 0.989 -> 0.992, s34-15 0.951 -> 0.956, S41-15 0.925 -> 0.924. Reversed kind pairs vs the designer 4 -> 6 (S29-15 and S46-15 add loop>star: the photos have the stars on top). Rulings: S5-15 accepted (the photo width is right); S33-15 accepted (the photo decides the stacking; a reference-differs case, see Known limitations); the split kept only when its outer part becomes trims, accepted.

## H13: the head's star base, rim blobs, rings by their color's band

Started at 105c9a8; commits 69de195 (rim blobs), f62a61a (star base), 6856829 (rings by band). Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h13-report.md`. Seen on S53-19: a dark-blue 8-point star is the base of the head (glitter triangles on its four axis arms, then a white pleated ring, a glitter ring, a blue ring and a white badge); the rule-based head had no blue in it at all, and its three pleated rings all ended at 84.5 units, so only the top one showed.

**Rim blobs are not motifs (`colors._rim_blobs`; `RIM_TOUCH` 0.3, `RIM_THICK_UNITS` 3).** The blue arms never reached the detector: the white ring cuts them off from the rest of the blue, and the motif absorption (blobs under `MOTIF_MAX_UNITS2`) gave the plain diagonal arms (824 and 891 units^2) to the white ring and the axis arms' blue edges (277-482 units^2) to the glitter, so the head's "white 4-point star" was the blue base painted white. A blob above the tails whose border touches the background on >= 0.3 of its pixels and that is at least 3 units thick is now kept: a ribbon piece cut off by the layers over it, while print and sparkles lie inside their ribbon and thinner rim blobs are edge blends. Over the 76 photos 46 blobs on 13 photos pass (A2-15, A3-15, A9-15B, S17-15, S31-15, S35-19, S37-15, S38-15, S53-19, S56-17B, S58-17, S69-19, S8-15); on the photos they are ribbon (S31-15 lilac backings, S38-15 the outer ruffle, S58-17 cream edging, S17-15 tips and edges, S69-19 the navy border of the white star) or, small, shading (S56-17B, A2-15). The bar cannot be raised past S69-19's navy border (3.9-4.3 units thick) without losing S53-19's axis-arm edges (3.5-4.8).

**The star base (`parts_head._base_star`, `_head_outline`, `_outline_symmetry`; `BASE_TIP_UNITS` 6, `BASE_TIP_SHARE` 0.75, `BASE_FLANK_UNITS` 2, `BASE_FLANK_TIP` 0.95, `BASE_REACH` 0.1, `BASE_Z` 15, `BASE_FIT_UNITS` 6).** Only without a head disc (an outline that is not round, `HEAD_ROUND`: the upper outline's 5-95 % radius spread under 0.12 head radii keeps the disc, so star tips standing less than about 12 % R past a round disc get no base), after the H12 stacking: the silhouette above the tails is followed out from 0.6 head radii; its order n (>= 3) and rotation come from the outline's turned self-correlation (the H11 rule; `_outline_order` now uses `_outline_symmetry`). The label in the 6 units under each visible tip must be one group on >= 0.75 of them. No base when the tips are drawn already by a points layer reaching within 0.1 of them: one of that color (a trim too: S80-22's backing) or one of another color on every tip axis (S29-15's loops, whose pink edge strips hold the tip label). A layer of that color whose count is a proper divisor of n, on tip axes and not a trim, is the base read in part and gives way (S53-19's blue 4-count: only the plain arms are wholly blue). The base is a star of count n, tips at the median tip radius, valleys of the straight-sided star that fits the outline's flanks outside the largest ring or center (kept notched), z 15 (the head disc is 10, points 20-29), confidence min(outline score, tip share, 1 - median flank-fit error / 6 units) so an outline with rounded lobes is flagged (fix round 1: S53-19 fits to 1.7 units, 0.72; a scalloped synthetic outline to 3.4, 0.43), part id `base`. A first version skipped trims and other colors and added bases under S29-15's loops and S80-22's backing.

Mined on the 76 photos: 37 round heads (head disc); of the 39 others 19 have no outline order >= 3 and 20 are lobed: S53-19 gains the base (blue, 8, r 89.0-136.1); 12 have their tips drawn already (S15-15, S15-15B, S29-15, S33-15, S56-17, S56-17B, S57-17, S59-17, S6-15, S61-19, S9-19, S80-22); 7 have tips of two or three colors (S1-15B, S18-15, S18-15B, S3-15, S32-15: two 4-point stars, S48-17, S69-19). The reference files draw a points layer at > 0.9 R on every lobed head.

**Rings by their color's band (`parts_head._rings_by_dominance`; `RING_OVERLAP_UNITS` 8, `RING_MAJORITY` 0.5).** On S53-19 the three pleated bands were dropped as bridges (their edges ramp), each color became a pleated layer over the whole zone, and the rings came out white 35.5-84.5, glitter 54.5-84.5, blue 35.5-62.5. Rings of different colors that overlap by more than a pleat's teeth (8 units) now take the run of radii where their color holds >= 0.5 of the circle (the run overlapping them most, the center's rows left out), extended over unheld rows halfway to the next run; a ring reaches under the pinked teeth of the ring inside it (`TOOTH`). Mined: 3 photos have such rings, A10-15 (pleated white 72.5-129.5 under blue 96.5-129.5 -> white 76.5-100.5, blue 94.5-126.5: the same picture), A9-15B (white 52.5-87.5, glitter 75.5-114.5 -> 54.5-82.5, 76.5-112.5) and S53-19 (-> blue 36.5-56.5, glitter 50.5-81.5, white 75.5-98.5; the photo's bands 0.26-0.43, 0.41-0.60, 0.56-0.75 R are 34-57, 54-79, 74-99 units).

**S53-19 before and after.** Before: a white 4-point star at 45 deg and a glitter 4-point star at 0 deg made the 8-point outline, no blue in the head; only the top one of the three overlapping rings showed (rule-based head fit 0.791 / 0.931). After: the blue 8-point base under everything, the glitter 4-point star on its axis arms, the white pleated ring outermost, then the glitter and the blue rings and the badge (0.811 / 0.922; the agreement is read against the corrected labels). Scoreboard agree 0.501 -> 0.579, shape 0.773 -> 0.768, headS 0.855 -> 0.843. Left: the glitter triangles are drawn as a 4-point star, too wide near their base (29 deg against 14-24 on the photo at 95 units; close from mid length out), the star kind's geometry; the reference's 8 glitter points are not on the photo (the diagonal arms are plain blue). A white 2-point "star" from the white pleat teeth (r 0-101.5, mostly under the rings) changes no score.

**Gates and rulings.** Scoreboard (34 IDs), new image: PASS 21/34 (held), color 0.852 -> 0.855, shape 0.919, head 0.842 -> 0.844, tail 0.955, silently wrong 2 (unchanged); known design PASS 21/34, color 0.856 -> 0.859, head 0.847 -> 0.848; uniform-scale successes 25 -> 26. Changed rows: S31-15 0.929 -> 0.986 (headS 0.844 -> 0.887), S58-17 0.972 -> 1.000 (0.868 -> 0.880), S69-19 0.993 -> 0.936 (its navy border, kept now, joins the navy front star's layer, which then reaches the tips over the white star; the reference has a navy backing star under it; still a success), S53-19 as above. With the base and before the rings rule S53-19's headS read 0.840: the base's tips are the photo's (median 136.1 units, the side arms 137-139) where the reference draws every tip at 1.0 R (132.4); capped there it read 0.856. Ruling: accepted as a reference-proportion effect, the base keeps the photo's tip radius. Rule-based head fits over the 76 photos: agreement 0.7508 -> 0.7483, silhouette 0.8854 -> 0.8860 (13 photos change; agreement is read against labels that changed on them).

**Fix round 1.** (1) Rings: a gap of unheld rows between two runs is split at one boundary row shared by both rings (the inner one takes an odd row; before, an odd gap left a one-row seam and the outer ring never reached under the inner one's teeth), a gap open on one side (no center band, or outside the outermost run) is not taken, a ring whose color holds no run under it stays as measured (before, it could jump to another run of its color: a white ring 35-84 moved onto 21-33), and rings left identical in color and radii are kept once. (2) Borders are backings (`parts_trims.split_borders`, `_border_bands`; `BORDER_ROW_SHARE` 0.5, `BORDER_BODY_UNITS` 10, `BORDER_UNITS` 8, `BORDER_MIN_SIDES` 0.5): on S69-19 the navy border of the white star, kept by the rim-blob rule, had joined the navy front star's layer, which then reached the tips and was drawn over the white star. A points layer of h whose outer rows are a band of h at most 8 units wide right outside the sides of another layer's points (same count, axes within `EDGETIP_AXIS_DEG`, not loop blocks; read from each axis that shows the other layer, on >= 0.5 of the visible sides) ends where that border starts (rows with >= 0.5 of h's pixels in the band, gaps of up to 8 units around the other layer's tips bridged, rows where the axis does not show the other layer skipped), keeping at least 10 units of body; the border becomes a backing trim of the other layer (its points, reaching h's outer end, wider by the band on each side), drawn behind it by the H5 order. S69-19: navy front star 71.5-138.5 -> 71.5-114.5, navy backing diamond 83.5-138.5 behind the white star (rule-based head fit 0.615 / 0.869 -> 0.757 / 0.907; before H13 0.793 / 0.889), scoreboard agree 0.936 -> 0.998 (0.993 before H13), shape 0.933 -> 0.924, headS 0.802 -> 0.775 (the reference's white star sits differently). On the 13 rim-blob photos and A10-15 it fires on S69-19 only. (3) A points layer of fewer than 3 lobes is never drawn (`tidy.drop_two_point_layers`, `MIN_POINTS` 3; a warning names it): S53-19's 2-point "star" from the white pleat teeth is gone (its scoreboard row is unchanged). (4) The base's confidence includes the flank fit (`BASE_FIT_UNITS`, above). (5) Counts in the mining paragraph corrected (12 drawn already, 7 mixed). Scoreboard after the fix round, new image: PASS 21/34, color 0.855 -> 0.857, shape 0.919, head 0.844 -> 0.843 (gate 0.842 held), silently wrong 2; known design PASS 21/34, color 0.861, shape 0.920, head 0.848; only S69-19's row changes. Samples byte-identical, both validate to `[]`.

## ML-1: a learned head-layer classifier (kind and count), pleated rings pinked

Started at b48abe0. Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/ml1-report.md`. User ruling (2026-11-12): the "no AI/ML" rule means no paid services or cloud APIs; free local models are allowed. scikit-learn 1.9.1 (with joblib 1.6.0, threadpoolctl 3.8.0) is pinned; nothing calls the network at run time.

**Dataset (`tools/build_layer_dataset.py` -> `rosette/data/layers.csv`).** Every head layer the rule-based detector draws on the 76 reference photos (`pipeline.detect(classify=False)`: head disc, rings, points layers with their trims and the base star, center, unknown parts inside 1.05 R): 331 layers, 325 labelled. Features (`rosette/vision/layer_features.py`, read off the detector's polar label unwrap between the layer's radii): extent in head radii and rim contact; coverage at 9 positions, fill share, full rows; the lobe-width profile at 9 positions and its ends / peak / fullness / width ratio (what the H4 kind rules threshold); the 72-bin angular profile at 5 radii as harmonic magnitudes 1..24 and a high-frequency share; circular runs at 5 radii; the turned self-correlation for every n in 2..24 (the rules' symmetry score); the color just inside and outside; category / pattern / texture; the layers overlapping it. Labels: the reference file is split into layers (classified pieces with `tools.stacking.piece_kind`; pieces of one Color and kind whose radial extents overlap are one layer; a scalloped annulus drawn as one shape is a pinked ring; a lobed shape from the middle is the center; a star under everything without a head disc is the base star), each with its rotational order as the count; a detected layer takes the reference layer seen on top on most of its photo pixels (its color in its extent near its drawn shape, in head radii; another Color counts half; under 0.25 it is `none`); detected trims are matched by the reference having that trim in their band and Color. Reference-differs photos (Known limitations): S29-15 and S53-19 take the rules' reading (confirmed photo-true in H12 / H13; S29-15's rings and unknowns are left out), s44-15's points layers are left out. Classes: disc 31, pinked ring 63, ring 7, loop 31, kite 22, petal 10, star 29, base star 4, backing 15, edge 5, center 99, none 9.

**Model (`rosette/layer_model.py`, `tools/train_layers.py`, saved `rosette/data/layer_model.joblib`, 3.2 MB, with the training set's sha256).** Kind: a random forest (400 trees, balanced class weights) over the measurements plus the rules' reading (class, role and count; it beats the measurements alone, 0.628 vs 0.609). Count: each n in 2..24 is scored by a second forest from features of that n (its symmetry against the best, its harmonic, the runs against n and 2n, whether n is the rules' count) and the best n wins. Evaluated leave one design out (fix round 1: every photo of the held-out design left out, as `tools/score.py` groups them, so a variant such as S36-15B never teaches S36-15; never leave-one-layer-out; confirmed downloads never enter it), the rules on the same labels. After the pleated-ring rule below (leave one photo out, before fix round 1, the model read 0.643 overall):

| class | n | rules | model |
|---|---|---|---|
| disc | 31 | 0.871 | 0.839 |
| pinked ring | 63 | 0.492 (0.111 before the rule) | 0.651 |
| ring | 7 | 0.143 | 0.000 |
| loop | 31 | 0.677 | 0.742 |
| kite | 22 | 0.091 | 0.136 |
| petal | 10 | 0.100 | 0.000 |
| star | 29 | 0.483 | 0.379 |
| base star | 4 | 0.250 | 0.000 |
| backing | 15 | 0.400 | 0.400 |
| edge | 5 | 1.000 | 0.800 |
| center | 99 | 0.667 | 0.869 |
| none | 9 | 0.556 | 0.444 |
| **kind, all** | 325 | **0.554** (0.498 before the rule) | **0.628** |
| balanced | | 0.477 | 0.438 |
| **count** (points labels) | 116 | **0.681** | **0.750** (loops 0.968, stars 0.724) |

On the point shapes alone (loop to edge) the model does not beat the rules (0.41 vs 0.43); its gain is the ring style and the center role. The `ring` labels are almost all "the head disc shows as a band here".

**The bars (`train_layers.bars`: from the leave-one-design-out data, each on the layers it is applied to; fix round 1).** Kind, on the 178 layers the rules read as a plain or pinked ring or a point shape: the model decides at probability >= 0.650, where its held-out precision is 0.923 on 14.6 % of them (the rules are right on 0.846 of those; model and rules disagree on 2, the model right on both). Count, on the 81 non-trim points layers: at score >= 0.784, held-out precision 0.908 on 80.2 % of them. (Before fix round 1, leave one photo out over all layers: 0.645 and 0.889. The count bar is a deviation from "the count replaces the rules everywhere": without it the full model turned S80-22's 6-point star into 3 and S69-19's 6-point star into 5.)

**Wiring (`rosette/vision/layer_classify.py`, run by `pipeline.detect` after tidy and assign, on the same rows the model was trained on).** Count: at or above its bar the scorer's n replaces the rules' count of every non-trim points layer, the rotation is re-read for n and the width ratio scaled by old / new; the edge strips and backings on its axes follow. Kind: at or above its bar, only between plain and pinked ring and among the point shapes, and only where it is a parameter change (loop / kite / petal / star set the kind, an edge strip loses its strip width, a backing takes its host's kind); the role never changes; making a layer an edge strip or the base star needs geometry the model does not give. Below a bar a disagreement adds "Part {id}: the layer kind is uncertain - check the head against the photo." or "Part {id}: the point count is uncertain - check the number of points against the photo." (flags). Without a saved model nothing changes; a model file that cannot be used (corrupt, unpicklable, another scikit-learn version, not a `LayerModel`) or a classifier that fails leaves the rules' reading untouched and adds "Layer classifier unavailable - rules only." (fix round 1; logged once). A model trained on other data than `layers.csv` (+ the confirmed rows) is used with a logged warning to retrain. Cost: about 0.6 s per photo (features 0.5 s, both forests 0.15 s) and a one-time model load of 2-5 s.

**Pleated rings are pinked (`parts_head.pleat_texture`, `PLEAT_TEXTURE` 0.035).** The dataset showed a rule defect: the reference files draw nearly every pleated ring pinked and the rules called 24 of them plain. A ring band whose photo brightness varies along the circle around its 10-degree running median by at least 0.035 of its mean (median of 8 circles over the middle of the band, the band's color only, outside 45 degrees of the tails) is now pinked. On the 38 labelled detected rings the 31 pinked ones read 0.053-0.68 and the one smooth satin band (S55-15) 0.020; the 6 other "plain" labels are the head disc showing as a band (0.16-0.55; A10-15's inner red band is pleated on the photo). **The basis is thin: one smooth ring below the bar.** A ring whose texture lies in `PLEAT_UNSURE` 0.03-0.05 (between the smooth ring and the lowest pleated one) is drawn by the bar and flagged: "The ring's pleating is uncertain - check the ring edge against the photo." (`parts_head.pleat_warnings`, run by `pipeline.detect`; H11 pleated edges, read from their teeth, are not checked). None of the 38 rings falls inside the band.

**Learning from confirmed downloads (`rosette/confirmed.py`, `/api/confirm`).** With "Learn from my downloads (kept on this computer)" on (the default; under the Download button, remembered in the browser), a download posts the final model; the detected head layers (before the classifier) are written to `rosette/data/confirmed_layers.csv` (git-ignored) with the downloaded part of the same id as the label (a deleted layer is `none`; a library head is not logged; a photo confirmed again replaces its rows). `tools/train_layers.py` trains on them together with the reference dataset; each row carries the design key of the download name (`s46-15.svg` -> `S46`) or else the photo id, and held-out evaluations (`train_layers.loo`, `layer_model.model_without`) never use confirmed rows. **Confirmed labels include layers the user did not review**: a download confirms the flags, not every layer, so an unflagged wrong layer is learned as right. The model improves as confirmed downloads accumulate (retrain with `tools/train_layers.py`).

**Scoreboard (34 IDs; fix round 1, measured on c13e2ee plus ML-1).** New image, the headline: the classifier is retrained without the photo's own design (`layer_model.model_without(design_key)`, so it never reads a photo it learned from). **At its bars the classifier changes no kind or count on these held-out photos**: with it and without it every new-image row draws the same (PASS 21/34, color 0.856, head 0.842, layer IoU mean 0.2243). It acts as a flag source: S33-15 (its two 6-point stars, which the reference draws as kites; a true flag, silently wrong 2 -> 1) and S37-15 (its 8 loops, which the reference draws as a star; true). With the saved model (as in production) the count warning also fires on S69-19 and S80-22, whose 6-point stars are right (false alarms). The round's measurable gain is the pleated-ring rule (rules pinked-ring accuracy 0.111 -> 0.492; S41-15 color 0.924 -> 0.931 at H13); the model improves as confirmed downloads accumulate. Known design (the full model, which has seen these photos; reported only): S33-15 0.886 -> 0.944 (its stars drawn as kites, headS 0.890 -> 0.863), S80-22 0.992 -> 1.000 (headS 0.879 -> 0.896), A14-17 0.949 -> 0.968, S41-15 0.931 -> 0.914, S67-19 0.964 -> 0.959, S8-15 0.640 -> 0.637. The 76-photo comparison scripts were not run.

## H14: every color shape is a visible piece

Started at 5b3d167; commits c13e2ee (the cut), e6cf888 (the head fit's render). Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/h14-report.md`. Seen on S29-15's outline row: our loops' outlines ran through the stars and the stars' through the loops, because every shape was drawn whole and only the stacking hid the overlaps; the reference draws each element as the piece that shows, so no line crosses another.

**The convention, mined.** Every solid-configuration head element of the 79 reference files rendered alone (`roles.element_masks`, 900 px, masks eroded by 1 px so antialiased shared edges do not count): 1527 elements, 24445 pairs; 566 pairs share more than 20 px, 282 of them the same element written twice (IoU > 0.9; the LLM repeats elements, S29-15 repeats 8). Of distinct elements, 257 pairs overlap by more than 5 % of the smaller one, in 7 files only: TEST (183, not a product), S48-17 (45, a reference-differs file) and 29 pairs in A16-17, S17-15, S11-15, S22-15 and S37-15, mostly a small element lying wholly on a larger one. 72 of 79 files draw visible pieces only.

**The rule (`geometry/clip.clip_visible`; `SLIVER` 0.5, `GRID` 0.001; `MIN_PIECE`, `SIMPLIFY` as before).** After the final stacking (z, with the H12 photo decisions), each color shape is cut by the union of every shape stacked above it, whatever their Colors or parts: higher z, or equal z and painted later by the shop (higher Color, then later in the file; ruling: accepted, the colored view is the same and equal-z shapes such as fan tails would cross otherwise). Before (H3 `clip_occluded`) a shape was cut only where the shop's layer order would have painted it over a shape meant to lie above it. A shape covered by less than 1 unit^2 stays whole, so uncovered circles and ring paths stay circles and arc paths (a ring around a center in its hole too). A cut piece is simplified, snapped to the written 3-decimal grid (rounding made two rings touch themselves, S56-17 and S70-19, before the snap), and its polygons under 1 unit^2 or under 0.5 units mean width are dropped (2 x area / perimeter; half the shop's 1-unit line; of the 1072 cut pieces on the 76 photos 263 are under 0.1, 87 in 0.1-0.5, 119 in 0.5-1). The pattern twin (Layer_3) is the same shape written twice. The tail tops rule (`_under_head`) is unchanged. Even-odd paths are read by pairwise XOR (`xor_all`), which took S61-19's build (library pieces of up to 42 rings) from 3.3-5.1 s to 1.2 s.

**The head fit keeps whole shapes (`build(model, pieces=False)`).** Antialiased seams where two pieces abut blend their colors, and `pipeline._head_render` reads such pixels as no color: the rule-based head's fit fell (S37-15 silhouette 0.953 -> 0.939) and a library head won on S37-15 (new image) and s34-15 (known design). The head fit's render therefore builds with the pre-H14 cut (`clip_occluded`), which leaves stacked shapes whole; the head fits are their pre-H14 values. Written files always use `clip_visible`.

**Size and runtime** (the same cached models of the 76 photos built at 5b3d167 and at c13e2ee):

| | before | after |
|---|---|---|
| SVG bytes mean / median / max | 37 386 / 34 124 / 238 497 | 38 206 (+2.2 %) / 36 249 (+6.2 %) / 90 479 |
| size after / before | | min 0.35 (S61-19), median 1.04, max 1.73 (A16-17) |
| build + write s, mean / p90 / max | 0.149 / 0.219 / 5.15 (S61-19) | 0.123 / 0.314 / 1.16 (S61-19) |
| validate issues | 0 | 0 |

Per photo the build is a fraction of a second of the 10-15 s analysis (S29-15 profile: 0.38 s for its 4 builds before, 0.46 s after).

**On the 76 photos (same models before and after).** Colored view: pixels changed by more than 200 summed RGB at most 0.079 % (S61-19, specks dropped between the glitter dots of its library pieces), every other ID under 0.01 %. Overlaps left: after shrinking each piece by 0.1 units, 2 pairs over 0.01 units^2 in all 76. Means: color 0.8400 -> 0.8397, shape and head unchanged, layer IoU 0.2213 -> 0.2303. **Layer IoU reads role labels on our pieces**: it rises where our pieces now match the designer's visible pieces (S33-15 +0.18, S3-15 +0.18, S28-15 +0.11) and drops where a cut piece changes its label (S55-15 -0.22, A7-15 -0.19, A1-15 -0.18, S40-15 -0.13): a head disc cut by its ring is a pinked band while the designer's matching annuli read "head" and our plain ring "ring" (the H3 weakness), a head disc cut into bands adds "ring" pieces, and one cut by a points layer has a lobed outer band and reads "points". Outline pages: S29-15's loops stop at the dark star's edge, the pink star sits inside the dark star with no line through it, and a hidden off-center arc in the center is gone; S53-19's star points stop at the outer pinked ring and its tail tops abut; S35-19 (library pieces) and A7-15 (concentric rings) look unchanged.

**Scoreboard (34 IDs, clean export of e6cf888).** New image PASS 21/34 (held), color 0.857 -> 0.856, shape 0.919, head 0.843 (held), layer 0.2163 -> 0.2164, shape_ok 26/34 (held); known design PASS 22/34 (held), color 0.862 -> 0.861. Rows moved: S46-15 0.751 -> 0.725 (known too; not a success before), S41-15 known 0.907 -> 0.914, S17-15 uni 0.843 -> 0.902 (uni success), S69-19 uni 1.000 -> 0.998. Each is a designer element whose two colors sit at 50/50 (S46-15 0.496 vs 0.498) flipped by some 400 seam pixels (0.08 % of the render): noise.

## B16: a library head only when it does not lose layer agreement

Full notes: `.superpowers/sdd/2026-11-05-rosette-svg-generator/b16-report.md`. Cause C7 of the diag52 diagnosis: in the new-image case the library cannot offer the photo's own design, and the outline path of `prefer_template` drew another design whenever its silhouette won by `TEMPLATE_MIN_GAIN` 0.02, even with much lower layer agreement (S27-15: rules 0.749 / 0.836 agreement / silhouette, A6-15 0.659 / 0.946, a plain disc replacing a purple 4-point star and 4 white bows).

**Rule (`pipeline.prefer_template`, `TEMPLATE_SIL_AGREE_LOSS` 0.03).** The outline path also needs the library head's agreement to be at most 0.03 below the rule-based head's (the mirror of `TEMPLATE_AGREE_SIL_LOSS` on the agreement path). The agreement path is unchanged. The override keeps its own conditions: when `override_allowed` holds (rule-based head a gross failure, outline gain >= `TEMPLATE_OVERRIDE_GAIN`, partition >= `TEMPLATE_OVERRIDE_PARTITION` or no head layer), `library_head_path` does not apply the new bar (`floor` or `override` as before). A library head that loses the bar is not "a better fit", so the rule-based head is drawn without the colors warning. The colors-floor path is gated by the bar too; only `override_allowed` is exempt (a gross rules failure with a library gain between 0.02 and 0.10 and a library agreement more than 0.03 lower used to give `floor` and now gives None).

**Measured** on the 76 photos, new image (own design left out of the library and the classifier), at HEAD 2cacb9f: per photo the rule-based head and the first library candidate were each drawn and scored once, then the decision was replayed (the replay with the old rule draws exactly the heads analyze draws on all 76). Losers: IDs whose drawn library head loses more than 0.02 head IoU, layer IoU or color, or drops color under 0.9, against the photo's own rule-based head.

| TEMPLATE_MIN_GAIN, agreement bar | library heads | head IoU | layer IoU | color mean | color OK | silently wrong | losers |
|---|---|---|---|---|---|---|---|
| 0.02, none (before) | 16 | 0.8511 | 0.2313 | 0.8444 | 47 | 2 | S17-15, S2-15B, S27-15, S35-19, S36-15, S40-15, S52-19, S56-17, S56-17B, S57-17, S65-18 |
| 0.00 or 0.01, 0.03 | 13 | 0.8493 | 0.2395 | 0.8453 | 47 | 2 | S11-15, S17-15, S35-19, S36-15, S40-15, S52-19, S56-17B, S57-17, S65-18 |
| **0.02, 0.03 (kept)** | 12 | 0.8494 | 0.2378 | 0.8461 | 47 | 2 | S17-15, S35-19, S36-15, S40-15, S52-19, S56-17B, S57-17, S65-18 |
| 0.03 to 0.06, 0.03 | 12 | 0.8494 | 0.2378 | 0.8461 | 47 | 2 | same |
| 0.10, 0.03 | 11 | 0.8482 | 0.2404 | 0.8460 | 47 | 2 | S17-15, S36-15, S40-15, S52-19, S56-17B, S57-17, S65-18 |

Every bar from 0 to 0.04 gives the same heads: S2-15B (S9-19), S27-15 (A6-15), S31-15 (S30-15) and S56-17 (S4-15) draw their rule-based heads; from 0.05 S31-15 keeps S30-15, at 0.10 S27-15 keeps A6-15. 0.03 sits in that plateau; 0.05 scores a little higher on head IoU (0.8505) and color mean (0.8468) only because S31-15 keeps S30-15, which the round asks to send back. The library heads that win on both measures stay (S11a-15, S40-15, S36-15B, S65-18). With the bar, `TEMPLATE_MIN_GAIN` 0.02-0.06 tie; 0.02 is kept. Per ID, library -> rule-based head (head IoU / layer IoU / color): S27-15 0.864 -> 0.793 / 0.088 -> 0.453 / 0.919 -> 1.000; S2-15B 0.820 -> 0.815 / 0.137 -> 0.211 / 0.613 -> 0.687; S56-17 0.856 -> 0.884 / 0.215 -> 0.270 / 0.951 -> 0.984; S31-15 0.887 -> 0.804 / 0.053 -> 0.055 / 0.986 -> 0.929 (still OK). **Head IoU falls 0.0017**: the outline of a plain disc or another design fits better than the drawn layers on S27-15 and S31-15; no bar that sends S27-15 back keeps head IoU. S17-15 keeps S53-19 through the override (rules 0.512 agreement is a gross failure, gain 0.174, partition 0.378); gating the override too would also send S17-15, S52-19, S56-17B and S57-17 back (head 0.8477, layer 0.2402, color 0.8494, color OK 49), against the brief's "the override keeps its own conditions".

**Scoreboard (34 IDs, new image).** PASS 21/34 (held), silently wrong 1 (held), color 0.856 (held), shape 0.919 -> 0.918, head 0.843 -> 0.840, layer IoU 0.2164 -> 0.2186, shape_ok 26/34 (held); the known-design column is unchanged (PASS 22/34). Rows moved: S2-15B agree 0.613 -> 0.687, headS 0.820 -> 0.815 (rule-based head instead of S9-19); S31-15 agree 0.986 -> 0.929 (still a success), headS 0.887 -> 0.804, shape 0.941 -> 0.918 (rule-based head instead of S30-15: its outline and the reference's agree less than S30-15's do). The 76-photo comparison scripts were not run.

## Final state (Task 16)

### Stages, modules and tuning constants

Constants are module-level names; read their comments in the source before changing one.

| Stage | Module | Tuning constants |
|---|---|---|
| prepare: decode, 1200 px, mask, axis, head circle | `rosette/vision/prepare.py` | `WORK_HEIGHT`, `CANVAS_SPAN`, `ALPHA_CUTOFF`, `MIN_DELTA_E`, `NOISE_FACTOR`, `MAX_BG_NOISE`, `OPEN_PX`, `CLOSE_PX`, `MIN_AREA_FRAC`, `MAX_AREA_FRAC`, `AXIS_SEARCH_FRAC`, `HEAD_PROFILE_FRAC`, `HEAD_RING_MARGIN`, H10: `TAILS_TRACK_FROM`, `TAILS_FIT_ROWS`, `TAILS_GAP_ROWS`, `TAILS_EDGE_TOL`, `TAILS_RUN_ROWS`, `FIT_CUT_MARGIN`, `FIT_GAP_CAP_DEG`, `FIT_TOL`, `FIT_COVER_GAP_DEG`, `FIT_TAKEOVER`, `FIT_MIN_COVERAGE` |
| color groups, stripe split, motif absorption, mixed-texture warning | `rosette/vision/colors.py` | `L_WEIGHT`, `SMOOTH_UNITS`, `FLAT_PCT`, `GRAIN_WEIGHT`, `SAMPLE_N`, `MIN_CENTER_SEP`, `PRINT_MAX_FRAC`, `TEXTURE_WIN_UNITS`, `PATTERN_T`, `MIXED_PATTERN_SHARE`, `BADGE_IN_HEAD`, `BADGE_MAX_AREA`, `BADGE_EDGE`, `BADGE_UNSURE`, `BADGE_ENCLOSED`, `BADGE_SHAPE_MIN`, `SEED`, H13: `RIM_TOUCH`, `RIM_THICK_UNITS` |
| part detection driver, leftovers as `unknown` | `rosette/vision/parts.py` | `LEFTOVER_OPEN_UNITS`, `TAIL_TOUCH_PX`, `HEAD_MARGIN`, `UNKNOWN_CONFIDENCE` |
| head: rings, points, bows, center, blob rings | `rosette/vision/parts_head.py`, `parts_common.py` | `RING_EXTEND_COV`, `FULL_COV`, `MIN_BAND_UNITS`, `MIN_LAYER_COV`, `MAX_COUNT`, `BOW_ENDS`, `PETAL_FULLNESS`, `BLOB_UNCONFIRMED_CONF`, `MIN_INNER_RATIO`, `HEAD_ROUND`, `PINKED_DEPTH`, `RING_MAX_GAP`, `PLEATED_FULL_MIN`, `PLEATED_REACH_UNITS`, H11: `COVER_CUT_SHARE`, `COVER_PROBE_UNITS`, `COVER_MAX_UNITS`, `STAR_MIN_ORDER`, `STAR_INSIDE_TOL`, `STAR_CONVEX`, `STAR_HIDDEN_DEPTH`, `PLEAT_MIN_COUNT`, `PLEAT_FILL`, `ALT_RADIUS_UNITS`, `ALT_OFFSET`, H12: `SPLIT_BANDS`, `SPLIT_FLOOR`, `SPLIT_MARGIN`, `SPLIT_MIN_INNER`, `BLOCK_MID`, `BLOCK_MIN_SHARE`, `STACK_SHARE`, `STACK_MIN_UNITS2`, `NEST_AXIS_DEG`, `NEST_UNITS`, H13: `BASE_TIP_UNITS`, `BASE_TIP_SHARE`, `BASE_FLANK_UNITS`, `BASE_FLANK_TIP`, `BASE_REACH`, `BASE_Z`, `BASE_FIT_UNITS`, `RING_OVERLAP_UNITS`, `RING_MAJORITY`, `BORDER_ROW_SHARE`, `BORDER_BODY_UNITS`, `BORDER_UNITS`, `BORDER_MIN_SIDES` (parts_trims) |
| tail stripes, glitter read as two tones, fans (H6) | `rosette/vision/parts_tails.py` | `ROW_FRACS`, `COUNT_ROWS`, `DENSE_SPAN`, `GAP_UNITS`, `ROW_BAND_PX`, `TOP_INSET`, `GLITTER_*`, `FAN_*` |
| head shape library and matcher (H7) | `rosette/vision/templates.py`, cache `rosette/data/head_templates.json`; decision in `rosette/pipeline.py` | `WEIGHTS`, `PROFILE_SCALE`, `LOBE_SCALE`, `LOBE_GATE`, `LOBE_BAND`, `LOBE_PROMINENCE`, `ROUND_GUARD_LOBES`, `WINDOW_V`, `GRID`, `EXTENT`, `TEMPLATE_MIN_GAIN`, `TEMPLATE_MIN_SCORE`, `TEMPLATE_MIN_PARTITION`, `TEMPLATE_OVERRIDE_GAIN`, `TEMPLATE_OVERRIDE_PARTITION`, `HEAD_LAYER_ROLES`, `TEMPLATE_CANDIDATES`, H11: `TEMPLATE_AGREE_GAIN`, `TEMPLATE_AGREE_SIL_LOSS`, `POINTS_COUNT_MIN`, B16: `TEMPLATE_SIL_AGREE_LOSS`, describe: `STRIP_ASPECT`, `TRIM_SOLIDITY`, `LOOP_END`, `STAR_DEPTH` |
| tidy: unknown specks, tail slivers (H6), points layers under 3 lobes (H13) | `rosette/tidy.py` | `SPECK_MAX_HEAD_SHARE`, `MIN_POINTS` (and `SLIVER_SHARE` from assign) |
| Color 1-4 assignment, confidence, warnings | `rosette/assign.py`, `rosette/roles.py`, rules in `rosette/data/assign_rules.json` (mined by `tools/mine_rules.py`) | `FLAG_BELOW`, `MAX_COLORS`, `PATTERN_MIN_RIBBON_SHARE`, `TAIL_POS_TOL`, `FLAGGED_CONF`, `PATTERN_MERGE_PENALTY`, `SLIVER_SHARE`, `RULE_LOGODDS` |
| geometry, shapes, occlusion cuts | `rosette/geometry/`, `rosette/shapes.py`, `rosette/build.py`, `rosette/geometry/clip.py` | `QUAD_SEGS`, `SIMPLIFY`, `MIN_PIECE`, `SLIVER`, `GRID` (H14 visible pieces, `clip_visible`) |
| SVG writer and contract validator | `rosette/writer.py`, `rosette/validate.py` | none (the contract is fixed) |
| learned head-layer classifier (ML-1): features, kind and count behind LOO-derived bars, confirmed-download rows | `rosette/vision/layer_features.py`, `rosette/vision/layer_classify.py`, `rosette/layer_model.py` (model `rosette/data/layer_model.joblib`, data `rosette/data/layers.csv`, `tools/build_layer_dataset.py`, `tools/train_layers.py`), `rosette/confirmed.py` | `KIND_TREES`, `COUNT_TREES`, `SEED`, `LayerModel.gate` and `count_gate` (derived with `train_layers.GATE_PRECISION`, `GATE_MIN_LAYERS`), `UNKNOWN_IN`, `AXIS_DEG`; pleated rings `PLEAT_TEXTURE`, `PLEAT_*` (parts_head) |
| orchestration, flagging, head fit | `rosette/pipeline.py` | `FLAG_BELOW` (imported from assign), `HEAD_FIT_MIN`, `HEAD_SILHOUETTE_MIN`, `CIRCLE_RATIO_MIN`, `CIRCLE_COVERAGE_MARGIN`, `HEAD_FIT_EXTENT`, `HEAD_FIT_WINDOW`, `HEAD_FIT_GRID` |
| shop simulation (solid and pattern previews) | `rosette/simulate.py` | none |

### Shop contract

Every generated file must satisfy section 3 of the [design spec](superpowers/specs/2026-11-05-rosette-svg-generator-design.md): fixed top-level layer order (`Layer_1`, `Layer_3`, `Layer_2_1`, `Layer_6`, `Layer_rose`; issue `group_order` when it is not kept), category classes for Color 1-4, Color 2 usable in solid and pattern mode, and Color 3 correct in both. `rosette/validate.py` checks it and `render` returns the issues; the UI does not offer a download while an issue is open. The shop's own coloring logic is in `reference/shop-configurator.js` (verbatim); `rosette/simulate.py` mirrors it for the UI previews.

### UI modes

`app/main.py` (FastAPI, started with `run.bat` or uvicorn on 127.0.0.1:8765) serves `app/static`. **Workbench** shows the photo, the detected model, the solid and pattern previews, edit controls and the flagged items on one screen. **Guided** walks through the same state one question at a time (`guided.js`); switching modes keeps the state. Edits go back through `render`, so the validator runs after every change.

### Current scoreboard (`tools/score.py`, 34 IDs, re-recorded for B16)

Re-recorded for H14 from a clean export of e6cf888 (visible pieces; the rows that moved are listed in the H14 section). Re-recorded for ML-1 (the new-image column runs the layer classifier trained without the photo's own design): the columns are the new-image case (each photo's own design left out of the head library, the headline); the last column is the known design (full library) agree / success / headS. `stack reversed` counts the kind pairs our file stacks the other way round from the designer file (reported only).

| ID | agree | uni | success | uni success | shape | headS | stack reversed | known design agree / success / headS |
|---|---|---|---|---|---|---|---|---|
| A10-15B | 0.906 | 0.942 | yes | yes | 0.870 | 0.684 | 0 | 0.906 / yes / 0.684 |
| A13-15B | 0.187 | 0.187 | no | no | 0.929 | 0.967 | 1 (head>center) | 0.187 / no / 0.967 |
| A14-17 | 0.949 | 0.949 | yes | yes | 0.957 | 0.963 | 0 | 0.968 / yes / 0.963 |
| A15-17 | 0.949 | 0.949 | yes | yes | 0.965 | 0.957 | 0 | 0.949 / yes / 0.957 |
| A16-17 | 0.837 | 0.926 | no | yes | 0.895 | 0.830 | 0 | 0.837 / no / 0.830 |
| A9-15B | 1.000 | 1.000 | yes | yes | 0.939 | 0.828 | 0 | 1.000 / yes / 0.828 |
| S10-19 | 0.942 | 0.955 | yes | yes | 0.900 | 0.767 | 0 | 0.942 / yes / 0.767 |
| S13-15 | 0.894 | 0.894 | no | no | 0.962 | 0.928 | 0 | 0.894 / no / 0.928 |
| S17-15 | 0.599 | 0.902 | no | yes | 0.886 | 0.730 | 0 | 0.669 / no / 0.737 |
| S2-15B | 0.687 | 0.687 | no | no | 0.934 | 0.815 | 0 | 0.687 / no / 0.920 |
| S26-15 | 0.800 | 0.985 | no | yes | 0.933 | 0.856 | 0 | 0.800 / no / 0.856 |
| S28-15 | 1.000 | 1.000 | yes | yes | 0.980 | 0.964 | 1 (star>loop) | 1.000 / yes / 0.964 |
| S29-15 | 0.301 | 0.301 | no | no | 0.908 | 0.765 | 1 (loop>star) | 0.301 / no / 0.765 |
| S30-15 | 0.972 | 0.972 | yes | yes | 0.945 | 0.879 | 0 | 0.983 / yes / 0.944 |
| S31-15 | 0.929 | 0.957 | yes | yes | 0.918 | 0.804 | 0 | 0.986 / yes / 0.887 |
| S33-15 | 0.886 | 0.960 | no | yes | 0.955 | 0.890 | 0 | 0.944 / yes / 0.863 |
| S35-19 | 0.963 | 0.963 | yes | yes | 0.927 | 0.845 | 0 | 0.942 / yes / 0.805 |
| S37-15 | 0.949 | 0.936 | yes | yes | 0.905 | 0.825 | 0 | 0.921 / yes / 0.825 |
| S40-15 | 0.893 | 0.889 | no | no | 0.927 | 0.879 | 0 | 0.893 / no / 0.879 |
| S41-15 | 0.931 | 0.951 | yes | yes | 0.906 | 0.914 | 0 | 0.914 / yes / 0.914 |
| S43-15 | 1.000 | 1.000 | yes | yes | 0.953 | 0.926 | 0 | 1.000 / yes / 0.926 |
| S45-15 | 0.935 | 0.935 | yes | yes | 0.926 | 0.809 | 0 | 0.935 / yes / 0.809 |
| S46-15 | 0.725 | 0.936 | no | yes | 0.835 | 0.660 | 1 (loop>star) | 0.725 / no / 0.660 |
| S5-15 | 0.868 | 0.988 | no | yes | 0.852 | 0.734 | 0 | 0.868 / no / 0.734 |
| S53-19 | 0.579 | 0.579 | no | no | 0.768 | 0.843 | 1 (star>ring) | 0.579 / no / 0.843 |
| S56-17B | 0.969 | 0.963 | yes | yes | 0.902 | 0.817 | 0 | 0.988 / yes / 0.848 |
| S58-17 | 1.000 | 1.000 | yes | yes | 0.943 | 0.880 | 0 | 1.000 / yes / 0.874 |
| S67-19 | 0.964 | 0.964 | yes | yes | 0.947 | 0.861 | 0 | 0.959 / yes / 0.861 |
| S68-19 | 0.966 | 0.620 | yes | no | 0.885 | 0.676 | 0 | 0.966 / yes / 0.676 |
| S69-19 | 0.998 | 0.998 | yes | yes | 0.924 | 0.775 | 0 | 0.998 / yes / 0.775 |
| S70-19 | 0.956 | 0.956 | yes | yes | 0.957 | 0.939 | 0 | 0.956 / yes / 0.939 |
| S8-15 | 0.640 | 0.640 | no | no | 0.936 | 0.841 | 0 | 0.637 / no / 0.841 |
| S80-22 | 0.992 | 1.000 | yes | yes | 0.934 | 0.879 | 0 | 1.000 / yes / 0.896 |
| s34-15 | 0.956 | 0.956 | yes | yes | 0.926 | 0.838 | 1 (backing>star) | 0.956 / yes / 0.841 |

Official (per-axis), tools/score.py summary: new image `success 21  flagged-not-success 12  silently-wrong 1  errors 0` / `PASS 21/34 (target >= 80%)` / `color agreement mean 0.856  shape: mean IoU 0.918  head 0.840  tail 0.955  shape_ok 26/34 (IoU >= 0.9)`, layer IoU mean 0.2186; known design (reported only) `PASS 22/34`, color 0.861, shape 0.920, head 0.847, shape_ok 26/34; `stacking: 6 kind pairs reversed vs the designer in 6/34 IDs (reported only)` / `success under uniform-scale alignment 26/34` (reported only). H12 (see its section for the rulings): S5-15 and S33-15 lose their success. H13: S53-19's headS 0.855 -> 0.843 accepted (reference proportions, see its section). ML-1: S33-15 is flagged now; the known-design column changes come from the full classifier (see its section). H14: S46-15 0.751 -> 0.725 (a 50/50 element flipped by seam pixels, see its section). B16: S2-15B and S31-15 draw their rule-based heads in the new-image column (see its section).

Flagged although correct: 20 of the 21 successes; S43-15 is an unflagged success with a visibly wrong head (see 5b); S28-15's wrong head is flagged since H11 (the point-count warning). The two silently wrong IDs are reference-vs-photo cases, not coloring misses: S46-15 (scorer alignment: the designer drew a larger head and narrower tails; its head fit 0.82 agreement, 0.94 outline is not a gross failure) and S33-15 (since H12 the photo's stacking puts the blue 6-point star over the purple glitter star, which its reference draws the other way; its head fit 0.71 is not a gross failure). S48-17 and s44-15 are the other reference-differs cases among the 76 photos (see Known limitations).

Over all 76 reference photos (scratch `compare_all.py`, match >= 0.9 and no issue): 36 -> 45 OK, mean color agreement 0.689 -> 0.832 (H2), then 0.832 -> 0.837 with OK still 45 (H3); mean whole shape IoU 0.902. Silently wrong (not OK, not flagged): A13-15 (designer differs from photo), S46-15 (scorer alignment), S48-17 (0.672, up from 0.382; the remaining miss is not re-diagnosed) and s44-15 (designer differs: much narrower outer tails). The every-head reminder is the UI checklist item, not a flag.

### Sample outputs

`output/s35-19.svg` and `output/s46-15.svg` were regenerated for H2 (and regenerated again for H3: byte-identical, so unchanged) from `reference/photos/S35-19.png` and `reference/photos/S46-15.png` with `rosette.pipeline.analyze` + `render`, the same code path the UI uses, and checked with `rosette.validate.validate` (both return `[]`). Regenerated again for H5 (pointed tail tops, visible-edge line art; both validate to `[]`), for H6 (`s35-19.svg` changed: its two tail slivers are merged; `s46-15.svg` byte-identical; both validate to `[]`) and for H7 (`s35-19.svg` changed: its head is the S35-19 library template, match 0.68; `s46-15.svg` byte-identical; both validate to `[]`), for H7 fix round 2 (`s35-19.svg` back to the rule-based head) and for fix round 3 (`s35-19.svg` drawn with its own S35-19 library template again, partition 0.68 over the 0.54 floor; `s46-15.svg` byte-identical; both validate to `[]`); for fix round 4 both are byte-identical and validate to `[]`. For H10 (704b6d8) `s35-19.svg` was regenerated after the H9 pose fix 5cb11e6 (H10 and the rebuilt library leave it unchanged); `s46-15.svg` is byte-identical; both validate to `[]`. For H11 both are byte-identical and validate to `[]`. For H12 `s46-15.svg` was regenerated (its loops drawn at their visible width and under its star, as the photo shows them; still unflagged, head fit 0.82 / outline 0.94); `s35-19.svg` is byte-identical; both validate to `[]`. For H13 (and its fix round 1) both were regenerated and are byte-identical; both validate to `[]`. For H14 (clean export of e6cf888) both were regenerated: `s46-15.svg` changed (every color shape is its visible piece; 30 301 -> 27 895 bytes), `s35-19.svg` is byte-identical (its library head's pieces were visible pieces already); both validate to `[]`, flags unchanged (S35-19's head-overlap warning now reads 69 %). They are the tool's unedited output, not confirmed by a person.

| Output | Flagged | Flagged items | Warnings |
|---|---|---|---|
| `output/s35-19.svg` | yes | the warnings only: the S35-19 library head (match 0.82) is above 0.6; no `unknown` parts; no color group below 0.6. | 2 tail slivers merged into their neighbours (H6); the library and rule-based heads overlap 71 % (the warning read 71 % at 105c9a8 already; heads differ; the head card offers S30-15 and S31-15 as well) |
| `output/s46-15.svg` | no | none (head fit 0.82 / outline 0.94 is information only) | none |

Before uploading either file to the shop, the user should open each photo in the UI, confirm the flags and compare the drawing with the photo. That confirmation has not been done; it is the user's step.

### Known limitations

- **Library heads pick the photo's own design for 54 of 76 photos (H10; H7 after fix round 4: 48).** S29-15 draws its rule-based head since H11 (since H12 a pink 6-point star over a larger dark 6-point star and the glitter loops; its own star template ranks 21st and its head fit loses to the rule-based head); S51-19 loses 0.059 color through the override (flagged; accepted). See the H7, H10 and H11 sections.

- **s44-15's two 6-point stars are read as 4-petal layers (H11).** Its head fit and its reference agree with the wrong head as well as with its own template; it is flagged by the point-count warning only when its own design is not in the library.
- **The success bar is not met: 21/34 succeed against 28 needed (80 %; 23/34 before H12, see the H12 rulings).** `pytest -m slow` fails accordingly. Tuning stopped; it is not finished.
- **S46-15 counts as silently wrong under per-axis scoring** (0.741, not flagged). It is a scorer alignment effect (the designer drew a larger head and narrower tails; uniform scale gives 0.926); its head fit (0.82) is not a gross failure.
- **Reference differs from the photo (silently wrong, not coloring misses).** S46-15 (above); S33-15 (H12: the photo shows the blue 6-point star over the purple glitter star, 0.89 of their overlap; its reference stacks them the other way: 0.905 -> 0.886, and its better head fit, 0.71, leaves it unflagged); S48-17 and s44-15 (their references differ from the photos; 76-photo list in the scoreboard notes).
- **Head shape errors are mostly not flagged.** The head flag fires on gross failures only (23 of 76 reference heads). S42-15, S43-15 and S28-15 have wrong head shapes that pixel overlap cannot tell from right heads; the fixed UI checklist item "Compare the head with the photo" is the safeguard until shape features land.
- The holdout is no longer clean (see Task 12 phase B), so the numbers on the tuned IDs overstate how the tool will do on new photos.
- Most successes are still flagged (22 of 23: placeholder `unknown` discs, or a part read below 0.6, mostly points rings). The flag is warranted but means most results need a human look.
- Remaining failure clusters:
  - Head structure not modeled: S17-15 (three layered stars, no head disc), S29-15 (since H12 a pink 6-point star over a larger dark 6-point star, over the glitter loops with pink edge strips; the pink pleated ring still reads as a dark ring and the pink star's satin body reads dark in the color groups, so the photo cannot order the two stars (49/51): the next color-grouping item), S68-19 (stacked chevron points), A16-17 (irregular 8-point star in the head's own black), S13-15 (loops drawn as a star because the navy ruffle merged into the blue group).
  - Mask: S53-19 (pure white stripes on a pure white background are cut out; one tail reads as three).
  - Color grouping: S2-15B (snowflake print merged with white satin). A13-15B's glitter is its own group since H6; its colors now differ from the reference's other scheme (royal Color 1 by the mined rules).
  - Designer differs from the photo: S8-15 (centre stripe the same red as the outer), S26-15 (glitter stripe widths), S46-15 (proportions).
- Photos need a single rosette on a plain background; a rosette whose colors match the background can fail the mask. The messages for no rosette and unreadable files come from `prepare`.
- Classical CV plus one small learned classifier (ML-1): unusual head constructions fall back to `unknown` placeholder discs that the user must fix in the UI. The classifier ties the rules on the point shapes and changes no new-image scoreboard row at its bars; its labels come from LLM-made reference files (see the ML-1 section).
