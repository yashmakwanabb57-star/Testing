# Mismatch analysis: photo vs designer SVG (76 photos)

Date 2026-11-06. Branch `feat/rosette-svg-generator`. Pipeline behavior unchanged by this work;
the only code change is the comparison tooling (`tools/rasterize.py`, `tools/score.py`,
`rosette/roles.py` rendering helpers) plus an unwired prototype (`tools/head_templates.py`).

## 1. Comparison tooling fix (class 3)

The shop (`reference/shop-configurator.js`, `setSvgImgColor`) toggles `#Layer_3` against
`#Layer_2_1` **only when both exist**. Our rasterizer and the scorer always hid `#Layer_3`.
26 designer files have `Layer_3` but no `Layer_2_1` (old layouts: `Layer_1` and `Layer_2` nested
inside `Layer_3`, or a top-level `Layer_3` holding the only copy of the Color 3 parts). Hiding it
made 3 files render blank (S1-15, S15-15B, S36-15) and dropped whole Color categories from 19
more (e.g. every Color 3 part of S61-19, A2-15, S48-17, S57-17). Separately, `score.compare`
only scored elements under `Layer_1` / `Layer_2_1_2..4`, so files keeping parts in `Layer_2`,
`Layer_2_1_1` or unnamed groups scored on a subset or not at all (the 14 reported 0.000 rows;
also S37-15 had 11 of 13 elements ignored).

Fix (commit "fix: render and score old-layout designer SVGs like the shop does"):
`rosette.roles.pattern_layer` implements the shop rule (first `#Layer_3` hidden only if a
`#Layer_2_1` exists); `rasterize`, `render_categories`, `element_masks` use it; `compare` scores
every solid-configuration color element. Our own output always has both groups, so it renders
exactly as before. None of the 35 canonical refs has the old layout, so `assign_rules.json`
mining input is unchanged.

After the fix: 36/76 at match >= 0.9 with no validation issue (same count as before, but now
measured on all files; several earlier "OK"s were measured on partial designer renders).
Mean color_agreement 0.689.

## 2. Per-photo diagnosis

Every comparison PNG (`compare_all.py` output, photo | ours | designer, both SVGs colored with our
detected per-category colors) was looked at. Columns:
- match / ok / flagged: color_agreement after the tooling fix, ok = match >= 0.9 and no issue,
  flagged = UI asks the user to check (NO = silent).
- out-in match: color_agreement if our groups were renumbered by the out-to-in tail rule
  (section 3; simulation only, not shipped).
- head columns: head agreement of the current output, the leave-one-out template and the
  full-library template (section 4); geo = score.py's per-part majority rule with the best
  relabeling (numbering ignored), pix = per-pixel share (stricter).

Sorted worst match first.

| ID | match | ok | flagged | classes | out-in match | head cur geo/pix | head LOO geo/pix (pick) | head full geo/pix (pick) | note |
|---|---|---|---|---|---|---|---|---|---|
| A7-15 | 0.000 | no | yes | head shape, tail numbering, designer differs from photo | 0.000 | 0.44/0.47 | 1.00/0.97 (A2-15) | 1.00/1.00 (A7-15) | Pleated 2-ring head drawn as pinked disc + loose rings; designer gives left/center/right tails 3 different Colors (photo 1-2-1), so no numbering can match |
| S1-15B | 0.000 | no | yes | head shape, tail numbering | 1.000 | 1.00/0.70 | 0.68/0.54 (S29-15) | 1.00/1.00 (S1-15B) | 6 two-tone bow loops drawn as squares; red/white Colors swapped vs designer (designer: outer tails = Color 1) |
| S18-15 | 0.000 | no | yes | head shape, tail numbering | 0.971 | 0.51/0.56 | 0.85/0.79 (S53-19) | 1.00/0.95 (S18-15B) | 8-point star + pleated rings drawn as square + diamonds; Colors swapped (outer tails should be Color 1) |
| S2-15 | 0.000 | no | yes | head shape, tail numbering | 1.000 | 1.00/0.73 | 1.00/0.72 (S5-15) | 1.00/0.73 (S2-15B) | Bow loops + star drawn as star + squares; Colors swapped |
| S2-15B | 0.000 | no | yes | head shape, tail numbering, tail geometry | 0.687 | 0.77/0.59 | 0.77/0.52 (S32-15) | 0.77/0.77 (S2-15B) | Bow loops drawn as star + petals; snowflake pattern stripe merged into white (2 colors found, designer has 3); numbering off |
| S9-19 | 0.000 | no | yes | head shape, tail numbering | 0.896 | 0.58/0.39 | 0.88/0.60 (S57-17) | 0.88/0.60 (S57-17) | Bow loops + petal layer drawn as plain disc + specks; Colors swapped |
| S36-15 | 0.066 | no | yes | head shape, tail numbering | 0.965 | 0.77/0.55 | 0.77/0.77 (S52-19) | 1.00/1.00 (S36-15) | 6 petal loops + rose drawn as overlapping circles; Colors swapped |
| S4-15 | 0.092 | no | yes | head shape, tail numbering | 0.908 | 0.69/0.71 | 0.83/0.68 (S10-19) | 0.83/0.68 (S10-19) | Layered petal fan drawn as one pinked disc; Colors swapped |
| S36-15B | 0.100 | no | yes | head shape, tail numbering | 0.747 | 0.23/0.30 | 0.62/0.57 (S31-15) | 1.00/1.00 (S36-15B) | Two-tone petal loops drawn as circle blobs; Colors swapped (ours red=Color 1 on center stripe, designer Color 1 on outer tails) |
| S32-15 | 0.142 | no | yes | head shape, tail numbering | 0.858 | 0.59/0.50 | 0.89/0.83 (S59-17) | 1.00/1.00 (S32-15) | Two-tone 8-point star + pleated ring drawn as diamond + 4 points; Colors swapped |
| S3-15 | 0.156 | no | yes | head shape, tail numbering | 0.844 | 0.53/0.55 | 0.82/0.61 (S18-15) | 1.00/1.00 (S3-15) | Two-tone 8-point star + pleated ring drawn as square + 4 points; Colors swapped |
| A2-15 | 0.174 | no | yes | head shape, designer differs from photo | 0.372 | 0.77/0.55 | 0.77/0.62 (S47-15) | 0.77/0.62 (S47-15) | Designer has asymmetric 3-Color tails (left/center/right), photo is 1-2-1 red/white/red; pleated rings drawn as ring + disc |
| A13-15B | 0.187 | no | yes | head shape, tail numbering | 0.272 | 0.76/0.84 | 0.76/0.87 (S47-15) | 0.76/0.87 (S47-15) | Asymmetric 3-tail photo; glitter middle tail split into thin stripes; pleated head drawn as one pinked disc; designer numbering of the 3 tails not out-to-in |
| S29-15 | 0.247 | no | yes | head shape, designer differs from photo | 0.247 | 0.38/0.37 | 0.49/0.35 (S57-17) | 0.49/0.35 (S57-17) | Tails right by photo, but designer puts Color 2 on the center stripe and Color 3 on the glitter stripes (pattern rule forces glitter = 2); star + loops head drawn as blobs |
| A9-15 | 0.268 | no | yes | head shape, designer differs from photo | 0.268 | 0.49/0.53 | 0.73/0.72 (A3-15) | 0.73/0.72 (A3-15) | Designer has asymmetric 3-Color tails, photo is symmetric blue/white/blue; stacked pleated rings drawn as one disc |
| S65-18 | 0.312 | no | yes | head shape, tail numbering, tail geometry | 1.000 | 0.89/0.68 | 0.89/0.84 (A6-15) | 0.89/0.84 (A6-15) | Center white stripe lost (tails drawn as two red overlapping tails with a gap); pleated disc drawn as ring; Colors swapped |
| A13-15 | 0.329 | no | NO | designer differs from photo | 0.329 | 0.76/0.75 | 0.76/0.75 (S11a-15) | 0.76/0.75 (S11a-15) | Photo is one red; designer splits it into 2 Colors (left tail + center vs right tail + head); silent: not flagged |
| A10-15 | 0.337 | no | yes | head shape, tail numbering | 0.000 | 0.75/0.74 | 0.82/0.73 (A15-17) | 1.00/1.00 (A10-15) | Asymmetric white/blue/red tails right by photo but Colors permuted vs designer; stacked pleated rings drawn as disc + diamond ring |
| S48-17 | 0.382 | no | yes | head shape, tail numbering | 0.672 | 0.76/0.61 | 0.87/0.57 (S61-19) | 1.00/0.99 (S48-17) | Tails right by photo; light blue and white Colors swapped (2nd stripe from outside should be Color 2); layered star head drawn as square + petals |
| S49-17 | 0.438 | no | yes | head shape, tail numbering | 0.983 | 0.93/0.72 | 1.00/0.74 (A9-15B) | 1.00/1.00 (S49-17) | Tails right by photo; dark green and white Colors swapped; layered head roughly right |
| S15-15 | 0.468 | no | yes | head shape, tail geometry, points count | 0.153 | 0.54/0.42 | 0.49/0.35 (S33-15) | 0.49/0.35 (S33-15) | Red split into two groups (pink-red / crimson) giving extra stripes; 2 layered 8-point stars drawn as one 8-star |
| S63-17 | 0.468 | no | yes | head shape, tail numbering | 0.941 | 0.72/0.72 | 0.66/0.64 (A1-15) | 1.00/1.00 (S63-17) | Tails right by photo, white / light blue Colors swapped; very complex layered head drawn as disc + 4 blobs |
| S53-19 | 0.513 | no | yes | head shape, tail geometry | 0.513 | 0.60/0.63 | 0.69/0.51 (S3-15) | 1.00/1.00 (S53-19) | Tail stripes misplaced (outer tails spread apart); 8-star + stacked pleated rings drawn as square + diamond + diamond ring |
| s44-15 | 0.519 | no | yes | head shape, designer differs from photo | 0.519 | 0.89/0.77 | 0.83/0.67 (S37-15) | 1.00/1.00 (s44-15) | Tails right by photo, designer draws outer tails much narrower; layered white/gold star drawn as petals |
| S68-19 | 0.600 | no | yes | head shape, tail numbering | 0.966 | 0.87/0.77 | 0.84/0.58 (A2-15) | 1.00/1.00 (S68-19) | 4 Colors; mid purple / light lavender swapped; layered stars drawn as 2 circles |
| S17-15 | 0.633 | no | yes | head shape | 0.633 | 0.42/0.34 | 0.54/0.49 (S3-15) | 0.54/0.49 (S3-15) | Tails right; layered 8-point stars + rose drawn as circle blobs |
| S8-15 | 0.636 | no | yes | head shape, designer differs from photo | 0.636 | 0.59/0.50 | 0.59/0.61 (S37-15) | 0.59/0.61 (S37-15) | Designer has a separate Color 3 center stripe, photo center is the same red as the outer tails; star points drawn as blocks |
| S46-15 | 0.741 | no | NO | head shape, points count | 0.741 | 0.93/0.69 | 0.96/0.74 (A14-17) | 0.96/0.74 (A14-17) | Tails right; gold layered star drawn as one 6-point star (photo/designer 8), glitter loops as blobs; silent: not flagged |
| S57-17 | 0.781 | no | yes | head shape | 0.781 | 0.36/0.34 | 0.49/0.40 (S22-15) | 1.00/1.00 (S57-17) | Tails right; striped bow loops + pleated rings drawn as broken blobs |
| S26-15 | 0.800 | no | yes | head shape | 0.800 | 0.84/0.71 | 0.94/0.66 (S70-19) | 1.00/1.00 (S26-15) | Tails right; 6-point star + glitter loops + hexagon center drawn as petals + disc |
| A3-15 | 0.824 | no | yes | head shape | 0.824 | 0.48/0.48 | 0.48/0.48 (A6-15) | 0.48/0.48 (A6-15) | Asymmetric tails right; 3 stacked pleated rings drawn as one pinked disc |
| S18-15B | 0.830 | no | yes | head shape | 0.830 | 0.76/0.55 | 0.84/0.59 (S15-15B) | 1.00/1.00 (S18-15B) | Tails right; 8-point two-tone star + pleated rings drawn as square + diamond |
| A16-17 | 0.837 | no | yes | head shape | 0.837 | 0.43/0.44 | 0.74/0.60 (A10-15B) | 0.74/0.74 (A16-17) | Tails right; pleated rings + black 8-star drawn as one black disc |
| S13-15 | 0.840 | no | yes | head shape | 0.840 | 0.59/0.44 | 0.92/0.68 (A15-17) | 1.00/1.00 (S13-15) | Tails right (thin extra stripe); loops + 6-star + pleated ring drawn as pinked disc + one star |
| S15-15B | 0.844 | no | yes | head shape, points count | 0.844 | 0.59/0.46 | 0.59/0.49 (S53-19) | 1.00/1.00 (S15-15B) | Tails right; two layered 8-point stars + pleated rings drawn as one 12-point star, no rings |
| S51-19 | 0.858 | no | yes | head shape | 0.858 | 0.46/0.32 | 0.61/0.57 (S47-15) | 0.61/0.57 (S47-15) | Tails right; layered petal loops drawn as blobs |
| S38-15 | 0.865 | no | yes | head shape | 0.865 | 0.79/0.55 | 0.70/0.59 (S26-15) | 0.70/0.59 (S26-15) | Tails right; 8-point star layer missing (glitter loops as blobs) |
| S59-17 | 0.876 | no | yes | head shape | 0.876 | 1.00/0.80 | 0.56/0.50 (S32-15) | 1.00/1.00 (S59-17) | Tails right; two layered 8-petal flowers drawn as petal ring + square star |
| A1-15 | 0.881 | no | yes | center | 0.881 | 0.69/0.80 | 1.00/0.82 (A12-15) | 1.00/0.82 (A12-15) | Single color right; center disc too small and sliver artifact |
| S52-19 | 0.899 | no | yes | head shape, tail geometry | 0.899 | 0.50/0.44 | 0.56/0.51 (S61-19) | 0.56/0.51 (S61-19) | White stripes lost (drawn as background gaps); striped bow loops drawn as one 6-star |
| S40-15 | 0.903 | yes | yes | head shape | 0.903 | 0.81/0.64 | 0.81/0.65 (A13-15) | 0.81/0.65 (A13-15) | Tails right; pinked disc + 8-star + ring drawn as ring + blobs |
| S10-19 | 0.904 | yes | yes | head shape | 0.904 | 0.60/0.52 | 0.79/0.54 (S57-17) | 0.79/0.54 (S57-17) | Tails right; bow loops + petals drawn as plain circle |
| S33-15 | 0.905 | yes | yes | head shape | 0.905 | 0.64/0.45 | 0.67/0.63 (S6-15) | 0.67/0.63 (S6-15) | Tails right; layered star close, pleated ring missing |
| A10-15B | 0.906 | yes | yes | head shape | 0.906 | 0.60/0.60 | 0.60/0.57 (S11a-15) | 0.60/0.57 (S11a-15) | Tails right; stacked pleated rings drawn as plain disc + blobs |
| S5-15 | 0.910 | yes | yes | head shape | 0.910 | 0.96/0.80 | 0.72/0.68 (S27-15) | 1.00/1.00 (S5-15) | Tails right; bow loops drawn as pinked disc + blocks |
| S6-15 | 0.919 | yes | yes | head shape | 0.919 | 0.73/0.74 | 1.00/0.93 (S33-15) | 1.00/1.00 (S6-15) | Tails right; layered stars close, pleated ring drawn as ring of fragments |
| A12-15 | 0.922 | yes | yes | head shape | 0.922 | 0.80/0.79 | 0.80/0.85 (s39-15) | 0.80/0.85 (s39-15) | Tails right; inner pleated ring missed (plain disc) |
| S61-19 | 0.924 | yes | NO | head shape | 0.924 | 0.90/0.77 | 0.50/0.51 (S48-17) | 1.00/1.00 (S61-19) | Tails right; outer black petal layer drawn as broken blobs; scores OK and is not flagged (silent) |
| S41-15 | 0.925 | yes | yes | head shape | 0.925 | 0.95/0.77 | 0.86/0.78 (S40-15) | 1.00/1.00 (S41-15) | Tails right; white 6-star drawn as blobs |
| S11-15 | 0.929 | yes | yes | head shape | 0.929 | 0.67/0.66 | 1.00/0.88 (s39-15) | 1.00/0.88 (s39-15) | Tails right; 12 star petals missed (ring instead) |
| S31-15 | 0.929 | yes | yes | head shape | 0.929 | 0.94/0.66 | 0.57/0.48 (S10-19) | 0.57/0.48 (S10-19) | Tails right; layered star + glitter loops roughly right |
| S45-15 | 0.933 | yes | yes | head shape | 0.933 | 0.84/0.65 | 0.90/0.77 (S8-15) | 1.00/1.00 (S45-15) | Tails right; star drawn as blocks, ring missing |
| S37-15 | 0.949 | yes | yes | head shape, center | 0.949 | 0.92/0.77 | 0.92/0.85 (S8-15) | 0.92/0.85 (S8-15) | Tails right; star points drawn as blocks, gold center missing |
| s34-15 | 0.951 | yes | yes | - | 0.951 | 0.81/0.66 | 0.64/0.55 (S33-15) | 0.64/0.55 (S33-15) | Close match |
| A15-17 | 0.954 | yes | yes | head shape | 0.954 | 0.68/0.62 | 0.81/0.70 (S43-15) | 1.00/1.00 (A15-17) | Tails right; black 6-star drawn as arrow fragments |
| S56-17 | 0.955 | yes | yes | head shape | 0.955 | 0.93/0.72 | 0.87/0.81 (S1-15) | 1.00/0.99 (S56-17B) | Tails right; bow loops drawn as 6-star + blobs |
| S70-19 | 0.956 | yes | yes | head shape | 0.956 | 0.74/0.60 | 0.62/0.62 (S41-15) | 0.62/0.62 (S41-15) | Tails right; star outline + glitter squares drawn as petals + blocks |
| S43-15 | 0.957 | yes | NO | head shape | 0.957 | 1.00/0.81 | 1.00/0.82 (S26-15) | 1.00/1.00 (S43-15) | Tails right; loops + petals drawn as blocks; not flagged |
| S35-19 | 0.958 | yes | yes | head shape | 0.958 | 0.82/0.61 | 0.87/0.67 (S31-15) | 0.87/0.67 (S31-15) | Tails right; loops + star roughly right |
| S42-15 | 0.960 | yes | NO | head shape | 0.960 | 0.84/0.84 | 1.00/0.93 (S40-15) | 1.00/1.00 (S42-15) | Tails right; petals + white star drawn as blocks; not flagged |
| S67-19 | 0.964 | yes | yes | head shape | 0.964 | 0.87/0.82 | 0.74/0.71 (A12-15) | 1.00/1.00 (S67-19) | Tails right; layered stars drawn as one star + petals |
| s39-15 | 0.966 | yes | yes | head shape | 0.966 | 1.00/0.80 | 0.95/0.76 (A14-17) | 1.00/1.00 (s39-15) | Tails right; white 6-star drawn as petals |
| S28-15 | 0.967 | yes | NO | - | 0.967 | 0.91/0.84 | 0.70/0.61 (A12-15) | 1.00/1.00 (S28-15) | Close match |
| A14-17 | 0.968 | yes | yes | head shape | 0.968 | 0.80/0.72 | 0.80/0.59 (A9-15) | 1.00/1.00 (A14-17) | Tails right; black 6-petal star drawn as diamonds + squares |
| S30-15 | 0.972 | yes | yes | head shape | 0.972 | 0.92/0.57 | 0.83/0.72 (S31-15) | 0.83/0.72 (S31-15) | Tails right; bow loops drawn as rays + diamonds |
| S56-17B | 0.972 | yes | NO | head shape | 0.972 | 0.91/0.66 | 0.87/0.81 (S1-15) | 1.00/1.00 (S56-17B) | Tails right; bow loops drawn as star + bars; not flagged |
| S80-22 | 0.979 | yes | yes | head shape | 0.979 | 0.96/0.78 | 0.67/0.47 (s34-15) | 1.00/1.00 (S80-22) | Tails right; layered star roughly right |
| S69-19 | 0.980 | yes | yes | head shape | 0.980 | 0.94/0.68 | 0.69/0.56 (S30-15) | 0.69/0.56 (S30-15) | Tails right; loops + star + ring drawn as star + blocks + ring fragments |
| A6-15 | 1.000 | yes | yes | - | 1.000 | 1.00/0.91 | 1.00/0.94 (A13-15) | 1.00/0.94 (A13-15) | Match |
| A9-15B | 1.000 | yes | yes | head shape | 1.000 | 1.00/0.76 | 1.00/0.74 (S49-17) | 1.00/1.00 (A9-15B) | Tails right; stacked pleated rings drawn as diamond ring, still scores 1.000 |
| S1-15 | 1.000 | yes | yes | head shape | 1.000 | 1.00/0.84 | 1.00/0.86 (S29-15) | 1.00/0.86 (S29-15) | Tails right; bow loops drawn as a ring, still scores 1.000 |
| S11a-15 | 1.000 | yes | NO | - | 1.000 | 0.92/0.70 | 1.00/0.83 (S41-15) | 1.00/0.83 (S41-15) | Close match |
| S27-15 | 1.000 | yes | yes | head shape | 1.000 | 1.00/0.75 | 1.00/0.93 (S5-15) | 1.00/0.93 (S5-15) | Tails right; bow loops drawn as square + blocks |
| S47-15 | 1.000 | yes | yes | - | 1.000 | 1.00/0.85 | 1.00/0.67 (A2-15) | 1.00/1.00 (S47-15) | Close match (petals as blocks) |
| S55-15 | 1.000 | yes | yes | head shape | 1.000 | 0.52/0.60 | 1.00/0.74 (A7-15) | 1.00/1.00 (S55-15) | Tails right; 5 white bow loops drawn as ring |
| S58-17 | 1.000 | yes | yes | - | 1.000 | 1.00/0.68 | 0.95/0.66 (S35-19) | 0.95/0.66 (S35-19) | Close match |

| class | IDs (all 76) | IDs among the 40 not OK |
|---|---|---|
| head shape | 68 | 38 |
| tail numbering | 18 | 18 |
| tail geometry | 5 | 5 |
| points count | 3 | 3 |
| center | 2 | 1 |
| designer differs from photo | 7 | 7 |
| other | 0 | 0 |
| none | 6 | 0 |

Class counts overlap (an ID can have several). "IDs among not OK" counts only the 40 rows under
the bar. Share of the lost agreement (sum of 1 - match over 76 = 23.6 rosette-equivalents):
- tail numbering: 10.1 (43%) is recovered by the out-to-in rule alone (simulated, section 3);
- designer differs from photo: 4.6 (19%) remains on 7 IDs whatever we do;
- tail geometry: 1.7 (7%) remains on 5 IDs;
- head shape, points, center and residual numbering misses (asymmetric A-series): 7.0 (30%).
  The color_agreement metric under-weights the head (it is small next to the tails): 30 of
  the 36 "OK" rows still have a visibly wrong head, e.g. S1-15, S27-15 and A9-15B score 1.000
  with bow loops or stacked pleated rings drawn as a ring, square or diamond ring.

Silent rows (wrong but not flagged): A13-15 (0.329, designer differs), S46-15 (0.741, 6-point
star where the rosette has 8). S61-19 now scores 0.924 (OK) because the old renderer dropped all
its designer Color 3 parts; its tails are right but the outer black petal layer is drawn as broken
blobs and nothing flags it. S42-15, S43-15, S56-17B and S28-15 are also unflagged with similar
partial head misses. The flagging has no head-shape signal at all.

## 3. Class 2: tail color numbering

Not geometry. In every case reported (S36-15B, S32-15, S3-15) and in S1-15B, S18-15, S2-15, S9-19,
S36-15, S4-15, S65-18, S48-17, S49-17, S63-17, S68-19, our stripe geometry and colors match the
photo; only which photo color is called Color 1/2/3 differs, so the designer file, filled with our
per-category colors, shows them on the other stripes.

**Designer rule, measured on all 79 designer SVGs** (rendered tail rows at 35/50/65% of the
tail length, categories met walking from each outer edge to the axis, runs under 3 px ignored):
Colors are numbered **from the outer tail edge inward**: Color 1 = outermost stripe, Color 2 =
next new color inward, Color 3 = next, Color 4 = next; colors that only appear in the head come
after. A color may repeat further in (1-2-1, 1-2-3-1, 1-2-3-4-1 all occur).
- exact on 73/79 files; Color 1 = outermost stripe on 77/79;
- 2-color files 24/25 exact, 3-color 48/53, the one 4-color file (S68-19) exact;
- exceptions: S22-15 and S29-15 (center stripe Color 2, middle stripes Color 3), A10-15,
  A13-15B and A7-15 (asymmetric "flag" tails, see below), D1 (not a product).
- The asymmetric A-series (A2-15, A3-15, A7-15, A9-15, A10-15, A13-15, A13-15B: left, center and
  right tails each a Color) follows no left-to-right rule (left-to-right Colors: 2,3,1 / 1,3,2 /
  2,1,3 / 1,3,2 / 3,2,1 / 1,2 / 1,2,3). Their numbering is not derivable from the photo.

**Why ours differs.** `rosette/assign.py` scores every Color mapping with
P(category | role) mined by `tools/mine_rules.py` from the 35 canonical refs only. Those are
mostly 3-color, where the inner stripe is Color 3, so `tail_inner` has P(1)=0.39, P(2)=0.08,
P(3)=0.50; `tail_outer` P(1)=0.69, P(2)=0.26. Renormalized over Colors 1-2 for a 2-color rosette,
outer=1/inner=2 scores log 0.69 + log 0.08 = -2.93 per unit area and outer=2/inner=1 scores
log 0.26 + log 0.39 = -2.29, so the assigner picks the reverse of the designer every time a
2-color rosette has a distinct center stripe. The miner is not conditioned on the number of
colors, and 1-2-1 files (S11-15, S15-15, S42-15, S6-15, S18-15 count their center stripe as
"inner = 1") push the same way. For 3-color rosettes the same prior swaps Colors 2 and 3 whenever
the area evidence is close (S48-17, S49-17, S63-17).

**Proposed rule (not shipped).** Number the used groups by first appearance walking the detected
tail stripes from the outside in (by |stripe_dx|, ignoring sliver stripes narrower than 35% of the
median stripe width), then head-only groups by area; keep the pattern rule as a check (a pattern
stripe that lands on a Color other than 2 is flagged, not forced). Keep the mined prior only as a
tie-break for rosettes without symmetric tails. Simulated on the 76 photos (scratch, our own
detections, designer files as truth):
- mean color_agreement 0.689 -> 0.825; IDs at match >= 0.9: 36 -> 45;
- 17 better, 2 worse: A10-15 0.337 -> 0.000 (asymmetric tails, rule does not apply) and S15-15
  0.468 -> 0.153 (our detection split the red into two groups, so the false group takes Color 2).
  Restricting the rule to symmetric tails removes the A10-15 loss; S15-15 needs the color split
  fixed first. Without the sliver filter the rule also broke S31-15 (0.929 -> 0.020) and S13-15
  (0.840 -> 0.418): a thin sliver stripe on the outer edge became Color 1. The filter is required.
- Not fixed by it: S29-15 / S22-15 (designer exception), the A-series, S2-15B (pattern stripe
  merged into white, only 2 groups found).

## 4. Head template library prototype

`tools/head_templates.py` (not wired into the pipeline). Each of the 77 product designer SVGs
(TEST and D1 left out) gives a template: every non-tail color piece (rosette.roles
classify_components) with its category, in paint order, normalized to the head circle that
`prepare` finds on the rendered file. A photo's head (color-group labels from `pixel_groups`,
normalized to the same `prepare` head circle) is compared on a 128x128 grid in head-radius units,
rows above 0.55 r below the center (tails cover the rest): one-to-one category-to-group partition
agreement (Hungarian on overlap), silhouette IoU, outer radius profile difference, lobe count.
The best template is placed on the photo head circle over our tails; its categories reach a Color
through the photo group they overlap.

Head agreement is measured like color_agreement but only over designer head pieces, with the
generated raster aligned to the designer by head circle (bounding-box alignment would mostly
measure the designer drawing the head larger than the photo). Means over 76 IDs:

| head output | geo (per-part rule, numbering ignored) | pix (per pixel) | pipe (pipeline numbering) |
|---|---|---|---|
| current detection | 0.758 | 0.640 | 0.629 |
| template, leave-one-out (design and variants removed) | 0.795 | 0.672 | 0.620 |
| template, full library | 0.886 | 0.836 | 0.671 |
| own designer head placed (sanity) | 0.980 | 0.979 | 0.720 |
| best possible LOO template (oracle pick) | 0.939 | 0.782 | - |

- LOO vs current: better on 33, worse on 23 (geo); better on 46, worse on 30 (pix). A gain, but
  small and not uniform: it helps where we draw a disc for loops or stars (S55-15 0.52 -> 1.00,
  S11-15 0.67 -> 1.00, A16-17 0.43 -> 0.74, S13-15 0.59 -> 0.92) and hurts where our detection was
  already close (S61-19 0.90 -> 0.50, S31-15 0.94 -> 0.57, S80-22 0.96 -> 0.67).
- The matcher is the weak part: with the full library it picks the photo's own design for only
  41/76, and LOO it reaches 0.795 of a possible 0.939. Partition agreement alone is the best
  single feature (LOO geo 0.821, pix 0.697); IoU, profile and lobes alone are 0.68-0.74 geo.
  Feature weights were not tuned; a weight grid was swept only to see the spread (in-sample,
  not a result).
- 'pipe' stays near 0.62-0.72 for every head source including the designer's own head: with the
  current numbering a perfect head still gets wrong Colors. Section 3 comes first.
- Whole rosette (bounding-box alignment as score.py, geo): current 0.869, LOO 0.876, full 0.899;
  the head is a small share of the rosette, as noted in section 2.

## 5. Recommendation for the next round

1. **Numbering by tails, out to in** (section 3). Smallest change, largest measured gain
   (+0.136 mean, +9 IDs over the bar, 43% of the lost agreement). Symmetric tails only, sliver
   filter, pattern rule becomes a flag. Keep the miner for the tie-break and condition it on the
   number of colors.
2. **Flag heads we cannot draw.** Every silent row is a head miss. A cheap check: compare the
   photo head segmentation with our rendered head on the same grid (the prototype's partition
   feature, against our own output instead of a template); low agreement flags the head. This
   turns silent-wrong into flagged-wrong before any head drawing improves.
3. **Head shapes from the library, as a suggestion.** Offer the top 3 templates (partition-ranked)
   in the Workbench for the user to pick, rather than auto-replacing: the LOO auto-pick gains
   only +0.03-0.04 and loses on 30 IDs. A human picking among a few candidates can approach the
   0.94 oracle. A better matcher (per-layer radius profiles, lobe count per color layer) is a
   research item, not a next-round fix.
4. **Tail geometry**: center stripe lost when two tails overlap (S65-18), white stripe taken as
   background (S52-19), one color split in two (S15-15), patterned stripe merged (S2-15B),
   stripes misplaced (S53-19). 5 IDs, 7% of the loss.
5. **Make the metric see heads**: report head agreement next to color_agreement in
   tools/score.py, or heads keep regressing unseen.

## 6. Realistic ceiling

- 7 of 76 photos differ from their designer SVG in ways no detector can recover: the A-series
  asymmetric 3-Color tails over symmetric photos (A2-15, A7-15, A9-15), a 2-Color design over a
  one-color photo (A13-15), a separate Color 3 center stripe the photo does not have (S8-15),
  Colors 2/3 placed against the photo's pattern (S29-15), much narrower outer tails (s44-15).
  Asymmetric-tail numbering (A10-15, A13-15B) is not derivable either. That caps the color bar at
  about 67-69 of 76 (88-91%).
- With numbering fixed, 45 of the 69 recoverable IDs pass the current bar; the remaining gap is
  mostly heads (layered stars, bow loops, stacked pleated rings, petals), which the current
  parametric head (disc, rings, one points ring, center) cannot express. Even a perfect pick from
  a library of other designs reaches 0.94 geo / 0.78 pix on heads, because the designer's head
  for a given rosette is often unique in the library.
- Expected after steps 1-2: about 45/76 OK by the current metric, no silent heads. Steps 3-4 could
  reach the mid 50s; the 67-69 ceiling needs heads close to the designer's own, which in practice
  means a human choosing or adjusting the head.

## Reproduce

- Comparison sheets: `.venv/Scripts/python.exe <SCRATCH>/compare_all.py "<repo>" <SCRATCH>/compare`
- Template prototype: `.venv/Scripts/python.exe -m tools.head_templates --cache <DIR>` (tens of
  minutes cold; per-ID lines and the totals above)
- The out-to-in simulation and the designer numbering census were scratch scripts (renumber2.py,
  outin2.py in the session scratch directory); the rule is fully described above.
