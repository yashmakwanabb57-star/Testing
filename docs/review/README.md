# Photo-by-photo review (how we work)

We go through the 76 photos one at a time, in the order in `photo-order.txt`. For each photo
we compare our generated SVG with the designer's SVG and fix the generator until the owner
approves. Never copy the designer SVG: every fix is a rule that reads the photo (designer files
only give proportions for parts the photo cannot show, for example hidden star valleys).

## Progress

- Owner-approved: lines 1-55 of `photo-order.txt` (A10-15 ... S49-17). Each approval is a commit.
- Done by Claude on its own (the owner asked it to continue alone; owner to review later):
  - 31 S28-15 - owner-approved after a fix: loops run in to the center, order loops, star, square (R48).
  - 32 S29-15 - owner-approved after a fix: a pinked ring of the inner star's color round the badge
    (R54: pleats read as light and shade). Colors 2 and 3 still swapped vs the designer (left as is).
  - 33 S30-15 (owner-approved, app run checked too) - diamonds between edged loops are a star behind them, the loops run in under the ring,
    no library head (R50; also S35-19, unchanged 0.963). 0.972 -> 0.979.
  - 34 S31-15 (owner-approved as is; the flower is smaller and plainer than the designer's two petal rings) - loops on a star around a bare center: both run in under the center, loops edged in the
    center's color, a 6-point flower of the star's color over the loops' bases (R51; only S31-15). 0.929 -> 0.933.
  - 35 S3-15, 36 S32-15, 37 S33-15 - already match the designer (1.000), no change. S32-15 approved as is
    (its white star is drawn over the red one; the photo and the designer have it behind). S33-15 approved
    after a fix: R42's long star goes under the other stars and is kept in the app (also S15-15B's app run).
  - 38 s34-15 - owner-approved after a fix: the black kites go between the lavender edge star and the white
    star, and the white star is the edge star scaled down (R55; only s34-15). 0.956 -> 0.961.
  - 39 S35-19 - structure matches the designer, no change (owner-approved).
  - 40 S36-15 - owner-approved after a fix: a one-color lobed head round a rose is a petal flower read from
    the outline round the rose, petals and rose in the designer's proportions (R56, geometry FLOWER_PROFILE;
    only S36-15). Head 0.846 -> 0.883; the app now draws it too (no library head).
  - 41 S36-15B - owner-approved after a fix: petals alternating two colors round a rose joined to one set
    are two petal sets half a step apart, the rose's color underneath (R57, the two-color branch of R56;
    only S36-15 and S36-15B). Head 0.803 -> 0.822; color 0.778 -> 0.713 because the designer's head
    colors disagree with its own tails (ours follow the photo).
  - 42 S37-15 - owner-approved after a fix: glitter points standing outside a pleated ring of the head's color
    are a star under a satin star of the head's color half a step off, the ring 0.6 of the head, the
    badge's gold rim the rose (R58; also S37-15's app run). 0.949 -> 0.957. R58 also fires on S8-15 (75,
    open): loops -> glitter star under a red star, 0.640 -> 0.633, head 0.841 -> 0.840.
  - 43 S38-15 - owner-approved after a fix: one group holding a dark satin star and lighter pleats is split
    by hue: pleats and rose the lighter group, a leaf star of the darker read from its hue runs, the loops
    run in under it edged in its color and end at 0.84 of the head (R59; only S38-15). 0.865 -> 0.901.
  - 44 s39-15 - owner-approved after a fix: a star under a shorter star half a step off takes its tips from
    its color runs; both are hexagrams with a smaller copy of each on top (R60; only s39-15). 0.951 -> 0.970.
  - 46 S41-15 - owner-approved after a fix: three stars of one count (two on one axis, one of the head's color,
    one half a step off) stack as the designer: other color, half-step, head color, hexagrams, upright, no
    ring (R62; only S41-15). 0.921 -> 0.943.
  - 48 S42-15 - owner-approved after a fix: star tips round a ring of the head's color are a hexagram under leaf
    petals of the head's color (geometry LEAF_PETAL_PROFILE), the rose the star's color (R64; only S42-15). 0.984 -> 1.000.
  - 49 S43-15 - owner-approved after a fix: loops round a ring of the head's color run in under leaf petals of the
    head's color, edged in the rose's color (R65, R64's cousin; only S43-15). 1.000 (structure now as the designer).
  - 
    56 S51-19 (0.905), 59-74 except 63/65/71 (0.96-1.00), 76 S9-19 (0.971) - structure close to the
    designer, no change.
  - 55 S49-17 - owner-approved after a fix: two twelve-point layers round a ring of the head's color are plain
    points at the rim, a pleated ring from the ruffle out, a leaf star on it (ROUND_COVER), the ruffle, the rose
    (R71; only S49-17). 0.897 -> 1.000.
  - 54 S48-17 - owner-approved after a fix: three four-point layers 30 degrees apart are the designer's stack:
    stepped squares, split arms with scaled bands (geometry SPLIT_ARM, traced), an eight-point star, the rose
    (R70; only S48-17). 0.672 -> 0.848.
  - 53 S47-15 - owner-approved after a fix: loops narrowing to points round a rose are satin petals (geometry
    SATIN_FRONT_PROFILE, 21 points), three in front of three, the rose 0.325 (R69; only S47-15). 1.000.
  - 45 S40-15 - owner-approved after a fix: glitter points between satin points of a third color: the outer
    pleats are the head disc, the satin star's color read from the hue between the glitter points, the glitter
    star on it, the pleated ring on both, the badge the satin color (R61; only S40-15). 0.893 -> 0.993.
  - 47 S4-15 - owner-approved after a fix: two close purples: dark loops read from lightness at the rim, wide
    blocks under a light ring and light loops half a step off, a dark rose (R63; only S4-15). 0.908 -> 0.980.
  - 50 s44-15 - owner-approved after a fix: four-point layers of one color on the axes and another on the diagonals
    are the designer's banded cross of stars, an eight-point star of the head's color inside (R66; only s44-15).
    Head as the designer; color 0.526 -> 0.504 (the tails: the photo's outer stripes are wider than the designer's).
  - 51 S45-15 - owner-approved after a fix: an eight-point star round a ring of a third color is the designer's
    stack: star, pleated ring, ruffle of the star's color, small star of the ring's color, rose (R67; only S45-15).
    0.935 -> 0.901 (against the designer's own reading 0.935 -> 0.956).
  - 52 S46-15 - owner-approved after a fix: a star with loops half a step off round a rose of the head's color is
    the designer's outer star, loops edged in the head's color, inner star, rose (R68, five or more points;
    only S46-15). Head as the designer; 0.725 -> 0.715.
  - 73 S70-19 - owner-approved after a fix: six loops between chevron stars (outer and inner of one color, the
    head's color between) round a pleated ring are the designer's stack: stars upright, loops edged in the head's
    color, the designer's radii (R72; only S70-19). 0.980 -> 0.993.
  - 72 S69-19 - owner-approved after a fix: six loops round chevron stars and a top star, no head disc, are the
    designer's stack: back and middle star, loops edged in the middle color, the top star over the loops, upright
    (R73; only S69-19). 0.972 -> 0.982, head 0.780 -> 0.790.
  - 71 S68-19 - owner-approved after a fix: a head read as a plain disc whose colors swing four times (the head's,
    the diagonals), twice (one axis), eight times (glitter pairs) is the designer's folded points: each point one V
    fold moved along the axis (new `chevron` kind), glitter pairs under the diagonals, the axes on top, a badge;
    R40 skips parts built whole (R74; only S68-19). Color 0.966 -> 1.000.
  - 70 S67-19 - owner-approved after a fix: a six-point star over three points half a step off, on a head round a
    center of its color, is the designer's split star (three points of that color, three of the head's) under
    the star, each edged; the owner's order: red, yellow, blue, then the inner lines red, yellow, blue (R75;
    only S67-19). 0.964 -> 0.960.
  - 71 S63-17 - owner-approved after a fix: an eight-point layer and a four-point layer leaning 20 degrees off
    the axes, three colors, are the designer's side folds: whole bands of the head's color and the second color,
    chevron corners on the diagonals with no gaps, a flower of petals, hexagon dots with a rose inside each, the
    center (R76; only S63-17). Color 1.000.
  - 72 S58-17 - owner-approved after a fix: two six-point bow layers half a step apart and a star in phase
    with one, no head disc, are the designer's loops between star points: the star tip up, straight-sided
    loops on a wider edge of the third color, the ruffle of that color, flat-sided hexagon roses on the points,
    the center with the rose (R77; only S58-17). Color 1.000.
  - Open (need new reading, not done):
    57 S5-15 (cross loops), 58 S52-19 (layered loops), 63 S57-17 (0.857),
    65 S59-17 (0.876), 75 S8-15 (star around the rose read as
    loops, 0.640).
- Owner rework of approved photos: A10-15B (line 2) - pleated rings of three colors round a rose (R78; only
  A10-15B). All pleated edges are now a smooth wave (geometry SCALLOP_*, as the designer draws them; the 34
  scored photos move by 0.01 at most), and the rose fills its circle with thin lines (geometry rose.py).
  A12-15 (line 4) - a blob of a second color at the badge is a pleated ring round it (R79; only A12-15).
  0.922 -> 1.000.
- Next (owner, at the office): verify Claude's work from line 56 (S51-19) one photo at a time with the
  normal loop below (show photo | ours | designer, wait for "works next" or a fix request). Then the open
  list above, worst first (S8-15, s44-15, S48-17, S68-19, S4-15).
- Update this section after each photo.

## Loop for one photo

1. Render: `.venv/Scripts/python.exe tools/review_show.py <ID>` -> `output/review/cur.png`
   (photo | ours | designer) and `output/review/z.png` (our head, close up). Open it for the owner.
2. Score: `.venv/Scripts/python.exe tools/score.py <ID>` (2nd column = color agreement, 3rd = head shape).
3. Describe the differences in plain words (designer colors: red = Color 1, blue = 2,
   yellow = 3) and propose a fix. Wait for the owner's answer.
4. Owner says "works next" -> commit, then the next photo. "not works fix" or a specific
   remark -> fix, re-render, show again.
5. Fix as a narrow, numbered rule (R-number in a comment, constants named, designer
   proportions cited). Latest rule: R79 (`rosette/pipeline.py` `_ring_round_badge`).
6. Before showing a fix: check which of the 76 photos the rule fires on
   (`.venv/Scripts/python.exe tools/review_fire.py <rule function>` or `--part <part id>`), and compare scores
   before/after (`git stash` baseline) for those photos. Narrow the rule until no other photo
   gets worse. Tell the owner what else changed.
7. Run `.venv/Scripts/python.exe -m pytest -q -m "not slow"` on touched areas before committing.
8. Commit message: `fix: <ID> - <what the rule does> (R<n>, <ID> <old> -> <new>; only <ID> over the 76)`.
   No AI attribution (CLAUDE.md).

## Useful

- Designer pieces in head radii: `rosette.vision.templates.extract_template(sid, svg)` -> pieces
  with `category`, `role`, `rings` (polygons). Good for measuring radii and angles.
- Photo colour groups along rays: `pixel_groups(prepare(bytes))` labels, sampled around `p.head`
  with `Frame.of(p).scale` (see `rosette/vision/stacked_stars.py` for an example).
- Library heads are skipped when a part has `params["tips"]`.
