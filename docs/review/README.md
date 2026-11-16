# Photo-by-photo review (how we work)

We go through the 76 photos one at a time, in the order in `photo-order.txt`. For each photo
we compare our generated SVG with the designer's SVG and fix the generator until the owner
approves. Never copy the designer SVG: every fix is a rule that reads the photo (designer files
only give proportions for parts the photo cannot show, for example hidden star valleys).

## Progress

- Owner-approved: lines 1-44 of `photo-order.txt` (A10-15 ... s39-15). Each approval is a commit.
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
  - 46 S41-15 (0.921), 48 S42-15 (0.968), 49 S43-15 (1.000), 55 S49-17 (0.897),
    56 S51-19 (0.905), 59-74 except 63/65/71 (0.96-1.00), 76 S9-19 (0.971) - structure close to the
    designer, no change.
  - 53 S47-15 - a small blunt star with no center is the rose: a center of its color, the loops under it
    (R53; only S47-15). Was a yellow 6-point star in the middle; head 0.767 -> 0.770.
  - Open (need new reading, not done): 45 S40-15 (light orange star not separated from the orange pleats),
    47 S4-15 (two purples too close: drawn as one disc), 50 s44-15 (layered black/gold star, 0.526),
    51 S45-15 (star read as square loops), 52 S46-15 (petal flower where the designer has a leaf star),
    54 S48-17 (stacked stars, 0.672), 57 S5-15 (cross loops), 58 S52-19 (layered loops), 63 S57-17 (0.857),
    65 S59-17 (0.876), 71 S68-19 (head drawn as a plain disc), 75 S8-15 (star around the rose read as
    loops, 0.640).
- Next (owner, at the office): verify Claude's work from line 45 (S40-15) one photo at a time with the
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
   proportions cited). Latest rule: R60 (`rosette/pipeline.py` `_under_star_tips`).
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
