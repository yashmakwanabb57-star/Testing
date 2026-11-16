# Photo-by-photo review (how we work)

We go through the 76 photos one at a time, in the order in `photo-order.txt`. For each photo
we compare our generated SVG with the designer's SVG and fix the generator until the owner
approves. Never copy the designer SVG: every fix is a rule that reads the photo (designer files
only give proportions for parts the photo cannot show, for example hidden star valleys).

## Progress

- Owner-approved: lines 1-32 of `photo-order.txt` (A10-15 ... S29-15). Each approval is a commit.
- Done by Claude on its own (the owner asked it to continue alone; owner to review later):
  - 31 S28-15 - owner-approved after a fix: loops run in to the center, order loops, star, square (R48).
  - 32 S29-15 - owner-approved after a fix: a pinked ring of the inner star's color round the badge
    (R54: pleats read as light and shade). Colors 2 and 3 still swapped vs the designer (left as is).
  - 33 S30-15 - diamonds between edged loops are a star behind them, the loops run in under the ring,
    no library head (R50; also S35-19, unchanged 0.963). 0.972 -> 0.979.
  - 34 S31-15 - loops on a star around a bare center: both run in under the center, loops edged in the
    center's color, a 6-point flower of the star's color over the loops' bases (R51; only S31-15). 0.929 -> 0.933.
  - 35 S3-15, 36 S32-15, 37 S33-15 - already match the designer (1.000), no change.
  - 38 s34-15, 39 S35-19, 40 S36-15 - structure matches the designer (0.956 / 0.962 / 1.000), no change.
    Small differences left: s34-15's dark points are longer than the designer's; S36-15's loop ends are
    square where the designer's are pointed petals.
  - 41 S36-15B - library head (S1-15B) is right in shape, but its red/white loop order is one step off
    the photo's (0.778); the rules head reads it worse. Left as is.
  - 42 S37-15 - the review run (design left out of the learned classifier) reads the glitter points as
    loops; with the full classifier (the shop's run) they are a star, as the designer draws. Left as is.
  - 43 S38-15 - colors read lighter than the designer's slots (orange pleats as Color 3); the star on top
    is missing. Open (0.865).
  - 44 s39-15 (0.951), 46 S41-15 (0.921), 48 S42-15 (0.968), 49 S43-15 (1.000), 55 S49-17 (0.897),
    56 S51-19 (0.905), 59-74 except 63/65/71 (0.96-1.00), 76 S9-19 (0.971) - structure close to the
    designer, no change.
  - 53 S47-15 - a small blunt star with no center is the rose: a center of its color, the loops under it
    (R53; only S47-15). Was a yellow 6-point star in the middle; head 0.767 -> 0.770.
  - Open (need new reading, not done): 45 S40-15 (light orange star not separated from the orange pleats),
    47 S4-15 (two purples too close: drawn as one disc), 50 s44-15 (layered black/gold star, 0.526),
    51 S45-15 (star read as square loops), 52 S46-15 (petal flower where the designer has a leaf star),
    54 S48-17 (stacked stars, 0.672), 57 S5-15 (cross loops), 58 S52-19 (layered loops), 63 S57-17 (0.857),
    65 S59-17 (0.876), 71 S68-19 (head drawn as a plain disc), 75 S8-15 (star around the rose read as
    loops, 0.640), and 43 S38-15 above.
- Next (owner, at the office): verify Claude's work from line 33 (S30-15) one photo at a time with the
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
   proportions cited). Latest rule: R54 (`rosette/pipeline.py` `_pleats_round_badge`).
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
