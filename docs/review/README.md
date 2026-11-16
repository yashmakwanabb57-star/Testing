# Photo-by-photo review (how we work)

We go through the 76 photos one at a time, in the order in `photo-order.txt`. For each photo
we compare our generated SVG with the designer's SVG and fix the generator until the owner
approves. Never copy the designer SVG: every fix is a rule that reads the photo (designer files
only give proportions for parts the photo cannot show, for example hidden star valleys).

## Progress

- Owner-approved: lines 1-30 of `photo-order.txt` (A10-15 ... S27-15). Each approval is a commit.
- Done by Claude on its own (the owner asked it to continue alone; owner to review later):
  - 31 S28-15 - R48: star on four loops, a square of the loops' color over its base, loops edged.
- Next: line 32.
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
   proportions cited). Latest rule: R47 (`rosette/pipeline.py` `_star_as_loops`).
6. Before showing a fix: check which of the 76 photos the rule fires on, and compare scores
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
