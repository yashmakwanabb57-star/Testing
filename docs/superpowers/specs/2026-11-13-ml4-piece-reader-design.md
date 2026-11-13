# ML-4: pieces named by a local vision-language model, counts from the signal, the render as the judge (design)

Date: 2026-11-13. Status: draft for the user's review; the measurements it rests on are the 52-photo probes of
2026-11-13 (session scratch `vlm/out_q4/table.md`, `vlm/marks_q4/table.md`).

## Intent

Eighteen rules rounds fixed one to three photos each; 36 of the 52 "not working" heads are still wrong, and the
causes left are structural (nested same-axis stacks, layers in the disc's colour, stacking order, badge versus rose,
loop heads without a disc). A vision-language model asked for the whole layer list (Qwen3.5-4B, free-form JSON)
miscounts most layers, invents repeated rings and breaks on a quarter of the photos. The same model asked one closed
question per SAM 2.1 piece (kind, texture, repeats) and one per touching pair (which lies on top) answers every time
and names the pieces right nearly everywhere, but still miscounts where each arm is its own piece. So the division of
labour is: pieces from SAM 2.1 and the rules, names and local order from the model, counts from the repeat signal, and
the final choice between candidates made by rendering each and comparing with the photo.

## Goals

1. A piece reader: for every head piece, kind (pleated disc, pleated ring, star arm, petal, loop, block, chevron, plain
   ring, badge, rose), texture and the model's local order against its touching neighbours, from a local model, cached
   per photo, with every answer stored as evidence (never as a silent decision).
2. Layers from piece groups: same-kind pieces on one radial band form a repeat ring; its count comes from the number of
   visible pieces completed by the strongest harmonic of the group's angular profile over the full circle (the R18
   arm-set reading, generalised to every kind and colour), never from the model's count alone.
3. Candidates and a judge: every decision the rules or the model leave uncertain (count k against 2k or k - 1, order of
   a pair, nested stack against a single star, badge against rose, loop head against disc) becomes a short list of
   candidate heads; each is rendered with the existing builder and compared with the photo inside the head mask on
   colour classes, edges and piece masks; the closest candidate wins and the margin becomes its confidence.
4. The scoreboards do not go down with the reader on; the reader off reproduces today's output byte for byte.

## Non-goals (this round)

- Fine-tuning the model (synthetic renders, a rented GPU): a later round, only if the measured reader is close but
  short.
- Tails, badge print reading, decorative small roses.
- Replacing the builder or the shop contract.

## Constraints (binding)

- No paid service, no cloud API, no network call at run time. Weights fetched once into `models/` (git-ignored).
- Model: Qwen3.5-4B, Apache-2.0 (official card: license apache-2.0, LICENSE file in the repo), GGUF from
  `ggml-org/Qwen3.5-4B-GGUF` (Q8_0 weights 4.48 GB, projector Q8_0 367 MB). Runtime: llama.cpp (MIT) `llama-server`
  on 127.0.0.1 only, started and stopped by the tool, thinking off (`--reasoning-budget 0`, `enable_thinking: false`),
  answers constrained by a JSON schema (enums for kind and texture, integer bounds for counts).
- Machine: 16 GB RAM, 16 threads, no usable GPU. Only one model process at a time; SAM 2.1 pieces come from the cache
  first and the reader never runs while a scoring run or a page rebuild runs.
- Python 3.14 in `.venv`; every package pinned. llama.cpp is an executable under `models/llama-cpp/` that the user
  installs (the setup README names the release and the checksum); the tool reports `reader: missing` with the install
  line when it is absent.
- Timing: the reader adds at most 4 minutes per photo on this CPU (cached afterwards); the UI shows it as one line.
- Settings: `ROSETTE_READER` (`on` | `off`, default `on` when the weights and the executable exist, else `off`),
  `ROSETTE_READER_PORT` (default 8089), `ROSETTE_READER_THREADS`.

## Decisions (made here)

- **Marks image.** The head crop at 768 px with each chosen piece's outline in its own colour and its number in a
  filled circle at the piece's interior point (`pieces._interior_point`). Chosen pieces: head pieces of at least
  0.4 % of the head area, largest first, at most 12 per image; when a photo has more, a second image carries the
  rest (the model answers per image; marks are global ids).
- **Questions.** Q1 per image: for every mark, kind (enum), texture (plain, glitter, patterned), repeats (integer 1-16:
  the model's count of identical pieces around the centre, stored as evidence only). Q2 per image: for every touching
  pair (masks overlap after a 3-pixel dilation; at most 12 pairs, largest contact first), which lies on top (a, b or
  unclear). Compact JSON, temperature 0.
- **Cache.** `.cache/reader/<key>.json` where key is the pieces cache key plus the prompt version; the comparison pages
  and the scorer never re-run the model for an unchanged photo.
- **Piece groups.** Pieces of one kind whose radial spans (2nd and 98th percentile radius) agree within
  `MIN_ZONE_UNITS` form a group; a group's count is the number of pieces when their angular steps fit one equal step
  (`_fit_steps`) with no gap, else the strongest harmonic of the group's union over the full circle with the R18
  n - 1 rule; a group of blocks, petals, loops or star arms becomes one points layer (kind from the group, rotation from
  the pieces' angles, radii from the union's polar profile, `source: "reader"`); a disc, ring, badge or rose piece
  becomes that part. Rules layers that a group explains are replaced; rules layers no group explains stay, flagged
  when the model named their pieces differently.
- **Candidates.** The candidate list for a head is the cross product of at most four open decisions (each with 2 to 3
  options), at most 24 heads; an open decision is one where the group count and the model's repeats disagree, where
  the pair order is unclear or contradicts the H12 vote, where a group could be a nested stack or a single star
  (R19's axis runs say so), or where the centre piece is badge or rose. The judge renders each candidate with
  `rosette.build` and `tools.rasterize` at the photo's head frame and scores it against the photo: colour-class
  agreement inside the head mask (the labels), edge agreement (Canny of the render against Canny of the photo, within
  3 px), and piece-mask agreement (each piece's mask against the candidate's part that covers it). The winner's
  margin over the runner-up, scaled, is the confidence; a margin under `JUDGE_MARGIN` keeps the rules' choice and
  flags the part.
- **Off path.** `ROSETTE_READER=off` or missing weights: the pipeline runs exactly as at ad56d5a (byte-identical
  output on the 76).

## Acceptance

- Reader on, new image: PASS >= 21/34, color >= 0.855, silently wrong <= 1; per-ID layer IoU changes listed, drops
  noted; the 52 head zooms (photo | ours | ours outline | reference outline) looked at, and the count of IDs that match
  or nearly match the reference outline rises from 13 by at least 10, with the per-ID before and after rows.
- Reader off: byte-identical to ad56d5a on the 76.
- Timing: reader under 4 minutes per photo uncached on this machine; cached analysis within the current timing test.
- Tests: the group counting, the candidate enumeration, the judge's scoring and the off path each have tests that run
  without the model (recorded answers as fixtures).

## Builder check: rulings (2026-11-13)

A fresh builder read the draft and asked eight questions; the answers are decisions of this document.

1. **The rules head is always candidate 0.** Reader groups never replace a rules layer directly; they produce
   candidates. The judge chooses among candidate 0 (today's output at ad56d5a) and the group candidates; a margin under
   `JUDGE_MARGIN` returns candidate 0 with a flag. "Reader off is byte-identical" and "scoreboards do not go down" are
   both tested against candidate 0.
2. **Kinds map onto today's parts.** star arm -> points `star`; petal -> points `petal`; loop -> points `bow`; block ->
   points `diamond` (today's blocks); chevron -> the R19 nested star stack on the host's axes; pleated disc and pleated
   ring -> the disc and `ring` parts; plain ring -> `ring` unpinked; badge and rose -> the `center` part with its
   badge/rose state. A loop head without a disc (S1-15, S4-15, S5-15) is not drawable today and stays out of this
   round, listed with its numbers; no new part type is added.
3. **The judge is calibrated against the photo, never the reference.** Its weights (colour classes, edges, piece masks)
   are mined so that candidate 0 wins on the 24 heads that score OK today and on the 13 of the 52 judged a match by
   eye, leave-one-out over those 37; the references are not used. Acceptance stays on the 34-ID scoreboard and the 52
   zooms as today.
4. **Rendering and budget.** Candidates are built with `rosette.build` and rasterized with `tools.rasterize` at the
   photo's head frame (centre, radius from `prepare`) at 400 px; colour-class agreement is on class ids (the photo's
   labels against the render's category map, as `tools/score.py` does), edges by Canny within 3 px, piece masks by
   IoU of each piece against the covering part; pixels outside the head mask and the tails are ignored. The reader's
   two questions and the judge's renders together fit the 4-minute budget: at most 24 candidates at 400 px.
5. **The model's order is evidence into the vote.** Where the H12 vote is undecided, a confident model answer (a or b)
   decides the pair as a vote at `STACK_SHARE`; where the vote and the model disagree, the pair is an open decision for
   the judge; where they agree, nothing changes. Entry point: `stack_votes` edges, same shape as the piece vote of R19.
6. **Count alternatives come from the signal.** A group's count k is the piece count when `_fit_steps` fits one equal
   step with no gap, else the strongest harmonic over the full circle with the R18 n - 1 rule, now for every kind.
   Alternatives for the judge: the model's `repeats` when it differs from k, and the harmonic's second peak when it is
   2k or k/2 within `HARMONIC_SECOND` of the first; at most three options. Photos that pass today are protected by
   ruling 1 (candidate 0 wins unless the judge's margin says otherwise).
7. **Runtime details.** llama.cpp release b12342 (Windows CPU x64 build, verified on this machine with the Q4_K_M
   GGUF, the Q8_0 projector and 768 px images); the README names that release and its SHA-256. The per-mark schema is
   built per image with `minItems` = `maxItems` = the mark count. The prompts are the probe's (session scratch
   `vlm/probe_marks.py`, handed to the builder) and live in `rosette/perception/reader.py` with a version string.
8. **"Match or near match" is a reviewer judgement** on the 52 zoom page, made by the controller and the user, so the
   +10 target is a reviewer checkpoint, not a test; the automatic gates are the scoreboard and the per-ID layer IoU.

## Left to the builder

The exact edge and colour weights of the judge (mined on the 76 against the known-good heads), the prompt wording
(kept in one module with a version string in the cache key), the second-image rule for more than 12 pieces, and how
the UI shows the reader's answers next to the flags. Ask by report, not by stopping, unless a constraint cannot be met.
