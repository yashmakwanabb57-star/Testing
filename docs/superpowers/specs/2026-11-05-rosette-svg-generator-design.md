# Rosette Photo to Configurator SVG - Design

Date: 2026-11-05
Source request: `task.md` (this design overrides it where they conflict; conflicts are listed in "Contract" below).

## 1. Intent

brroset.dk sells "design selv" rosettes (award ribbons). Each product page has a
configurator: an inline SVG whose parts the shop recolors (Color 1/2/3), switches
between solid and pattern for Color 2, and shows/hides a rose in the center.
Today a designer hand-draws each SVG in Illustrator.

The tool takes a **photo of a brand-new rosette** (no designer SVG exists yet) and
produces an SVG **the way the designer would draw it**: clean, symmetric,
schematic geometry, not a trace of satin folds and glitter. It must drop into the
shop configurator and behave correctly.

**The one thing it must not get wrong:** colors in the wrong layers/classes.
Imperfect geometry is acceptable; a part that recolors with the wrong color, or
disappears in pattern/solid mode, is a failure even if the picture looks right.

## 2. Goals, non-goals, anti-goals

Goals
- Upload PNG/WEBP/JPG/JPEG photo, get a structured SVG that works in the shop configurator.
- Geometry rebuilt from rosette parts (head, points, center, tails), symmetric and clean.
- Up to **four** customer colors (Color 1-4; user confirmed customers can change a 4th color). Color 4 is emitted only when the design needs it.
- Every color part correctly placed in both the pattern and solid configurations.
- A person can see and correct every detection (part counts, types, color per part) without drawing.
- A configurator simulator in the tool that uses the shop's own coloring/toggle logic.
- A hard validation gate before download.
- A regression test set built from the shop's existing photo + designer SVG pairs.

Non-goals (for now)
- Hosting / multi-user / login. Runs locally on the user's machine.
- Colors inside the SVG. Output carries shapes and classes only; the shop applies colors.
- Tracing text, paw prints, logos, satin shading, or background.
- Uploading into the shop admin automatically.
- Paid services or cloud APIs. Free local models are allowed (user ruling 2026-11-12: the
  `task.md` "no AI/ML" rule means no paid services); nothing calls the network at run time.

Anti-goals
- A raster-to-SVG tracer that "looks similar" but has one path per glitter speck or fold.
- Silently guessing a color category when unsure.
- Downloading an SVG that fails any structural check.

## 3. The contract (what the shop actually requires)

Derived from the live configurator code on the S46-15 product page
(`globalMethods.setSvgImgColor`, `setSvgImgRoseVisibility`) and from 79 live
designer SVGs fetched from `/h/data/product-configurators-svg.axd?productid=br<ID>`.
Where this differs from `task.md`, this section wins.

### 3.1 Document
- Root `<svg version="1.1" xmlns=... xmlns:xlink=... x="0px" y="0px" viewBox="0 0 600 850" style="enable-background:new 0 0 600 850;" xml:space="preserve">`.
  - 69 of 79 live files use 600x850; the tool always emits 600x850.
  - No `id` on the root `<svg>` (one legacy file has `id="Layer_1"` on root, a duplicate-ID bug).
  - The photo's 1000x2000 size is NOT preserved; the photo is a preview, not the artwork (user confirmed).
- One `<style type="text/css">` block declaring each used class with `fill:none` plus the outline/rose stroke styles. No Illustrator comment, no metadata.

### 3.2 Layers (top-level order = paint order)
```
<g id="Layer_1">                      Color 1 solid parts        class stN_1
<g id="Layer_3">                      shown when Color 2 = pattern
    <g id="Layer_3_2">                Color 2 pattern parts      class stN_2_pattern
        <defs><pattern id="image_stN_2_pattern" patternUnits="userSpaceOnUse" width="100" height="100">
                <image xlink:href="" x="0" y="0" width="100" height="100"/></pattern></defs>
    <g id="Layer_3_3">                Color 3 parts              class stN_3
    <g id="Layer_3_4">                Color 4 parts (only if used) class stN_4
<g id="Layer_2_1">                    shown when Color 2 = solid
    <g id="Layer_2_1_2">              Color 2 solid parts        class stN_2
    <g id="Layer_2_1_3">              Color 3 parts              class stN_3
    <g id="Layer_2_1_4">              Color 4 parts (only if used) class stN_4
<g id="Layer_6">                      black outline line art     class st6
<g id="Layer_rose">                   rose line art              class st4
```
- The shop toggles `#Layer_3` vs `#Layer_2_1` by ID. `task.md` says `Layer_2`; that is wrong for this site.
- Color 4 mirrors Color 3 exactly (live reference: S68-19, `Layer_3_4` / `Layer_2_1_4`, class `st3_4`). The shop's coloring function is generic over the color index, so `_4` needs no site change.
- **Empty groups:** `Layer_1`, `Layer_3`/`Layer_3_2`/`Layer_3_3`, `Layer_2_1`/`Layer_2_1_2`/`Layer_2_1_3`, `Layer_6`, `Layer_rose` are always emitted, even if empty, so the shop's toggles always find them (builder's call). `_4` groups only when Color 4 is used. Pattern defs only for pattern classes actually used.
- **Intentional duplication:** every Color 2 shape appears once in `Layer_3_2` (as `stN_2_pattern`) and once in `Layer_2_1_2` (as `stN_2`) with identical geometry. Every Color 3 (and Color 4) shape appears once in `Layer_3_3` (`Layer_3_4`) and once in `Layer_2_1_3` (`Layer_2_1_4`). "No duplicate paths" means no duplicates *within* a layer.
- Paint order inside the color layers follows physical stacking (back to front): tails, head disc, points, inner rings, center disc. Parts of different colors interleave by stacking, which is why Color 3 lives inside both toggled groups.

### 3.3 Classes
- Color classes: `stN_1`, `stN_2`, `stN_2_pattern`, `stN_3`, `stN_4`, where **N is one of 0, 1, 2, 3, 5 only**. The shop's coloring loop iterates exactly `['st0','st1','st2','st3','st5']`; any other N is never colored. (`task.md` allows 0-6; 4 and 6 are reserved.)
- `st4` = rose. The shop shows/hides every `.st4` element with the Rose option. Never use `st4` for anything else.
- `st6` = black outline line art. Never colored by the shop.
- `st_circel` (sic) = element shown when the rose is OFF, hidden when ON. Optional; emitted only if the design needs a different center when the rose is off.
- For every N used with `_2_pattern`, a matching `<pattern id="image_stN_2_pattern">` with an `<image>` child must exist (the shop writes `xlink:href` into it at runtime; an empty href is correct, not a broken reference).
- Elements: `path`, `polygon`, `polyline`, `circle`, `ellipse`. Color parts must be **closed, fillable** shapes (the shop sets `style.fill`).

### 3.4 Rose
- One master rose (taken from a live reference, e.g. the S46-15 `Layer_rose` group: one `path` + two `polyline`, class `st4`) stored as an asset, scaled and translated to the detected center. Not re-derived from the photo.

## 4. How it works

```
photo -> 1 prepare -> 2 detect parts -> 3 color groups -> 4 assign Color 1/2/3
      -> 5 build geometry -> 6 write SVG -> 7 validate -> UI review/correct -> download
```
Each step is its own module with a typed input/output so it can be improved independently.
The intermediate result is a **RosetteModel** (JSON-serializable): the parts list with
measurements and color assignment. The UI edits the RosetteModel; steps 5-7 rerun from it
instantly without re-analyzing the photo.

1. **Prepare** - load (Pillow; WEBP supported), keep alpha, remove the near-white/transparent background, find the rosette mask, its vertical symmetry axis, bounding box, head circle (largest roughly-circular blob near the top) and tails region (below the head).
2. **Detect parts** (OpenCV/scikit-image; since ML-1 a small local scikit-learn classifier, trained on the reference photos, reads each head layer's kind and count behind a confidence bar):
   - Head: outer radius; edge type pinked vs smooth; inner concentric rings via radial color profile from the center.
   - Points ring: type (star tips / diamonds / square bows / petals) and count N via an angular color profile at a few radii (count of periodic transitions, FFT peak); rotation offset; inner/outer radius.
   - Center: rose vs plain disc (texture/edge density), radius.
   - Tails: count, top/bottom x-extents, fan angle, stripes per tail (horizontal color profile along the tail), pinked bottom with tooth size.
   Each detection carries a confidence.
3. **Color groups** - cluster pixels per part in CIELAB with shading-robust features (chroma/hue weighted over lightness) so satin highlights and shadows of one ribbon merge; small k (2-6). Pattern detection per part: high local texture energy (glitter) or a repeating motif (autocorrelation peaks), vs low-texture solid satin.
4. **Assign Color 1-4** - there is no documented house rule, so the rules are mined from the 35 full-structure reference files (see 6): which part roles (tails, head disc, points, rings, center disc, stripes) tend to be Color 1/2/3/4 and which carry the pattern. A pattern-textured group is Color 2. Parts sharing a color group share a category. If the photo has more than 4 color groups, the closest groups are merged and the merge is flagged. Low-confidence assignments are always flagged for confirmation; the human-correction UI is the safety net, not an afterthought.
5. **Build geometry** - parametric generators per part type on the 600x850 canvas, mirrored / rotationally symmetric, with pinked edges as zigzag polylines matching designer tooth size (~5.4 units, as in live files). Plus `st6` outlines for part boundaries and the scaled master rose.
6. **Write SVG** - emit exactly the structure in section 3 (lxml), compact numbers (3 decimals), no metadata.
7. **Validate** - the hard gate (section 5).

## 5. Validation (blocks download on any failure)
- Well-formed XML; root attributes and viewBox `0 0 600 850`.
- Required groups present with correct parent/child: `Layer_1`, `Layer_3 > Layer_3_2, Layer_3_3`, `Layer_2_1 > Layer_2_1_2, Layer_2_1_3`, `Layer_6`, `Layer_rose`. Unique IDs.
- Every element's class matches its layer (`_1` only in Layer_1, `_2_pattern` only in Layer_3_2, `_2` only in Layer_2_1_2, `_3` only in Layer_3_3/Layer_2_1_3, `_4` only in Layer_3_4/Layer_2_1_4, `st4` only in Layer_rose, `st6` only in Layer_6).
- Color class N in {0,1,2,3,5}.
- Pattern def exists for every `stN_2_pattern` used; no unused defs.
- Layer_3_2 and Layer_2_1_2 contain the same geometry set; Layer_3_3 and Layer_2_1_3 the same set; Layer_3_4 and Layer_2_1_4 the same set (when present, both must be present).
- No duplicate geometry within a layer; every color element is closed.
- Every style-block class is used and every used class is declared.
- **Simulation check:** run the shop's coloring logic (ported) on the SVG for both Color 2 modes and rose on/off; every color element must receive a fill, and nothing visible may be left unfilled.

## 6. Reference / test set
- 79 live designer SVGs and 76 matching product photos (`/files/data/img/products/<ID>.png`, 1000x2000), already fetched during brainstorming. Stored in the repo under `reference/` (svg + photo per product ID).
- 35 of them follow the full structure (`Layer_1 + Layer_3 + Layer_2_1`); these are the scoring set and the source for color-assignment rules. Legacy-structure files are used only for shape vocabulary.
- Scoring per pair (photo in, generated SVG vs designer SVG): primary = color-category agreement per part region (rasterize both with category colors, compare per pixel inside the rosette mask); secondary = part counts (points, tails, rings) and shape IoU.
- **Bar (user agreed):** on the 35 scoring rosettes, color categories and layers are correct with zero manual edits in at least 8 of 10; every remaining case must be flagged in the UI, never silently wrong.
- Photo-to-canvas registration for scoring: align the photo's rosette mask to the designer SVG's rasterized mask by bounding box (builder's call).
- A test command prints the scoreboard; it is the regression gate for pipeline changes.
- The shop's configurator JS (`setSvgImgColor`, `setSvgImgRoseVisibility`, layer toggle) is copied verbatim from the fetched S46-15 product page into `reference/shop-configurator.js` and is the source for the simulator port.

## 7. UI (local web app)
Two modes over the same data, user asked for both:
- **Workbench (default):** one screen. Left: photo. Middle: generated SVG with configurator simulator controls (pick Color 1/2/3, Color 2 solid/pattern, rose on/off, using the shop's own logic). Right: detected parts (editable: counts, types, radii) and color groups (reassign category per group or per part; flagged low-confidence items highlighted). Validation status and Download button. Optional: before/after overlay slider, SVG source view.
- **Guided:** steps Upload > Parts > Colors > Test in simulator > Download, one decision at a time, cannot skip a flagged item.
- Error messages for unsupported files, no rosette found, or validation failure (what failed and where).
- Output filename: input stem, lowercased, `.svg`. Saved via browser download; the tool never writes next to the input (the reference SVGs in Downloads have the same names).

## 8. Stack (builder's call, stated to the user)
- Python 3 in a project virtualenv: FastAPI + uvicorn, OpenCV (opencv-python-headless), NumPy, scikit-image, Pillow, lxml; pytest. All free, commercial-friendly licenses.
- Frontend: plain HTML/CSS/JS served by FastAPI, no build step. The simulator reuses a port of the shop's `setSvgImgColor` / rose / layer-toggle logic.
- Start with one command; opens on localhost.

## 9. Builder's calls already made (check these)
- Canvas layout: rosette fitted to 600x850 like S35-19, centered at x=300, top near y=25, tail ends near y=825.
- Expected input: product-style shots (one rosette, plain/light or transparent background, head up). Anything unrecognized becomes a simple flagged placeholder shape for manual correction, never silently dropped.
- Always emit all required groups even when empty (see 3.2).
- Stack in section 8.

## 10. Left to the builder
- Exact CV algorithms and thresholds per detector, tuned against the scoring set.
- Parametric shape generators per part type; which point types to support first (by frequency in the reference set).
- How confidence is computed and the threshold for flagging.
- Internal numbering of N per part within {0,1,2,3,5}.
