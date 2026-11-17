// Pure workbench logic: no DOM, no network. Shared by app.js and
// loaded directly by the Node tests in tests/js.

export const FLAG_BELOW = 0.6; // rosette/assign.py FLAG_BELOW
export const MAX_COUNT = 200; // rosette/model.py MAX_COUNT
export const POINT_KINDS = ['star', 'diamond', 'bow', 'petal'];
// Base width of a loop (bow) as a share of its widest width when the model sets no
// inner_ratio: rosette/geometry/points.py LOOP_PROFILE[0].
export const LOOP_BASE = 0.85;

// Value shown in a part's param field: a bow's inner_ratio is the drawn base width.
export function paramShown(params, key) {
  if (key === 'inner_ratio' && !(key in params)) return LOOP_BASE;
  return params[key];
}
export const TAIL_SHARED = ['tails_count', 'top_y', 'bottom_y', 'top_width', 'bottom_width', 'spread_deg'];
export const TAIL_KEYS = [...TAIL_SHARED, 'stripes', 'stripes_bottom']; // rosette/build.py _TAIL_KEYS

// Readable text for an API error: detail is a string for our own errors and a list
// for FastAPI request validation errors.
export function errorText(status, body) {
  const d = body && body.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    const bits = d.map((e) => {
      const loc = (e && Array.isArray(e.loc) ? e.loc : []).filter((x) => x !== 'body').join('.');
      const msg = e && e.msg ? e.msg : 'invalid value';
      return loc ? `${loc}: ${msg}` : msg;
    });
    return `The request was rejected - ${bits.join('; ')}`;
  }
  if (status === 413) return 'File is too large. The limit is 25 MB.';
  return `The request failed (HTTP ${status}).`;
}

export const hex = (rgb) => '#' + rgb.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');

const LOW_TEXT = {
  head: 'head read with low confidence', ring: 'ring read with low confidence',
  points: 'points read with low confidence', center: 'center read with low confidence',
  template: 'library head shape is a weak match',
};

// ------------------------------------------------------------ library heads (H7)

// The part that uses a color group: a template part maps reference Colors to groups
// in params group_1..group_4 (rosette/build.py template_groups).
export function partGroups(p) {
  if (p.role !== 'template') return [p.color_group];
  const out = new Set([p.color_group]);
  for (const [k, v] of Object.entries(p.params)) if (k.startsWith('group_')) out.add(Number(v));
  return [...out];
}

export function usesGroup(p, gid) {
  return partGroups(p).includes(gid);
}

// Library heads the matcher offers on the head card (rosette/pipeline.py TEMPLATE_CANDIDATES);
// a higher rank was chosen from the whole library.
const TOP_CANDIDATES = 3;

// Head card text of a library head.
export function templateText(p) {
  if (Number(p.params.rank) > TOP_CANDIDATES) {
    const desc = p.params.description ? ` - ${p.params.description}` : '';
    return `Head shape: chosen from the library: ${p.params.template_id}${desc}`;
  }
  const score = Number(p.params.score);
  const pct = Number.isFinite(score) ? `${Math.round(score * 100)}%` : '-';
  const pts = Number(p.params.points) > 0 ? `, ${p.params.points} points` : '';
  return `Head shape: from library template ${p.params.template_id}, match ${pct}${pts}`;
}

// The head card heading (H9): the head the conversion chose (choice: 'rules' or a
// template_id, kept from the analyze response); '' without a choice. The UI has no head
// picker; heads are switched only through the API (rosette/pipeline.py use_head).
export function automaticHeadText(model, choice) {
  if (!model || !choice) return '';
  if (choice === 'rules') return 'Automatic result: rule-based head';
  const p = [...model.parts, ...(model.alternatives || [])]
    .find((q) => q.role === 'template' && String(q.params.template_id) === choice);
  const score = p ? Number(p.params.score) : NaN;
  const pct = Number.isFinite(score) ? `, match ${Math.round(score * 100)}%` : '';
  return `Automatic result: library template ${choice}${pct}`;
}

// Body of /api/render: the model, and the head to draw when a script asks the server for one.
export function renderBody(model, head) {
  return JSON.stringify(head === undefined ? { model } : { model, head });
}

// Mirrors rosette/pipeline.py is_flagged: one item per thing the user must confirm.
// A part of any role below FLAG_BELOW is flagged; tail stripes are confirmed per tails
// card (tailCheckKey), so a low-confidence tail group gives one item.
export function flagItems(model, issues, serverFlagged) {
  if (!model) return [];
  const items = [];
  for (const g of model.groups) {
    if (g.confidence < FLAG_BELOW) {
      items.push({ key: `group:${g.id}`, text: `Color group ${g.id}: is Color ${g.category ?? '-'} right?` });
    }
  }
  const tails = tailGroups(model);
  const tailFirst = new Map(tails.map((ps, n) => [ps[0], n]));
  for (const p of model.parts) {
    if (p.role === 'tail_stripe') {
      if (!tailFirst.has(p)) continue;
      const n = tailFirst.get(p);
      if (tails[n].some((q) => q.confidence < FLAG_BELOW)) {
        items.push({ key: tailCheckKey(n), text: 'Tails: stripes read with low confidence' });
      }
    } else if (p.role !== 'unknown' && p.confidence < FLAG_BELOW) {
      items.push({ key: `part:${p.id}`, text: `${p.id}: ${LOW_TEXT[p.role] || 'read with low confidence'}` });
    } else if (p.role === 'unknown') {
      // build warns "part <id>: unrecognised shape..."; that warning is shown on this part
      // card and confirmed with it, so it is not listed twice.
      const w = model.warnings.find((x) => x.startsWith(`part ${p.id}: `));
      items.push({ key: `part:${p.id}`, text: w || `part ${p.id}: unrecognised shape, drawn as a plain disc` });
    }
  }
  const unknown = new Set(model.parts.filter((p) => p.role === 'unknown').map((p) => p.id));
  for (const w of model.warnings) {
    const m = /^part (\S+): /.exec(w);
    if (m && unknown.has(m[1])) continue;
    items.push({ key: `warning:${w}`, text: w });
  }
  if (serverFlagged && !items.length && !issues.length) {
    items.push({ key: 'server:flagged', text: 'The server flagged this result. Check it before download.' });
  }
  return items;
}

const NEEDS_CONVERT = 'Convert the photo to SVG first.';

// The fixed checklist: always present once a model exists, confirmed like a flag but
// not a flag (is_flagged knows nothing of it). The head fit cannot tell a petal from a
// block (rosette/pipeline.py), so the user compares every head with the photo.
export const HEAD_CHECK = { key: 'check:head', text: 'Compare the head with the photo' };

// Head fit measured on the photo at conversion (api/analyze head_fit): information only.
export function headFitText(fit) {
  if (!fit) return 'No head match measured for this model.';
  const pct = (v) => `${Math.round(v * 100)}%`;
  return `Head match ${pct(fit.agreement)}, outline ${pct(fit.silhouette)} - measured on the photo when converted; information, not a flag.`;
}

export function checklistItems(s) {
  if (!s.model) return [];
  return [{ ...HEAD_CHECK, info: headFitText(s.headFit) }];
}

// Why "Convert to SVG" is disabled; null means it may run. s.pendingFile is the chosen
// photo (choosing or dropping a file only previews it; Convert calls /api/analyze).
export function convertBlocker(s) {
  if (!s.pendingFile) return 'Choose a photo first.';
  if (s.analyzing) return 'Converting...';
  return null;
}

// Clear everything derived from the previous photo, so a newly chosen file can never
// leave the old model downloadable under the new name. Run on choosing a file and again
// when Convert starts. Bumping the sequence numbers drops any analysis or render still
// in flight. The caller clears its render timer.
export function resetForFile(s) {
  s.analyzeSeq++;
  s.render.seq++;
  Object.assign(s.render, { dirty: false, inFlight: false, error: null });
  Object.assign(s, { filename: null, model: null, svg: null, issues: [], flagged: false, analyzing: false, photoFrame: null, headFit: null });
  s.confirmed.clear();
  s.invalid.clear();
  s.sim.tried.clear();
}

// Reasons Download is disabled for a workbench state; empty means ready.
export function downloadBlockers(s) {
  if (s.analyzing) return ['Converting the photo...'];
  if (!s.model || !s.svg) return [s.pendingFile ? NEEDS_CONVERT : 'Upload a photo first.'];
  const out = [];
  if (s.invalid.size) out.push('Fix the invalid values (red fields).');
  if (s.render.dirty || s.render.inFlight) out.push('Updating the SVG...');
  else if (s.render.error) out.push('The last update failed. Change a value to retry.');
  if (s.issues.length) out.push(`${s.issues.length} check(s) failed.`);
  return out;
}

// Flagged items and checklist ticks not yet given. They no longer block Download (the owner's
// call): the user may download without confirming; the download box lists them.
export function openConfirmations(s) {
  if (!s.model || !s.svg) return [];
  const out = [];
  const flags = flagItems(s.model, s.issues, s.flagged);
  const open = flags.filter((f) => !s.confirmed.has(f.key)).length;
  if (open) out.push(`${open} of ${flags.length} flagged item(s) not confirmed.`);
  for (const c of checklistItems(s)) {
    if (!s.confirmed.has(c.key)) out.push(`${c.text} (not ticked).`);
  }
  return out;
}

export const SOURCE_NOT_READY = 'Not ready for the shop - fix the items listed first.';

// Banner over the "SVG source" view: the source stays readable for debugging, but while
// Download is blocked it must not pass for a shop-ready file. null: no banner.
export function sourceNotice(s) {
  if (!s.svg) return null;
  return downloadBlockers(s).length ? SOURCE_NOT_READY : null;
}

export function groupColor(model, category) {
  const gs = model.groups.filter((g) => g.category === category).sort((a, b) => b.area - a.area);
  return gs.length ? hex(gs[0].rgb) : null;
}

function tailKey(p) {
  return JSON.stringify(TAIL_KEYS.map((k) => p.params[k] ?? ''));
}

// Tail stripe parts grouped the way rosette/build.py groups them (same _TAIL_KEYS).
export function tailGroups(model) {
  const groups = new Map();
  for (const p of model.parts) {
    if (p.role !== 'tail_stripe') continue;
    const k = tailKey(p);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(p);
  }
  return [...groups.values()];
}

// Parse one input value. Returns {ok, value}.
export function parseValue(type, raw) {
  if (type === 'kind') return { ok: POINT_KINDS.includes(raw), value: raw };
  if (type === 'bool') return { ok: true, value: raw ? 1 : 0 };
  const t = String(raw).trim();
  if (t === '') return { ok: false };
  const v = Number(t);
  if (!Number.isFinite(v)) return { ok: false };
  if (type === 'int' && (!Number.isInteger(v) || v < 1 || v > MAX_COUNT)) return { ok: false };
  return { ok: true, value: v };
}

export const tailCheckKey = (n) => `tails:${n}`;

