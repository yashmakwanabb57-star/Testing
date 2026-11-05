// Pure workbench logic: no DOM, no network. Shared by app.js (and a guided mode) and
// loaded directly by the Node tests in tests/js.

export const FLAG_BELOW = 0.6; // rosette/assign.py FLAG_BELOW
export const MAX_COUNT = 200; // rosette/model.py MAX_COUNT
export const POINT_KINDS = ['star', 'diamond', 'bow', 'petal'];
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
};

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
  Object.assign(s, { filename: null, model: null, svg: null, issues: [], flagged: false, analyzing: false, photoFrame: null });
  s.confirmed.clear();
  s.invalid.clear();
  s.sim.tried.clear();
  Object.assign(s.guided, { step: 1, reached: 1 });
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
  const flags = flagItems(s.model, s.issues, s.flagged);
  const open = flags.filter((f) => !s.confirmed.has(f.key)).length;
  if (open) out.push(`${open} of ${flags.length} flagged item(s) still to confirm.`);
  return out;
}

export const SOURCE_NOT_READY = 'Not ready for the shop - fix or confirm the items listed first.';

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

// ---------------------------------------------------------------- guided mode

export const GUIDED_STEPS = [
  { n: 1, label: 'Upload', title: 'Upload a photo' },
  { n: 2, label: 'Parts', title: 'Check the parts' },
  { n: 3, label: 'Colors', title: 'Check colors' },
  { n: 4, label: 'Test', title: 'Test on site colors' },
  { n: 5, label: 'Download', title: 'Download' },
];

export const tailCheckKey = (n) => `tails:${n}`;

// One "Looks right" check per part card, the way the workbench lays them out: each
// non-tail part, and one per tail group. Part keys are the same as the flag keys, so
// confirming a flagged part here confirms it everywhere.
export function partChecks(model) {
  if (!model) return [];
  const out = [];
  const tails = tailGroups(model);
  const tailFirst = new Map(tails.map((ps, n) => [ps[0], n]));
  for (const p of model.parts) {
    if (p.role !== 'tail_stripe') out.push({ key: `part:${p.id}`, label: p.id });
    else if (tailFirst.has(p)) out.push({ key: tailCheckKey(tailFirst.get(p)), label: 'Tails' });
  }
  return out;
}

const ROLE_NOUNS = { head: 'head', ring: 'ring', points: 'point', center: 'center', tail_stripe: 'stripe', unknown: 'shape' };

// One question per color group, flagged groups first, otherwise in model order.
// usedOn lists the part kinds using the color ("stripe"); noun is set when there is one.
// A plain s is the plural: these words all take one.
export function colorQuestions(model, issues = [], serverFlagged = false) {
  if (!model) return [];
  const flagged = new Set(flagItems(model, issues, serverFlagged).map((f) => f.key));
  const qs = model.groups.map((g, order) => {
    const users = model.parts.filter((p) => p.color_group === g.id);
    const nouns = [...new Set(users.map((p) => ROLE_NOUNS[p.role] || 'shape'))];
    const key = `group:${g.id}`;
    return { key, group: g, noun: nouns.length === 1 ? nouns[0] : null, usedOn: nouns, uses: users.length, flagged: flagged.has(key), order };
  });
  return qs.sort((a, b) => (b.flagged - a.flagged) || (a.order - b.order));
}

// Why Next is blocked on a guided step, in plain words; null means the user may go on.
export function advanceBlocker(step, s) {
  if (step >= GUIDED_STEPS.length) return 'This is the last step.';
  if (s.analyzing) return 'Wait for the conversion to finish.';
  if (!s.model || !s.svg) return s.pendingFile ? NEEDS_CONVERT : 'Upload a photo first.';
  if (step === 2) {
    if (s.invalid.size) return 'Fix the invalid values (red fields) first.';
    const checks = partChecks(s.model);
    const open = checks.filter((c) => !s.confirmed.has(c.key)).length;
    if (open) return `Mark every part "Looks right" first (${open} of ${checks.length} left).`;
  }
  if (step === 3) {
    // Every color question needs an answer (Yes), flagged or not: a wrong color in a wrong
    // layer is the worst failure. A pick through Change... clears the answer, so it gets a
    // fresh Yes. Warnings (and the server-flagged fallback) must be confirmed as well;
    // part and tails flags are confirmed in step 2.
    const qs = colorQuestions(s.model, s.issues, s.flagged);
    const open = qs.filter((q) => !s.confirmed.has(q.key)).length;
    if (open) return `Answer every color question first (${open} of ${qs.length} left).`;
    const warnings = flagItems(s.model, s.issues, s.flagged)
      .filter((f) => !/^(part|tails|group):/.test(f.key) && !s.confirmed.has(f.key)).length;
    if (warnings) return `Confirm ${warnings} warning(s) first.`;
  }
  if (step === 4) {
    const tried = (s.sim && s.sim.tried) || new Set();
    const missing = [];
    if (!tried.has('pattern')) missing.push('Color 2 Solid / Pattern');
    if (!tried.has('rose')) missing.push('Rose On / Off');
    if (missing.length) return `Try ${missing.join(' and ')} at least once first.`;
  }
  return null;
}

export const canAdvance = (step, s) => advanceBlocker(step, s) === null;

// The guided step to show: the given step, or an earlier one whose gate no longer passes
// (e.g. a part's "Looks right" was removed in the workbench while guided sat on step 5).
export function clampStep(step, s) {
  for (let k = 1; k < step; k++) if (!canAdvance(k, s)) return k;
  return step;
}

// Breadcrumb jumps: back is always allowed; forward only to a step already reached, and
// only if every step in between could be passed with Next right now.
export function canJump(from, to, s, reached) {
  if (to < 1 || to > GUIDED_STEPS.length) return false;
  if (to <= from) return true;
  if (to > reached) return false;
  for (let k = from; k < to; k++) if (!canAdvance(k, s)) return false;
  return true;
}
