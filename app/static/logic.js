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

// Mirrors rosette/pipeline.py is_flagged: one item per thing the user must confirm.
export function flagItems(model, issues, serverFlagged) {
  if (!model) return [];
  const items = [];
  for (const g of model.groups) {
    if (g.confidence < FLAG_BELOW) {
      items.push({ key: `group:${g.id}`, text: `Color group ${g.id}: is Color ${g.category ?? '-'} right?` });
    }
  }
  for (const p of model.parts) {
    if (p.role === 'points' && p.confidence < FLAG_BELOW) {
      items.push({ key: `part:${p.id}`, text: `${p.id}: points read with low confidence` });
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

// Reasons Download is disabled for a workbench state; empty means ready.
export function downloadBlockers(s) {
  if (s.analyzing) return ['Analyzing the photo...'];
  if (!s.model || !s.svg) return ['Upload a photo first.'];
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
