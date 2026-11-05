// Workbench UI. Plain ES module, no framework, no network access beyond the local API.
// Sections: constants, state, api, derived (pure), actions (mutate state), views (DOM),
// init (event wiring). A guided mode can reuse state, api, derived and actions and add
// its own views.
import { applyConfig, glitterPattern, COLOR_INDICES } from './simulator.js';

// ---------------------------------------------------------------- constants

const FLAG_BELOW = 0.6; // rosette/assign.py FLAG_BELOW
const MAX_COUNT = 200; // rosette/model.py MAX_COUNT
const DEBOUNCE_MS = 200;
const POINT_KINDS = ['star', 'diamond', 'bow', 'petal'];
const DEFAULT_COLORS = { 1: '#2e6b4a', 2: '#9fb7c4', 3: '#d9b26f', 4: '#8a5a9e' };
const TAIL_SHARED = ['tails_count', 'top_y', 'bottom_y', 'top_width', 'bottom_width', 'spread_deg'];
const TAIL_KEYS = [...TAIL_SHARED, 'stripes', 'stripes_bottom']; // rosette/build.py _TAIL_KEYS

// Validation codes from rosette/validate.py, in check order.
const CHECKS = [
  ['xml', 'Well-formed XML'],
  ['root', 'Root element and viewBox'],
  ['dup_id', 'Unique IDs'],
  ['missing_group', 'Required layer groups present'],
  ['bad_parent', 'Layer groups nested correctly'],
  ['bad_element', 'Only allowed elements'],
  ['class_layer', 'Classes in the right layers'],
  ['class_number', 'Class numbers 0, 1, 2, 3, 5'],
  ['pattern_def_missing', 'Pattern definitions present'],
  ['pattern_def_unused', 'No unused pattern definitions'],
  ['toggle_mismatch', 'Solid and pattern layers match'],
  ['dup_geometry', 'No duplicate shapes'],
  ['not_closed', 'All color shapes closed'],
  ['style_mismatch', 'Style rules match the classes'],
  ['sim_unfilled', 'Shop coloring fills every shape'],
];

const ROLE_LABELS = {
  head: 'Head', ring: 'Ring', points: 'Points', center: 'Center',
  tail_stripe: 'Tails', unknown: 'Unrecognised shape',
};

// Editable params per role. type: num | int | kind | bool.
const PARAMS = {
  head: [['r', 'Radius', 'num'], ['pinked', 'Pinked edge', 'bool'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  ring: [['r_outer', 'Outer radius', 'num'], ['r_inner', 'Inner radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  points: [['kind', 'Kind', 'kind'], ['count', 'Count', 'int'], ['r_inner', 'Inner radius', 'num'],
    ['r_outer', 'Outer radius', 'num'], ['rotation_deg', 'Rotation (deg)', 'num'],
    ['width_ratio', 'Width ratio', 'num'], ['inner_ratio', 'Inner ratio (bow)', 'num'],
    ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  center: [['r', 'Radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  unknown: [['r', 'Radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  tail_stripe: [['tails_count', 'Tails', 'int'], ['top_y', 'Top y', 'num'], ['bottom_y', 'Bottom y', 'num'],
    ['top_width', 'Top width', 'num'], ['bottom_width', 'Bottom width', 'num'], ['spread_deg', 'Spread (deg)', 'num']],
};

// ---------------------------------------------------------------- state

export const state = {
  photoUrl: null,
  uploadName: '',
  filename: null, // download name from /api/analyze
  model: null, // RosetteModel JSON, edited in place
  svg: null, // clean SVG text from the last successful render
  issues: [],
  flagged: false, // server is_flagged of the last render
  confirmed: new Set(), // flag keys the user confirmed
  invalid: new Set(), // input field keys holding an invalid value
  analyzing: false,
  analyzeSeq: 0, // latest upload; older analyze responses are dropped
  render: { seq: 0, timer: null, dirty: false, inFlight: false, error: null },
  sim: {
    colors: { ...DEFAULT_COLORS },
    touched: new Set(), // color indices the user picked by hand
    pattern: false, // Color 2 uses the sample pattern
    rose: true,
    overlay: false,
    opacity: 50,
    source: false,
  },
  photoFrame: null, // overlay placement of the photo in the 600x850 frame
};

// ---------------------------------------------------------------- api

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

async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error('Could not reach the local server. Is it still running?');
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) throw new Error(errorText(res.status, body));
  if (!body) throw new Error('The server sent an unreadable response.');
  return body;
}

export const api = {
  analyze(file) {
    const form = new FormData();
    form.append('file', file, file.name);
    return request('/api/analyze', { method: 'POST', body: form });
  },
  render(model) {
    return request('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
  },
};

// ---------------------------------------------------------------- derived (pure)

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

export function downloadBlockers() {
  const s = state;
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

function groupColor(model, category) {
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

// ---------------------------------------------------------------- actions

export const actions = {
  async loadFile(file) {
    if (!file) return;
    const token = ++state.analyzeSeq;
    clearTimeout(state.render.timer);
    state.render.seq++; // drop any render still in flight
    Object.assign(state.render, { dirty: false, inFlight: false, error: null });
    if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
    Object.assign(state, {
      photoUrl: URL.createObjectURL(file), uploadName: file.name, filename: null, model: null,
      svg: null, issues: [], flagged: false, analyzing: true, photoFrame: null,
    });
    state.confirmed.clear();
    state.invalid.clear();
    hideError();
    views.all();
    try {
      const res = await api.analyze(file);
      if (token !== state.analyzeSeq) return;
      Object.assign(state, {
        model: res.model, svg: res.svg, issues: res.issues || [], flagged: !!res.flagged,
        filename: res.filename,
      });
      state.sim.touched.clear();
      state.sim.pattern = false;
      state.sim.rose = !!res.model.rose;
      actions.defaultColors();
    } catch (e) {
      if (token === state.analyzeSeq) showError(e.message);
    } finally {
      if (token === state.analyzeSeq) {
        state.analyzing = false;
        views.all();
      }
    }
  },

  defaultColors() {
    for (const i of COLOR_INDICES) {
      if (state.sim.touched.has(i)) continue;
      state.sim.colors[i] = groupColor(state.model, i) || DEFAULT_COLORS[i];
    }
  },

  // Set one param of one part; field is the input key for invalid tracking.
  setParam(part, key, type, raw, field) {
    const r = parseValue(type, raw);
    if (!r.ok) {
      state.invalid.add(field);
      views.download();
      return false;
    }
    state.invalid.delete(field);
    part.params[key] = r.value;
    actions.scheduleRender();
    return true;
  },

  // Shared tail params apply to every stripe of the tail group, so build keeps them together.
  setTailParam(parts, key, raw, field) {
    if (key === 'tails_count') return actions.setTailsCount(parts, raw, field);
    const r = parseValue('num', raw);
    if (!r.ok) {
      state.invalid.add(field);
      views.download();
      return false;
    }
    state.invalid.delete(field);
    for (const p of parts) p.params[key] = r.value;
    actions.scheduleRender();
    return true;
  },

  // Rebuild the stripe parts for a new tail count: every tail gets the stripes of the
  // first tail; tails that already existed keep their stripe colors.
  setTailsCount(parts, raw, field) {
    const r = parseValue('int', raw);
    if (!r.ok) {
      state.invalid.add(field);
      views.download();
      return false;
    }
    state.invalid.delete(field);
    const n = r.value;
    const minTail = Math.min(...parts.map((p) => p.params.tail_index));
    const template = parts.filter((p) => p.params.tail_index === minTail)
      .sort((a, b) => a.params.stripe_index - b.params.stripe_index);
    const existing = new Map(parts.map((p) => [`${p.params.tail_index}-${p.params.stripe_index}`, p]));
    const mine = new Set(parts);
    const taken = new Set(state.model.parts.filter((p) => !mine.has(p)).map((p) => p.id));
    const fresh = [];
    for (let t = 0; t < n; t++) {
      for (const tp of template) {
        const s = tp.params.stripe_index;
        const old = existing.get(`${t}-${s}`);
        let id = old ? old.id : `tail-${t}-${s}`;
        while (taken.has(id)) id += 'b';
        taken.add(id);
        fresh.push({
          ...tp, id,
          params: { ...tp.params, tail_index: t, tails_count: n },
          color_group: old ? old.color_group : tp.color_group,
        });
      }
    }
    const at = state.model.parts.findIndex((p) => mine.has(p));
    const rest = state.model.parts.filter((p) => !mine.has(p));
    rest.splice(at, 0, ...fresh);
    state.model.parts = rest;
    views.parts();
    actions.scheduleRender();
    return true;
  },

  setPartGroup(parts, groupId) {
    for (const p of parts) p.color_group = groupId;
    actions.scheduleRender();
  },

  setGroupCategory(group, category) {
    group.category = category;
    actions.defaultColors();
    views.simControls();
    views.parts(); // group option labels show the category
    actions.scheduleRender();
  },

  setModelRose(on) {
    state.model.rose = on;
    actions.scheduleRender();
  },

  confirm(key, on) {
    if (on) state.confirmed.add(key);
    else state.confirmed.delete(key);
    views.flagsMarked();
    views.download();
  },

  setSimColor(i, value) {
    state.sim.colors[i] = value;
    state.sim.touched.add(i);
    views.preview();
  },

  setSim(patch) {
    Object.assign(state.sim, patch);
    views.simControls();
    views.preview();
  },

  scheduleRender() {
    if (!state.model) return;
    clearTimeout(state.render.timer);
    state.render.dirty = true;
    state.render.timer = setTimeout(actions.renderNow, DEBOUNCE_MS);
    views.download();
    views.status();
  },

  async renderNow() {
    if (state.invalid.size) {
      state.render.dirty = false;
      views.download();
      views.status();
      return;
    }
    const seq = ++state.render.seq;
    Object.assign(state.render, { dirty: false, inFlight: true });
    views.status();
    const model = JSON.parse(JSON.stringify(state.model));
    try {
      const res = await api.render(model);
      if (seq !== state.render.seq) return; // a newer request superseded this one
      Object.assign(state, { svg: res.svg, issues: res.issues || [], flagged: !!res.flagged });
      state.render.error = null;
      hideError();
    } catch (e) {
      if (seq !== state.render.seq) return;
      state.render.error = e.message;
      showError(e.message);
    } finally {
      if (seq === state.render.seq) {
        state.render.inFlight = false;
        views.afterRender();
      }
    }
  },

  // Saves the clean server SVG of the last render, never the simulator-colored DOM.
  download() {
    if (downloadBlockers().length) return;
    const blob = new Blob([state.svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = state.filename || 'rosette.svg';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};

// ---------------------------------------------------------------- views

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) e.append(c);
  return e;
}

function showError(msg) {
  $('error-text').textContent = msg;
  $('error-banner').hidden = false;
}

function hideError() {
  $('error-banner').hidden = true;
}

let patternCache = { base: null, url: null };
function patternImage() {
  const base = state.sim.colors[2];
  if (patternCache.base !== base) patternCache = { base, url: glitterPattern(base) };
  return patternCache.url;
}

function groupOptions(selected) {
  return state.model.groups.map((g) => el('option', {
    value: String(g.id), selected: g.id === selected,
    text: `Group ${g.id} - Color ${g.category ?? '-'} (${hex(g.rgb)})`,
  }));
}

function confirmBox(key) {
  return el('label', { class: 'confirm' },
    el('input', {
      type: 'checkbox', 'data-confirm': key, checked: state.confirmed.has(key),
      onchange: (ev) => actions.confirm(key, ev.target.checked),
    }),
    ' Confirmed');
}

function paramField(type, label, value, field, onInput) {
  let input;
  if (type === 'kind') {
    input = el('select', { 'data-field': field, onchange: (ev) => onInput(ev.target.value, ev.target) },
      ...POINT_KINDS.map((k) => el('option', { value: k, selected: k === value, text: k })));
  } else if (type === 'bool') {
    input = el('input', {
      type: 'checkbox', 'data-field': field, checked: !!Number(value),
      onchange: (ev) => onInput(ev.target.checked, ev.target),
    });
  } else {
    const shown = value === undefined ? '' : (type === 'num' ? String(Math.round(value * 1000) / 1000) : String(value));
    input = el('input', {
      type: 'number', 'data-field': field, value: shown,
      step: type === 'int' ? '1' : 'any', min: type === 'int' ? '1' : null, max: type === 'int' ? String(MAX_COUNT) : null,
      class: state.invalid.has(field) ? 'invalid' : null,
      oninput: (ev) => onInput(ev.target.value, ev.target),
    });
  }
  return el('label', { class: 'param' }, el('span', { text: label }), input);
}

function markInvalid(target, ok) {
  target.classList.toggle('invalid', !ok);
}

function partCard(p) {
  const key = `part:${p.id}`;
  const fields = PARAMS[p.role] || PARAMS.unknown;
  const grid = el('div', { class: 'params' });
  for (const [k, label, type] of fields) {
    if (k === 'inner_ratio' && p.params.kind !== 'bow') continue;
    if (!(k in p.params) && k !== 'inner_ratio') continue;
    const field = `${p.id}.${k}`;
    const value = k === 'inner_ratio' && !(k in p.params) ? 1 : p.params[k];
    grid.append(paramField(type, label, value, field, (raw, target) => {
      const ok = actions.setParam(p, k, type, raw, field);
      if (target.type === 'number') markInvalid(target, ok);
      if (k === 'kind') views.parts();
    }));
  }
  const summary = p.role === 'points' ? `${p.params.kind} x ${p.params.count}` : '';
  const card = el('div', { class: 'card', 'data-flag': key },
    el('div', { class: 'card-head' },
      el('strong', { text: ROLE_LABELS[p.role] || p.role }),
      el('span', { class: 'muted', text: ` ${p.id}${summary ? ' - ' + summary : ''}` }),
      el('span', { class: 'conf muted', text: `${Math.round(p.confidence * 100)}%` })),
    el('label', { class: 'param wide' }, el('span', { text: 'Color group' }),
      el('select', { onchange: (ev) => actions.setPartGroup([p], Number(ev.target.value)) }, ...groupOptions(p.color_group))),
    grid);
  card.append(el('p', { class: 'flag-text', text: 'Please confirm' }), confirmBox(key));
  return card;
}

function tailCard(parts, n) {
  const first = parts[0];
  const grid = el('div', { class: 'params' });
  for (const [k, label, type] of PARAMS.tail_stripe) {
    const field = `tails${n}.${k}`;
    grid.append(paramField(type, label, first.params[k], field, (raw, target) => {
      const ok = actions.setTailParam(parts, k, raw, field);
      if (target.isConnected) markInvalid(target, ok);
    }));
  }
  // Stripe colors: one select per stripe, applied to that stripe on every tail.
  const minTail = Math.min(...parts.map((p) => p.params.tail_index));
  const stripes = parts.filter((p) => p.params.tail_index === minTail)
    .sort((a, b) => a.params.stripe_index - b.params.stripe_index);
  const stripeList = el('div', { class: 'stripes' });
  for (const sp of stripes) {
    const same = parts.filter((p) => p.params.stripe_index === sp.params.stripe_index);
    stripeList.append(el('label', { class: 'param wide' },
      el('span', { text: `Stripe ${sp.params.stripe_index + 1}` }),
      el('select', { onchange: (ev) => actions.setPartGroup(same, Number(ev.target.value)) },
        ...groupOptions(sp.color_group))));
  }
  return el('div', { class: 'card' },
    el('div', { class: 'card-head' },
      el('strong', { text: 'Tails' }),
      el('span', { class: 'muted', text: ` ${first.params.tails_count} x ${stripes.length} stripe(s)` })),
    grid, stripeList);
}

export const views = {
  all() {
    views.photo();
    views.simControls();
    views.parts();
    views.groups();
    views.afterRender();
  },

  afterRender() {
    views.preview();
    views.warnings();
    views.flagsMarked();
    views.checks();
    views.download();
    views.status();
  },

  status() {
    const s = state;
    let text = '';
    if (s.analyzing) text = 'Analyzing...';
    else if (s.render.dirty || s.render.inFlight) text = 'Updating...';
    $('status').textContent = text;
    $('file-name').textContent = s.uploadName || 'No photo loaded';
  },

  photo() {
    const img = $('photo');
    const over = $('overlay-photo');
    if (state.photoUrl) {
      img.src = state.photoUrl;
      over.src = state.photoUrl;
      img.hidden = false;
      $('drop-hint').hidden = true;
    } else {
      img.hidden = true;
      $('drop-hint').hidden = false;
    }
  },

  simControls() {
    const box = $('color-pickers');
    box.replaceChildren(...COLOR_INDICES.map((i) => el('label', { class: 'picker' },
      el('span', { text: `Color ${i}` }),
      el('input', {
        type: 'color', value: state.sim.colors[i], 'aria-label': `Color ${i}`,
        oninput: (ev) => actions.setSimColor(i, ev.target.value),
      }))));
    $('c2-solid').classList.toggle('active', !state.sim.pattern);
    $('c2-pattern').classList.toggle('active', state.sim.pattern);
    $('rose-on').classList.toggle('active', state.sim.rose);
    $('rose-off').classList.toggle('active', !state.sim.rose);
    $('overlay-toggle').checked = state.sim.overlay;
    $('overlay-slider').disabled = !state.sim.overlay;
    $('overlay-slider').value = String(state.sim.opacity);
    $('source-toggle').checked = state.sim.source;
  },

  preview() {
    const box = $('preview');
    const src = $('svg-source');
    src.hidden = !state.sim.source;
    src.textContent = state.svg || '';
    if (!state.svg) {
      box.replaceChildren(el('p', { class: 'muted center', text: state.analyzing ? 'Analyzing the photo...' : 'The generated SVG appears here.' }));
      views.overlay();
      return;
    }
    const doc = new DOMParser().parseFromString(state.svg, 'image/svg+xml');
    if (doc.getElementsByTagName('parsererror').length) {
      box.replaceChildren(el('p', { class: 'muted center', text: 'The SVG could not be displayed.' }));
      return;
    }
    const svg = document.importNode(doc.documentElement, true);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    applyConfig(svg, {
      colors: state.sim.colors,
      patternIndex: state.sim.pattern ? 2 : null,
      patternImage: state.sim.pattern ? patternImage() : null,
      rose: state.sim.rose,
    });
    box.replaceChildren(svg);
    views.overlay();
  },

  overlay() {
    const over = $('overlay-photo');
    const on = state.sim.overlay && !!state.photoUrl;
    over.hidden = !on;
    $('preview').style.opacity = on ? String(state.sim.opacity / 100) : '1';
    const f = state.photoFrame;
    if (on && f) {
      Object.assign(over.style, { left: `${f.left}%`, top: `${f.top}%`, width: `${f.width}%`, height: 'auto', objectFit: '' });
    } else {
      Object.assign(over.style, { left: '0', top: '0', width: '100%', height: '100%', objectFit: 'contain' });
    }
  },

  parts() {
    const box = $('parts');
    const rose = $('model-rose');
    if (!state.model) {
      box.replaceChildren(el('p', { class: 'muted', text: state.analyzing ? 'Analyzing...' : 'No parts yet.' }));
      rose.disabled = true;
      rose.checked = false;
      return;
    }
    rose.disabled = false;
    state.invalid.clear(); // fresh inputs show the model values, which are valid
    rose.checked = !!state.model.rose;
    const cards = [];
    const tails = tailGroups(state.model);
    const tailFirst = new Map(tails.map((ps, n) => [ps[0], n]));
    for (const p of state.model.parts) {
      if (p.role !== 'tail_stripe') cards.push(partCard(p));
      else if (tailFirst.has(p)) cards.push(tailCard(tails[tailFirst.get(p)], tailFirst.get(p)));
    }
    box.replaceChildren(...cards);
    views.flagsMarked();
  },

  groups() {
    const box = $('groups');
    if (!state.model) {
      box.replaceChildren(el('p', { class: 'muted', text: 'No colors yet.' }));
      return;
    }
    box.replaceChildren(...state.model.groups.map((g) => {
      const key = `group:${g.id}`;
      const uses = state.model.parts.filter((p) => p.color_group === g.id).length;
      return el('div', { class: 'card group', 'data-flag': key },
        el('div', { class: 'card-head' },
          el('span', { class: 'swatch', style: `background:${hex(g.rgb)}`, title: hex(g.rgb) }),
          el('strong', { text: `Group ${g.id}` }),
          el('span', { class: 'muted', text: ` ${hex(g.rgb)}${g.pattern ? ', patterned' : ''}, ${Math.round(g.area * 100)}% area, ${uses} part(s)` }),
          el('span', { class: 'conf muted', text: `${Math.round(g.confidence * 100)}%` })),
        el('label', { class: 'param wide' }, el('span', { text: 'Category' }),
          el('select', { onchange: (ev) => actions.setGroupCategory(g, Number(ev.target.value)) },
            g.category === null ? el('option', { value: '', selected: true, disabled: true, text: '-' }) : null,
            ...[1, 2, 3, 4].map((c) => el('option', { value: String(c), selected: g.category === c, text: `Color ${c}` })))),
        el('p', { class: 'flag-text', text: 'Please confirm' }),
        confirmBox(key));
    }));
    views.flagsMarked();
  },

  warnings() {
    const flags = flagItems(state.model, state.issues, state.flagged)
      .filter((f) => f.key.startsWith('warning:') || f.key.startsWith('server:'));
    $('warnings-block').hidden = !flags.length;
    $('warnings').replaceChildren(...flags.map((f) => el('div', { class: 'card flagged', 'data-flag': f.key },
      el('p', { class: 'warn-text', text: f.text }),
      confirmBox(f.key))));
    views.flagsMarked();
  },

  // Toggle the flagged / confirmed look on every card carrying a data-flag key.
  flagsMarked() {
    const items = new Map(flagItems(state.model, state.issues, state.flagged).map((f) => [f.key, f]));
    for (const card of document.querySelectorAll('[data-flag]')) {
      const key = card.getAttribute('data-flag');
      const flagged = items.has(key);
      card.classList.toggle('flagged', flagged);
      card.classList.toggle('confirmed', flagged && state.confirmed.has(key));
      const box = card.querySelector(`input[data-confirm]`);
      if (box) box.checked = state.confirmed.has(key);
      const text = card.querySelector('.flag-text');
      if (text && flagged) text.textContent = `Please confirm: ${items.get(key).text}`;
    }
  },

  checks() {
    const list = $('checks');
    if (!state.svg) {
      list.replaceChildren(el('li', { class: 'muted', text: 'No SVG yet.' }));
      return;
    }
    const byCode = new Map();
    for (const i of state.issues) {
      if (!byCode.has(i.code)) byCode.set(i.code, []);
      byCode.get(i.code).push(i);
    }
    const known = new Set(CHECKS.map(([c]) => c));
    const rows = [...CHECKS, ...[...byCode.keys()].filter((c) => !known.has(c)).map((c) => [c, c])];
    const passed = rows.filter(([c]) => !byCode.has(c)).length;
    list.replaceChildren(
      el('li', { class: 'summary', text: `Checks: ${passed}/${rows.length} passed` }),
      ...rows.map(([code, label]) => {
        const fails = byCode.get(code) || [];
        const li = el('li', { class: fails.length ? 'fail' : 'pass', title: code },
          el('span', { class: 'mark', text: fails.length ? 'FAIL' : 'OK' }), ` ${label}`);
        if (fails.length) {
          li.append(el('ul', {}, ...fails.slice(0, 5).map((f) => el('li', { text: f.message + (f.element ? ` (${f.element})` : '') })),
            fails.length > 5 ? el('li', { text: `and ${fails.length - 5} more` }) : null));
        }
        return li;
      }));
  },

  download() {
    const blockers = downloadBlockers();
    $('download').disabled = blockers.length > 0;
    $('download-reason').textContent = blockers.length ? blockers.join(' ') : `Ready: ${state.filename}`;
  },
};

// ---------------------------------------------------------------- photo frame

// Approximate where the photo sits in the 600x850 SVG frame, the way
// rosette/vision/prepare.py frames it: the rosette's vertical extent maps to y 25..825
// and its middle to x 300. The foreground is found from alpha or the border color.
export function photoFrame(img) {
  const H = img.naturalHeight;
  const W = img.naturalWidth;
  if (!H || !W) return null;
  const h = Math.min(300, H);
  const w = Math.max(1, Math.round((W * h) / H));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const n = w * h;
  let transparent = 0;
  for (let i = 0; i < n; i++) if (d[i * 4 + 3] < 128) transparent++;
  let isFg;
  if (transparent / n >= 0.01) {
    isFg = (i) => d[i * 4 + 3] >= 128;
  } else {
    const border = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (x < 4 || y < 4 || x >= w - 4 || y >= h - 4) border.push(y * w + x);
      }
    }
    const mean = [0, 1, 2].map((k) => border.reduce((s, i) => s + d[i * 4 + k], 0) / border.length);
    const dist = (i) => Math.hypot(d[i * 4] - mean[0], d[i * 4 + 1] - mean[1], d[i * 4 + 2] - mean[2]);
    const spread = border.map(dist).sort((a, b) => a - b)[Math.floor(border.length * 0.95)] || 0;
    const thr = Math.max(30, 3 * spread);
    isFg = (i) => dist(i) > thr;
  }
  const minRun = Math.max(2, Math.round(w * 0.01));
  let top = -1;
  let bottom = -1;
  let left = w;
  let right = -1;
  for (let y = 0; y < h; y++) {
    let count = 0;
    let lo = -1;
    let hi = -1;
    for (let x = 0; x < w; x++) {
      if (isFg(y * w + x)) {
        count++;
        if (lo < 0) lo = x;
        hi = x;
      }
    }
    if (count >= minRun) {
      if (top < 0) top = y;
      bottom = y;
      left = Math.min(left, lo);
      right = Math.max(right, hi);
    }
  }
  if (top < 0 || bottom - top < h * 0.2) return null;
  const s = (bottom + 1 - top) / 800; // canvas px per SVG unit
  const axis = (left + right + 1) / 2;
  return {
    left: ((300 - axis / s) / 600) * 100,
    top: ((25 - top / s) / 850) * 100,
    width: (w / s / 600) * 100,
  };
}

// ---------------------------------------------------------------- init

function init() {
  $('file-input').addEventListener('change', (ev) => {
    actions.loadFile(ev.target.files[0]);
    ev.target.value = '';
  });
  const zone = $('drop-zone');
  zone.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    zone.classList.add('over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (ev) => {
    ev.preventDefault();
    zone.classList.remove('over');
    if (ev.dataTransfer.files.length) actions.loadFile(ev.dataTransfer.files[0]);
  });
  $('photo').addEventListener('load', () => {
    state.photoFrame = photoFrame($('photo'));
    views.overlay();
  });
  $('error-close').addEventListener('click', hideError);
  $('c2-solid').addEventListener('click', () => actions.setSim({ pattern: false }));
  $('c2-pattern').addEventListener('click', () => actions.setSim({ pattern: true }));
  $('rose-on').addEventListener('click', () => actions.setSim({ rose: true }));
  $('rose-off').addEventListener('click', () => actions.setSim({ rose: false }));
  $('overlay-toggle').addEventListener('change', (ev) => actions.setSim({ overlay: ev.target.checked }));
  $('overlay-slider').addEventListener('input', (ev) => {
    state.sim.opacity = Number(ev.target.value);
    views.overlay();
  });
  $('source-toggle').addEventListener('change', (ev) => actions.setSim({ source: ev.target.checked }));
  $('model-rose').addEventListener('change', (ev) => actions.setModelRose(ev.target.checked));
  $('download').addEventListener('click', () => actions.download());
  views.all();
}

init();
