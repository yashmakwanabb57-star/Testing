// Workbench UI. Plain ES module, no framework, no network access beyond the local API.
// Sections: constants, state, api, derived (pure), actions (mutate state), views (DOM),
// init (event wiring). A guided mode can reuse state, api, derived and actions and add
// its own views.
import { applyConfig, glitterPattern, COLOR_INDICES } from './simulator.js';
import {
  MAX_COUNT, POINT_KINDS, errorText, hex, flagItems,
  downloadBlockers as blockersFor, groupColor, tailGroups, parseValue, tailCheckKey, sourceNotice,
  convertBlocker, resetForFile, checklistItems, paramShown, headChoice, switchHead, templateText, usesGroup,
} from './logic.js';

export { errorText, flagItems, parseValue, tailGroups, hex };
export const downloadBlockers = () => blockersFor(state);

// Called after every change that can affect the blockers or the simulator (guided mode
// listens here to keep its Next button and checklists current).
export const listeners = [];
function notify() {
  for (const f of listeners) f();
}

// ---------------------------------------------------------------- constants

const DEBOUNCE_MS = 200;
const DEFAULT_COLORS = { 1: '#2e6b4a', 2: '#9fb7c4', 3: '#d9b26f', 4: '#8a5a9e' };

// Validation codes from rosette/validate.py, in check order.
const CHECKS = [
  ['xml', 'Well-formed XML'],
  ['root', 'Root element and viewBox'],
  ['dup_id', 'Unique IDs'],
  ['missing_group', 'Required layer groups present'],
  ['bad_parent', 'Layer groups nested correctly'],
  ['group_order', 'Layer groups in shop paint order'],
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
  tail_stripe: 'Tails', unknown: 'Unrecognised shape', template: 'Head (library)',
};

// Editable params per role. type: num | int | kind | bool.
const PARAMS = {
  head: [['r', 'Radius', 'num'], ['pinked', 'Pinked edge', 'bool'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  ring: [['r_outer', 'Outer radius', 'num'], ['r_inner', 'Inner radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  points: [['kind', 'Kind', 'kind'], ['count', 'Count', 'int'], ['r_inner', 'Inner radius', 'num'],
    ['r_outer', 'Outer radius', 'num'], ['rotation_deg', 'Rotation (deg)', 'num'],
    ['width_ratio', 'Width ratio', 'num'], ['inner_ratio', 'Inner ratio (bow)', 'num'],
    ['edge_width', 'Edge strip width', 'num'],
    ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  center: [['r', 'Radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  template: [['r', 'Head radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num'],
    ['rotation_deg', 'Rotation (deg)', 'num']],
  unknown: [['r', 'Radius', 'num'], ['cx', 'Center x', 'num'], ['cy', 'Center y', 'num']],
  tail_stripe: [['tails_count', 'Tails', 'int'], ['top_y', 'Top y', 'num'], ['bottom_y', 'Bottom y', 'num'],
    ['top_width', 'Top width', 'num'], ['bottom_width', 'Bottom width', 'num'], ['spread_deg', 'Spread (deg)', 'num']],
};

// ---------------------------------------------------------------- state

export const state = {
  photoUrl: null,
  uploadName: '',
  pendingFile: null, // chosen photo; Convert to SVG sends it to /api/analyze
  filename: null, // download name from /api/analyze
  model: null, // RosetteModel JSON, edited in place
  svg: null, // clean SVG text from the last successful render
  issues: [],
  flagged: false, // server is_flagged of the last render
  headFit: null, // head fit measured at conversion (information, not a flag)
  confirmed: new Set(), // flag keys the user confirmed
  invalid: new Set(), // input field keys holding an invalid value
  analyzing: false,
  analyzeSeq: 0, // latest conversion; older analyze responses are dropped
  render: { seq: 0, timer: null, dirty: false, inFlight: false, error: null },
  sim: {
    colors: { ...DEFAULT_COLORS },
    touched: new Set(), // color indices the user picked by hand
    pattern: false, // Color 2 uses the sample pattern
    rose: true,
    tried: new Set(), // 'pattern' / 'rose' once the user has toggled them since the upload
    overlay: false,
    opacity: 50,
    source: false,
  },
  photoFrame: null, // overlay placement of the photo in the 600x850 frame
  guided: { mode: 'workbench', step: 1, reached: 1 }, // guided mode view state
};

// ---------------------------------------------------------------- api


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

// ---------------------------------------------------------------- actions

export const actions = {
  // Choosing or dropping a file only shows the photo; Convert to SVG analyzes it. The
  // old model is cleared right away so it cannot be downloaded under the new name.
  chooseFile(file) {
    if (!file) return;
    clearTimeout(state.render.timer);
    resetForFile(state);
    if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
    Object.assign(state, { photoUrl: URL.createObjectURL(file), uploadName: file.name, pendingFile: file });
    hideError();
    views.all();
  },

  async convert() {
    if (convertBlocker(state)) return;
    const file = state.pendingFile;
    clearTimeout(state.render.timer);
    resetForFile(state);
    state.analyzing = true;
    const token = state.analyzeSeq;
    let scrollTop = false;
    hideError();
    views.all();
    try {
      const res = await api.analyze(file);
      if (token !== state.analyzeSeq) return;
      Object.assign(state, {
        model: res.model, svg: res.svg, issues: res.issues || [], flagged: !!res.flagged,
        filename: res.filename, headFit: res.head_fit || null,
      });
      state.sim.touched.clear();
      state.sim.pattern = false;
      state.sim.rose = !!res.model.rose;
      actions.defaultColors();
      scrollTop = true;
    } catch (e) {
      if (token === state.analyzeSeq) showError(e.message);
    } finally {
      if (token === state.analyzeSeq) {
        state.analyzing = false;
        views.all();
        if (scrollTop) views.scrollToTop();
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
    actions.unconfirm(`part:${part.id}`);
    actions.scheduleRender();
    return true;
  },

  // A confirmation covers what was on screen when it was given: editing the item clears it.
  unconfirm(...keys) {
    let changed = false;
    for (const k of keys) changed = state.confirmed.delete(k) || changed;
    if (changed) {
      views.flagsMarked();
      views.download();
    }
  },

  // Validate a value without applying it (marks the field invalid or valid).
  checkField(type, raw, field) {
    const ok = parseValue(type, raw).ok;
    if (ok) state.invalid.delete(field);
    else state.invalid.add(field);
    views.download();
    return ok;
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
    actions.unconfirm(...tailConfirmations());
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
    actions.unconfirm(...tailConfirmations());
    rebuildParts(field);
    actions.scheduleRender();
    return true;
  },

  // Library head <-> rule-based head (the other one waits in model.alternatives).
  switchHead() {
    const before = state.model.parts.filter((p) => p.role !== 'tail_stripe').map((p) => `part:${p.id}`);
    switchHead(state.model);
    actions.unconfirm(...before, 'check:head');
    views.parts();
    views.groups();
    actions.scheduleRender();
  },

  // A template part's reference Color c is drawn in color group groupId.
  setTemplateGroup(part, c, groupId) {
    part.params[`group_${c}`] = groupId;
    actions.unconfirm(`part:${part.id}`);
    actions.scheduleRender();
  },

  setPartGroup(parts, groupId) {
    for (const p of parts) p.color_group = groupId;
    const tails = parts.some((p) => p.role === 'tail_stripe') ? tailConfirmations() : [];
    actions.unconfirm(...parts.map((p) => `part:${p.id}`), ...tails);
    actions.scheduleRender();
  },

  setGroupCategory(group, category) {
    group.category = category;
    // Also clears warnings that name this group's color (e.g. "Color #xxxxxx is partly patterned").
    const h = hex(group.rgb);
    actions.unconfirm(`group:${group.id}`,
      ...state.model.warnings.filter((w) => w.includes(h)).map((w) => `warning:${w}`));
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
    for (const k of ['pattern', 'rose']) {
      if (k in patch && patch[k] !== state.sim[k]) state.sim.tried.add(k);
    }
    Object.assign(state.sim, patch);
    views.simControls();
    views.preview();
    notify();
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

// Guided-mode "Looks right" keys of the tail groups. A tail edit can regroup or reorder
// the stripes, so it clears all of them.
function tailConfirmations() {
  return [...state.confirmed].filter((k) => k.startsWith('tails:'));
}

// ---------------------------------------------------------------- views

const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...children) {
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
    el('span', { class: 'lbl-wb', text: ' Confirmed' }),
    el('span', { class: 'lbl-guided', text: ' Looks right' }));
}

// onInput(raw, target) applies a value. With onCommit set, typing only validates
// (onInput) and the value is applied on change (onCommit), for fields whose edit
// rebuilds the card and would otherwise take the focus away mid-typing.
function paramField(type, label, value, field, onInput, onCommit = null) {
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
      onchange: onCommit ? (ev) => onCommit(ev.target.value, ev.target) : null,
    });
  }
  return el('label', { class: 'param' }, el('span', { text: label }), input);
}

function markInvalid(target, ok) {
  target.classList.toggle('invalid', !ok);
}

function partSummary(p) {
  if (p.role === 'template') return ` ${templateText(p)}`;
  const summary = p.role === 'points' ? `${p.params.kind} x ${p.params.count}` : '';
  return ` ${p.id}${summary ? ' - ' + summary : ''}`;
}

// Re-render the parts list; if the edited field had focus (Enter in a number field, a
// select change), focus its replacement so keyboard work can continue.
function rebuildParts(field) {
  const active = document.activeElement;
  const had = active && active.getAttribute && active.getAttribute('data-field') === field;
  views.parts();
  if (had) {
    const next = document.querySelector(`[data-field="${CSS.escape(field)}"]`);
    if (next) next.focus();
  }
}

function partCard(p) {
  const key = `part:${p.id}`;
  const fields = PARAMS[p.role] || PARAMS.unknown;
  const grid = el('div', { class: 'params' });
  for (const [k, label, type] of fields) {
    if (k === 'inner_ratio' && p.params.kind !== 'bow') continue;
    if (!(k in p.params) && k !== 'inner_ratio') continue;
    const field = `${p.id}.${k}`;
    const value = paramShown(p.params, k);
    grid.append(paramField(type, label, value, field, (raw, target) => {
      const ok = actions.setParam(p, k, type, raw, field);
      if (target.type === 'number') markInvalid(target, ok);
      if (k === 'kind') rebuildParts(field);
      else if (ok) card.querySelector('.part-summary').textContent = partSummary(p);
    }));
  }
  const colors = p.role === 'template'
    ? Object.keys(p.params).filter((k) => /^group_\d$/.test(k)).sort().map((k) => {
      const c = Number(k.slice(6));
      return el('label', { class: 'param wide' }, el('span', { text: `Template Color ${c} drawn as` }),
        el('select', { onchange: (ev) => actions.setTemplateGroup(p, c, Number(ev.target.value)) },
          ...groupOptions(Number(p.params[k]))));
    })
    : [el('label', { class: 'param wide' }, el('span', { text: 'Color group' }),
      el('select', { onchange: (ev) => actions.setPartGroup([p], Number(ev.target.value)) }, ...groupOptions(p.color_group)))];
  const card = el('div', { class: 'card', 'data-flag': key },
    el('div', { class: 'card-head' },
      el('strong', { text: ROLE_LABELS[p.role] || p.role }),
      el('span', { class: 'muted part-summary', text: partSummary(p) }),
      el('span', { class: 'conf muted', text: `${Math.round(p.confidence * 100)}%` })),
    ...colors,
    grid);
  if (p.role === 'template') {
    card.append(el('button', { type: 'button', class: 'btn small', text: 'Use the rule-based head instead',
      onclick: () => actions.switchHead() }));
  }
  card.append(el('p', { class: 'flag-text', text: 'Please confirm' }), confirmBox(key));
  return card;
}

function tailCard(parts, n) {
  const first = parts[0];
  const grid = el('div', { class: 'params' });
  for (const [k, label, type] of PARAMS.tail_stripe) {
    const field = `tails${n}.${k}`;
    if (k === 'tails_count') {
      grid.append(paramField(type, label, first.params[k], field,
        (raw, target) => markInvalid(target, actions.checkField(type, raw, field)),
        (raw, target) => {
          const ok = actions.setTailParam(parts, k, raw, field);
          if (target.isConnected) markInvalid(target, ok);
        }));
      continue;
    }
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
  const key = tailCheckKey(n);
  return el('div', { class: 'card', 'data-flag': key },
    el('div', { class: 'card-head' },
      el('strong', { text: 'Tails' }),
      el('span', { class: 'muted', text: ` ${first.params.tails_count} x ${stripes.length} stripe(s)` })),
    grid, stripeList, el('p', { class: 'flag-text', text: 'Please confirm' }), confirmBox(key));
}

export const views = {
  // A new upload starts at the top of the page and of the parts panel.
  scrollToTop() {
    window.scrollTo(0, 0);
    for (const id of ['col-model', 'parts']) $(id).scrollTop = 0;
  },

  all() {
    views.photo();
    views.simControls();
    views.parts();
    views.groups();
    views.afterRender();
  },

  afterRender() {
    views.preview();
    views.checklist();
    views.warnings();
    views.flagsMarked();
    views.checks();
    views.download();
    views.status();
  },

  status() {
    const s = state;
    let text = '';
    if (s.analyzing) text = 'Converting...';
    else if (s.render.dirty || s.render.inFlight) text = 'Updating...';
    $('status').textContent = text;
    $('file-name').textContent = s.uploadName || 'No photo loaded';
    $('convert').disabled = !!convertBlocker(s);
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
    views.source();
    if (!state.svg) {
      box.replaceChildren(el('p', { class: 'muted center', text: state.analyzing ? 'Converting the photo...' : (state.pendingFile ? 'Press Convert to SVG to make the SVG.' : 'The generated SVG appears here.') }));
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

  // The source of the last render, with a not-ready banner while Download is blocked.
  source() {
    $('svg-source').hidden = !state.sim.source;
    $('svg-source-text').textContent = state.svg || '';
    const note = sourceNotice(state);
    const box = $('svg-source-note');
    box.hidden = !note;
    box.textContent = note ? `${note} ${downloadBlockers().join(' ')}` : '';
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
      box.replaceChildren(el('p', { class: 'muted', text: state.analyzing ? 'Converting...' : 'No parts yet.' }));
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
    if (headChoice(state.model) === 'rules') {
      const t = state.model.alternatives.find((p) => p.role === 'template');
      cards.push(el('div', { class: 'card' },
        el('p', { class: 'muted', text: `${templateText(t)} (not used: the rule-based head is drawn).` }),
        el('button', { type: 'button', class: 'btn small', text: 'Use the library head instead',
          onclick: () => actions.switchHead() })));
    }
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
      const uses = state.model.parts.filter((p) => usesGroup(p, g.id)).length;
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

  // The fixed checklist (logic.js checklistItems): always shown with a model, confirmed
  // like a flag, never counted as one.
  checklist() {
    const items = checklistItems(state);
    $('checklist-block').hidden = !items.length;
    $('checklist').replaceChildren(...items.map((c) => el('div', { class: 'card checklist', 'data-flag': c.key },
      el('p', { class: 'check-text', text: c.text }),
      el('p', { class: 'muted', text: c.info }),
      confirmBox(c.key))));
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
      card.classList.toggle('checked', state.confirmed.has(key)); // guided "Looks right"
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
    views.source();
    $('download-reason').textContent = blockers.length ? blockers.join(' ') : `Ready: ${state.filename}`;
    notify();
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
    actions.chooseFile(ev.target.files[0]);
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
    if (ev.dataTransfer.files.length) actions.chooseFile(ev.dataTransfer.files[0]);
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
  $('convert').addEventListener('click', () => actions.convert());
  views.all();
}

init();
