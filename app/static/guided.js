// Guided mode: a step-by-step view over the workbench. It owns no model logic: state,
// actions (edits, confirmations, renders), views and the download blockers all come from
// app.js and logic.js. Each step borrows the workbench's own DOM nodes (photo, preview,
// parts list, warnings, checks, download) and puts them back when switching to Workbench,
// so edits and confirmations are the same in both modes.
import { state, actions, views, listeners, el, hex } from './app.js';
import { GUIDED_STEPS, advanceBlocker, canJump, colorQuestions, flagItems, partChecks } from './logic.js';

const MODE_KEY = 'rosette.mode';
const LAST = GUIDED_STEPS.length;
const $ = (id) => document.getElementById(id);

// Workbench nodes a step may borrow; each gets a placeholder so it can go back home.
const MOVABLE = ['drop-zone', 'stage', 'sim-controls', 'svg-source', 'parts', 'model-rose-row', 'warnings-block', 'checks', 'download-box'];
const home = new Map();

const INTRO = {
  1: 'Upload a photo of the rosette: drop it on the photo area or use Upload photo.',
  2: 'Compare every part with the photo. Fix any value that is off, then tick "Looks right" on each card.',
  3: 'One question per color, flagged colors first. Answer Yes, or use Change... to pick the right color.',
  4: 'Try the shop options on the preview: switch Color 2 between Solid and Pattern, and the Rose on and off.',
  5: 'Last checks. Download is enabled once every check passes and every flagged item is confirmed.',
};

let rendered = 0; // step whose layout is on screen (0: none)
let questionSig = '';
const changing = new Set(); // color question keys with the category select open

// ---------------------------------------------------------------- borrowing nodes

function rememberHomes() {
  for (const id of MOVABLE) {
    const node = $(id);
    const ph = document.createComment(`home of #${id}`);
    node.before(ph);
    home.set(id, ph);
  }
}

function sendHome() {
  for (const [id, ph] of home) ph.after($(id));
  rendered = 0;
}

// ---------------------------------------------------------------- step content

function uploadStatus() {
  const box = el('div', { class: 'g-status', id: 'g-upload-status' });
  fillUploadStatus(box);
  return box;
}

function fillUploadStatus(box) {
  let text = 'No photo yet.';
  if (state.analyzing) text = 'Analyzing the photo...';
  else if (state.model) {
    const flags = flagItems(state.model, state.issues, state.flagged).length;
    text = `Found ${partChecks(state.model).length} part card(s) and ${state.model.groups.length} color(s).`
      + (flags ? ` ${flags} item(s) are flagged for a closer look.` : '');
  }
  box.textContent = text;
}

function listWords(ws) {
  return ws.length < 2 ? ws.join('') : `${ws.slice(0, -1).join(', ')} and ${ws[ws.length - 1]}`;
}

function questionCard(q) {
  const g = q.group;
  const confirmed = state.confirmed.has(q.key);
  const swatch = el('span', { class: 'swatch', style: `background:${hex(g.rgb)}`, title: hex(g.rgb) });
  // "Is the <swatch> stripe Color 3?"; a color on mixed parts: "Is <swatch> Color 1?"
  const subject = q.noun ? ['the ', swatch, ` ${q.noun}`] : [swatch];
  const text = g.category === null || g.category === undefined
    ? el('p', { class: 'q-text' }, 'Which color is ', ...subject, '?')
    : el('p', { class: 'q-text' }, 'Is ', ...subject, ` Color ${g.category}?`);
  const used = q.usedOn.length ? `used on ${listWords(q.usedOn.map((w) => `${w}s`))}` : 'not used';
  const meta = el('p', {
    class: 'muted q-meta',
    text: `${hex(g.rgb)}${g.pattern ? ', patterned' : ''}, ${used}, ${Math.round(g.area * 100)}% area, `
      + `${Math.round(g.confidence * 100)}% sure${q.flagged ? ' - please check' : ''}`,
  });
  const yes = el('button', {
    type: 'button', class: confirmed ? 'btn small' : 'btn small primary',
    disabled: confirmed || g.category === null || g.category === undefined,
    text: confirmed ? 'Confirmed' : 'Yes',
    onclick: () => actions.confirm(q.key, true),
  });
  const change = el('button', {
    type: 'button', class: 'btn small', text: 'Change...', 'aria-expanded': changing.has(q.key) ? 'true' : 'false',
    onclick: () => {
      if (changing.has(q.key)) changing.delete(q.key);
      else changing.add(q.key);
      renderQuestions(true);
    },
  });
  const row = el('div', { class: 'q-actions' }, yes, change);
  if (changing.has(q.key)) {
    row.append(el('select', {
      'aria-label': `Color for group ${g.id}`,
      onchange: (ev) => {
        changing.delete(q.key);
        actions.setGroupCategory(g, Number(ev.target.value));
        renderQuestions(true);
      },
    },
    el('option', { value: '', selected: true, disabled: true, text: 'Pick a color' }),
    ...[1, 2, 3, 4].map((c) => el('option', { value: String(c), text: `Color ${c}${c === g.category ? ' (current)' : ''}` }))));
  }
  const cls = ['card', 'question'];
  if (q.flagged) cls.push('flagged');
  if (confirmed) cls.push('checked');
  if (q.flagged && confirmed) cls.push('confirmed');
  return el('div', { class: cls.join(' ') }, text, meta, row);
}

function renderQuestions(force = false) {
  const box = $('g-questions');
  if (!box) return;
  const qs = colorQuestions(state.model, state.issues, state.flagged);
  const sig = JSON.stringify([qs.map((q) => [q.key, q.group.category, q.flagged, state.confirmed.has(q.key), q.uses]),
    [...changing]]);
  if (!force && sig === questionSig) return;
  questionSig = sig;
  box.replaceChildren(...(qs.length ? qs.map(questionCard) : [el('p', { class: 'muted', text: 'No colors yet.' })]));
}

function checklistItem(done, text) {
  return el('li', { class: done ? 'pass' : 'todo' }, el('span', { class: 'mark', text: done ? 'DONE' : 'TO DO' }), ` ${text}`);
}

function renderChecklist() {
  const box = $('g-checklist');
  if (!box) return;
  box.replaceChildren(
    checklistItem(state.sim.tried.has('pattern'), 'Switch Color 2 between Solid and Pattern'),
    checklistItem(state.sim.tried.has('rose'), 'Switch the Rose on and off'));
}

// Per step: workbench nodes for the visual column, then the task column content.
function layout(step) {
  const intro = el('p', { class: 'g-intro', text: INTRO[step] });
  switch (step) {
    case 1:
      return { visual: ['drop-zone'], task: [intro, el('label', { class: 'btn', for: 'file-input', text: 'Choose photo...' }), uploadStatus()] };
    case 2:
      return { visual: ['drop-zone', 'stage'], task: [intro, 'parts', 'model-rose-row'] };
    case 3:
      return { visual: ['drop-zone', 'stage'], task: [intro, el('div', { class: 'list', id: 'g-questions' }), 'warnings-block'] };
    case 4:
      return { visual: ['drop-zone', 'stage'], task: [intro, 'sim-controls', el('ul', { class: 'checks g-checklist', id: 'g-checklist' }), 'svg-source'] };
    default:
      return { visual: ['stage'], task: [intro, 'checks', 'download-box'] };
  }
}

const node = (x) => (typeof x === 'string' ? $(x) : x);

function renderStep() {
  const step = state.guided.step;
  sendHome();
  const { visual, task } = layout(step);
  $('g-visual').replaceChildren(...visual.map(node));
  $('g-task').replaceChildren(...task.map(node));
  $('g-visual').className = `g-visual n${visual.length}`;
  const info = GUIDED_STEPS[step - 1];
  $('g-title').textContent = `Step ${step} of ${LAST} - ${info.title}`;
  rendered = step;
  questionSig = '';
  changing.clear();
  renderQuestions(true);
  renderChecklist();
  refreshNav();
}

// ---------------------------------------------------------------- navigation

function refreshNav() {
  const { step, reached } = state.guided;
  const crumbs = [];
  GUIDED_STEPS.forEach((info, i) => {
    if (i) crumbs.push(el('span', { class: 'sep', 'aria-hidden': 'true', text: '>' }));
    const current = info.n === step;
    crumbs.push(el('button', {
      type: 'button', class: current ? 'crumb current' : 'crumb',
      'aria-current': current ? 'step' : null,
      disabled: !current && !canJump(step, info.n, state, reached),
      text: `${info.n} ${info.label}`,
      onclick: () => go(info.n),
    }));
  });
  $('g-crumbs').replaceChildren(...crumbs);
  const blocker = advanceBlocker(step, state);
  $('g-back').disabled = step === 1;
  $('g-next').hidden = step === LAST;
  $('g-next').disabled = !!blocker;
  $('g-reason').textContent = step === LAST ? '' : (blocker || '');
}

function go(step) {
  const g = state.guided;
  if (step === g.step || !canJump(g.step, step, state, g.reached)) return;
  g.step = step;
  renderStep();
  window.scrollTo(0, 0);
}

function next() {
  const g = state.guided;
  if (advanceBlocker(g.step, state)) return;
  g.reached = Math.max(g.reached, g.step + 1); // Next is what reaches a step
  go(g.step + 1);
}

// Called by app.js after every relevant change, in either mode.
function refresh() {
  if (state.guided.mode !== 'guided') return;
  if (rendered !== state.guided.step) {
    renderStep(); // a new upload moved the guided view back to step 1
    return;
  }
  const status = $('g-upload-status');
  if (status) fillUploadStatus(status);
  renderQuestions();
  renderChecklist();
  refreshNav();
}

// ---------------------------------------------------------------- mode switch

function setMode(mode, save = true) {
  state.guided.mode = mode;
  if (save) {
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      // storage unavailable (private mode): the choice just is not remembered
    }
  }
  const guided = mode === 'guided';
  document.body.classList.toggle('mode-guided', guided);
  $('workbench').hidden = guided;
  $('guided').hidden = !guided;
  for (const [id, on] of [['mode-workbench', !guided], ['mode-guided', guided]]) {
    $(id).classList.toggle('active', on);
    $(id).setAttribute('aria-pressed', String(on));
  }
  if (guided) renderStep();
  else {
    sendHome();
    views.flagsMarked();
  }
}

function savedMode() {
  try {
    return localStorage.getItem(MODE_KEY) === 'guided' ? 'guided' : 'workbench';
  } catch {
    return 'workbench';
  }
}

function init() {
  rememberHomes();
  listeners.push(refresh);
  $('mode-workbench').addEventListener('click', () => setMode('workbench'));
  $('mode-guided').addEventListener('click', () => setMode('guided'));
  $('g-back').addEventListener('click', () => go(state.guided.step - 1));
  $('g-next').addEventListener('click', next);
  setMode(savedMode(), false);
}

init();
