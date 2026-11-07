// Node tests for the library head (H7) logic in logic.js.
// Run: node --test tests/js/template.test.mjs (tests/test_js.py runs it when node is on PATH)
import test from 'node:test';
import assert from 'node:assert/strict';
import { colorQuestions, flagItems, headCandidates, headChoice, automaticHeadText, headFitOf, libraryChoices, libraryOverride, pickNeedsServer, renderBody, partGroups, switchHead, templateText, usesGroup } from '../../app/static/logic.js';

function model(score = 0.8) {
  return {
    rose: false,
    warnings: [],
    groups: [
      { id: 0, rgb: [38, 82, 64], pattern: false, texture: 1, area: 0.5, category: 1, confidence: 0.93 },
      { id: 1, rgb: [208, 173, 114], pattern: false, texture: 1, area: 0.3, category: 3, confidence: 0.9 },
    ],
    parts: [
      { id: 't0', role: 'tail_stripe', params: { tail_index: 0, stripe_index: 0, tails_count: 1 }, color_group: 0, z: 0, confidence: 0.9 },
      { id: 'template', role: 'template', params: { template_id: 'S29-15', cx: 300, cy: 150, r: 120, rotation_deg: 0, score, group_1: 1, group_2: 0 }, color_group: 1, z: 10, confidence: score },
    ],
    alternatives: [
      { id: 'head', role: 'head', params: { cx: 300, cy: 150, r: 120, pinked: 1 }, color_group: 0, z: 10, confidence: 0.4 },
    ],
  };
}

test('template card text names the template and the match', () => {
  assert.equal(templateText(model().parts[1]), 'Head shape: from library template S29-15, match 80%');
  assert.ok(!/[–—]/.test(templateText(model().parts[1])));
  const p = { ...model().parts[1], params: { ...model().parts[1].params, points: 12 } };
  assert.equal(templateText(p), 'Head shape: from library template S29-15, match 80%, 12 points');
});

test('a template part uses every mapped group', () => {
  const p = model().parts[1];
  assert.deepEqual(partGroups(p).sort(), [0, 1]);
  assert.ok(usesGroup(p, 0) && usesGroup(p, 1) && !usesGroup(p, 2));
  const q = colorQuestions(model()).find((x) => x.group.id === 1);
  assert.equal(q.noun, 'head shape');
});

test('switchHead swaps the template and the rule-based head, tails stay', () => {
  const m = model();
  assert.equal(headChoice(m), 'template');
  switchHead(m);
  assert.deepEqual(m.parts.map((p) => p.id), ['t0', 'head']);
  assert.deepEqual(m.alternatives.map((p) => p.id), ['template']);
  assert.equal(headChoice(m), 'rules');
  switchHead(m);
  assert.deepEqual(m.parts.map((p) => p.id), ['t0', 'template']);
  assert.equal(headChoice({ ...m, parts: m.parts.slice(0, 1), alternatives: [] }), null);
});

test('switchHead moves the hidden head colors out of the colors list and back', () => {
  const m = model();
  m.alternatives[0].color_group = 2;
  m.alternative_groups = [{ id: 2, rgb: [1, 2, 3], pattern: false, texture: 1, area: 0.1, category: 2, confidence: 0.9 }];
  switchHead(m);
  assert.deepEqual(m.groups.map((g) => g.id), [0, 2]);
  assert.deepEqual(m.alternative_groups.map((g) => g.id), [1]);
  switchHead(m);
  assert.deepEqual(m.groups.map((g) => g.id), [0, 1]);
  assert.deepEqual(m.alternative_groups.map((g) => g.id), [2]);
});

test('a weak template match is flagged, a good one is not', () => {
  assert.ok(flagItems(model(0.45), [], false).some((f) => f.key === 'part:template' && /weak match/.test(f.text)));
  assert.ok(!flagItems(model(0.8), [], false).some((f) => f.key === 'part:template'));
});

// The head card's picker (fix round 4): the rule-based head and the library candidates.
function pickerModel() {
  const m = model(0.8);
  m.parts[1].params = { ...m.parts[1].params, rank: 1, points: 6, description: '6 loops, 6 kites, ring, center' };
  m.alternatives.push({ id: 'template_3', role: 'template', params: { template_id: 'S57-17', cx: 300, cy: 150, r: 120, rotation_deg: 0, score: 0.55, rank: 3, group_1: 0 }, color_group: 0, z: 10, confidence: 0.55 });
  m.alternatives.push({ id: 'template_2', role: 'template', params: { template_id: 'S30-15', cx: 300, cy: 150, r: 120, rotation_deg: 0, score: 0.7, rank: 2, group_1: 1, group_2: 2 }, color_group: 1, z: 10, confidence: 0.7 });
  m.alternative_groups = [{ id: 2, rgb: [1, 2, 3], pattern: false, texture: 1, area: 0.1, category: 2, confidence: 0.9 }];
  return m;
}

test('headCandidates lists the rule-based head and the library candidates by rank', () => {
  const c = headCandidates(pickerModel());
  assert.deepEqual(c.map((x) => x.choice), ['rules', 'S29-15', 'S30-15', 'S57-17']);
  assert.deepEqual(c.map((x) => x.drawn), [false, true, false, false]);
  assert.equal(c[1].text, 'Library template S29-15 - match 80%, 6 points - 6 loops, 6 kites, ring, center');
  assert.equal(c[2].text, 'Library template S30-15 - match 70%');
  assert.equal(c[0].text, 'Rule-based head (the detected parts)');
  assert.ok(c.every((x) => !/[–—]/.test(x.text)));
  assert.deepEqual(headCandidates({ ...model(), parts: model().parts.slice(0, 1), alternatives: [] }), []);
});

test('switchHead draws the picked candidate, the others wait, colors follow', () => {
  const m = pickerModel();
  switchHead(m, 'S30-15');
  assert.deepEqual(m.parts.map((p) => p.id), ['t0', 'template_2']);
  assert.deepEqual(m.alternatives.map((p) => p.id).sort(), ['head', 'template', 'template_3']);
  assert.deepEqual(m.groups.map((g) => g.id), [0, 1, 2]);
  assert.equal(headChoice(m), 'template');
  assert.equal(headCandidates(m).find((x) => x.drawn).choice, 'S30-15');
  switchHead(m, 'rules');
  assert.deepEqual(m.parts.map((p) => p.id), ['t0', 'head']);
  assert.deepEqual(m.alternative_groups.map((g) => g.id), [1, 2]);
  // the toggle without a choice goes back to the best match (rank 1)
  switchHead(m);
  assert.deepEqual(m.parts.map((p) => p.id), ['t0', 'template']);
  assert.throws(() => switchHead(m, 'S99-15'));
});

test('a picked weak candidate is flagged like any library head', () => {
  const m = pickerModel();
  switchHead(m, 'S57-17');
  assert.ok(flagItems(m, [], false).some((f) => f.key === 'part:template_3' && /weak match/.test(f.text)));
});

test('switchHead swaps the head-dependent warnings and keeps the library head placement', () => {
  const m = pickerModel();
  m.warnings = ['general', 'heads differ'];
  m.head_states = {
    'S29-15': { warnings: ['heads differ'], fit: { agreement: 0.8, silhouette: 0.9 } },
    'S30-15': { warnings: [], fit: { agreement: 0.7, silhouette: 0.85 } },
    rules: { warnings: ['far: head match 42%'], fit: { agreement: 0.42, silhouette: 0.6 } },
  };
  m.parts[1].params.cx = 310;
  m.parts[1].params.r = 99;
  switchHead(m, 'S30-15');
  assert.deepEqual(m.warnings, ['general']);
  assert.equal(m.parts[1].params.cx, 310);
  assert.equal(m.parts[1].params.r, 99);
  switchHead(m, 'rules');
  assert.deepEqual(m.warnings, ['general', 'far: head match 42%']);
  assert.deepEqual(headFitOf(m), { agreement: 0.42, silhouette: 0.6 });
  switchHead(m, 'S29-15');
  assert.deepEqual(m.warnings, ['general', 'heads differ']);
  assert.equal(headFitOf({ ...m, head_states: {} }), undefined);
});

// H8: the library chooser (the list comes from /api/library; ids outside the candidates
// are placed and checked by the server).
const LIBRARY = [
  { id: 'S9-19', description: '4 loops, 6-point star, center' },
  { id: 'A1-15', description: 'disc, center' },
  { id: 'S29-15', description: '6 loops, 6 kites, 12 strips, 6-point star, ring, center' },
  { id: 'S11a-15', description: '2 points, 10 kites, ring, center' },
];

test('libraryChoices lists the candidates first, then every other template by id, filtered', () => {
  const c = libraryChoices(pickerModel(), LIBRARY);
  assert.deepEqual(c.map((x) => x.choice), ['S29-15', 'S30-15', 'S57-17', 'A1-15', 'S11a-15', 'S9-19']);
  assert.equal(c[3].text, 'A1-15 - disc, center');
  assert.equal(c[0].text, 'S29-15 - 6 loops, 6 kites, ring, center');
  assert.deepEqual(libraryChoices(pickerModel(), LIBRARY, 's1').map((x) => x.choice), ['S11a-15']);
  assert.deepEqual(libraryChoices(pickerModel(), LIBRARY, ' 9-19 ').map((x) => x.choice), ['S9-19']);
  assert.ok(c.every((x) => !/[–—]/.test(x.text)));
  assert.deepEqual(libraryChoices(null, LIBRARY), []);
});

test('a pick outside the candidates goes to the server, a candidate switches locally', () => {
  const m = pickerModel();
  assert.equal(pickNeedsServer(m, 'S9-19'), true);
  assert.equal(pickNeedsServer(m, 'S30-15'), false);
  assert.equal(pickNeedsServer(m, 'rules'), false);
  // the server's answer: the chosen head joins the candidates, then it switches locally
  const t = { id: 'library_S9-19', role: 'template', params: { template_id: 'S9-19', cx: 300, cy: 150, r: 120, rotation_deg: 0, score: 0.3, rank: 4, description: '4 loops, 6-point star, center', group_1: 1 }, color_group: 1, z: 10, confidence: 0.5 };
  m.alternatives.push(t);
  assert.equal(pickNeedsServer(m, 'S9-19'), false);
  switchHead(m, 'S9-19');
  assert.equal(templateText(m.parts.find((p) => p.role === 'template')), 'Head shape: chosen from the library: S9-19 - 4 loops, 6-point star, center');
  assert.ok(headCandidates(m).some((x) => x.choice === 'S9-19' && x.drawn));
  assert.equal(renderBody(m, 'S9-19'), JSON.stringify({ model: m, head: 'S9-19' }));
  assert.equal(renderBody(m), JSON.stringify({ model: m }));
});

// H9: the automatic result first, the library chooser as a closed manual override.
test('the library chooser is a closed manual override under the automatic result', () => {
  const o = libraryOverride(pickerModel(), LIBRARY);
  assert.equal(o.label, 'Other library heads (manual override)');
  assert.equal(o.open, false);
  assert.equal(o.choices.length, 6);
  assert.equal(automaticHeadText(pickerModel(), 'S29-15'), 'Automatic result: library template S29-15, match 80% (drawn)');
  const m = pickerModel();
  switchHead(m, 'S30-15');
  assert.equal(automaticHeadText(m, 'S29-15'), 'Automatic result: library template S29-15, match 80% (not drawn: another head is picked)');
  switchHead(m, 'rules');
  assert.equal(automaticHeadText(m, 'rules'), 'Automatic result: rule-based head (drawn)');
  assert.equal(automaticHeadText(m, null), '');
  for (const t of [o.label, automaticHeadText(m, 'S29-15')]) assert.ok(!/[–—]/.test(t));
});
