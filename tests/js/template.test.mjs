// Node tests for the library head (H7) logic in logic.js.
// Run: node --test tests/js/template.test.mjs (tests/test_js.py runs it when node is on PATH)
import test from 'node:test';
import assert from 'node:assert/strict';
import { colorQuestions, flagItems, automaticHeadText, renderBody, partGroups, templateText, usesGroup } from '../../app/static/logic.js';

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

test('a weak template match is flagged, a good one is not', () => {
  assert.ok(flagItems(model(0.45), [], false).some((f) => f.key === 'part:template' && /weak match/.test(f.text)));
  assert.ok(!flagItems(model(0.8), [], false).some((f) => f.key === 'part:template'));
});

// A model with a library head drawn and two more candidates waiting in alternatives.
function pickerModel() {
  const m = model(0.8);
  m.parts[1].params = { ...m.parts[1].params, rank: 1, points: 6, description: '6 loops, 6 kites, ring, center' };
  m.alternatives.push({ id: 'template_3', role: 'template', params: { template_id: 'S57-17', cx: 300, cy: 150, r: 120, rotation_deg: 0, score: 0.55, rank: 3, group_1: 0 }, color_group: 0, z: 10, confidence: 0.55 });
  m.alternatives.push({ id: 'template_2', role: 'template', params: { template_id: 'S30-15', cx: 300, cy: 150, r: 120, rotation_deg: 0, score: 0.7, rank: 2, group_1: 1, group_2: 2 }, color_group: 1, z: 10, confidence: 0.7 });
  m.alternative_groups = [{ id: 2, rgb: [1, 2, 3], pattern: false, texture: 1, area: 0.1, category: 2, confidence: 0.9 }];
  return m;
}

// H9 (amended): the head card shows the automatic head only; no picker in the UI.
test('the head card heading names the automatic head', () => {
  assert.equal(automaticHeadText(pickerModel(), 'S29-15'), 'Automatic result: library template S29-15, match 80%');
  assert.equal(automaticHeadText(pickerModel(), 'S30-15'), 'Automatic result: library template S30-15, match 70%');
  assert.equal(automaticHeadText(pickerModel(), 'rules'), 'Automatic result: rule-based head');
  assert.equal(automaticHeadText(pickerModel(), null), '');
  assert.ok(!/[–—]/.test(automaticHeadText(pickerModel(), 'S29-15')));
  const m = pickerModel();
  assert.equal(renderBody(m, 'S9-19'), JSON.stringify({ model: m, head: 'S9-19' }));
  assert.equal(renderBody(m), JSON.stringify({ model: m }));
});
