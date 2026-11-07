// Node tests for the library head (H7) logic in logic.js.
// Run: node --test tests/js/template.test.mjs (tests/test_js.py runs it when node is on PATH)
import test from 'node:test';
import assert from 'node:assert/strict';
import { colorQuestions, flagItems, headChoice, partGroups, switchHead, templateText, usesGroup } from '../../app/static/logic.js';

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
