// Node tests for the guided mode's pure logic (step gates, jumps, part checks, color questions).
// Run: node --test tests/js/guided.test.mjs (tests/test_js.py runs it when node is on PATH)
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GUIDED_STEPS, advanceBlocker, canAdvance, canJump, clampStep, colorQuestions, partChecks, tailCheckKey,
} from '../../app/static/logic.js';

// ------------------------------------------------------------ fixtures

function stripe(t, s, group) {
  return {
    id: `tail-${t}-${s}`, role: 'tail_stripe', color_group: group, z: 1, confidence: 0.9,
    params: { tails_count: 2, top_y: 300, bottom_y: 800, top_width: 80, bottom_width: 120, spread_deg: 10,
      stripes: [0.5, 0.5], stripes_bottom: [0.5, 0.5], tail_index: t, stripe_index: s },
  };
}

function model() {
  return {
    rose: true,
    warnings: [],
    groups: [
      { id: 0, rgb: [38, 82, 64], pattern: false, texture: 1, area: 0.5, category: 1, confidence: 0.93 },
      { id: 1, rgb: [208, 173, 114], pattern: false, texture: 1, area: 0.3, category: 3, confidence: 0.4 },
      { id: 2, rgb: [160, 180, 200], pattern: true, texture: 1, area: 0.2, category: 2, confidence: 0.9 },
    ],
    parts: [
      { id: 'head', role: 'head', params: { cx: 300, cy: 150, r: 120, pinked: 1 }, color_group: 0, z: 10, confidence: 0.9 },
      { id: 'points-1', role: 'points', params: { cx: 300, cy: 150, kind: 'star', count: 6, r_inner: 50, r_outer: 100, rotation_deg: 0, width_ratio: 1 }, color_group: 2, z: 20, confidence: 0.9 },
      stripe(0, 0, 1), stripe(0, 1, 0), stripe(1, 0, 1), stripe(1, 1, 0),
    ],
  };
}

function st(overrides = {}) {
  return {
    analyzing: false, model: model(), svg: '<svg/>', issues: [], flagged: true,
    invalid: new Set(), confirmed: new Set(),
    render: { dirty: false, inFlight: false, error: null },
    sim: { tried: new Set() },
    ...overrides,
  };
}

const ALL_PARTS = ['part:head', 'part:points-1', tailCheckKey(0)];

// ------------------------------------------------------------ steps

test('five steps in order', () => {
  assert.deepEqual(GUIDED_STEPS.map((s) => s.label), ['Upload', 'Parts', 'Colors', 'Test', 'Download']);
  assert.equal(GUIDED_STEPS[2].title, 'Check colors');
});

// ------------------------------------------------------------ partChecks

test('one check per non-tail part and one per tail group', () => {
  assert.deepEqual(partChecks(model()).map((c) => c.key), ALL_PARTS);
  assert.deepEqual(partChecks(null), []);
});

// ------------------------------------------------------------ step 1

test('step 1 needs an analyzed photo', () => {
  assert.match(advanceBlocker(1, st({ model: null, svg: null })), /Upload a photo/);
  assert.match(advanceBlocker(1, st({ analyzing: true })), /analysis/);
  assert.equal(canAdvance(1, st()), true);
});

// ------------------------------------------------------------ step 2

test('step 2 is blocked until every part looks right', () => {
  assert.match(advanceBlocker(2, st()), /Looks right.*3 of 3 left/);
  assert.match(advanceBlocker(2, st({ confirmed: new Set(['part:head']) })), /2 of 3 left/);
  assert.equal(canAdvance(2, st({ confirmed: new Set(ALL_PARTS) })), true);
});

test('step 2 counts unflagged parts too, not just flagged ones', () => {
  // no part here is flagged, yet each still needs its own check
  assert.equal(canAdvance(2, st({ confirmed: new Set(['part:head', 'part:points-1']) })), false);
});

test('step 2 is blocked by an invalid field', () => {
  assert.match(advanceBlocker(2, st({ confirmed: new Set(ALL_PARTS), invalid: new Set(['head.r']) })), /invalid/);
});

test('step 2 needs a model', () => {
  assert.match(advanceBlocker(2, st({ model: null, svg: null })), /Upload/);
});

// ------------------------------------------------------------ step 3

const ALL_GROUPS = ['group:0', 'group:1', 'group:2'];

test('step 3 needs an answer on every color question, flagged or not', () => {
  assert.match(advanceBlocker(3, st()), /^Answer every color question first \(3 of 3 left\)\.$/);
  // answering only the flagged group (group 1, confidence 0.4) is not enough
  assert.match(advanceBlocker(3, st({ confirmed: new Set(['group:1']) })), /\(2 of 3 left\)/);
  assert.match(advanceBlocker(3, st({ confirmed: new Set(['group:1', 'group:0']) })), /\(1 of 3 left\)/);
  assert.equal(canAdvance(3, st({ confirmed: new Set(ALL_GROUPS) })), true);
});

test('step 3 with no flags at all still needs every answer', () => {
  const m = model();
  m.groups[1].confidence = 0.9;
  assert.match(advanceBlocker(3, st({ model: m, flagged: false })), /3 of 3 left/);
  assert.equal(canAdvance(3, st({ model: m, flagged: false, confirmed: new Set(ALL_GROUPS) })), true);
});

test('step 3 needs warnings acknowledged as well', () => {
  const m = model();
  m.warnings = ['Color #a0b4c8 is partly patterned'];
  const b = advanceBlocker(3, st({ model: m, confirmed: new Set(ALL_GROUPS) }));
  assert.equal(b, 'Confirm 1 warning(s) first.');
  assert.equal(canAdvance(3, st({ model: m, confirmed: new Set([...ALL_GROUPS, 'warning:Color #a0b4c8 is partly patterned']) })), true);
});

test('step 3 ignores part flags (those are step 2)', () => {
  const m = model();
  m.parts[1].confidence = 0.3; // flags part:points-1
  assert.equal(canAdvance(3, st({ model: m, confirmed: new Set(ALL_GROUPS) })), true);
});

test('step 3 includes the server-flagged fallback', () => {
  const m = model();
  m.groups[1].confidence = 0.9; // nothing flagged locally, but the server says flagged
  assert.match(advanceBlocker(3, st({ model: m, confirmed: new Set(ALL_GROUPS) })), /warning/);
  assert.equal(canAdvance(3, st({ model: m, confirmed: new Set([...ALL_GROUPS, 'server:flagged']) })), true);
});

// ------------------------------------------------------------ step 4

test('step 4 needs Pattern and Rose toggled once each', () => {
  assert.match(advanceBlocker(4, st()), /Color 2 Solid \/ Pattern and Rose On \/ Off/);
  const one = advanceBlocker(4, st({ sim: { tried: new Set(['pattern']) } }));
  assert.match(one, /Rose/);
  assert.doesNotMatch(one, /Pattern/);
  assert.equal(canAdvance(4, st({ sim: { tried: new Set(['pattern', 'rose']) } })), true);
});

// ------------------------------------------------------------ step 5

test('step 5 is the last step', () => {
  assert.equal(canAdvance(5, st()), false);
  assert.match(advanceBlocker(5, st()), /last step/);
});

// ------------------------------------------------------------ canJump

test('breadcrumb: back always, forward only to reached steps that pass their gates', () => {
  const s = st();
  assert.equal(canJump(3, 1, s, 3), true);
  assert.equal(canJump(1, 2, s, 1), false); // not reached yet
  assert.equal(canJump(1, 2, s, 2), true); // step 1 passes
  assert.equal(canJump(1, 3, s, 3), false); // step 2 is not passed
  const done = st({ confirmed: new Set([...ALL_PARTS, ...ALL_GROUPS]), sim: { tried: new Set(['pattern', 'rose']) } });
  assert.equal(canJump(1, 5, done, 5), true);
  assert.equal(canJump(1, 5, done, 4), false);
  assert.equal(canJump(1, 0, done, 5), false);
  assert.equal(canJump(1, 6, done, 5), false);
});

test('canJump turns false after a confirmation is removed', () => {
  const s = st({ confirmed: new Set([...ALL_PARTS, ...ALL_GROUPS]), sim: { tried: new Set(['pattern', 'rose']) } });
  assert.equal(canJump(2, 5, s, 5), true);
  assert.equal(canJump(3, 5, s, 5), true);
  s.confirmed.delete('part:head'); // e.g. the part was edited
  assert.equal(canJump(2, 5, s, 5), false);
  assert.equal(canJump(2, 3, s, 5), false);
  assert.equal(canJump(3, 5, s, 5), true); // the gates from step 3 on still pass
  assert.equal(canJump(5, 2, s, 5), true); // going back is always allowed
  s.confirmed.delete('group:2');
  assert.equal(canJump(3, 5, s, 5), false);
});

// ------------------------------------------------------------ clampStep

test('clampStep keeps a step whose earlier gates pass', () => {
  const s = st({ confirmed: new Set([...ALL_PARTS, ...ALL_GROUPS]), sim: { tried: new Set(['pattern', 'rose']) } });
  for (const k of [1, 2, 3, 4, 5]) assert.equal(clampStep(k, s), k);
});

test('clampStep: at step 5, removing a part Looks right check clamps back to step 2', () => {
  const s = st({ confirmed: new Set([...ALL_PARTS, ...ALL_GROUPS]), sim: { tried: new Set(['pattern', 'rose']) } });
  assert.equal(clampStep(5, s), 5);
  s.confirmed.delete(tailCheckKey(0));
  assert.equal(clampStep(5, s), 2);
  assert.equal(clampStep(2, s), 2); // the step itself may be blocked; only earlier gates clamp
});

test('clampStep goes to the first failing gate', () => {
  const s = st({ confirmed: new Set(ALL_PARTS), sim: { tried: new Set() } });
  assert.equal(clampStep(5, s), 3); // colors unanswered, and test not tried
  assert.equal(clampStep(5, st({ model: null, svg: null })), 1);
});

// ------------------------------------------------------------ colorQuestions

test('color questions: flagged first, then model order, with a part noun', () => {
  const qs = colorQuestions(model(), [], false);
  assert.deepEqual(qs.map((q) => q.key), ['group:1', 'group:0', 'group:2']);
  assert.equal(qs[0].flagged, true);
  assert.equal(qs[0].noun, 'stripe'); // group 1 is used only by tail stripes
  assert.equal(qs[1].noun, null); // group 0: head and stripes
  assert.deepEqual(qs[1].usedOn, ['head', 'stripe']);
  assert.equal(qs[2].noun, 'point');
  assert.equal(qs[0].uses, 2);
  assert.deepEqual(colorQuestions(null), []);
});

// ------------------------------------------------------------ flagged parts in step 2

test('a flagged tails card still needs its Looks right in step 2, and is not a step 3 warning', () => {
  const m = model();
  for (const p of m.parts) if (p.role === 'tail_stripe') p.confidence = 0.545;
  const partsOnly = new Set(['part:head', 'part:points-1']);
  assert.match(advanceBlocker(2, st({ model: m, confirmed: partsOnly })), /1 of 3 left/);
  assert.equal(canAdvance(2, st({ model: m, confirmed: new Set(ALL_PARTS) })), true);
  // step 3 does not count the tails flag as a warning
  assert.equal(canAdvance(3, st({ model: m, confirmed: new Set(ALL_GROUPS) })), true);
});

test('a low-confidence head is flagged and needs its Looks right', () => {
  const m = model();
  m.parts[0].confidence = 0.4;
  assert.match(advanceBlocker(2, st({ model: m, confirmed: new Set(['part:points-1', tailCheckKey(0)]) })), /1 of 3 left/);
});
