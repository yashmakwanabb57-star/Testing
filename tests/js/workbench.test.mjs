// Node tests for the workbench's pure logic and the shop-coloring port.
// Run: node --test tests/js/workbench.test.mjs (tests/test_js.py runs it when node is on PATH)
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SOURCE_NOT_READY, downloadBlockers, errorText, flagItems, parseValue, sourceNotice,
} from '../../app/static/logic.js';
import { applyConfig } from '../../app/static/simulator.js';

// ------------------------------------------------------------ fixtures

function model() {
  return {
    rose: true,
    warnings: [],
    groups: [
      { id: 0, rgb: [38, 82, 64], pattern: false, texture: 1, area: 0.5, category: 1, confidence: 0.93 },
      { id: 1, rgb: [208, 173, 114], pattern: false, texture: 1, area: 0.3, category: 3, confidence: 0.4 },
    ],
    parts: [
      { id: 'head', role: 'head', params: { cx: 300, cy: 150, r: 120, pinked: 1 }, color_group: 0, z: 10, confidence: 0.9 },
      { id: 'points-1', role: 'points', params: { cx: 300, cy: 150, kind: 'star', count: 6, r_inner: 50, r_outer: 100, rotation_deg: 0, width_ratio: 1 }, color_group: 1, z: 20, confidence: 0.9 },
    ],
  };
}

function ready(overrides = {}) {
  return {
    analyzing: false, model: model(), svg: '<svg/>', issues: [], flagged: true,
    invalid: new Set(), confirmed: new Set(['group:1']),
    render: { dirty: false, inFlight: false, error: null },
    ...overrides,
  };
}

// ------------------------------------------------------------ downloadBlockers

test('ready state has no blockers', () => {
  assert.deepEqual(downloadBlockers(ready()), []);
});

test('an unconfirmed flag blocks download', () => {
  const b = downloadBlockers(ready({ confirmed: new Set() }));
  assert.equal(b.length, 1);
  assert.match(b[0], /1 of 1 flagged/);
});

test('a dirty or in-flight render blocks download', () => {
  assert.match(downloadBlockers(ready({ render: { dirty: true, inFlight: false, error: null } }))[0], /Updating/);
  assert.match(downloadBlockers(ready({ render: { dirty: false, inFlight: true, error: null } }))[0], /Updating/);
});

test('a failed render blocks download', () => {
  assert.match(downloadBlockers(ready({ render: { dirty: false, inFlight: false, error: 'x' } }))[0], /failed/);
});

test('validation issues block download', () => {
  const b = downloadBlockers(ready({ issues: [{ code: 'dup_id', message: 'm', element: null }] }));
  assert.ok(b.some((x) => /1 check\(s\) failed/.test(x)));
});

test('an invalid field blocks download', () => {
  assert.match(downloadBlockers(ready({ invalid: new Set(['points-1.count']) }))[0], /invalid values/);
});

test('no model, or analysis running, blocks download', () => {
  assert.equal(downloadBlockers(ready({ model: null })).length, 1);
  assert.match(downloadBlockers(ready({ analyzing: true }))[0], /Analyzing/);
});

// ------------------------------------------------------------ flagItems

test('flagItems mirrors is_flagged', () => {
  const m = model();
  m.parts[1].confidence = 0.5;
  m.parts.push({ id: 'unknown-3', role: 'unknown', params: { cx: 1, cy: 1, r: 1 }, color_group: 0, z: 1, confidence: 0.1 });
  m.warnings = ['part unknown-3: unrecognised shape, drawn as a plain disc', 'Color #265240 is partly patterned'];
  const keys = flagItems(m, [], true).map((f) => f.key);
  assert.deepEqual(keys, ['group:1', 'part:points-1', 'part:unknown-3', 'warning:Color #265240 is partly patterned']);
});

test('a server flag alone still needs a confirmation', () => {
  const m = model();
  m.groups[1].confidence = 0.9;
  assert.deepEqual(flagItems(m, [], true).map((f) => f.key), ['server:flagged']);
  assert.deepEqual(flagItems(m, [], false), []);
});

// ------------------------------------------------------------ parseValue

test('parseValue rejects bad counts', () => {
  for (const raw of ['', 'NaN', '0', '201', '2.5', '-1', 'abc', '1e999']) {
    assert.equal(parseValue('int', raw).ok, false, raw);
  }
  assert.deepEqual(parseValue('int', '12'), { ok: true, value: 12 });
  assert.deepEqual(parseValue('int', '200'), { ok: true, value: 200 });
});

test('parseValue numbers and kinds', () => {
  assert.equal(parseValue('num', '').ok, false);
  assert.equal(parseValue('num', 'NaN').ok, false);
  assert.equal(parseValue('num', 'Infinity').ok, false);
  assert.deepEqual(parseValue('num', '-2.5'), { ok: true, value: -2.5 });
  assert.equal(parseValue('kind', 'bow').ok, true);
  assert.equal(parseValue('kind', 'hexagon').ok, false);
});

// ------------------------------------------------------------ errorText

test('errorText with a string detail', () => {
  assert.equal(errorText(422, { detail: 'No rosette found in the image.' }), 'No rosette found in the image.');
});

test('errorText with a FastAPI list detail', () => {
  const t = errorText(422, { detail: [{ loc: ['body', 'model'], msg: 'Field required' }, { loc: ['body', 0], msg: 'JSON decode error' }] });
  assert.equal(t, 'The request was rejected - model: Field required; 0: JSON decode error');
});

test('errorText without a body', () => {
  assert.match(errorText(413, null), /too large/);
  assert.equal(errorText(500, null), 'The request failed (HTTP 500).');
});

// ------------------------------------------------------------ applyConfig (shop port)

// Minimal DOM: enough for the selectors the shop code uses (.class, #id, tag).
class El {
  constructor(tag, { id = null, cls = null } = {}, children = []) {
    this.tag = tag;
    this.id = id;
    this.classes = cls ? cls.split(' ') : [];
    this.attrs = {};
    this.style = {};
    this.children = children;
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }

  * walk() {
    for (const c of this.children) {
      yield c;
      yield* c.walk();
    }
  }

  matches(sel) {
    if (sel.startsWith('.')) return this.classes.includes(sel.slice(1));
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    return this.tag === sel;
  }

  querySelectorAll(sel) { return [...this.walk()].filter((e) => e.matches(sel)); }

  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

// The structure rosette/writer.py emits, one shape per layer.
function svgTree() {
  const image = new El('image');
  image.attrs = { 'xlink:href': '', width: '100', height: '100' };
  const pattern = new El('pattern', { id: 'image_st0_2_pattern' }, [image]);
  return new El('svg', {}, [
    new El('g', { id: 'Layer_1' }, [new El('polygon', { cls: 'st0_1' })]),
    new El('g', { id: 'Layer_3' }, [
      new El('g', { id: 'Layer_3_2' }, [new El('defs', {}, [pattern]), new El('polygon', { cls: 'st0_2_pattern' })]),
      new El('g', { id: 'Layer_3_3' }, [new El('polygon', { cls: 'st1_3' })]),
    ]),
    new El('g', { id: 'Layer_2_1' }, [
      new El('g', { id: 'Layer_2_1_2' }, [new El('polygon', { cls: 'st0_2' })]),
      new El('g', { id: 'Layer_2_1_3' }, [new El('polygon', { cls: 'st1_3' })]),
    ]),
    new El('g', { id: 'Layer_6' }, [new El('polyline', { cls: 'st6' })]),
    new El('g', { id: 'Layer_rose' }, [new El('path', { cls: 'st4' }), new El('circle', { cls: 'st_circel' })]),
  ]);
}

const COLORS = { 1: '#111111', 2: '#222222', 3: '#333333', 4: '#444444' };
const one = (svg, sel) => svg.querySelector(sel);

test('solid colors fill the solid layer and hide the pattern layer', () => {
  const svg = svgTree();
  applyConfig(svg, { colors: COLORS, patternIndex: null, patternImage: null, rose: true });
  assert.equal(one(svg, '.st0_1').style.fill, '#111111');
  assert.equal(one(svg, '.st0_2').style.fill, '#222222');
  assert.equal(one(svg, '.st0_2_pattern').style.fill, undefined);
  for (const e of svg.querySelectorAll('.st1_3')) assert.equal(e.style.fill, '#333333');
  assert.equal(one(svg, '#Layer_3').style.display, 'none');
  assert.equal(one(svg, '#Layer_2_1').style.display, 'inline');
});

test('pattern on Color 2 fills the pattern layer with the image and shows it', () => {
  const svg = svgTree();
  const url = 'data:image/png;base64,AAAA';
  applyConfig(svg, { colors: COLORS, patternIndex: 2, patternImage: url, rose: true });
  assert.equal(one(svg, '.st0_2_pattern').style.fill, 'url(#image_st0_2_pattern)');
  const pattern = one(svg, '#image_st0_2_pattern');
  assert.equal(pattern.attrs.width, '80');
  assert.equal(pattern.attrs.height, '80');
  const image = one(pattern, 'image');
  assert.equal(image.attrs['xlink:href'], url);
  assert.equal(image.attrs.width, '80');
  assert.equal(one(svg, '#Layer_3').style.display, 'inline');
  assert.equal(one(svg, '#Layer_2_1').style.display, 'none');
  assert.equal(one(svg, '.st0_2').style.fill, undefined);
});

test('the rose toggle shows st4 or st_circel', () => {
  const on = svgTree();
  applyConfig(on, { colors: COLORS, patternIndex: null, patternImage: null, rose: true });
  assert.equal(one(on, '.st4').style.display, 'inline');
  assert.equal(one(on, '.st_circel').style.display, 'none');
  const off = svgTree();
  applyConfig(off, { colors: COLORS, patternIndex: null, patternImage: null, rose: false });
  assert.equal(one(off, '.st4').style.display, 'none');
  assert.equal(one(off, '.st_circel').style.display, 'inline');
});

// ------------------------------------------------------------ part confidence (all roles)

function stripePart(t, s, conf) {
  return {
    id: `tail-${t}-${s}`, role: 'tail_stripe', color_group: 0, z: 1, confidence: conf,
    params: { tails_count: 2, top_y: 300, bottom_y: 800, top_width: 80, bottom_width: 120, spread_deg: 10,
      stripes: [0.5, 0.5], stripes_bottom: [0.5, 0.5], tail_index: t, stripe_index: s },
  };
}

test('a low-confidence part of any role is flagged, like is_flagged', () => {
  for (const role of ['head', 'ring', 'center']) {
    const m = model();
    m.groups[1].confidence = 0.9;
    assert.deepEqual(flagItems(m, [], false), [], role);
    m.parts[0].role = role;
    m.parts[0].confidence = 0.55;
    assert.deepEqual(flagItems(m, [], false).map((f) => f.key), ['part:head'], role);
  }
});

test('low-confidence tail stripes give one flag for their tails card', () => {
  const m = model();
  m.groups[1].confidence = 0.9;
  m.parts.push(stripePart(0, 0, 0.545), stripePart(0, 1, 0.545), stripePart(1, 0, 0.545), stripePart(1, 1, 0.545));
  const flags = flagItems(m, [], false);
  assert.deepEqual(flags.map((f) => f.key), ['tails:0']);
  assert.match(flags[0].text, /Tails/);
  // confirming the tails card clears the blocker
  const s = ready({ model: m, confirmed: new Set() });
  assert.match(downloadBlockers(s)[0], /1 of 1 flagged/);
  assert.deepEqual(downloadBlockers(ready({ model: m, confirmed: new Set(['tails:0']) })), []);
});

test('confident tail stripes are not flagged', () => {
  const m = model();
  m.groups[1].confidence = 0.9;
  m.parts.push(stripePart(0, 0, 0.9), stripePart(1, 0, 0.9));
  assert.deepEqual(flagItems(m, [], false), []);
});

// ------------------------------------------------------------ SVG source gate

test('the SVG source carries a not-ready banner while download is blocked', () => {
  assert.equal(SOURCE_NOT_READY, 'Not ready for the shop - fix or confirm the items listed first.');
  assert.equal(sourceNotice(ready()), null);
  assert.equal(sourceNotice(ready({ confirmed: new Set() })), SOURCE_NOT_READY);
  assert.equal(sourceNotice(ready({ issues: [{ code: 'group_order', message: 'm', element: null }] })), SOURCE_NOT_READY);
  assert.equal(sourceNotice(ready({ render: { dirty: true, inFlight: false, error: null } })), SOURCE_NOT_READY);
  // nothing to show yet: no banner over an empty source
  assert.equal(sourceNotice(ready({ model: null, svg: null })), null);
});
