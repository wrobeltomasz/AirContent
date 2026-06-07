// test/measures.test.js — unit tests for the per-measure analytics and the
// dirty-data guard. Pure synthetic data, so these always run (no CSV needed).
//
// Run: npm test

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMeasures, cleanMatrix, DEFAULT_SIGMA } from '../measures.js';

const FIELDS = [
  { key: 'a', label: 'A' },
  { key: 'b', label: 'B' },
  { key: 'c', label: 'C' },
];

// Feature `a` drives price; `b` oscillates (weak); `c` cycles (near-zero).
function synthetic(n = 50) {
  const vectors = [];
  const prices = [];
  for (let i = 0; i < n; i++) {
    const a = i;
    const b = (i * 7) % 11;
    const c = i % 5;
    vectors.push([a, b, c]);
    prices.push(100000 + 5000 * a + 20 * b);
  }
  return { vectors, prices };
}

test('measure strength ranks the strongest input first', () => {
  const { vectors, prices } = synthetic();
  const { measures } = computeMeasures(vectors, prices, { fields: FIELDS });
  const byKey = Object.fromEntries(measures.map((m) => [m.key, m]));
  assert.equal(measures.length, 3);
  assert.ok(byKey.a.strength > byKey.b.strength, 'a stronger than b');
  assert.ok(byKey.a.strength > byKey.c.strength, 'a stronger than c');
  // Strength is a correlation magnitude, so it stays within [0, 1].
  for (const m of measures) assert.ok(m.strength >= 0 && m.strength <= 1);
});

test('total strength is the sum of kept measure strengths', () => {
  const { vectors, prices } = synthetic();
  const fieldStats = {
    a: { missingRate: 0 },
    b: { missingRate: 0.9 }, // too dirty → excluded
    c: { missingRate: 0.02 },
  };
  const result = computeMeasures(vectors, prices, {
    fields: FIELDS,
    fieldStats,
  });

  assert.deepEqual(
    result.excluded,
    ['b'],
    'the 90%-missing measure is excluded'
  );
  const kept = result.measures.filter((m) => m.included);
  const sum = kept.reduce((acc, m) => acc + m.strength, 0);
  assert.ok(
    Math.abs(result.totalStrength - sum) < 1e-3,
    'totalStrength == Σ kept strengths'
  );
});

test('hierarchy runs from purest to dirtiest with cumulative strength', () => {
  const { vectors, prices } = synthetic();
  const fieldStats = {
    a: { missingRate: 0 },
    b: { missingRate: 0.9 },
    c: { missingRate: 0.02 },
  };
  const { hierarchy } = computeMeasures(vectors, prices, {
    fields: FIELDS,
    fieldStats,
  });

  for (let i = 1; i < hierarchy.length; i++) {
    assert.ok(
      hierarchy[i - 1].purity >= hierarchy[i].purity,
      'purity non-increasing'
    );
    assert.ok(
      hierarchy[i].cumulativeStrength >= hierarchy[i - 1].cumulativeStrength,
      'cumulative strength non-decreasing'
    );
  }
  assert.equal(
    hierarchy[0].slopeDeltaVsPrev,
    0,
    'first measure has no previous'
  );
  assert.equal(
    hierarchy[hierarchy.length - 1].key,
    'b',
    'dirtiest measure ranked last'
  );
});

test('cleanMatrix neutralises an excluded measure to a constant', () => {
  const vectors = [
    [1, 5],
    [2, 9],
    [3, 50],
    [4, 7],
  ];
  const measures = [{ included: true }, { included: false }];
  const cleaned = cleanMatrix(vectors, measures, DEFAULT_SIGMA);
  const col1 = cleaned.map((r) => r[1]);
  const mean = (5 + 9 + 50 + 7) / 4;
  for (const v of col1)
    assert.equal(v, mean, 'excluded column collapses to its mean');
});

test('cleanMatrix clamps out-of-norm cells back to the band', () => {
  // Nine zeros and one spike → mean 10, sample std ≈ 31.62.
  const vectors = Array.from({ length: 10 }, (_, i) => [i === 9 ? 100 : 0]);
  const measures = [{ included: true }];
  const sigma = 2;
  const cleaned = cleanMatrix(vectors, measures, sigma);
  const std = Math.sqrt(vectors.reduce((a, r) => a + (r[0] - 10) ** 2, 0) / 9);
  const hi = 10 + sigma * std;
  assert.ok(cleaned[9][0] < 100, 'spike was clamped down');
  assert.ok(Math.abs(cleaned[9][0] - hi) < 1e-6, 'clamped to mean + sigma*std');
});
