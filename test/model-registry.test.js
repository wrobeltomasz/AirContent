// test/model-registry.test.js — verifies that several models can be built from
// different slices of data and predicted against independently, that every
// model carries a business description + measures, and that the model output
// aggregates measure strength.
//
// Run: npm test   (uses node --test)
// Data-dependent tests skip automatically if the local (gitignored) dataset is
// absent; the description-required check needs no data and always runs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'fs';
import {
  buildModelFromCSV,
  listModels,
  predictWith,
  getMeta,
  clearRegistry,
} from '../model-registry.js';

const CSV = 'csv_example/Melbourne_housing_FULL.csv';
const hasData = existsSync(CSV);
const DESC = 'Test model — valuation baseline.';

test('a model cannot be built without a business description', async () => {
  clearRegistry();
  await assert.rejects(
    () => buildModelFromCSV('no-desc', CSV, {}),
    /business description is required/
  );
  await assert.rejects(
    () => buildModelFromCSV('blank-desc', CSV, { description: '   ' }),
    /business description is required/
  );
});

test(
  'builds several models from different data slices',
  { skip: !hasData },
  async () => {
    clearRegistry();

    const houses = await buildModelFromCSV('houses', CSV, {
      description: 'Detached houses — buy-side valuation baseline.',
      filter: (row) => row.Type === 'h',
    });
    const units = await buildModelFromCSV('units', CSV, {
      description: 'Units / apartments — rental yield screening.',
      filter: (row) => row.Type === 'u',
    });

    assert.equal(listModels().length, 2, 'two models registered');
    assert.ok(houses.trainedRows > 0, 'houses model trained on real rows');
    assert.ok(units.trainedRows > 0, 'units model trained on real rows');
    assert.notEqual(houses.trainedRows, units.trainedRows);

    // Each model carries its business context and feature-vector definition.
    assert.match(houses.description, /houses/i);
    assert.equal(houses.featureKeys.length, houses.measures.length);

    // The model output aggregates the strengths of the measures it kept.
    const keptSum = houses.measures
      .filter((m) => m.included)
      .reduce((acc, m) => acc + m.strength, 0);
    assert.ok(
      Math.abs(houses.totalStrength - keptSum) < 1e-3,
      'totalStrength == Σ kept strengths'
    );
    assert.ok(houses.totalStrength >= 0);

    const ph = await predictWith('houses', [120, 4, 2]);
    const pu = await predictWith('units', [60, 2, 1]);
    assert.ok(Number.isFinite(ph) && Number.isFinite(pu));

    assert.ok(getMeta('houses').finalMae >= 0, 'training MAE recorded');
  }
);

test('rejects an empty slice', { skip: !hasData }, async () => {
  clearRegistry();
  await assert.rejects(
    () =>
      buildModelFromCSV('none', CSV, {
        description: DESC,
        filter: () => false,
      }),
    /no valid rows/
  );
});

test('unknown model name is reported', async () => {
  clearRegistry();
  assert.equal(getMeta('missing'), null);
  await assert.rejects(() => predictWith('missing', [1, 2, 3]), /not found/);
});
