// test/model-registry.test.js — verifies that models can be built from
// different model TYPES (each with its own property schema) and from different
// slices of one dataset, predicted against independently, that every model
// carries a business description + measures, and that the model output
// aggregates measure strength.
//
// Run: npm test   (uses node --test)
// Data-dependent tests skip automatically if the local (gitignored) example
// datasets are absent; the description-required check needs no data.

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
import { getType } from '../model-types.js';

const RE_CSV = 'csv_example/properties.csv';
const CARS_CSV = 'csv_example/cars.csv';
const PAY_CSV = 'csv_example/payroll.csv';
const hasData =
  existsSync(RE_CSV) && existsSync(CARS_CSV) && existsSync(PAY_CSV);
const DESC = 'Test model — valuation baseline.';

test('a model cannot be built without a business description', async () => {
  clearRegistry();
  await assert.rejects(
    () => buildModelFromCSV('no-desc', RE_CSV, { type: getType('realestate') }),
    /business description is required/
  );
  await assert.rejects(
    () =>
      buildModelFromCSV('blank-desc', RE_CSV, {
        type: getType('realestate'),
        description: '   ',
      }),
    /business description is required/
  );
});

test(
  'builds a model per type, each with its own property schema',
  { skip: !hasData },
  async () => {
    clearRegistry();

    const re = await buildModelFromCSV('homes', RE_CSV, {
      description: 'Property valuation baseline.',
      type: getType('realestate'),
    });
    const cars = await buildModelFromCSV('autos', CARS_CSV, {
      description: 'Used-car pricing.',
      type: getType('cars'),
    });
    const pay = await buildModelFromCSV('salaries', PAY_CSV, {
      description: 'Salary estimation.',
      type: getType('payroll'),
    });

    assert.equal(listModels().length, 3, 'three models registered');

    // Each model trains on its type's own properties.
    assert.deepEqual(re.featureKeys, ['area', 'rooms', 'floor']);
    assert.deepEqual(cars.featureKeys, ['age', 'mileage', 'power']);
    assert.deepEqual(pay.featureKeys, ['experience', 'age', 'education']);
    assert.equal(re.typeId, 'realestate');
    assert.equal(cars.target.key, 'price');
    assert.equal(pay.target.key, 'salary');

    // Each model carries its measures and aggregates kept strength.
    assert.equal(cars.featureKeys.length, cars.measures.length);
    const keptSum = cars.measures
      .filter((m) => m.included)
      .reduce((acc, m) => acc + m.strength, 0);
    assert.ok(
      Math.abs(cars.totalStrength - keptSum) < 1e-3,
      'totalStrength == Σ kept strengths'
    );

    // Predict with each, using that type's feature vector length.
    const pr = await predictWith('homes', [85, 3, 2]);
    const pc = await predictWith('autos', [5, 60000, 150]);
    const ps = await predictWith('salaries', [10, 35, 4]);
    assert.ok(
      Number.isFinite(pr) && Number.isFinite(pc) && Number.isFinite(ps)
    );
    assert.ok(getMeta('autos').finalMae >= 0, 'training MAE recorded');
  }
);

test(
  'builds several models from different slices of one dataset',
  { skip: !hasData },
  async () => {
    clearRegistry();
    const type = getType('realestate');

    const small = await buildModelFromCSV('small', RE_CSV, {
      description: 'Smaller properties.',
      type,
      filter: (row) => Number(row.rooms) <= 3,
    });
    const large = await buildModelFromCSV('large', RE_CSV, {
      description: 'Larger properties.',
      type,
      filter: (row) => Number(row.rooms) > 3,
    });

    assert.ok(small.trainedRows > 0 && large.trainedRows > 0);
    assert.notEqual(small.trainedRows, large.trainedRows);
  }
);

test('rejects an empty slice', { skip: !hasData }, async () => {
  clearRegistry();
  await assert.rejects(
    () =>
      buildModelFromCSV('none', RE_CSV, {
        description: DESC,
        type: getType('realestate'),
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
