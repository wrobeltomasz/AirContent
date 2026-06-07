// test/model-types.test.js — verifies the model-type catalogue that drives the
// wizard: built-ins are present and well-formed, and a user definition is
// validated/normalised before it can reach the trainer.
//
// These tests touch no disk state (they exercise normalizeType + the built-in
// listing only), so they are safe to run anywhere.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { listTypes, getType, normalizeType } from '../model-types.js';

test('ships the built-in types', () => {
  const ids = listTypes().map((t) => t.id);
  for (const id of ['realestate', 'cars', 'payroll']) {
    assert.ok(ids.includes(id), `built-in "${id}" present`);
  }
  const cars = getType('cars');
  assert.equal(cars.builtin, true);
  assert.deepEqual(
    cars.fields.map((f) => f.key),
    ['age', 'mileage', 'power']
  );
  assert.equal(cars.target.key, 'price');
});

test('normalizeType defaults a property column to its key', () => {
  const t = normalizeType({
    id: 'trucks',
    label: 'Trucks',
    fields: [{ key: 'axles', label: 'Axles', integer: true }],
  });
  assert.equal(t.fields[0].column, 'axles');
  assert.equal(t.fields[0].required, true);
  assert.equal(t.target.key, 'price');
});

test('normalizeType rejects malformed definitions', () => {
  assert.throws(
    () => normalizeType({ id: 'X bad', label: 'x', fields: [{ key: 'a' }] }),
    /type id/
  );
  assert.throws(
    () => normalizeType({ id: 'ok', label: '', fields: [{ key: 'a' }] }),
    /label is required/
  );
  assert.throws(
    () => normalizeType({ id: 'ok', label: 'Ok', fields: [] }),
    /at least one property/
  );
  assert.throws(
    () => normalizeType({ id: 'ok', label: 'Ok', fields: [{ key: '1bad' }] }),
    /valid identifier/
  );
  assert.throws(
    () =>
      normalizeType({
        id: 'ok',
        label: 'Ok',
        fields: [{ key: 'a', min: 10, max: 5 }],
      }),
    /min .* > max/
  );
});
