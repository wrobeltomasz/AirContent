// csv-dataset.js — shared CSV parsing + mapping onto the model feature vector.
//
// Single source of truth for turning a raw housing CSV into validated training
// data. Consumed by csv-mapping-test.js (evaluation) and model-registry.js
// (building several models from different data slices).

import { readFileSync } from 'fs';
import { validateRecord, ValidationError } from './validation.js';
import { getConfig } from './config-store.js';

// Default projection of Melbourne columns onto the model's [area, rooms, floor].
export const DEFAULT_COLUMN_MAP = {
  area: 'BuildingArea',
  rooms: 'Rooms',
  floor: 'Bathroom',
};
export const DEFAULT_TARGET_COLUMN = 'Price';

export function parseCSVLine(line) {
  const out = [];
  let cur = '';
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (q && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else q = !q;
    } else if (c === ',' && !q) {
      out.push(cur.trim());
      cur = '';
    } else cur += c;
  }
  out.push(cur.trim());
  return out;
}

export function loadRows(filePath) {
  const text = readFileSync(filePath, 'utf8');
  const lines = text.split('\n').filter((l) => l.trim());
  const headers = parseCSVLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCSVLine(line);
    const row = {};
    headers.forEach((h, i) => (row[h] = values[i]));
    return row;
  });
}

// Map + validate one row. Returns a mapped record or a structured rejection so
// callers can tally exactly why rows drop out.
export function mapRow(
  row,
  columnMap = DEFAULT_COLUMN_MAP,
  target = DEFAULT_TARGET_COLUMN
) {
  const fields = getConfig().validation.fields.map((f) => f.key);
  const record = {};
  for (const key of fields) {
    const raw = row[columnMap[key]];
    const num = parseFloat(raw);
    if (raw === undefined || raw === '' || Number.isNaN(num)) {
      return {
        ok: false,
        reason: 'mapping',
        detail: `${key}: empty/non-numeric`,
      };
    }
    record[key] = num;
  }
  try {
    validateRecord(record);
  } catch (e) {
    if (e instanceof ValidationError) {
      return { ok: false, reason: 'validation', detail: e.message };
    }
    throw e;
  }
  const price = parseFloat(row[target]);
  if (Number.isNaN(price)) {
    return { ok: false, reason: 'target', detail: 'price empty/non-numeric' };
  }
  return { ok: true, record, vector: fields.map((k) => record[k]), price };
}

// Load a CSV and return validated training data plus mapping statistics.
// `options.filter(row)` optionally restricts which rows are considered — this is
// how several models are built from different slices of the same file.
export function mapDataset(filePath, options = {}) {
  const {
    columnMap = DEFAULT_COLUMN_MAP,
    target = DEFAULT_TARGET_COLUMN,
    filter = null,
  } = options;

  const allRows = loadRows(filePath);
  const rows = filter ? allRows.filter(filter) : allRows;

  const fieldKeys = getConfig().validation.fields.map((f) => f.key);
  const fieldStats = {};
  for (const key of fieldKeys) fieldStats[key] = { considered: 0, present: 0 };

  const vectors = [];
  const prices = [];
  const records = [];
  const rejected = { mapping: 0, validation: 0, target: 0 };

  for (const row of rows) {
    // Per-field completeness: count whether each measure's raw cell is usable,
    // independent of whether the whole row is ultimately accepted. This is what
    // measure purity/dirtiness is based on.
    for (const key of fieldKeys) {
      fieldStats[key].considered++;
      const raw = row[columnMap[key]];
      const num = parseFloat(raw);
      if (!(raw === undefined || raw === '' || Number.isNaN(num))) {
        fieldStats[key].present++;
      }
    }

    const r = mapRow(row, columnMap, target);
    if (r.ok) {
      vectors.push(r.vector);
      prices.push(r.price);
      records.push(r.record);
    } else {
      rejected[r.reason]++;
    }
  }

  for (const key of fieldKeys) {
    const s = fieldStats[key];
    s.missing = s.considered - s.present;
    s.missingRate = s.considered ? s.missing / s.considered : 0;
  }

  return {
    total: rows.length,
    totalFileRows: allRows.length,
    columnMap,
    vectors,
    prices,
    records,
    rejected,
    fieldStats,
    mappingSuccessRate: rows.length ? vectors.length / rows.length : 0,
  };
}
