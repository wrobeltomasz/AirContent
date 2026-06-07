// csv-dataset.js — shared CSV parsing + mapping onto a model feature vector.
//
// Single source of truth for turning a raw CSV into validated training data.
// The schema (which properties, which CSV columns, which target) is supplied by
// the caller — model-registry.js passes the chosen model type's schema — so the
// same loader serves real estate, cars, payroll and any user-defined type.

import { readFileSync } from 'fs';
import { validateAgainstFields, ValidationError } from './validation.js';
import { getConfig } from './config-store.js';

// Default projection of CSV columns onto the model's [area, rooms, floor].
export const DEFAULT_COLUMN_MAP = {
  area: 'area',
  rooms: 'rooms',
  floor: 'floor',
};
export const DEFAULT_TARGET_COLUMN = 'price';

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

// Resolve the field definitions, column map and target for a mapping run.
// Defaults to the global validation schema so existing callers keep working;
// the model-type pipeline passes an explicit schema instead.
function resolveSchema(options) {
  const fields = options.fields || getConfig().validation.fields;
  const columnMap =
    options.columnMap ||
    fields.reduce((m, f) => {
      m[f.key] = f.column || f.key;
      return m;
    }, {});
  const target = options.target || DEFAULT_TARGET_COLUMN;
  const targetColumn = typeof target === 'string' ? target : target.column;
  return { fields, columnMap, targetColumn };
}

// Map + validate one row against a field schema. Returns a mapped record or a
// structured rejection so callers can tally exactly why rows drop out.
export function mapRow(
  row,
  columnMap = DEFAULT_COLUMN_MAP,
  target = DEFAULT_TARGET_COLUMN,
  fields = getConfig().validation.fields
) {
  const targetColumn = typeof target === 'string' ? target : target.column;
  const record = {};
  for (const field of fields) {
    const raw = row[columnMap[field.key]];
    const num = parseFloat(raw);
    if (raw === undefined || raw === '' || Number.isNaN(num)) {
      return {
        ok: false,
        reason: 'mapping',
        detail: `${field.key}: empty/non-numeric`,
      };
    }
    record[field.key] = num;
  }
  try {
    validateAgainstFields(record, fields);
  } catch (e) {
    if (e instanceof ValidationError) {
      return { ok: false, reason: 'validation', detail: e.message };
    }
    throw e;
  }
  const price = parseFloat(row[targetColumn]);
  if (Number.isNaN(price)) {
    return { ok: false, reason: 'target', detail: 'target empty/non-numeric' };
  }
  return {
    ok: true,
    record,
    vector: fields.map((f) => record[f.key]),
    price,
  };
}

// Load a CSV and return validated training data plus mapping statistics.
//   options.fields    — field definitions (defaults to global validation schema)
//   options.columnMap — key → CSV column (defaults to each field's `column`)
//   options.target    — target column name or { column } (defaults to "price")
//   options.filter(row) — optionally restrict which rows are considered, so
//     several models can be built from different slices of the same file.
export function mapDataset(filePath, options = {}) {
  const { fields, columnMap, targetColumn } = resolveSchema(options);
  const { filter = null } = options;

  const allRows = loadRows(filePath);
  const rows = filter ? allRows.filter(filter) : allRows;

  const fieldKeys = fields.map((f) => f.key);
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

    const r = mapRow(row, columnMap, targetColumn, fields);
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
