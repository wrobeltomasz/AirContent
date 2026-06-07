// model-types.js — the "model type" catalogue that drives the wizard.
//
// A model type is a named schema describing WHAT a model is trained on: an
// ordered list of properties (each property → one numeric input of the network
// and one CSV column) plus the target column to predict. Built-in types ship
// with the project (real estate, cars, payroll); users define their own from
// the admin panel. Custom types and edited built-ins are persisted to
// model-types.json so they survive a restart.
//
// This decouples training from the single global validation schema in
// config.js: every model can now be trained on a different set of properties.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TYPES_PATH = path.join(__dirname, 'model-types.json');

const ID_RE = /^[a-z][a-z0-9_-]*$/;
const KEY_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

// Properties carry an optional CSV `column`; when omitted it defaults to the
// property key, so a tidy CSV (header == key) needs no per-column mapping.
export const BUILTIN_TYPES = [
  {
    id: 'realestate',
    label: 'Real Estate',
    icon: 'apartment.png',
    description: 'Property valuation from size and layout.',
    target: { key: 'price', label: 'Price', column: 'price' },
    fields: [
      {
        key: 'area',
        label: 'Area (m²)',
        column: 'area',
        min: 10,
        max: 1000,
        integer: false,
        required: true,
      },
      {
        key: 'rooms',
        label: 'Rooms',
        column: 'rooms',
        min: 1,
        max: 20,
        integer: true,
        required: true,
      },
      {
        key: 'floor',
        label: 'Floor',
        column: 'floor',
        min: 0,
        max: 100,
        integer: true,
        required: true,
      },
    ],
  },
  {
    id: 'cars',
    label: 'Cars',
    icon: 'car_gear.png',
    description: 'Used-car price from age, mileage and power.',
    target: { key: 'price', label: 'Price', column: 'price' },
    fields: [
      {
        key: 'age',
        label: 'Age (years)',
        column: 'age',
        min: 0,
        max: 40,
        integer: true,
        required: true,
      },
      {
        key: 'mileage',
        label: 'Mileage (km)',
        column: 'mileage',
        min: 0,
        max: 500000,
        integer: false,
        required: true,
      },
      {
        key: 'power',
        label: 'Power (hp)',
        column: 'power',
        min: 30,
        max: 800,
        integer: true,
        required: true,
      },
    ],
  },
  {
    id: 'payroll',
    label: 'Payroll',
    icon: 'payments.png',
    description: 'Salary estimate from experience and education.',
    target: { key: 'salary', label: 'Salary', column: 'salary' },
    fields: [
      {
        key: 'experience',
        label: 'Experience (years)',
        column: 'experience',
        min: 0,
        max: 45,
        integer: true,
        required: true,
      },
      {
        key: 'age',
        label: 'Age',
        column: 'age',
        min: 18,
        max: 70,
        integer: true,
        required: true,
      },
      {
        key: 'education',
        label: 'Education level (1–5)',
        column: 'education',
        min: 1,
        max: 5,
        integer: true,
        required: true,
      },
    ],
  },
];

const BUILTIN_IDS = new Set(BUILTIN_TYPES.map((t) => t.id));

function loadCustom() {
  if (!fs.existsSync(TYPES_PATH)) return {};
  try {
    return JSON.parse(fs.readFileSync(TYPES_PATH, 'utf8')) || {};
  } catch {
    console.warn(`Failed to parse ${TYPES_PATH}, ignoring custom model types`);
    return {};
  }
}

function writeCustom(custom) {
  fs.writeFileSync(TYPES_PATH, JSON.stringify(custom, null, 2), 'utf8');
}

// Validate + normalise a type definition coming from the wizard. Throws on any
// structural problem so a broken schema can never reach the trainer.
export function normalizeType(input) {
  if (!input || typeof input !== 'object') {
    throw new Error('model type must be an object');
  }
  const id = String(input.id || '').trim();
  if (!ID_RE.test(id)) {
    throw new Error(
      'type id must start with a letter and contain only lowercase letters, digits, "-" or "_"'
    );
  }
  const label = String(input.label || '').trim();
  if (!label) throw new Error('type label is required');

  const fields = Array.isArray(input.fields) ? input.fields : [];
  if (fields.length < 1) throw new Error('a type needs at least one property');

  const seen = new Set();
  const normFields = fields.map((f, i) => {
    const key = String(f.key || '').trim();
    if (!KEY_RE.test(key)) {
      throw new Error(`property #${i + 1}: key must be a valid identifier`);
    }
    if (seen.has(key)) throw new Error(`duplicate property key "${key}"`);
    seen.add(key);
    const min = f.min === '' || f.min == null ? null : Number(f.min);
    const max = f.max === '' || f.max == null ? null : Number(f.max);
    if (min != null && max != null && min > max) {
      throw new Error(`property "${key}": min (${min}) > max (${max})`);
    }
    return {
      key,
      label: String(f.label || key).trim(),
      column: String(f.column || key).trim() || key,
      min,
      max,
      integer: !!f.integer,
      required: f.required !== false,
    };
  });

  const t = input.target || {};
  const targetKey = String(t.key || 'price').trim() || 'price';
  const target = {
    key: targetKey,
    label: String(t.label || 'Target').trim() || 'Target',
    column: String(t.column || targetKey).trim() || targetKey,
  };

  return {
    id,
    label,
    icon: String(input.icon || 'ballot.png').trim() || 'ballot.png',
    description: String(input.description || '').trim(),
    builtin: BUILTIN_IDS.has(id),
    target,
    fields: normFields,
  };
}

// Built-ins layered with any persisted custom/overridden types.
export function listTypes() {
  const custom = loadCustom();
  const byId = new Map();
  for (const t of BUILTIN_TYPES) {
    byId.set(t.id, { ...normalizeType(t), builtin: true, customized: false });
  }
  for (const [id, def] of Object.entries(custom)) {
    try {
      const norm = normalizeType({ ...def, id });
      byId.set(id, {
        ...norm,
        builtin: BUILTIN_IDS.has(id),
        customized: true,
      });
    } catch {
      /* skip a corrupt persisted entry */
    }
  }
  return [...byId.values()];
}

export function getType(id) {
  return listTypes().find((t) => t.id === id) || null;
}

export function saveType(input) {
  const type = normalizeType(input);
  const custom = loadCustom();
  custom[type.id] = type;
  writeCustom(custom);
  return getType(type.id);
}

// Remove a custom type, or revert an edited built-in to its shipped defaults.
// A pristine built-in (no override on disk) cannot be removed.
export function removeType(id) {
  const custom = loadCustom();
  if (Object.prototype.hasOwnProperty.call(custom, id)) {
    delete custom[id];
    writeCustom(custom);
    return true;
  }
  return false;
}
