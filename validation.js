// validation.js — JavaScript-based input validation.

import { getConfig } from './config-store.js';

export class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

// Validate a record against an explicit list of field definitions. Shared by
// the global validator (config.validation.fields) and the model-type pipeline,
// which validates against the properties of a chosen model type instead.
export function validateAgainstFields(metadata, fields, enabled = true) {
  if (!enabled) return true;

  const errors = [];

  for (const field of fields) {
    const value = metadata[field.key];

    // Required check
    if (field.required && (value === undefined || value === null)) {
      errors.push(`${field.label} is required`);
      continue;
    }

    if (value === undefined || value === null) continue;

    // Type check
    if (typeof value !== 'number') {
      errors.push(`${field.label} must be a number`);
      continue;
    }

    // Integer check
    if (field.integer && !Number.isInteger(value)) {
      errors.push(`${field.label} must be an integer`);
      continue;
    }

    // Range check
    if (typeof field.min === 'number' && value < field.min) {
      errors.push(`${field.label} must be >= ${field.min}`);
    }
    if (typeof field.max === 'number' && value > field.max) {
      errors.push(`${field.label} must be <= ${field.max}`);
    }
  }

  if (errors.length > 0) {
    throw new ValidationError(errors.join('; '));
  }

  return true;
}

export function validateRecord(metadata) {
  const cfg = getConfig();
  return validateAgainstFields(
    metadata,
    cfg.validation.fields,
    cfg.validation.enabled
  );
}

export function metadataToFeatureVector(metadata) {
  const cfg = getConfig();
  validateRecord(metadata);

  return cfg.validation.fields.map((field) => {
    const value = metadata[field.key];
    if (value === undefined || value === null) {
      throw new ValidationError(`${field.label} is missing`, field.key);
    }
    return value;
  });
}
