// validation.js — JavaScript-based input validation.

import { getConfig } from './config-store.js';

export class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

export function validateRecord(metadata) {
  const cfg = getConfig();

  if (!cfg.validation.enabled) {
    return true;
  }

  const errors = [];

  for (const field of cfg.validation.fields) {
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
