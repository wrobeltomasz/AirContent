// model-registry.js — build and hold several models, each trained on a
// different dataset (or a different slice of the same file).
//
// Each model is created from the shared architecture (createModel) and trained
// on real, mapped + validated CSV data via trainModelOnData. Before training,
// the feature matrix is analysed into per-input "measures" and cleaned so dirty
// vectors/measures cannot alter the model. Every model carries a business
// description (required) plus its measures, hierarchy and summed measure
// strength so the admin panel can explain what was trained and why. Models are
// stored in-memory keyed by a caller-chosen name so they can be predicted
// against independently.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';
import {
  mkdirSync,
  existsSync,
  rmSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createModel, trainModelOnData, predict } from './model.js';
import { mapDataset } from './csv-dataset.js';
import { computeMeasures, cleanMatrix } from './measures.js';
import { getConfig } from './config-store.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const MODELS_DIR = path.join(__dirname, 'models');
export const TMP_DIR = path.join(__dirname, 'tmp');

const registry = new Map();

function ensureDir(dir) {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

async function saveModelToDisk(name, model, meta) {
  const dir = path.join(MODELS_DIR, name);
  ensureDir(dir);
  await model.save(`file://${dir}`);
  writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta, null, 2), 'utf8');
}

export async function initRegistry() {
  ensureDir(MODELS_DIR);
  ensureDir(TMP_DIR);
  if (!existsSync(MODELS_DIR)) return;
  const entries = readdirSync(MODELS_DIR, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(MODELS_DIR, entry.name);
    const metaPath = path.join(dir, 'meta.json');
    const modelPath = path.join(dir, 'model.json');
    if (!existsSync(metaPath) || !existsSync(modelPath)) continue;
    try {
      const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
      const model = await tf.loadLayersModel(`file://${modelPath}`);
      registry.set(entry.name, { model, meta });
    } catch (e) {
      console.warn(`[registry] failed to load model "${entry.name}": ${e.message}`);
    }
  }
}

// Build (train) one named model from a CSV file.
//   options.description           — business context for the model (required)
//   options.columnMap/filter/target — forwarded to mapDataset (data slice)
//   options.sigma/dirtinessThreshold — measure cleaning thresholds
export async function buildModelFromCSV(name, filePath, options = {}) {
  const { description, sigma, dirtinessThreshold, ...mapOptions } = options;

  if (!name || !String(name).trim()) {
    throw new Error('buildModelFromCSV: a model name is required');
  }
  if (!description || !String(description).trim()) {
    throw new Error(
      `buildModelFromCSV("${name}"): a business description is required`
    );
  }

  const data = mapDataset(filePath, mapOptions);
  if (!data.vectors.length) {
    throw new Error(
      `buildModelFromCSV("${name}"): no valid rows after mapping`
    );
  }

  const fields = getConfig().validation.fields;
  const analysis = computeMeasures(data.vectors, data.prices, {
    fields,
    fieldStats: data.fieldStats,
    sigma,
    dirtinessThreshold,
  });

  // Guard the model: clamp outliers and neutralise too-dirty measures before
  // the network ever sees the data.
  const cleaned = cleanMatrix(data.vectors, analysis.measures, analysis.sigma);

  const model = createModel();
  const { finalMae } = await trainModelOnData(model, cleaned, data.prices);

  const meta = {
    name,
    description: String(description).trim(),
    source: filePath,
    featureKeys: fields.map((f) => f.key),
    featureLabels: fields.map((f) => f.label),
    trainedRows: cleaned.length,
    totalRows: data.total,
    mappingSuccessRate: +data.mappingSuccessRate.toFixed(4),
    rejected: data.rejected,
    finalMae: finalMae == null ? null : +finalMae.toFixed(2),
    measures: analysis.measures,
    hierarchy: analysis.hierarchy,
    totalStrength: analysis.totalStrength,
    excludedMeasures: analysis.excluded,
    sigma: analysis.sigma,
    dirtinessThreshold: analysis.dirtinessThreshold,
    trainedAt: new Date().toISOString(),
  };
  registry.set(name, { model, meta });
  await saveModelToDisk(name, model, meta);
  return meta;
}

export function getModel(name) {
  const entry = registry.get(name);
  if (!entry) throw new Error(`model "${name}" not found in registry`);
  return entry.model;
}

export function getMeta(name) {
  const entry = registry.get(name);
  return entry ? entry.meta : null;
}

export function listModels() {
  return [...registry.values()].map((e) => e.meta);
}

// Update the business description of an existing model (the model itself is
// untouched). Throws if the model is unknown.
export function setDescription(name, description) {
  const entry = registry.get(name);
  if (!entry) throw new Error(`model "${name}" not found in registry`);
  if (!description || !String(description).trim()) {
    throw new Error('a business description is required');
  }
  entry.meta.description = String(description).trim();
  const metaPath = path.join(MODELS_DIR, name, 'meta.json');
  if (existsSync(metaPath)) {
    writeFileSync(metaPath, JSON.stringify(entry.meta, null, 2), 'utf8');
  }
  return entry.meta;
}

export async function predictWith(name, input) {
  return predict(getModel(name), input);
}

export function removeModel(name) {
  const dir = path.join(MODELS_DIR, name);
  if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  return registry.delete(name);
}

export function clearRegistry() {
  registry.clear();
}
