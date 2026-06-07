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

// The browser `@tensorflow/tfjs` build has no file:// IO handler (that ships
// only with tfjs-node), so we persist with custom in-memory IO handlers and
// write the artifacts to disk ourselves: model.json (topology + weight specs)
// + weights.bin (raw weight bytes), matching the standard TF.js layout.
async function saveModelToDisk(name, model, meta) {
  const dir = path.join(MODELS_DIR, name);
  ensureDir(dir);
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const modelJson = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [
          { paths: ['weights.bin'], weights: artifacts.weightSpecs },
        ],
      };
      writeFileSync(
        path.join(dir, 'model.json'),
        JSON.stringify(modelJson),
        'utf8'
      );
      writeFileSync(
        path.join(dir, 'weights.bin'),
        Buffer.from(artifacts.weightData)
      );
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      };
    })
  );
  writeFileSync(
    path.join(dir, 'meta.json'),
    JSON.stringify(meta, null, 2),
    'utf8'
  );
}

async function loadModelFromDisk(dir) {
  const modelJson = JSON.parse(
    readFileSync(path.join(dir, 'model.json'), 'utf8')
  );
  const weightBuf = readFileSync(path.join(dir, 'weights.bin'));
  const weightData = weightBuf.buffer.slice(
    weightBuf.byteOffset,
    weightBuf.byteOffset + weightBuf.byteLength
  );
  const weightSpecs = modelJson.weightsManifest[0].weights;
  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs,
      weightData,
    })
  );
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
      const model = await loadModelFromDisk(dir);
      registry.set(entry.name, { model, meta });
    } catch (e) {
      console.warn(
        `[registry] failed to load model "${entry.name}": ${e.message}`
      );
    }
  }
}

// Build (train) one named model from a CSV file.
//   options.description           — business context for the model (required)
//   options.type                  — model type (schema of properties + target);
//                                    defaults to the global validation schema
//   options.filter                — restrict rows (build models from data slices)
//   options.sigma/dirtinessThreshold — measure cleaning thresholds
export async function buildModelFromCSV(name, filePath, options = {}) {
  const {
    description,
    type,
    sigma,
    dirtinessThreshold,
    filter,
    ...mapOptions
  } = options;

  if (!name || !String(name).trim()) {
    throw new Error('buildModelFromCSV: a model name is required');
  }
  if (!description || !String(description).trim()) {
    throw new Error(
      `buildModelFromCSV("${name}"): a business description is required`
    );
  }

  // A type defines the properties (feature vector) and target column. Without
  // one the model falls back to the global validation schema in config.js.
  const fields = type ? type.fields : getConfig().validation.fields;
  const targetMeta = type
    ? type.target
    : { key: 'price', label: 'Price', column: 'price' };

  const data = mapDataset(filePath, {
    ...mapOptions,
    fields,
    target: targetMeta,
    filter,
  });
  if (!data.vectors.length) {
    throw new Error(
      `buildModelFromCSV("${name}"): no valid rows after mapping`
    );
  }

  const analysis = computeMeasures(data.vectors, data.prices, {
    fields,
    fieldStats: data.fieldStats,
    sigma,
    dirtinessThreshold,
  });

  // Guard the model: clamp outliers and neutralise too-dirty measures before
  // the network ever sees the data.
  const cleaned = cleanMatrix(data.vectors, analysis.measures, analysis.sigma);

  const model = createModel(fields.length);
  const { finalMae } = await trainModelOnData(model, cleaned, data.prices);

  const meta = {
    name,
    description: String(description).trim(),
    typeId: type ? type.id : null,
    typeLabel: type ? type.label : 'Default (config schema)',
    target: targetMeta,
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
