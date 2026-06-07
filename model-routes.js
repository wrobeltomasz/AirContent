// model-routes.js — admin API for the model-type wizard plus building and
// inspecting models.
//
// A model is trained on a server-side CSV against a chosen model type (the
// schema of properties + target). Types are managed via /api/model-types;
// models are persisted to disk and reloaded on startup. The admin panel uses
// these endpoints to manage types, populate its model-selection menu, show
// per-model statistics + measures, and predict with a chosen model.

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';
import {
  buildModelFromCSV,
  listModels,
  getMeta,
  setDescription,
  predictWith,
  removeModel,
} from './model-registry.js';
import { listTypes, getType, saveType, removeType } from './model-types.js';
import { getConfig } from './config-store.js';

const router = express.Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Default dataset per built-in type, so picking a type in the wizard needs no
// manual path. Custom types must supply their own data source.
const DEFAULT_DATASETS = {
  realestate: 'csv_example/properties.csv',
  cars: 'csv_example/cars.csv',
  payroll: 'csv_example/payroll.csv',
};
const DEFAULT_DATASET = DEFAULT_DATASETS.realestate;

function resolveDataset(source, typeId) {
  const fallback = DEFAULT_DATASETS[typeId] || DEFAULT_DATASET;
  const rel = source && source.trim() ? source.trim() : fallback;
  return path.isAbsolute(rel) ? rel : path.join(__dirname, rel);
}

// ── Model types (the wizard catalogue) ──────────────────────────────────────
router.get('/api/model-types', (req, res) => {
  const types = listTypes();
  res.json({ types, count: types.length });
});

router.get('/api/model-types/:id', (req, res) => {
  const type = getType(req.params.id);
  if (!type) {
    res.status(404).json({ error: `type "${req.params.id}" not found` });
    return;
  }
  res.json(type);
});

// Create or update a (custom or overridden) model type.
router.post('/api/model-types', express.json(), (req, res) => {
  try {
    res.status(201).json(saveType(req.body || {}));
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Delete a custom type, or revert an edited built-in to its shipped defaults.
router.delete('/api/model-types/:id', (req, res) => {
  if (!removeType(req.params.id)) {
    res
      .status(404)
      .json({ error: `no custom definition for "${req.params.id}"` });
    return;
  }
  res.json({ success: true });
});

// List trained models — drives the panel's model-selection menu.
router.get('/api/models', (req, res) => {
  const models = listModels();
  res.json({ models, count: models.length });
});

// Full metadata (incl. measures + hierarchy) for one model.
router.get('/api/models/:name', (req, res) => {
  const meta = getMeta(req.params.name);
  if (!meta) {
    res.status(404).json({ error: `model "${req.params.name}" not found` });
    return;
  }
  res.json(meta);
});

// Build (train) a model. A business description and a model type are mandatory.
router.post('/api/models', express.json(), async (req, res) => {
  try {
    const {
      name,
      description,
      typeId,
      source,
      segmentColumn,
      segmentValue,
      sigma,
      dirtinessThreshold,
    } = req.body || {};
    if (!name || !name.trim()) {
      res.status(400).json({ error: 'name is required' });
      return;
    }
    if (!description || !description.trim()) {
      res.status(400).json({ error: 'a business description is required' });
      return;
    }

    const type = getType(typeId || 'realestate');
    if (!type) {
      res.status(400).json({ error: `unknown model type: ${typeId}` });
      return;
    }

    const dataset = resolveDataset(source, type.id);
    if (!existsSync(dataset)) {
      res.status(400).json({ error: `dataset not found: ${dataset}` });
      return;
    }

    const options = { description, type };
    // Optional generic slice: train only on rows where a column equals a value.
    if (segmentColumn && segmentValue !== undefined && segmentValue !== '') {
      options.filter = (row) =>
        String(row[segmentColumn]) === String(segmentValue);
    }
    if (Number.isFinite(sigma)) options.sigma = sigma;
    if (Number.isFinite(dirtinessThreshold))
      options.dirtinessThreshold = dirtinessThreshold;

    const meta = await buildModelFromCSV(name.trim(), dataset, options);
    res.status(201).json(meta);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Edit a model's business description.
router.patch('/api/models/:name', express.json(), (req, res) => {
  try {
    const meta = setDescription(req.params.name, (req.body || {}).description);
    res.json(meta);
  } catch (error) {
    const code = /not found/.test(error.message) ? 404 : 400;
    res.status(code).json({ error: error.message });
  }
});

// Predict a price with a chosen model.
router.post('/api/models/:name/predict', express.json(), async (req, res) => {
  try {
    const body = req.body;
    const input = Array.isArray(body) ? body : body && body.input;
    // Expected feature count comes from the model's own type, not the global
    // config — each model can have a different number of properties.
    const meta = getMeta(req.params.name);
    const featureCount = meta
      ? meta.featureKeys.length
      : getConfig().model.featureCount;
    if (!Array.isArray(input) || input.length !== featureCount) {
      res
        .status(400)
        .json({ error: `input must be an array of ${featureCount} numbers` });
      return;
    }
    const value = await predictWith(req.params.name, input);
    res.json({
      model: req.params.name,
      input,
      prediction: Number(value.toFixed(2)),
    });
  } catch (error) {
    const code = /not found/.test(error.message) ? 404 : 400;
    res.status(code).json({ error: error.message });
  }
});

// Remove a model from the registry.
router.delete('/api/models/:name', (req, res) => {
  if (!removeModel(req.params.name)) {
    res.status(404).json({ error: `model "${req.params.name}" not found` });
    return;
  }
  res.json({ success: true });
});

export default router;
