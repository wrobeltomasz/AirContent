// model-routes.js — admin API for building and inspecting several models.
//
// Models are trained on a server-side CSV (default: the local Melbourne file)
// and held in the in-memory registry for the lifetime of the server process.
// The admin panel uses these endpoints to populate its model-selection menu,
// show per-model training statistics + measures, and predict with a chosen
// model.

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
import { getConfig } from './config-store.js';

const router = express.Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_DATASET = 'csv_example/properties.csv';

function resolveDataset(source) {
  const rel = source && source.trim() ? source.trim() : DEFAULT_DATASET;
  return path.isAbsolute(rel) ? rel : path.join(__dirname, rel);
}

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

// Build (train) a model. A business description is mandatory.
router.post('/api/models', express.json(), async (req, res) => {
  try {
    const { name, description, type, source, sigma, dirtinessThreshold } =
      req.body || {};
    if (!name || !name.trim()) {
      res.status(400).json({ error: 'name is required' });
      return;
    }
    if (!description || !description.trim()) {
      res.status(400).json({ error: 'a business description is required' });
      return;
    }

    const dataset = resolveDataset(source);
    if (!existsSync(dataset)) {
      res.status(400).json({ error: `dataset not found: ${dataset}` });
      return;
    }

    const options = { description };
    if (type && type !== 'all') options.filter = (row) => row.Type === type;
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
    const featureCount = getConfig().model.featureCount;
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
