// model.js — shared ML model definition, training, and inference helpers.
// Single source of truth used by both ml-model.js (CLI) and ml-api.js (server),
// so the architecture and data logic can never drift between the two.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';
import { config } from './config.js';

// Number of input features the model expects. Defined in config.js (re-exported
// here so existing callers can keep importing it from model.js).
export const FEATURE_COUNT = config.model.featureCount;

// Weights used to synthesize labels from features (house-price style target).
const TARGET_WEIGHTS = config.training.targetWeights;

// Build and compile the regression network from the settings in config.model.
export function createModel() {
  const { hiddenUnits, dropoutRate, learningRate } = config.model;
  const [firstUnits, ...restUnits] = hiddenUnits;

  const layers = [
    tf.layers.dense({
      inputShape: [FEATURE_COUNT],
      units: firstUnits,
      activation: 'relu',
    }),
    tf.layers.dropout({ rate: dropoutRate }),
    ...restUnits.map((units) =>
      tf.layers.dense({ units, activation: 'relu' })
    ),
    tf.layers.dense({ units: 1, activation: 'linear' }),
  ];

  const model = tf.sequential({ layers });

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'meanSquaredError',
    metrics: ['mae'],
  });

  return model;
}

// Generate synthetic training data.
export function generateData(numSamples) {
  const features = tf.randomNormal([numSamples, FEATURE_COUNT]);
  const labels = features.mul(tf.tensor1d(TARGET_WEIGHTS)).sum(1, true);
  return { features, labels };
}

// Train the model on freshly generated synthetic data. Disposes its own tensors.
// Hyperparameters default to config.training but can be overridden per call.
export async function trainModel(
  model,
  {
    numSamples = config.training.numSamples,
    epochs = config.training.epochs,
    batchSize = config.training.batchSize,
  } = {}
) {
  const { features, labels } = generateData(numSamples);
  await model.fit(features, labels, { epochs, batchSize, verbose: 0 });
  features.dispose();
  labels.dispose();
  return model;
}

// Run a single prediction for a feature array; returns the raw numeric output.
// Cleans up the tensors it allocates so callers don't leak memory.
export async function predict(model, input) {
  const tensor = tf.tensor2d([input]);
  const output = model.predict(tensor);
  const [value] = await output.data();
  tensor.dispose();
  output.dispose();
  return value;
}
