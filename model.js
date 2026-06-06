// model.js — shared ML model definition, training, and inference helpers.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-layers';
import { getConfig } from './config-store.js';

export function createModel() {
  const { featureCount, hiddenUnits, dropoutRate, learningRate } = getConfig().model;
  const [firstUnits, ...restUnits] = hiddenUnits;

  const layers = [
    tf.layers.dense({
      inputShape: [featureCount],
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

export function generateData(numSamples) {
  const cfg = getConfig();
  const features = tf.randomNormal([numSamples, cfg.model.featureCount]);
  const labels = features.mul(tf.tensor1d(cfg.training.targetWeights)).sum(1, true);
  return { features, labels };
}

export async function trainModel(
  model,
  {
    numSamples = getConfig().training.numSamples,
    epochs = getConfig().training.epochs,
    batchSize = getConfig().training.batchSize,
  } = {}
) {
  const { features, labels } = generateData(numSamples);
  await model.fit(features, labels, { epochs, batchSize, verbose: 0 });
  features.dispose();
  labels.dispose();
  return model;
}

export async function predict(model, input) {
  const tensor = tf.tensor2d([input]);
  const output = model.predict(tensor);
  const [value] = await output.data();
  tensor.dispose();
  output.dispose();
  return value;
}
