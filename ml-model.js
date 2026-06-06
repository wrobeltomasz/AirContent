import { createModel, trainModel, predict } from './model.js';

// Sample inputs to demonstrate predictions after training.
const SAMPLE_INPUTS = [
  [1, 2, 1.5],
  [0.5, 1.5, 2],
  [2, 3, 1],
];

async function main() {
  console.log('TensorFlow.js ML Model Example');
  console.log('==============================\n');

  const model = createModel();
  console.log('Model created with architecture:');
  model.summary();
  console.log();

  console.log('Training neural network...');
  await trainModel(model);
  console.log('Training complete!\n');

  console.log('Making predictions:');
  for (const input of SAMPLE_INPUTS) {
    const value = await predict(model, input);
    console.log(
      `Input: [${input.join(', ')}] → Predicted price: $${value.toFixed(2)}`
    );
  }
}

main().catch(console.error);
