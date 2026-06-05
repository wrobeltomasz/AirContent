// config.js — central settings for this project's most common functions.
//
// Single source of truth for the variables that drive the shared ML helpers
// (createModel, trainModel, predict in model.js) and the HTTP server in
// ml-api.js. Tune behavior here instead of editing the functions themselves.

export const config = {
  // Architecture — consumed by createModel().
  model: {
    featureCount: 3, // number of input features the model expects
    hiddenUnits: [32, 16], // size of each hidden dense layer (relu)
    dropoutRate: 0.2, // dropout applied after the first hidden layer
    learningRate: 0.01, // Adam optimizer learning rate
  },

  // Training hyperparameters — consumed by trainModel() / generateData().
  training: {
    numSamples: 200, // synthetic samples generated per training run
    epochs: 50,
    batchSize: 32,
    targetWeights: [50, 100, 75], // weights used to synthesize labels (len must equal featureCount)
  },

  // HTTP prediction server — consumed by ml-api.js.
  server: {
    port: 3000, // overridable at runtime with the PORT env var
  },

  // Observability metrics — consumed by ml-api.js.
  metrics: {
    // Timing feature: how long each request spends between arriving from the
    // browser and the response being sent. The settings below are the "rules"
    // that drive the measurement, kept here as variables rather than hardcoded.
    timing: {
      enabled: true, // master switch for response-time tracking
      slowResponseMs: 500, // responses slower than this count as "slow" — a software-stability signal
    },

    // Business KPIs — domain counters surfaced via /health. Seed the starting
    // values here; wire them to real domain events as the product grows.
    business: {
      cowsInField: 0, // example business metric: head of cattle currently in the field
    },
  },
};

// Fail fast on inconsistent settings rather than producing a silently broken model.
if (config.training.targetWeights.length !== config.model.featureCount) {
  throw new Error(
    `config: training.targetWeights (${config.training.targetWeights.length}) ` +
      `must have one weight per feature (model.featureCount = ${config.model.featureCount})`
  );
}

export default config;
