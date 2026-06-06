// config.js — default settings for every tunable part of the project.
//
// This file holds the DEFAULTS. At runtime they are layered with overrides that
// the admin panel persists to `runtime-config.json` (see config-store.js), so
// you normally tune behaviour from the browser admin panel rather than by
// editing this file. Editing the defaults here is still supported and is the
// source of truth whenever no runtime override exists.
//
// The feature vector the model consumes is defined by `validation.fields`: the
// fields, in order, map 1:1 to the numeric inputs of the network. With the
// defaults below a property is described by [area, rooms, floor].

export const config = {
  // Architecture — consumed by createModel().
  model: {
    featureCount: 3, // number of input features (must equal validation.fields.length)
    hiddenUnits: [32, 16], // size of each hidden dense layer (relu)
    dropoutRate: 0.2, // dropout applied after the first hidden layer
    learningRate: 0.01, // Adam optimizer learning rate
  },

  // Training hyperparameters — consumed by trainModel() / generateData().
  training: {
    numSamples: 200, // synthetic samples generated when no DB data is available
    epochs: 50,
    batchSize: 32,
    targetWeights: [50, 100, 75], // weights used to synthesize labels (len must equal featureCount)
  },

  // HTTP prediction + admin server — consumed by ml-api.js.
  server: {
    port: 3000, // overridable at runtime with the PORT env var
  },

  // Input validation — consumed by validation.js. Configurable from the admin
  // panel. Each field maps, in order, to one numeric input of the model and to
  // one key inside a record's metadata JSON.
  validation: {
    enabled: true, // master switch; when false, range/required checks are skipped
    fields: [
      {
        key: 'area',
        label: 'Square meters',
        min: 10,
        max: 1000,
        integer: false,
        required: true,
      },
      {
        key: 'rooms',
        label: 'Rooms',
        min: 1,
        max: 20,
        integer: true,
        required: true,
      },
      {
        key: 'floor',
        label: 'Floor',
        min: 0,
        max: 100,
        integer: true,
        required: true,
      },
    ],
  },

  // PostgreSQL connection + storage — consumed by db.js. Values here are the
  // defaults; the admin panel and PG* environment variables override them.
  database: {
    host: 'localhost',
    port: 5432,
    database: 'aircontent',
    user: 'postgres',
    password: '',
    table: 'properties', // single table holding all structured test data
    ssl: false,
  },

  // Configurable tag vocabulary — consumed by the admin panel and CSV import for
  // labelling/searching records. Records may also carry ad-hoc tags.
  tags: ['apartment', 'house', 'studio', 'rental', 'sale', 'renovated', 'new'],

  // Observability metrics — consumed by ml-api.js.
  metrics: {
    // Timing feature: how long each request spends between arriving from the
    // browser and the response being sent.
    timing: {
      enabled: true, // master switch for response-time tracking
      slowResponseMs: 500, // responses slower than this count as "slow"
    },
    // Business KPIs — domain counters surfaced via /health.
    business: {
      recordsIngested: 0, // running count of records imported via CSV/API
    },
  },
};

// Validate that a config object is internally consistent. Exported so both this
// file (on the defaults) and config-store.js (on the merged runtime config) can
// fail fast rather than building a silently broken model.
export function assertConfigConsistent(cfg) {
  const featureCount = cfg.model.featureCount;
  if (cfg.validation.fields.length !== featureCount) {
    throw new Error(
      `config: validation.fields (${cfg.validation.fields.length}) must have ` +
        `one entry per feature (model.featureCount = ${featureCount}).`
    );
  }
  if (cfg.training.targetWeights.length !== featureCount) {
    throw new Error(
      `config: training.targetWeights (${cfg.training.targetWeights.length}) ` +
        `must have one weight per feature (model.featureCount = ${featureCount}).`
    );
  }
  for (const field of cfg.validation.fields) {
    if (
      typeof field.min === 'number' &&
      typeof field.max === 'number' &&
      field.min > field.max
    ) {
      throw new Error(
        `config: validation field "${field.key}" has min (${field.min}) > max (${field.max}).`
      );
    }
  }
}

assertConfigConsistent(config);

export default config;
