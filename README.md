# projekty

A Node.js machine-learning project built with [TensorFlow.js](https://www.tensorflow.org/js). It trains a small regression neural network and serves predictions over a lightweight HTTP API.

## Stack

- **Runtime:** Node.js (ES modules)
- **ML:** `@tensorflow/tfjs` + `@tensorflow/tfjs-layers` (pure-JS CPU backend)
- **Tooling:** Prettier (formatting) + ESLint (linting)

## Requirements

- Node.js 18+
- npm

## Installation

```bash
npm install
```

## Usage

### Train the model and print sample predictions

```bash
npm run ml:train
```

Builds the network, trains it on synthetic data for 50 epochs, and prints predictions for a few sample inputs.

### Start the prediction API server

```bash
npm run ml:api
```

The server trains the model on startup, then listens on **http://localhost:3000**.

## API

| Method | Endpoint   | Description                                        |
| ------ | ---------- | -------------------------------------------------- |
| GET    | `/`        | API documentation                                  |
| GET    | `/health`  | Live server state (status, uptime, counters, etc.) |
| POST   | `/predict` | Send a 3-number array, get a prediction            |

### Example

```bash
curl -X POST -H "Content-Type: application/json" \
  -d "[1, 2, 1.5]" \
  http://localhost:3000/predict
```

Response:

```json
{
  "input": [1, 2, 1.5],
  "prediction": "352.89"
}
```

Sending an array that is not exactly 3 numbers returns HTTP `400` with an error message. Requests received before the model finishes training return HTTP `503`.

### Server state and logging

The server logs every lifecycle event and request to stdout with timestamps, e.g.:

```
[2026-06-04T20:35:19.941Z] INFO  ML API server listening on http://localhost:3000
[2026-06-04T20:35:31.244Z] INFO  prediction {"input":[1,2,1.5],"output":352.85}
[2026-06-04T20:35:31.255Z] INFO  request {"method":"POST","url":"/predict","status":200,"ms":23}
[2026-06-04T20:35:31.536Z] WARN  invalid input rejected {"body":"[1, 2]"}
```

`GET /health` returns the current state at any time:

```json
{
  "status": "ready",
  "model": "ready",
  "uptimeSeconds": 14,
  "startedAt": "2026-06-04T20:35:17.096Z",
  "modelTrainedAt": "2026-06-04T20:35:19.931Z",
  "requests": 4,
  "predictions": 2,
  "errors": 0,
  "lastPrediction": {
    "input": [2, 3, 1],
    "output": 454.54,
    "at": "2026-06-04T20:35:31.530Z"
  }
}
```

The port can be overridden with the `PORT` environment variable (default `3000`).

## Configuration

All tunable settings for the project's most common functions live in a single
file, **`config.js`** — change them there instead of editing the functions:

| Group      | Setting                  | Used by                  | Default        |
| ---------- | ------------------------ | ------------------------ | -------------- |
| `model`    | `featureCount`           | `createModel`, validation | `3`            |
| `model`    | `hiddenUnits`            | `createModel`            | `[32, 16]`     |
| `model`    | `dropoutRate`            | `createModel`            | `0.2`          |
| `model`    | `learningRate`           | `createModel`            | `0.01`         |
| `training` | `numSamples`             | `trainModel`             | `200`          |
| `training` | `epochs`                 | `trainModel`             | `50`           |
| `training` | `batchSize`              | `trainModel`             | `32`           |
| `training` | `targetWeights`          | `generateData`           | `[50, 100, 75]`|
| `server`   | `port`                   | `ml-api.js`              | `3000`         |

`hiddenUnits` accepts any number of layers (e.g. `[64, 32, 16]`), and
`targetWeights` must have exactly `featureCount` entries — `config.js` throws on
startup if it does not. `trainModel(model, { epochs, batchSize, numSamples })`
arguments still override the config defaults per call, and `PORT` env var still
overrides `server.port`.

## Model

A regression neural network (~673 parameters):

| Layer           | Output shape | Activation |
| --------------- | ------------ | ---------- |
| Dense (input 3) | 32           | relu       |
| Dropout (0.2)   | 32           | —          |
| Dense           | 16           | relu       |
| Dense (output)  | 1            | linear     |

- **Optimizer:** Adam (learning rate 0.01)
- **Loss:** mean squared error
- **Training:** 50 epochs, batch size 32, on synthetic data

> **Note:** The model is trained in-memory on startup and is not yet persisted to disk — each run retrains from scratch.

## Project structure

```
.
├── config.js           # Central settings for the common functions (model/training/server)
├── model.js            # Shared model: createModel/trainModel/predict (single source of truth)
├── index.js            # Sample entry file
├── ml-model.js         # Train + predict CLI (npm run ml:train)
├── ml-api.js           # HTTP prediction server (npm run ml:api)
├── eslint.config.js    # ESLint flat config
├── .prettierrc.json    # Prettier config
└── package.json
```

## Scripts

| Script                 | Description                       |
| ---------------------- | --------------------------------- |
| `npm run ml:train`     | Train model and print predictions |
| `npm run ml:api`       | Start the prediction API server   |
| `npm run format`       | Format all files with Prettier    |
| `npm run format:check` | Check formatting without writing  |
| `npm run lint`         | Lint with ESLint                  |
| `npm run lint:fix`     | Lint and auto-fix                 |

## License

ISC
