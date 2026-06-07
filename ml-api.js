import express from 'express';
import { createModel, trainModel, predict } from './model.js';
import { getConfig } from './config-store.js';
import { metadataToFeatureVector, ValidationError } from './validation.js';
import * as db from './db.js';
import adminRoutes from './admin.js';
import modelRoutes from './model-routes.js';
import { initRegistry } from './model-registry.js';
import path from 'path';
import { fileURLToPath } from 'url';
import os from 'os';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
let trainedModel = null;
const startMs = Date.now();

const state = {
  status: 'starting',
  startedAt: new Date().toISOString(),
  modelTrainedAt: null,
  requests: 0,
  predictions: 0,
  errors: 0,
  lastPrediction: null,
  responses: 0,
  bytesSent: 0,
  lastContentLength: null,
  maxContentLength: 0,
  totalResponseTimeMs: 0,
  lastResponseTimeMs: null,
  maxResponseTimeMs: 0,
  slowResponses: 0,
};

function log(level, message, data) {
  const ts = new Date().toISOString();
  const tail = data ? ' ' + JSON.stringify(data) : '';
  console.log(`[${ts}] ${level.toUpperCase().padEnd(5)} ${message}${tail}`);
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  const contentLength = Buffer.byteLength(body, 'utf8');

  state.responses += 1;
  state.bytesSent += contentLength;
  state.lastContentLength = contentLength;
  if (contentLength > state.maxContentLength) {
    state.maxContentLength = contentLength;
  }

  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Content-Length': contentLength,
  });
  res.end(body);
}

async function initializeModel() {
  state.status = 'training';
  log('info', 'Initializing ML model...');
  const model = createModel();
  await trainModel(model);
  trainedModel = model;
  state.status = 'ready';
  state.modelTrainedAt = new Date().toISOString();
  log('info', 'Model trained and ready');
}

async function startServer() {
  try {
    await db.initSchema();
  } catch (error) {
    log('warn', 'Database init failed (continuing offline)', {
      message: error.message,
    });
  }

  app.use(express.json());
  app.use(express.static(path.join(__dirname, 'public')));

  app.use(adminRoutes);
  app.use(modelRoutes);

  app.get('/admin', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'admin.html'));
  });

  app.get('/', (req, res) => {
    sendJson(res, 200, {
      message: 'AirContent ML API',
      version: '2.0.0',
      endpoints: {
        'POST /predict': 'Send feature vector for prediction',
        'GET /health': 'System status and metrics',
        'GET /admin': 'Admin panel (browser)',
        'POST /api/records': 'Create record (admin)',
        'GET /api/records': 'List records (admin)',
        'GET /api/models': 'List trained models (admin)',
        'POST /api/models': 'Build a model from a dataset (admin)',
        'POST /api/models/:name/predict': 'Predict with a named model (admin)',
      },
      example:
        'curl -X POST -H "Content-Type: application/json" -d "[85, 3, 2]" http://localhost:3000/predict',
    });
  });

  app.get('/health', (req, res) => {
    const cfg = getConfig();
    sendJson(res, 200, {
      status: state.status,
      model: state.status === 'ready' ? 'ready' : 'not-ready',
      uptimeSeconds: Math.floor((Date.now() - startMs) / 1000),
      startedAt: state.startedAt,
      modelTrainedAt: state.modelTrainedAt,
      requests: state.requests,
      predictions: state.predictions,
      errors: state.errors,
      lastPrediction: state.lastPrediction,
      responses: state.responses,
      bytesSent: state.bytesSent,
      lastContentLength: state.lastContentLength,
      maxContentLength: state.maxContentLength,
      lastResponseTimeMs: state.lastResponseTimeMs,
      maxResponseTimeMs: state.maxResponseTimeMs,
      avgResponseTimeMs:
        state.responses > 0
          ? Number((state.totalResponseTimeMs / state.responses).toFixed(2))
          : null,
      slowResponses: state.slowResponses,
      recordsIngested: cfg.metrics.business.recordsIngested || 0,
    });
  });

  app.get('/api/system', (req, res) => {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;
    const cpus = os.cpus();
    const loadAvg = os.loadavg();
    const mem = process.memoryUsage();
    const platform = os.platform();

    sendJson(res, 200, {
      system: {
        platform,
        arch: os.arch(),
        hostname: os.hostname(),
        nodeVersion: process.version,
        osUptimeSeconds: Math.floor(os.uptime()),
      },
      memory: {
        totalMB: Math.round(totalMem / 1048576),
        freeMB: Math.round(freeMem / 1048576),
        usedMB: Math.round(usedMem / 1048576),
        usagePct: parseFloat(((usedMem / totalMem) * 100).toFixed(1)),
      },
      cpu: {
        count: cpus.length,
        model: cpus[0]?.model || 'unknown',
        loadAvgSupported: platform !== 'win32',
        loadAvg1m: parseFloat(loadAvg[0].toFixed(2)),
        loadAvg5m: parseFloat(loadAvg[1].toFixed(2)),
        loadAvg15m: parseFloat(loadAvg[2].toFixed(2)),
      },
      process: {
        heapUsedMB: parseFloat((mem.heapUsed / 1048576).toFixed(1)),
        heapTotalMB: parseFloat((mem.heapTotal / 1048576).toFixed(1)),
        heapUsagePct: parseFloat(((mem.heapUsed / mem.heapTotal) * 100).toFixed(1)),
        rssMB: parseFloat((mem.rss / 1048576).toFixed(1)),
        externalMB: parseFloat((mem.external / 1048576).toFixed(1)),
        uptimeSeconds: Math.floor(process.uptime()),
      },
      server: {
        uptimeSeconds: Math.floor((Date.now() - startMs) / 1000),
        status: state.status,
        requests: state.requests,
        predictions: state.predictions,
        errors: state.errors,
        responses: state.responses,
        bytesSentKB: parseFloat((state.bytesSent / 1024).toFixed(1)),
        avgResponseTimeMs:
          state.responses > 0
            ? Number((state.totalResponseTimeMs / state.responses).toFixed(2))
            : null,
        lastResponseTimeMs: state.lastResponseTimeMs,
        maxResponseTimeMs: state.maxResponseTimeMs,
        slowResponses: state.slowResponses,
      },
    });
  });

  app.post('/predict', (req, res) => {
    const reqStart = Date.now();
    state.requests += 1;

    res.on('finish', () => {
      const responseTimeMs = Date.now() - reqStart;
      if (getConfig().metrics.timing.enabled) {
        state.totalResponseTimeMs += responseTimeMs;
        state.lastResponseTimeMs = responseTimeMs;
        if (responseTimeMs > state.maxResponseTimeMs) {
          state.maxResponseTimeMs = responseTimeMs;
        }
        if (responseTimeMs > getConfig().metrics.timing.slowResponseMs) {
          state.slowResponses += 1;
        }
      }
      log('info', 'request', {
        method: 'POST',
        url: '/predict',
        status: res.statusCode,
        ms: responseTimeMs,
      });
    });

    if (state.status !== 'ready') {
      sendJson(res, 503, { error: 'Model not ready', status: state.status });
      return;
    }

    try {
      const input = req.body;

      const featureCount = getConfig().model.featureCount;
      if (!Array.isArray(input) || input.length !== featureCount) {
        state.errors += 1;
        log('warn', 'invalid input rejected', { body: JSON.stringify(input) });
        sendJson(res, 400, {
          error: `Input must be an array of ${featureCount} numbers`,
        });
        return;
      }

      predict(trainedModel, input)
        .then((value) => {
          const output = Number(value.toFixed(2));
          state.predictions += 1;
          state.lastPrediction = {
            input,
            output,
            at: new Date().toISOString(),
          };
          log('info', 'prediction', { input, output });
          sendJson(res, 200, { input, prediction: output.toFixed(2) });
        })
        .catch((error) => {
          state.errors += 1;
          log('error', 'prediction failed', { message: error.message });
          sendJson(res, 500, { error: error.message });
        });
    } catch (error) {
      state.errors += 1;
      log('error', 'prediction failed', { message: error.message });
      sendJson(res, 500, { error: error.message });
    }
  });

  app.post('/api/predict-from-metadata', (req, res) => {
    const reqStart = Date.now();
    state.requests += 1;

    res.on('finish', () => {
      const responseTimeMs = Date.now() - reqStart;
      if (getConfig().metrics.timing.enabled) {
        state.totalResponseTimeMs += responseTimeMs;
        state.lastResponseTimeMs = responseTimeMs;
        if (responseTimeMs > state.maxResponseTimeMs) {
          state.maxResponseTimeMs = responseTimeMs;
        }
        if (responseTimeMs > getConfig().metrics.timing.slowResponseMs) {
          state.slowResponses += 1;
        }
      }
    });

    if (state.status !== 'ready') {
      sendJson(res, 503, { error: 'Model not ready', status: state.status });
      return;
    }

    try {
      const metadata = req.body;
      const input = metadataToFeatureVector(metadata);
      predict(trainedModel, input)
        .then((value) => {
          const output = Number(value.toFixed(2));
          state.predictions += 1;
          state.lastPrediction = {
            metadata,
            prediction: output,
            at: new Date().toISOString(),
          };
          sendJson(res, 200, {
            metadata,
            featureVector: input,
            prediction: output.toFixed(2),
          });
        })
        .catch((error) => {
          state.errors += 1;
          log('error', 'prediction failed', { message: error.message });
          sendJson(res, 500, { error: error.message });
        });
    } catch (error) {
      state.errors += 1;
      const message =
        error instanceof ValidationError
          ? error.message
          : 'Prediction failed: ' + error.message;
      sendJson(res, 400, { error: message });
    }
  });

  const PORT = process.env.PORT || getConfig().server.port;
  app.listen(PORT, () => {
    log('info', `ML API server listening on http://localhost:${PORT}`);
    log('info', `Admin panel at http://localhost:${PORT}/admin`);
  });
}

(async () => {
  log('info', 'Server process starting', { startedAt: state.startedAt });
  await initRegistry();
  await initializeModel();
  await startServer();
})().catch((error) => {
  state.status = 'error';
  log('error', 'fatal startup error', { message: error.message });
  process.exitCode = 1;
});

process.on('SIGTERM', async () => {
  log('info', 'SIGTERM received, shutting down gracefully');
  await db.closePool();
  process.exit(0);
});
