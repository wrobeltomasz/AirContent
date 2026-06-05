import http from 'http';
import { createModel, trainModel, predict, FEATURE_COUNT } from './model.js';
import { config } from './config.js';

let trainedModel = null;
const startMs = Date.now();

// Live server state — queryable at any time via GET /health.
const state = {
  status: 'starting', // starting -> training -> ready
  startedAt: new Date().toISOString(),
  modelTrainedAt: null,
  requests: 0,
  predictions: 0,
  errors: 0,
  lastPrediction: null,
  // Content-length accounting for responses sent from server to browser.
  responses: 0, // number of responses sent
  bytesSent: 0, // cumulative response body bytes sent to browsers
  lastContentLength: null, // byte length of the most recent response
  maxContentLength: 0, // largest single response seen, in bytes
  // Timing — measures the time between a request arriving from the browser and
  // its response being sent. Aggregates feed the software-stability picture.
  totalResponseTimeMs: 0, // cumulative response time across all responses
  lastResponseTimeMs: null, // time taken by the most recent response
  maxResponseTimeMs: 0, // slowest single response seen, in ms
  slowResponses: 0, // responses slower than config.metrics.timing.slowResponseMs
  // Business KPIs — domain counters seeded from config.metrics.business.
  cowsInField: config.metrics.business.cowsInField,
};

// Timestamped structured logger.
function log(level, message, data) {
  const ts = new Date().toISOString();
  const tail = data ? ' ' + JSON.stringify(data) : '';
  console.log(`[${ts}] ${level.toUpperCase().padEnd(5)} ${message}${tail}`);
}

// Measure the content length, in bytes, of a payload that will travel from the
// server to the browser. Uses UTF-8 byte length (not string .length) so the
// figure matches the Content-Length header and the bytes actually sent.
function measureContentLength(payload) {
  return Buffer.byteLength(payload, 'utf8');
}

// Serialize a JSON payload, measure its content length, fold that into the live
// server state, advertise it via the Content-Length header, and send it. All
// server-to-browser responses go through here so the accounting stays complete.
function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  const contentLength = measureContentLength(body);

  state.responses += 1;
  state.bytesSent += contentLength;
  state.lastContentLength = contentLength;
  if (contentLength > state.maxContentLength) {
    state.maxContentLength = contentLength;
  }

  res.writeHead(statusCode, { 'Content-Length': contentLength });
  res.end(body);
}

// Create and train model
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

// Create HTTP server
async function startServer() {
  const server = http.createServer((req, res) => {
    const reqStart = Date.now();
    state.requests += 1;

    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    // Log every request once the response is sent (method, url, status, timing).
    res.on('finish', () => {
      // Immediate measure: time between the browser's request and our response.
      const responseTimeMs = Date.now() - reqStart;

      // Fold the measure into the running timing stats (software-stability view).
      if (config.metrics.timing.enabled) {
        state.totalResponseTimeMs += responseTimeMs;
        state.lastResponseTimeMs = responseTimeMs;
        if (responseTimeMs > state.maxResponseTimeMs) {
          state.maxResponseTimeMs = responseTimeMs;
        }
        if (responseTimeMs > config.metrics.timing.slowResponseMs) {
          state.slowResponses += 1;
        }
      }

      log('info', 'request', {
        method: req.method,
        url: req.url,
        status: res.statusCode,
        ms: responseTimeMs,
      });
    });

    // Handle predict endpoint
    if (req.url === '/predict' && req.method === 'POST') {
      // Guard against requests arriving before the model is ready.
      if (state.status !== 'ready') {
        sendJson(res, 503, { error: 'Model not ready', status: state.status });
        return;
      }

      let body = '';

      req.on('data', (chunk) => {
        body += chunk.toString();
      });

      req.on('end', async () => {
        try {
          const input = JSON.parse(body);

          if (!Array.isArray(input) || input.length !== FEATURE_COUNT) {
            log('warn', 'invalid input rejected', { body });
            sendJson(res, 400, {
              error: `Input must be an array of ${FEATURE_COUNT} numbers`,
            });
            return;
          }

          const value = await predict(trainedModel, input);
          const output = Number(value.toFixed(2));

          state.predictions += 1;
          state.lastPrediction = {
            input,
            output,
            at: new Date().toISOString(),
          };
          log('info', 'prediction', { input, output });

          sendJson(res, 200, { input, prediction: output.toFixed(2) });
        } catch (error) {
          state.errors += 1;
          log('error', 'prediction failed', { message: error.message });
          sendJson(res, 500, { error: error.message });
        }
      });

      return;
    }

    // Health check endpoint — reports the full current server state.
    if (req.url === '/health' && req.method === 'GET') {
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
        cowsInField: state.cowsInField,
      });
      return;
    }

    // Documentation endpoint
    if (req.url === '/' && req.method === 'GET') {
      sendJson(res, 200, {
        message: 'ML Model API',
        endpoints: {
          'POST /predict': 'Send [num1, num2, num3] to get prediction',
          'GET /health': 'Check API status and live server state',
        },
        example:
          'curl -X POST -H "Content-Type: application/json" -d "[1, 2, 1.5]" http://localhost:3000/predict',
      });
      return;
    }

    sendJson(res, 404, { error: 'Not found' });
  });

  const PORT = process.env.PORT || config.server.port;
  server.listen(PORT, () => {
    log('info', `ML API server listening on http://localhost:${PORT}`);
  });
}

// Main
(async () => {
  log('info', 'Server process starting', { startedAt: state.startedAt });
  await initializeModel();
  await startServer();
})().catch((error) => {
  state.status = 'error';
  log('error', 'fatal startup error', { message: error.message });
  process.exitCode = 1;
});
