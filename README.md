# AirContent — Multi-Domain Price Prediction with Admin Panel

A production-ready Node.js + TensorFlow.js ML system that trains regression models for any domain — real estate, cars, payroll, or your own — through a **model-type wizard**, with a browser-based admin panel, PostgreSQL persistence, configurable validation, and CSV import.

## Features

- **Model-Type Wizard:** Define what each model trains on — real estate, cars, payroll, or any user-defined type with its own properties and target
- **ML Model:** TensorFlow.js regression network with configurable architecture; the input size adapts to the chosen type
- **Disk Persistence:** Trained models are saved to `models/` and reloaded automatically on startup
- **Database:** PostgreSQL storage with JSONB metadata and tagging
- **Validation:** JavaScript-based input validation (configurable via admin panel)
- **Admin Panel:** Browser-based UI to configure model types, model, validation, database, and manage records
- **CSV Import:** Bulk record upload with field mapping
- **REST API:** Prediction endpoints supporting both raw features and structured metadata
- **Health Checks:** Live server state and metrics endpoint
- **Self-Contained:** No external ML services; all computation in Node.js

## Stack

- **Runtime:** Node.js 18+ (ES modules)
- **Framework:** Express.js
- **ML:** TensorFlow.js with tfjs-layers
- **Database:** PostgreSQL with pg driver
- **Frontend:** Vanilla HTML/CSS/JS admin panel
- **Code Quality:** ESLint + Prettier
- **License:** MIT 2.0

## Installation

```bash
npm install
```

### Database Setup

PostgreSQL must be running. Configure connection via environment variables or admin panel:

```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_DATABASE=aircontent
export PG_USER=postgres
export PG_PASSWORD=yourpassword

npm run db:init
```

Or configure directly in the admin panel at `http://localhost:3000/admin` under "Database" tab.

## Quick Start

```bash
# Start the API server with trained model
npm run ml:api

# Server starts on http://localhost:3000
# Admin panel: http://localhost:3000/admin
# Health check: curl http://localhost:3000/health
```

## Usage

### Admin Panel

Open **http://localhost:3000/admin** in your browser.

**Tabs:**

- **Dashboard** — System overview and metrics
- **System** — Live server hardware stats: RAM, CPU, process heap, request counters, uptime (auto-refreshes every 5 s)
- **Models** — Train models from a chosen type (and optional data slice) and inspect each one's measures
- **Model Types** — Wizard to define what each model trains on: properties + target, for built-in or custom types
- **Validation** — Configure input validation rules and allowed ranges
- **Model** — Tune hyperparameters (learning rate, hidden units, epochs, etc.)
- **Database** — Configure PostgreSQL connection
- **Records** — View and delete stored records; filter by tag
- **CSV Import** — Bulk upload CSV files with field mapping

All changes are persisted to `runtime-config.json` and applied immediately.

### Model Types (Wizard)

A **model type** is a named schema that decides what a model trains on: an
ordered list of **properties** (each property maps to one numeric input of the
network and to one CSV column) plus the **target** column to predict. This is
what lets the system train on more than one static schema.

Three built-in types ship with the project:

| Type            | Properties                       | Target   | Default dataset              |
| --------------- | -------------------------------- | -------- | ---------------------------- |
| **Real Estate** | `area`, `rooms`, `floor`         | `price`  | `csv_example/properties.csv` |
| **Cars**        | `age`, `mileage`, `power`        | `price`  | `csv_example/cars.csv`       |
| **Payroll**     | `experience`, `age`, `education` | `salary` | `csv_example/payroll.csv`    |

From the **Model Types** tab you can create your own type (e.g. trucks,
laptops), edit a built-in, or revert an edited built-in to its shipped defaults.
Custom and edited types are persisted to `model-types.json` so they survive a
restart. Each property defines its key, label, CSV column, min/max range,
integer flag, and required flag — the same validation the records API enforces.

### Models

The **Models** tab builds and compares named models. For each model you pick a
**type**, give it a name and a required **business description** (a model with no
business context cannot be trained), and optionally restrict training to a data
slice (rows where a chosen column equals a value). The model trains on its
type's properties, is saved to `models/<name>/`, and is reloaded on the next
start. Trained models populate a selection menu; choosing one shows:

- **Measures** — one row per input property of the type with its **strength**
  (how strongly it drives the target), slope, deviation, slope/deviation deltas
  vs. the previous measure, and purity (data quality).
- **Integration hierarchy** — measures ordered from the purest to the dirtiest,
  with the running (cumulative) strength.
- **Total strength** — the model output: the sum of the strengths of the
  measures it kept.
- A **predict box** for ad-hoc valuations with the selected model.

Before training, the feature matrix is cleaned: outlier cells are clamped back
to the norm and any measure too dirty to trust (default: >80% missing) is
neutralised, so no dirty vector or measure can alter the model. Models are saved
to disk (`models/`) and reloaded on startup. See [GLOSSARY.md](GLOSSARY.md) for
the full term reference and API.

### REST API

#### Prediction from raw feature vector

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d "[85, 3, 2]"
```

Response:

```json
{
  "input": [85, 3, 2],
  "prediction": "352.89"
}
```

#### Prediction from structured metadata

Metadata is validated against configured rules, then converted to feature vector:

```bash
curl -X POST http://localhost:3000/api/predict-from-metadata \
  -H "Content-Type: application/json" \
  -d '{"area": 85, "rooms": 3, "floor": 2}'
```

Response:

```json
{
  "metadata": { "area": 85, "rooms": 3, "floor": 2 },
  "featureVector": [85, 3, 2],
  "prediction": "352.89"
}
```

#### Create record (validated metadata → database)

```bash
curl -X POST http://localhost:3000/api/records \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"area": 85, "rooms": 3, "floor": 2}, "tags": ["apartment", "sale"]}'
```

#### List records

```bash
curl "http://localhost:3000/api/records?limit=10&offset=0"

# Filter by tag
curl "http://localhost:3000/api/records?tag=apartment"
```

#### Health check

```bash
curl http://localhost:3000/health
```

Response:

```json
{
  "status": "ready",
  "model": "ready",
  "uptimeSeconds": 42,
  "startedAt": "2026-06-06T10:00:00.000Z",
  "modelTrainedAt": "2026-06-06T10:00:05.000Z",
  "requests": 15,
  "predictions": 12,
  "errors": 0,
  "recordsIngested": 50,
  "lastResponseTimeMs": 23,
  "maxResponseTimeMs": 145,
  "avgResponseTimeMs": 42.5,
  "slowResponses": 0
}
```

## Configuration

### Defaults

All configurable settings have defaults in **`config.js`**. At runtime they are layered with:

1. Overrides saved in `runtime-config.json` (from admin panel)
2. Environment variables (highest priority)

### Configuration Sections

#### Model Architecture

| Setting              | Default  | Description                          |
| -------------------- | -------- | ------------------------------------ |
| `model.featureCount` | 3        | Number of input features             |
| `model.hiddenUnits`  | [32, 16] | Hidden layer sizes (relu activation) |
| `model.dropoutRate`  | 0.2      | Dropout after first layer            |
| `model.learningRate` | 0.01     | Adam optimizer learning rate         |

#### Training

| Setting                  | Default       | Description                        |
| ------------------------ | ------------- | ---------------------------------- |
| `training.epochs`        | 50            | Training iterations                |
| `training.batchSize`     | 32            | Batch size for training            |
| `training.numSamples`    | 200           | Synthetic samples per training run |
| `training.targetWeights` | [50, 100, 75] | Label synthesis weights            |

#### Server

| Setting       | Default | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `server.port` | 3000    | HTTP listen port (override with `PORT` env var) |

#### Validation

Each field in `validation.fields` defines one input feature:

```javascript
{
  key: 'area',
  label: 'Square meters',
  min: 10,
  max: 1000,
  integer: false,
  required: true
}
```

- **key** — JSON field name in metadata
- **label** — Display name in admin panel
- **min/max** — Allowed range (validation rejects outside)
- **integer** — If true, rejects non-integers
- **required** — If true, rejects missing/null values

#### Database

| Setting             | Default    | Description            |
| ------------------- | ---------- | ---------------------- |
| `database.host`     | localhost  | PostgreSQL host        |
| `database.port`     | 5432       | PostgreSQL port        |
| `database.database` | aircontent | Database name          |
| `database.user`     | postgres   | Database user          |
| `database.password` | ""         | Database password      |
| `database.table`    | properties | Table name for records |
| `database.ssl`      | false      | Use SSL connection     |

#### Tags

```javascript
tags: ['apartment', 'house', 'studio', 'rental', 'sale', 'renovated', 'new'];
```

Configurable vocabulary for tagging records. Records can also have ad-hoc tags not in this list.

### Environment Variables

All settings can be overridden at runtime:

```bash
PORT=4000 \
PG_HOST=db.example.com \
PG_DATABASE=prod_aircontent \
PG_USER=admin \
PG_PASSWORD=secret \
npm run ml:api
```

## PostgreSQL Schema

Single table with JSONB metadata and tags:

```sql
CREATE TABLE properties (
  id SERIAL PRIMARY KEY,
  metadata JSONB NOT NULL,
  tags TEXT[] DEFAULT ARRAY[]::TEXT[],
  price FLOAT8,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX properties_tags_idx ON properties USING GIN(tags);
CREATE INDEX properties_created_idx ON properties (created_at DESC);
```

**Columns:**

- `id` — Auto-increment record ID
- `metadata` — JSONB containing validation fields (e.g., `{"area": 85, "rooms": 3, "floor": 2}`)
- `tags` — Array of labels for filtering/searching
- `price` — Optional predicted or reference price
- `created_at`, `updated_at` — Timestamps

### Example Record

```json
{
  "id": 1,
  "metadata": {
    "area": 85,
    "rooms": 3,
    "floor": 2
  },
  "tags": ["apartment", "sale", "renovated"],
  "price": 352.89,
  "created_at": "2026-06-06T10:05:00Z",
  "updated_at": "2026-06-06T10:05:00Z"
}
```

## CSV Import

### Format

CSV file with headers (configurable):

```csv
area,rooms,floor,tags,price
85,3,2,"apartment;sale",350000
120,4,1,"house;new",500000
45,1,3,"studio",150000
```

### Via Admin Panel

1. Navigate to **CSV Import** tab
2. Map CSV columns to required fields (area, rooms, floor)
3. Optionally specify tags and price columns
4. Select file and click **Import Records**

### Field Mapping

The CSV importer maps columns to your configured validation fields **in order**:

- Column mapping is done by name (e.g., "area", "rooms", "floor")
- Missing or invalid data causes import errors (listed in response)
- Successful records are stored in PostgreSQL

### Example Import Response

```json
{
  "imported": [
    {
      "id": 1,
      "metadata": { "area": 85, "rooms": 3, "floor": 2 },
      "tags": ["apartment", "sale"],
      "price": 350000,
      "created_at": "2026-06-06T10:10:00Z"
    }
  ],
  "errors": [
    {
      "message": "Field \"rooms\" must be an integer",
      "row": 5
    }
  ]
}
```

## Validation

Input validation happens at two points:

1. **Admin Panel** — When saving configuration, validates that ranges are sensible
2. **API / Database** — When creating records, validates metadata against rules

### Validation Rules

For each field:

- **Required** check: reject if missing/null
- **Type** check: must be a number
- **Integer** check: if marked integer, reject decimals
- **Range** check: must be within [min, max]

### Example Validation

Config:

```javascript
{
  key: 'area',
  label: 'Square meters',
  min: 10,
  max: 1000,
  integer: false,
  required: true
}
```

Valid inputs:

```javascript
{
  area: 85.5;
} // ✓ decimal within range
{
  area: 500;
} // ✓ integer within range
```

Invalid inputs:

```javascript
{
  area: null;
} // ✗ required
{
  area: 5;
} // ✗ below min
{
  area: 2000;
} // ✗ above max
{
  area: 'large';
} // ✗ not a number
```

## Model Architecture

Default 3-input regression network:

| Layer           | Output | Activation | Notes               |
| --------------- | ------ | ---------- | ------------------- |
| Dense (input 3) | 32     | ReLU       | First hidden layer  |
| Dropout         | 32     | —          | 20% dropout         |
| Dense           | 16     | ReLU       | Second hidden layer |
| Dense (output)  | 1      | Linear     | Prediction          |

**Parameters:** ~673  
**Optimizer:** Adam (lr=0.01)  
**Loss:** Mean Squared Error  
**Training:** 50 epochs, batch size 32, synthetic data

### Customization

Via admin panel or `config.js`:

```javascript
model: {
  hiddenUnits: [64, 32, 16],  // Add more layers
  dropoutRate: 0.3,            // Increase dropout
  learningRate: 0.001,         // Lower learning rate
}
```

Save via the admin panel, then restart the server. The model is trained once at startup, so a restart is required to apply architecture changes.

## Project Structure

```
.
├── config.js              # Default configuration
├── config-store.js        # Runtime config management (env vars + overrides)
├── validation.js          # Input validation logic (global + per-type fields)
├── model.js               # ML model (createModel, trainModel, predict)
├── model-types.js         # Model-type catalogue (wizard): built-in + custom types
├── db.js                  # PostgreSQL database layer
├── csv-import.js          # CSV parsing and import (DB)
├── csv-dataset.js         # Schema-driven CSV → feature-vector mapping
├── measures.js            # Per-measure analytics + dirty-data cleaning
├── model-registry.js      # Build/persist models from a type; reload on startup
├── model-routes.js        # Admin API for model types + building/inspecting models
├── admin.js               # Admin panel Express routes
├── ml-api.js              # Main API server (Express)
├── public/
│   ├── admin.html         # Browser admin panel UI
│   └── images/            # Admin panel icons
├── models/                # Trained models persisted to disk (gitignored)
├── tmp/                   # Temporary files (gitignored)
├── csv_example/           # Local example datasets (gitignored)
├── test/                  # Unit tests (node --test)
├── package.json
├── eslint.config.js
├── .prettierrc.json
├── GLOSSARY.md            # Term reference + how-to
├── LICENSE                # MIT License
└── README.md
```

## Scripts

| Script                 | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `npm start`            | Start the prediction API server (port 3000)       |
| `npm run ml:api`       | Alias for `npm start`                             |
| `npm run db:init`      | Initialize PostgreSQL schema                      |
| `npm test`             | Run the unit tests (model types + model registry) |
| `npm run format`       | Format with Prettier                              |
| `npm run format:check` | Check formatting                                  |
| `npm run lint`         | Lint with ESLint                                  |
| `npm run lint:fix`     | Lint and auto-fix                                 |

## API Endpoints

| Method | Path                         | Description                                                       |
| ------ | ---------------------------- | ----------------------------------------------------------------- |
| GET    | `/`                          | API documentation                                                 |
| GET    | `/health`                    | Server status and metrics                                         |
| GET    | `/api/system`                | OS-level resource stats (RAM, CPU, process heap, server counters) |
| POST   | `/predict`                   | Predict from feature vector                                       |
| POST   | `/api/predict-from-metadata` | Predict from validated metadata                                   |
| GET    | `/api/config`                | Get current config                                                |
| POST   | `/api/config`                | Save config changes                                               |
| POST   | `/api/config/reset`          | Reset to defaults                                                 |
| GET    | `/api/records`               | List records (paginated, filterable)                              |
| POST   | `/api/records`               | Create record                                                     |
| GET    | `/api/records/:id`           | Get record by ID                                                  |
| DELETE | `/api/records/:id`           | Delete record                                                     |
| POST   | `/api/csv/import`            | Import CSV file                                                   |
| GET    | `/api/model-types`           | List model types (wizard catalogue)                               |
| GET    | `/api/model-types/:id`       | One model-type definition                                         |
| POST   | `/api/model-types`           | Create/update a (custom or overridden) model type                 |
| DELETE | `/api/model-types/:id`       | Delete a custom type / revert an edited built-in                  |
| GET    | `/api/models`                | List trained models (selection menu)                              |
| GET    | `/api/models/:name`          | Model metadata + measures                                         |
| POST   | `/api/models`                | Build a model from a type + dataset (requires description)        |
| PATCH  | `/api/models/:name`          | Update a model's business description                             |
| POST   | `/api/models/:name/predict`  | Predict with a named model                                        |
| DELETE | `/api/models/:name`          | Remove a model                                                    |
| GET    | `/admin`                     | Admin panel (HTML)                                                |

## Error Handling

### Validation Errors

```json
{
  "error": "area must be >= 10; area must be <= 1000"
}
```

Status: `400 Bad Request`

### Model Not Ready

```json
{
  "error": "Model not ready",
  "status": "training"
}
```

Status: `503 Service Unavailable`

### Database Errors

```json
{
  "error": "Failed to connect to database"
}
```

Status: `500 Internal Server Error`

## Logging

All events are logged to stdout with ISO timestamps:

```
[2026-06-06T10:00:05.123Z] INFO  Model trained and ready
[2026-06-06T10:00:10.456Z] INFO  prediction {"input":[85,3,2],"output":352.89}
[2026-06-06T10:00:10.467Z] INFO  request {"method":"POST","url":"/predict","status":200,"ms":11}
[2026-06-06T10:00:15.789Z] WARN  invalid input rejected {"body":"[1, 2]"}
[2026-06-06T10:00:20.012Z] ERROR prediction failed {"message":"Model not ready"}
```

## Development

### Formatting

```bash
npm run format
```

### Linting

```bash
npm run lint:fix
```

### Local Testing

```bash
# Terminal 1: Start server
npm run ml:api

# Terminal 2: Make requests
curl http://localhost:3000/health
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" -d "[85, 3, 2]"
```

### Offline Mode

If PostgreSQL is not available, the API still runs but database operations return errors. The ML model and predictions work independently of the database.

## Architecture Notes

### Single Source of Truth

- **config.js** — Defaults
- **runtime-config.json** — Admin panel overrides (optional, auto-created)
- **Environment variables** — Runtime overrides (highest priority)

Validation is enforced at three levels to prevent invalid configs from ever reaching the model.

### Memory Management

- TensorFlow.js tensors are explicitly disposed after use
- Database connections are pooled via pg
- Model is trained once at startup, then reused

### Extensibility

To train on a new domain, add a **model type** (no code change): open the
**Model Types** tab, define its properties and target, then train a model from
it on the Models tab. Built-in types live in `model-types.js`; custom ones are
saved to `model-types.json`.

To change the startup demo model's schema (`/predict`,
`/api/predict-from-metadata`):

1. Add entry to `validation.fields` in `config.js`
2. Ensure `model.featureCount` matches field count
3. Update `training.targetWeights` (one weight per field)
4. Restart server

To add new API endpoints:

1. Create route in `admin.js` or `ml-api.js`
2. Admin routes live in `admin.js` (mounted at root); ML routes in `ml-api.js`
3. Validate input via `validation.js` or custom logic

## Glossary & How-To

See **[GLOSSARY.md](GLOSSARY.md)** for term definitions and step-by-step guides
on defining model types, building models from them (including data slices), and
testing them.

## License

MIT License 2.0 — See LICENSE file for details.

## Support

For issues, questions, or contributions:

- File issues on GitHub
- Check admin panel logs (`/health` endpoint)
- Review PostgreSQL connection settings in Database tab
- Validate CSV format before import

---

**Version:** 2.1.0  
**Updated:** 2026-06-07
