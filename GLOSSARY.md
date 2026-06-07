# Glossary & How-To

Reference for the terms and workflows added during the CSV evaluation and
multi-model work.

## Terms

| Term | Meaning |
|------|---------|
| **Feature vector** | The ordered numeric inputs the model consumes, defined by `validation.fields` in `config.js`. Default: `[area, rooms, floor]`. |
| **Column map** | Projection of raw CSV columns onto the feature vector, e.g. `BuildingArea → area`, `Rooms → rooms`, `Bathroom → floor` (`DEFAULT_COLUMN_MAP` in `csv-dataset.js`). |
| **Mapping success rate** | Fraction of CSV rows that map to a valid, fully-populated feature vector with a usable target price. |
| **Rejection reasons** | Why a row is dropped: `mapping` (empty/non-numeric cell), `validation` (out of configured range/type), `target` (missing price). |
| **EV (expected value)** | Mean of the model's predictions over the accepted set; centre of the norm band. |
| **Norm band (±3σ)** | `[EV − 3·std, EV + 3·std]`. Predictions outside it are auto-corrected (clamped) back to the bound. |
| **EV correction** | Clamping out-of-band predictions so the model stops emitting unpredictable values. |
| **Matrix normalization** | Column-wise z-scoring of the feature matrix so each measured object is expressed relative to its peers. |
| **Out-of-norm rejection** | An object whose normalized features still exceed ±3σ is rejected as a validation-logic violation rather than scored. |
| **Stability (CV)** | Coefficient of variation of accepted predictions (`std / mean`). Lower = more stable, repeatable output. |
| **Reduction rate** | Fraction of the original dataset removed by all filters before final scoring. |
| **Model registry** | In-memory store of several named models, each trained on a different dataset/slice (`model-registry.js`). |
| **Final MAE** | Mean absolute error at the end of training; recorded per model in its registry metadata. |
| **Measure** | One input parameter of the feature vector (area, rooms, floor). Each measure is individual and carries attributes: mean, deviation, slope, correlation, min/max, out-of-norm count, completeness (`measures.js`). |
| **Measure strength** | How strongly a measure drives price — the absolute Pearson correlation `|r|` (0–1) between that input and the target. |
| **Total strength** | The model's aggregate output measure: the sum of the strengths of the measures it kept (`totalStrength = Σ strength`). Each measure also reports its `weightShare` of this total. |
| **Slope / deviation deltas** | Each measure is differentiated from the previous one in the hierarchy by the change in its slope (price per unit) and deviation (spread). |
| **Purity / dirtiness** | Data quality of a measure: purity = completeness of its source column, dirtiness = its missing fraction. `dirtiness = 1 − purity`. |
| **Dirtiness threshold** | A measure dirtier than this (default `0.8`) is *neutralised* — replaced by a constant — so it cannot alter the model. Outlier *cells* in kept measures are clamped to ±σ instead of removed. |
| **Integration hierarchy** | Measures ranked from purest to dirtiest. The model is "integrated" in that order; `cumulativeStrength` shows the running sum of strength as each measure is added. |
| **Business description** | A required plain-language statement of what a model is for and how its output should be used. A model with no business context cannot be built. |

## How to build models

Models share the architecture from `createModel()` (`config.js` → `model.*`)
and are trained on real, mapped + validated CSV data.

### One model, real data

```js
import { createModel, trainModelOnData } from './model.js';
import { mapDataset } from './csv-dataset.js';

const data = mapDataset('csv_example/Melbourne_housing_FULL.csv');
const model = createModel();
await trainModelOnData(model, data.vectors, data.prices);
```

### Several models from different data

Use the registry. Each call builds and stores an independent model; a
**business description is required**. Pass a `filter` (or a different
`columnMap` / `target`, or a different file) to train on a different data slice:

```js
import { buildModelFromCSV, predictWith, listModels } from './model-registry.js';

const CSV = 'csv_example/Melbourne_housing_FULL.csv';

// Two models from the same file, different slices:
await buildModelFromCSV('houses', CSV, {
  description: 'Detached houses — buy-side valuation baseline.',
  filter: (row) => row.Type === 'h',
});
await buildModelFromCSV('units', CSV, {
  description: 'Units / apartments — rental yield screening.',
  filter: (row) => row.Type === 'u',
});

await predictWith('houses', [120, 4, 2]); // → predicted price
listModels(); // → metadata for every registered model
```

`buildModelFromCSV(name, filePath, options)` returns metadata including
`description`, `featureKeys`, `trainedRows`, `mappingSuccessRate`, `rejected`,
`finalMae`, `measures`, `hierarchy`, `totalStrength`, `excludedMeasures` and
`trainedAt`. Options: `description` (**required**), `filter` / `columnMap` /
`target` (forwarded to `mapDataset`), and `sigma` / `dirtinessThreshold`
(measure cleaning). Before training, the matrix is analysed into measures and
cleaned (`measures.js`): outlier cells are clamped and any measure dirtier than
the threshold is neutralised, so dirty data cannot alter the model.

### From the admin panel

Open **Models** in the admin panel (`/admin`). Fill in a name, pick a segment,
and write a business description (required), then **Train Model**. Trained
models populate the **selection menu**; choosing one shows its measures table,
the purest→dirtiest integration hierarchy, the summed measure strength, and a
predict box. Models live in the server process memory, so the menu lists
whatever has been trained since the server last started.

Endpoints behind the panel:

```
GET    /api/models                  list trained models (menu)
GET    /api/models/:name            full metadata + measures
POST   /api/models                  build a model { name, description, type?, source? }
PATCH  /api/models/:name            update { description }
POST   /api/models/:name/predict    { input: [area, rooms, floor] }
DELETE /api/models/:name            remove a model
```

## How to test models

### Mapping & stability evaluation

```bash
npm run test:csv                       # defaults to the local Melbourne file
node csv-mapping-test.js path/to.csv   # any CSV
```

Prints mapping-success, EV-correction, and stability stats and writes
`results/csv-mapping-results.json`. See `results/csv-mapping-report.md` for the
business interpretation.

### Unit tests

```bash
npm test    # node --test
```

`test/model-registry.test.js` builds several models from different slices and
checks they train, register, and predict independently. Tests needing the local
dataset **skip automatically** when `csv_example/` is absent (it is gitignored),
so the suite stays green in CI without the data.
