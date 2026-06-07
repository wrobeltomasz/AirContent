# Glossary & How-To

Reference for the terms and workflows added during the CSV evaluation and
multi-model work.

## Terms

| Term                         | Meaning                                                                                                                                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ------------------------------------------ |
| **Model type**               | A named schema describing what a model trains on: an ordered list of properties (each → one input + one CSV column) plus the target column. Built-ins: real estate, cars, payroll; users add their own (`model-types.js`). |
| **Feature vector**           | The ordered numeric inputs the model consumes, defined by the chosen model type's properties (or, for the startup demo model, by `validation.fields` in `config.js`).                                                      |
| **Column map**               | Projection of raw CSV columns onto the feature vector. Each property carries a CSV `column` (defaulting to its key), e.g. real estate maps `area → area`, `rooms → rooms`, `floor → floor`.                                |
| **Mapping success rate**     | Fraction of CSV rows that map to a valid, fully-populated feature vector with a usable target price.                                                                                                                       |
| **Rejection reasons**        | Why a row is dropped: `mapping` (empty/non-numeric cell), `validation` (out of configured range/type), `target` (missing price).                                                                                           |
| **EV (expected value)**      | Mean of the model's predictions over the accepted set; centre of the norm band.                                                                                                                                            |
| **Norm band (±3σ)**          | `[EV − 3·std, EV + 3·std]`. Predictions outside it are auto-corrected (clamped) back to the bound.                                                                                                                         |
| **EV correction**            | Clamping out-of-band predictions so the model stops emitting unpredictable values.                                                                                                                                         |
| **Matrix normalization**     | Column-wise z-scoring of the feature matrix so each measured object is expressed relative to its peers.                                                                                                                    |
| **Out-of-norm rejection**    | An object whose normalized features still exceed ±3σ is rejected as a validation-logic violation rather than scored.                                                                                                       |
| **Stability (CV)**           | Coefficient of variation of accepted predictions (`std / mean`). Lower = more stable, repeatable output.                                                                                                                   |
| **Reduction rate**           | Fraction of the original dataset removed by all filters before final scoring.                                                                                                                                              |
| **Model registry**           | In-memory store of several named models, each trained on a different dataset/slice (`model-registry.js`).                                                                                                                  |
| **Final MAE**                | Mean absolute error at the end of training; recorded per model in its registry metadata.                                                                                                                                   |
| **Measure**                  | One input parameter of the feature vector (area, rooms, floor). Each measure is individual and carries attributes: mean, deviation, slope, correlation, min/max, out-of-norm count, completeness (`measures.js`).          |
| **Measure strength**         | How strongly a measure drives price — the absolute Pearson correlation `                                                                                                                                                   | r   | ` (0–1) between that input and the target. |
| **Total strength**           | The model's aggregate output measure: the sum of the strengths of the measures it kept (`totalStrength = Σ strength`). Each measure also reports its `weightShare` of this total.                                          |
| **Slope / deviation deltas** | Each measure is differentiated from the previous one in the hierarchy by the change in its slope (price per unit) and deviation (spread).                                                                                  |
| **Purity / dirtiness**       | Data quality of a measure: purity = completeness of its source column, dirtiness = its missing fraction. `dirtiness = 1 − purity`.                                                                                         |
| **Dirtiness threshold**      | A measure dirtier than this (default `0.8`) is _neutralised_ — replaced by a constant — so it cannot alter the model. Outlier _cells_ in kept measures are clamped to ±σ instead of removed.                               |
| **Integration hierarchy**    | Measures ranked from purest to dirtiest. The model is "integrated" in that order; `cumulativeStrength` shows the running sum of strength as each measure is added.                                                         |
| **Business description**     | A required plain-language statement of what a model is for and how its output should be used. A model with no business context cannot be built.                                                                            |

## How to build models

Models share the architecture from `createModel()` (`config.js` → `model.*`)
and are trained on real, mapped + validated CSV data.

### Pick a model type

A model type is the schema a model trains on. List or define types first:

```js
import { listTypes, getType, saveType } from './model-types.js';

listTypes(); // → real estate, cars, payroll + any custom types
const cars = getType('cars'); // { fields:[age,mileage,power], target:{key:'price'} }

// Define a custom type (persisted to model-types.json):
saveType({
  id: 'laptops',
  label: 'Laptops',
  target: { key: 'price', label: 'Price', column: 'price' },
  fields: [
    {
      key: 'ram',
      label: 'RAM (GB)',
      column: 'ram',
      min: 2,
      max: 128,
      integer: true,
    },
    {
      key: 'cpu',
      label: 'CPU score',
      column: 'cpu',
      min: 1000,
      max: 50000,
      integer: true,
    },
  ],
});
```

### Build models from a type

Each call builds and stores an independent model; a **business description is
required**. The model is trained on its type's properties and persisted to
`models/<name>/` so it survives a restart. Pass a `filter` to train on a slice:

```js
import {
  buildModelFromCSV,
  predictWith,
  listModels,
} from './model-registry.js';
import { getType } from './model-types.js';

await buildModelFromCSV('used-cars', 'csv_example/cars.csv', {
  description: 'Used-car pricing baseline.',
  type: getType('cars'),
});
await buildModelFromCSV('homes', 'csv_example/properties.csv', {
  description: 'Property valuation baseline.',
  type: getType('realestate'),
  filter: (row) => Number(row.rooms) <= 3, // a data slice
});

await predictWith('used-cars', [3, 40000, 180]); // → predicted price
listModels(); // → metadata for every registered model
```

`buildModelFromCSV(name, filePath, options)` returns metadata including
`description`, `typeId`, `target`, `featureKeys`, `trainedRows`,
`mappingSuccessRate`, `rejected`, `finalMae`, `measures`, `hierarchy`,
`totalStrength`, `excludedMeasures` and `trainedAt`. Options: `description`
(**required**), `type` (schema; defaults to the global config schema), `filter`
(data slice), and `sigma` / `dirtinessThreshold` (measure cleaning). Before
training, the matrix is analysed into measures and cleaned (`measures.js`):
outlier cells are clamped and any measure dirtier than the threshold is
neutralised, so dirty data cannot alter the model.

### From the admin panel

Open **Model Types** in the admin panel (`/admin`) to add or edit a type and its
properties. Then open **Models**: pick a type, give the model a name and a
required business description, and **Train Model**. Trained models populate the
**selection menu**; choosing one shows its measures table, the purest→dirtiest
integration hierarchy, the summed measure strength, and a predict box. Models
are saved to disk and reloaded the next time the server starts.

Endpoints behind the panel:

```
GET    /api/model-types             list model types (wizard catalogue)
GET    /api/model-types/:id         one type definition
POST   /api/model-types             create/update a type { id, label, target, fields }
DELETE /api/model-types/:id         delete a custom type / revert an edited built-in
GET    /api/models                  list trained models (menu)
GET    /api/models/:name            full metadata + measures
POST   /api/models                  build a model { name, description, typeId, source?, segmentColumn?, segmentValue? }
PATCH  /api/models/:name            update { description }
POST   /api/models/:name/predict    { input: [...properties of the type] }
DELETE /api/models/:name            remove a model
```

## How to test models

```bash
npm test    # node --test
```

`test/model-registry.test.js` builds a model per type and several models from
different slices, checking they train, register, persist and predict
independently. `test/model-types.test.js` checks the built-in types and the
wizard's validation. Tests needing the local example datasets **skip
automatically** when `csv_example/` is absent (it is gitignored), so the suite
stays green in CI without the data.
