# CSV → Model Mapping Evaluation

Run with `npm run test:csv` (defaults to the local `csv_example/` dataset, which
is gitignored and not distributed). Machine-readable output:
`results/csv-mapping-results.json`.

## What the test does

1. **Maps** each CSV row onto the model feature vector `[area, rooms, floor]`
   (`BuildingArea → area`, `Rooms → rooms`, `Bathroom → floor`, `Price` = target).
2. **Validates** every mapped row with the project's own `validateRecord` logic
   (range / type / integer / required), tallying *why* each row is rejected.
3. **Normalizes** the accepted feature matrix column-wise (z-score) so each
   measured object is expressed relative to its peers.
4. **EV-corrects** predictions: the expected value (EV) and a ±3σ "norm band"
   are computed; any prediction outside the band is clamped back to it, and any
   object whose normalized features sit beyond 3σ is rejected as an
   out-of-norm validation-logic violation.
5. **Reports** mapping-success, stability, and dataset-reduction statistics.

## Results (Melbourne_housing_FULL.csv, 34,857 rows)

| Metric | Value |
|--------|-------|
| Total rows | 34,857 |
| Mapped successfully | 10,522 (**30.2%**) |
| Rejected — mapping (empty/non-numeric) | 21,115 |
| Rejected — validation (out of range) | 162 |
| Rejected — missing price (target) | 3,058 |
| Expected value (EV) | 7,465.73 |
| Norm band (±3σ) | [-4,438.26, 19,369.72] |
| Predictions EV-corrected | 0 |
| Out-of-norm objects rejected | 323 |
| Accepted for scoring | 10,199 |
| Stability (coeff. of variation, lower = better) | **0.434** |
| Dataset reduction | **70.7%** |

## Reading the results

- **Mapping success is only 30.2%.** The dominant loss is the *mapping* stage
  (21,115 rows) — mostly empty `BuildingArea`. Real listings simply do not carry
  a measured building area, so the field most predictive of price is also the
  most often missing.
- **Validation rejections are small (162).** The configured ranges fit the data
  well; the model is rarely fed structurally impossible objects.
- **EV correction fired 0 times, but 323 objects were rejected up-front.**
  Because out-of-norm *inputs* are filtered before prediction, the model never
  had to emit an out-of-band *output* — the normalization step did the work the
  EV clamp would otherwise have done. The clamp remains the safety net.
- **Stability CV = 0.434** on the 10,199 accepted objects: predictions cluster
  reasonably tightly around the EV, so the model output is stable rather than
  erratic.

## Business interpretation — real-estate purchase decisions

- **Treat 30% as the trustworthy slice.** For ~70% of listings the data is too
  incomplete to score reliably. A buyer/investor should make automated,
  model-driven offers **only** on the mapped-and-accepted segment, and route the
  rest to manual appraisal. Acting on the full feed would mean pricing the
  majority of properties on guesses.
- **Data quality is the bottleneck, not the model.** The cheapest way to expand
  coverage is to source/clean `BuildingArea`, not to retune hyperparameters.
  Each percentage point of mapping success directly enlarges the addressable
  deal pipeline.
- **The norm band is a risk guardrail.** Predictions outside ±3σ are corrected
  and the 323 out-of-norm objects are quarantined — exactly the listings most
  likely to be data-entry errors or genuine outliers (mansions, mispriced
  bargains). For acquisitions, those should be reviewed by a human before any
  capital is committed.
- **Stability supports portfolio-scale use.** A CV of 0.43 means valuations are
  consistent enough to compare properties against one another and to set
  repeatable bid thresholds, rather than swinging unpredictably between runs.

## Predicted next steps when the model cannot predict accurately

When inputs fall outside what the model can score reliably, the pipeline is
designed to **degrade safely rather than emit noise**:

1. **Reject, don't guess** — rows that fail mapping/validation are dropped with a
   logged reason instead of being imputed and scored.
2. **Auto-correct the EV** — predictions drifting past the ±3σ band are clamped
   to the band, so a single bad input cannot produce a wild valuation.
3. **Normalize the object's matrix sequence** — features are z-scored against the
   peer distribution before scoring, stopping scale-driven outliers from
   dominating.
4. **Quarantine out-of-norm objects** — anything still beyond 3σ after
   normalization is rejected as a validation-logic violation and excluded from
   scoring.
5. **Reduce and report** — the surviving set (the `reductionRate`) and the
   `stability` CV are surfaced so the model's effective coverage and reliability
   are always visible, not hidden behind an optimistic headline accuracy.
