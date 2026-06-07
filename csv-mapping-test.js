// csv-mapping-test.js — maps a real-world housing CSV onto the model's feature
// vector, validates every row, applies expected-value (EV) correction and
// feature-matrix normalization, then reports mapping-success and stability
// statistics for the model evaluation.
//
// Standalone runner (no PostgreSQL required):
//   node csv-mapping-test.js [path/to/file.csv]
//
// Defaults to csv_example/Melbourne_housing_FULL.csv (gitignored local data).
// Writes a machine-readable report to results/csv-mapping-results.json.

import { writeFileSync, existsSync } from 'fs';
import { pathToFileURL } from 'url';
import { createModel, trainModel, predict } from './model.js';
import { mapDataset, DEFAULT_COLUMN_MAP } from './csv-dataset.js';

const DEFAULT_CSV = 'csv_example/Melbourne_housing_FULL.csv';

// Number of standard deviations beyond which a value is "deviating from the
// norm" and is corrected (predictions) or rejected (already-failed inputs).
const SIGMA = 3;

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}
function std(xs, m = mean(xs)) {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

// Column-wise z-score normalization of the feature matrix. "Normalizing the
// matrix sequence associated with the measured object" — each measured object
// (row) is expressed relative to the distribution of its peers so that the
// stability metrics are not dominated by raw scale differences.
function normalizeMatrix(matrix) {
  const cols = matrix[0]?.length ?? 0;
  const stats = [];
  for (let c = 0; c < cols; c++) {
    const col = matrix.map((r) => r[c]);
    const m = mean(col);
    stats.push({ mean: m, std: std(col, m) || 1 });
  }
  const normalized = matrix.map((r) =>
    r.map((v, c) => (v - stats[c].mean) / stats[c].std)
  );
  return { normalized, stats };
}

export async function runEvaluation(filePath) {
  const data = mapDataset(filePath);
  const total = data.total;
  const rejected = data.rejected;
  const matrix = data.vectors;
  const { normalized } = normalizeMatrix(matrix);

  // Train the project model and predict on every successfully mapped object.
  const model = createModel();
  await trainModel(model);
  const predictions = [];
  for (const v of matrix) predictions.push(await predict(model, v));

  // EV (expected-value) correction: clamp any prediction that deviates more
  // than SIGMA standard deviations from the expected value back to that bound,
  // so the model stops emitting unpredictable out-of-norm values. Inputs whose
  // *features* already deviate beyond SIGMA AND sit at a normalized extreme are
  // rejected as a validation-logic violation rather than silently predicted.
  const ev = mean(predictions);
  const sd = std(predictions, ev) || 1;
  const lo = ev - SIGMA * sd;
  const hi = ev + SIGMA * sd;

  let corrected = 0;
  let outOfNormRejected = 0;
  const finalPreds = [];
  for (let i = 0; i < predictions.length; i++) {
    const p = predictions[i];
    const featureDeviation = Math.max(...normalized[i].map(Math.abs));
    if (featureDeviation > SIGMA) {
      // out-of-norm measured object → reject (violation of validation logic)
      outOfNormRejected++;
      continue;
    }
    if (p < lo || p > hi) {
      finalPreds.push(Math.min(hi, Math.max(lo, p))); // EV auto-correction
      corrected++;
    } else {
      finalPreds.push(p);
    }
  }

  const accepted = finalPreds.length;
  const mappedCount = matrix.length;
  const mappingSuccessRate = data.mappingSuccessRate;
  // Stability: how tightly the accepted predictions cluster around the EV.
  // Lower coefficient of variation == more stable model output.
  const stabilityCV = accepted ? std(finalPreds, mean(finalPreds)) / (mean(finalPreds) || 1) : 0;
  // Reduction: fraction of the original dataset removed before final scoring,
  // by all filters combined (mapping + validation + out-of-norm).
  const reductionRate = total ? 1 - accepted / total : 0;

  return {
    dataset: filePath,
    totalRows: total,
    columnMap: DEFAULT_COLUMN_MAP,
    mapping: {
      mapped: mappedCount,
      rejectedMapping: rejected.mapping,
      rejectedValidation: rejected.validation,
      rejectedMissingTarget: rejected.target,
      mappingSuccessRate: +mappingSuccessRate.toFixed(4),
    },
    evCorrection: {
      expectedValue: +ev.toFixed(2),
      stdDev: +sd.toFixed(2),
      lowerBound: +lo.toFixed(2),
      upperBound: +hi.toFixed(2),
      predictionsCorrected: corrected,
      outOfNormRejected,
    },
    stability: {
      acceptedForScoring: accepted,
      coefficientOfVariation: +stabilityCV.toFixed(4),
      reductionRate: +reductionRate.toFixed(4),
    },
    generatedAt: new Date().toISOString(),
  };
}

async function main() {
  const filePath = process.argv[2] || DEFAULT_CSV;
  if (!existsSync(filePath)) {
    console.error(`CSV not found: ${filePath}`);
    process.exit(1);
  }
  const report = await runEvaluation(filePath);

  const m = report.mapping;
  const e = report.evCorrection;
  const s = report.stability;
  console.log(`\nCSV → DB mapping evaluation: ${report.dataset}`);
  console.log(`  Total rows ................. ${report.totalRows}`);
  console.log(`  Mapped successfully ........ ${m.mapped} (${(m.mappingSuccessRate * 100).toFixed(1)}%)`);
  console.log(`  Rejected (mapping) ......... ${m.rejectedMapping}`);
  console.log(`  Rejected (validation) ...... ${m.rejectedValidation}`);
  console.log(`  Rejected (missing price) ... ${m.rejectedMissingTarget}`);
  console.log(`  Expected value (EV) ........ ${e.expectedValue} (norm band [${e.lowerBound}, ${e.upperBound}])`);
  console.log(`  Predictions EV-corrected ... ${e.predictionsCorrected}`);
  console.log(`  Out-of-norm rejected ....... ${e.outOfNormRejected}`);
  console.log(`  Accepted for scoring ....... ${s.acceptedForScoring}`);
  console.log(`  Stability (CV, lower=better) ${s.coefficientOfVariation}`);
  console.log(`  Dataset reduction .......... ${(s.reductionRate * 100).toFixed(1)}%\n`);

  const out = 'results/csv-mapping-results.json';
  writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`Report written to ${out}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
