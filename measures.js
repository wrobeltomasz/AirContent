// measures.js — per-measure analytics for a trained model.
//
// A "measure" is one input parameter of the feature vector (e.g. area, rooms,
// floor). For every measure this module computes:
//   • strength      — how strongly the input drives the price (|Pearson r|),
//                     i.e. the measure strength by which the parameter is
//                     calculated. The model output aggregates these as
//                     totalStrength = Σ strength over the measures kept.
//   • attributes    — mean, deviation, slope, correlation, min/max, outliers.
//   • purity/dirty  — data quality of the measure (completeness of its source
//                     column). Purer = cleaner data.
//   • hierarchy     — measures ranked from the purest to the dirtiest, each one
//                     differentiated from the previous by its slope and
//                     deviation deltas, with the running (cumulative) strength
//                     showing integration order.
//
// cleanMatrix() then guards the model: outlier cells are clamped back to the
// norm and any measure too dirty to trust is neutralised to a constant, so no
// dirty vector or measure can alter the trained model.

const r2 = (x) => Math.round(x * 100) / 100;
const r4 = (x) => Math.round(x * 10000) / 10000;

// A measure whose source column is dirtier than this (default 80% unusable) is
// excluded from the model: it carries almost no trustworthy signal, so letting
// it train the network would corrupt rather than inform it.
export const DEFAULT_DIRTINESS_THRESHOLD = 0.8;
// Cells beyond this many standard deviations are treated as out-of-norm.
export const DEFAULT_SIGMA = 3;

function columnStats(col) {
  const n = col.length;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of col) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = n ? sum / n : 0;
  let variance = 0;
  if (n > 1) {
    for (const v of col) variance += (v - mean) ** 2;
    variance /= n - 1;
  }
  return { mean, std: Math.sqrt(variance), min: n ? min : 0, max: n ? max : 0 };
}

// Compute one measure per feature column, the purest→dirtiest hierarchy, and
// the summed measure strength used to prepare the model.
//   vectors    — mapped feature matrix (rows × features)
//   prices     — target value per row
//   options.fields      — config validation.fields (for key/label per column)
//   options.fieldStats  — per-field completeness from mapDataset (drives purity)
export function computeMeasures(vectors, prices, options = {}) {
  const {
    fields = null,
    fieldStats = null,
    sigma = DEFAULT_SIGMA,
    dirtinessThreshold = DEFAULT_DIRTINESS_THRESHOLD,
  } = options;

  const rows = vectors.length;
  const cols = vectors[0]?.length ?? 0;
  const price = columnStats(prices);

  const measures = [];
  for (let c = 0; c < cols; c++) {
    const col = vectors.map((row) => row[c]);
    const { mean, std, min, max } = columnStats(col);

    // Covariance with price → slope (price per unit) and Pearson correlation.
    let cov = 0;
    if (rows > 1) {
      for (let i = 0; i < rows; i++)
        cov += (col[i] - mean) * (prices[i] - price.mean);
      cov /= rows - 1;
    }
    const slope = std > 0 ? cov / (std * std) : 0;
    const correlation = std > 0 && price.std > 0 ? cov / (std * price.std) : 0;
    const strength = Math.abs(correlation);

    // Statistical outliers within this measure (used by cleanMatrix to clamp).
    let outOfNorm = 0;
    if (std > 0) {
      for (const v of col) if (Math.abs((v - mean) / std) > sigma) outOfNorm++;
    }

    // Purity = completeness of the source column. Falls back to fully-present
    // when no field stats were supplied.
    const key = fields?.[c]?.key ?? `f${c}`;
    const stat = fieldStats?.[key] ?? null;
    const missingRate = stat ? stat.missingRate : 0;
    const dirtiness = missingRate;
    const purity = 1 - dirtiness;
    const included = dirtiness <= dirtinessThreshold;
    const tier =
      dirtiness <= 0.05 ? 'clean' : included ? 'acceptable' : 'dirty';

    measures.push({
      index: c,
      key,
      label: fields?.[c]?.label ?? key,
      strength: r4(strength),
      purity: r4(purity),
      dirtiness: r4(dirtiness),
      included,
      tier,
      attributes: {
        mean: r2(mean),
        deviation: r2(std),
        slope: r2(slope),
        correlation: r4(correlation),
        min: r2(min),
        max: r2(max),
        outOfNorm,
        completeness: r4(1 - missingRate),
      },
    });
  }

  // Hierarchy: purest first; ties broken by the more useful (stronger) measure.
  const totalStrength = measures.reduce(
    (a, m) => a + (m.included ? m.strength : 0),
    0
  );
  const ordered = [...measures].sort(
    (a, b) => b.purity - a.purity || b.strength - a.strength
  );

  let prev = null;
  let cumulative = 0;
  const hierarchy = ordered.map((m, i) => {
    if (m.included) cumulative += m.strength;
    const node = {
      rank: i + 1,
      key: m.key,
      label: m.label,
      purity: m.purity,
      strength: m.strength,
      included: m.included,
      tier: m.tier,
      // Differentiation relative to the previous (purer) measure.
      slopeDeltaVsPrev: prev
        ? r2(m.attributes.slope - prev.attributes.slope)
        : 0,
      deviationDeltaVsPrev: prev
        ? r2(m.attributes.deviation - prev.attributes.deviation)
        : 0,
      cumulativeStrength: r4(cumulative),
    };
    prev = m;
    return node;
  });

  // Mirror rank, deltas and weight share back onto each measure for tables.
  for (const node of hierarchy) {
    const m = measures.find((x) => x.key === node.key);
    m.rank = node.rank;
    m.slopeDeltaVsPrev = node.slopeDeltaVsPrev;
    m.deviationDeltaVsPrev = node.deviationDeltaVsPrev;
    m.weightShare =
      totalStrength && m.included ? r4(m.strength / totalStrength) : 0;
  }

  return {
    rows,
    sigma,
    dirtinessThreshold,
    measures,
    hierarchy,
    totalStrength: r4(totalStrength),
    excluded: measures.filter((m) => !m.included).map((m) => m.key),
  };
}

// Guard the model before training: clamp out-of-norm cells back to the ±sigma
// band, and neutralise any excluded (too-dirty) measure to its column mean so a
// constant — carrying no signal — replaces untrustworthy values. Either way no
// dirty vector or measure can move the trained weights.
export function cleanMatrix(vectors, measures, sigma = DEFAULT_SIGMA) {
  const cols = vectors[0]?.length ?? 0;
  const stats = [];
  for (let c = 0; c < cols; c++)
    stats.push(columnStats(vectors.map((row) => row[c])));

  return vectors.map((row) =>
    row.map((v, c) => {
      if (!measures[c]?.included) return stats[c].mean; // neutralise dirty measure
      if (stats[c].std === 0) return v;
      const lo = stats[c].mean - sigma * stats[c].std;
      const hi = stats[c].mean + sigma * stats[c].std;
      return Math.min(hi, Math.max(lo, v)); // clamp outlier back to the norm
    })
  );
}
