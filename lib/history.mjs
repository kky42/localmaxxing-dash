// The in-memory chart history: what /api/state sends for the selected range, and the active output average. The chart
// fields are totals over the model servers; with several, each point also has every server's own under `servers`.

export const HISTORY_FIELDS = ["outputTokensPerSecond", "promptTokensPerSecond", "runningRequests", "queue"];
export const HISTORY_NODE_FIELDS = ["temperature", "memoryAvailableBytes"];

// A history point carries the last prompt pass its server reported: prefillAt (ms, prefillUpdatedAt) and prefillRate.
// The chart draws each pass once, at its own time, so a pass held over many points is one dot.
export function prefillPass(inference) {
  const at = Date.parse(inference?.ok ? inference.prefillUpdatedAt ?? "" : "");
  const rate = inference?.promptComputeTokensPerSecond ?? inference?.promptTokensPerSecond;
  return Number.isFinite(at) && Number.isFinite(rate) && rate > 0 ? { prefillAt: at, prefillRate: rate } : { prefillAt: null, prefillRate: null };
}
export function latestPass(passes) {
  return passes.reduce((latest, pass) => (Number.isFinite(pass.prefillAt) && !(pass.prefillAt <= latest.prefillAt) ? pass : latest), { prefillAt: null, prefillRate: null });
}
// The passes first seen in a bucket (after `before`, the pass held when it began) fold into one at the latest time.
function bucketPass(bucket, before, pick = (point) => point) {
  const fresh = new Map();
  for (const point of bucket) {
    const { prefillAt, prefillRate } = pick(point) ?? {};
    if (Number.isFinite(prefillAt) && Number.isFinite(prefillRate) && !(prefillAt <= before)) fresh.set(prefillAt, prefillRate);
  }
  if (!fresh.size) return { prefillAt: pick(bucket.at(-1))?.prefillAt ?? null, prefillRate: pick(bucket.at(-1))?.prefillRate ?? null };
  return { prefillAt: Math.max(...fresh.keys()), prefillRate: mean([...fresh.values()]) };
}

const mean = (values) => {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null;
};

// At most targetCount points: consecutive samples are averaged into one point stamped with the last one's time.
// Missing values are left out of an average, and a bucket with none stays null so the chart line breaks.
export function downsampleHistory(points, targetCount = 360) {
  if (points.length <= targetCount) return points;
  const bucketSize = Math.ceil(points.length / targetCount);
  const output = [];
  for (let index = 0; index < points.length; index += bucketSize) {
    const bucket = points.slice(index, index + bucketSize);
    const last = bucket.at(-1);
    const before = points[index - 1];
    const averaged = { ...last, ...bucketPass(bucket, before?.prefillAt) };
    for (const key of HISTORY_FIELDS) averaged[key] = mean(bucket.map((point) => point[key]));
    averaged.nodes = Object.fromEntries(Object.keys(last.nodes ?? {}).map((id) => [id,
      Object.fromEntries(HISTORY_NODE_FIELDS.map((field) => [field, mean(bucket.map((point) => point.nodes?.[id]?.[field]))]))]));
    // With several model servers, each server's own fields are averaged the same way.
    if (last.servers) {
      averaged.servers = Object.fromEntries(Object.keys(last.servers).map((id) => [id, {
        ...Object.fromEntries(HISTORY_FIELDS.map((field) => [field, mean(bucket.map((point) => point.servers?.[id]?.[field]))])),
        ...bucketPass(bucket, before?.servers?.[id]?.prefillAt, (point) => point.servers?.[id]),
      }]));
    }
    output.push(averaged);
  }
  return output;
}

// The mean output rate over the samples with requests running (idle periods would pull it towards zero).
export function summarizeHistory(points, minutes = 60) {
  const activeOutputRates = points
    .filter((point) => point.runningRequests > 0 && Number.isFinite(point.outputTokensPerSecond))
    .map((point) => point.outputTokensPerSecond);
  return {
    activeOutputTokensPerSecond: activeOutputRates.length
      ? activeOutputRates.reduce((sum, value) => sum + value, 0) / activeOutputRates.length
      : null,
    activeSamples: activeOutputRates.length,
    windowMinutes: minutes,
  };
}

