import test from "node:test";
import assert from "node:assert/strict";
import { downsampleHistory, summarizeHistory, prefillPass } from "../lib/history.mjs";

const point = (at, output, running, temperature) => ({
  at, outputTokensPerSecond: output, promptTokensPerSecond: null, runningRequests: running, queue: 0,
  nodes: { 1: { temperature, memoryAvailableBytes: null } },
});

test("a long history is averaged down to at most the target count, keeping gaps as gaps", () => {
  const points = Array.from({ length: 10 }, (_, i) => point(i * 1000, i < 4 ? 10 * (i + 1) : null, 1, i < 2 ? null : 50 + i));
  assert.equal(downsampleHistory(points, 20), points);
  const down = downsampleHistory(points, 5);
  assert.equal(down.length, 5);
  // Each point stands for two samples and carries the later one's time.
  assert.deepEqual(down.map((p) => p.at), [1000, 3000, 5000, 7000, 9000]);
  assert.deepEqual(down.map((p) => p.outputTokensPerSecond), [15, 35, null, null, null]);
  assert.deepEqual(down.map((p) => p.nodes[1].temperature), [null, 52.5, 54.5, 56.5, 58.5]);
  assert.equal(down[0].nodes[1].memoryAvailableBytes, null);
});

test("the active average counts only samples with requests running", () => {
  const stats = summarizeHistory([point(0, 0, 0, 40), point(1, 60, 2, 41), point(2, 80, 1, 42), point(3, null, 1, 43)], 15);
  assert.deepEqual(stats, { activeOutputTokensPerSecond: 70, activeSamples: 2, windowMinutes: 15 });
  assert.equal(summarizeHistory([point(0, 0, 0, 40)]).activeOutputTokensPerSecond, null);
});

test("with several model servers, each server's own fields are averaged next to the totals", () => {
  const points = Array.from({ length: 4 }, (_, i) => ({ ...point(i * 1000, 30, 2, 50), servers: { a: { outputTokensPerSecond: 20 + i, promptTokensPerSecond: null, runningRequests: 1, queue: 0 }, b: { outputTokensPerSecond: i < 2 ? null : 10, promptTokensPerSecond: null, runningRequests: 1, queue: 0 } } }));
  const down = downsampleHistory(points, 2);
  assert.deepEqual(down.map((p) => p.servers.a.outputTokensPerSecond), [20.5, 22.5]);
  assert.deepEqual(down.map((p) => p.servers.b.outputTokensPerSecond), [null, 10]);
  assert.equal(downsampleHistory(points.map(({ servers, ...rest }) => rest), 2)[0].servers, undefined);
});

test("prompt passes survive downsampling: the passes first seen in a bucket fold into one, a held pass is not repeated", () => {
  const pass = (at, prefillAt, prefillRate) => ({ ...point(at, 10, 1, 50), prefillAt, prefillRate });
  const points = [pass(0, null, null), pass(1000, 500, 1000), pass(2000, 500, 1000), pass(3000, 500, 1000), pass(4000, 3500, 3000), pass(5000, 4500, 2000)];
  const down = downsampleHistory(points, 3);
  assert.deepEqual(down.map((p) => [p.prefillAt, p.prefillRate]), [[500, 1000], [500, 1000], [4500, 2500]]);
  assert.deepEqual(prefillPass({ ok: true, prefillUpdatedAt: new Date(7000).toISOString(), promptComputeTokensPerSecond: 1597.4 }), { prefillAt: 7000, prefillRate: 1597.4 });
  assert.deepEqual(prefillPass({ ok: true, prefillUpdatedAt: null, promptComputeTokensPerSecond: 0 }), { prefillAt: null, prefillRate: null });
  assert.deepEqual(prefillPass({ ok: false, prefillUpdatedAt: new Date(7000).toISOString(), promptComputeTokensPerSecond: 5 }), { prefillAt: null, prefillRate: null });
});

test("a downsampled bucket keeps the prompt pass running at its latest point that had one", async () => {
  const { runningPass } = await import("../lib/history.mjs");
  assert.deepEqual(runningPass({ ok: true, activePrefill: { startedAt: new Date(500).toISOString(), seconds: 1 } }), { prefillStartedAt: 500 });
  assert.deepEqual(runningPass({ ok: true, activePrefill: null }), { prefillStartedAt: null });
  const points = [0, 1, 2, 3].map((i) => ({ ...point(i * 1000, 10, 1, 50), prefillStartedAt: i === 2 ? 1500 : null }));
  const down = downsampleHistory(points, 2);
  assert.deepEqual(down.map((p) => p.prefillStartedAt), [null, 1500]);
});
