// Bench engine: decode / prefill / agent suites against the inference API.
// Zero-dependency (global fetch). One job at a time; results append to
// bench-history.json next to the token ledger. TTFT and tok/s here are measured
// only from the bench's own requests — not the live panel's rolling window.
//
// Measurement protocols follow the sparkDash Decode/Prefill benches:
// - decode tok/s = (completion_tokens − 1) / (last − first token), temperature 0,
//   top_p 1, thinking off, 32-token warmup so JIT/drafters are not billed.
// - prefill tps = (prompt_tokens − cached) / (TTFT − fixed overhead), where the
//   overhead is the median TTFT of tiny prompts, capped at half the measured TTFT.
// - aggregate rates: each request's own real wall time is its denominator, and a
//   wave's aggregate is Σ tokens / Σ wall seconds.

import { readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";

const SUITES = new Set(["decode", "prefill", "agent"]);
const ALLOWED_CONCURRENCY = new Set([1, 2]);
const HISTORY_FILE = "bench-history.json";
const HISTORY_LIMIT = 30;
const REQUEST_TIMEOUT_MS = 900_000;
const DECODE_WARMUP_TOKENS = 32;
const DECODE_DEFAULT_TOKENS = 384;
const DECODE_MAX_TOKENS = 2048;
const PREFILL_SIZES = [8192, 32768, 65536];
/** words ≈ tokens / 1.3 for English filler; good enough to pick chunk sizes. */
const WORDS_PER_TOKEN = 1 / 1.3;

const FILLER = "The lighthouse keeper climbed the spiral stairs each evening to light the lamp before the fishing fleet returned through the narrow strait. ";
const CODE_FILLER = "def process_batch(items, *, strict=False, limit=None):\n    results = []\n    for item in items[:limit or len(items)]:\n        try:\n            results.append(transform(item, strict=strict))\n        except ValueError:\n            continue\n    return results\n\n";

function now() {
  return Date.now();
}

/** ~4 chars per token, the same fallback estimate the sparkDash reference uses. */
function estimateTokens(text) {
  return text && typeof text === "string" ? Math.max(1, Math.round(text.length / 4)) : 0;
}

function round2(value) {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : null;
}

function percentile(sorted, share) {
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1));
  return sorted[index];
}

function stats(values) {
  const clean = values.filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);
  if (!clean.length) return null;
  return { n: clean.length, median: round2(clean[Math.floor(clean.length / 2)]), p95: round2(percentile(clean, 0.95)) };
}

function thinkingOff(body) {
  body.chat_template_kwargs = {
    ...(body.chat_template_kwargs ?? {}),
    enable_thinking: false,
    thinking: false,
    thinking_mode: "disabled",
  };
  return body;
}

/** One streaming chat request. Returns timings, usage and any tool calls. */
async function chatStream({ apiUrl, model, messages, tools = undefined, maxTokens, signal, onDelta }) {
  const body = thinkingOff({
    model,
    messages,
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1,
    stream: true,
    stream_options: { include_usage: true },
  });
  if (tools) body.tools = tools;
  const started = now();
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => controller.abort(new Error("request timeout")), REQUEST_TIMEOUT_MS);
  let firstTokenAt = null;
  let lastTokenAt = null;
  let completionTokens = null;
  let promptTokens = null;
  let cachedTokens = null;
  let usageEstimated = false;
  let content = "";
  /** Characters of streamed tool-call JSON, for the token estimate when usage is missing. */
  let toolChars = "";
  let promptChars = 0;
  /** Token-bearing delta events; 1 means everything arrived in a single chunk (no window). */
  let tokenChunks = 0;
  /** @type {Record<number, { name?: string, arguments: string }>} */
  const calls = {};
  let finishReason = null;
  try {
    const response = await fetch(`${apiUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        let chunk;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue;
        }
        const choice = chunk.choices?.[0];
        if (choice?.delta?.content) {
          firstTokenAt ??= now();
          lastTokenAt = now();
          tokenChunks += 1;
          content += choice.delta.content;
          onDelta?.();
        }
        if (choice?.delta?.tool_calls) {
          firstTokenAt ??= now();
          tokenChunks += 1;
          lastTokenAt = now();
          for (const call of choice.delta.tool_calls) {
            const slot = (calls[call.index ?? 0] ??= { name: undefined, arguments: "" });
            if (call.function?.name) { slot.name = call.function.name; toolChars += call.function.name; }
            if (call.function?.arguments) { slot.arguments += call.function.arguments; toolChars += call.function.arguments; }
          }
        }
        if (choice?.finish_reason) finishReason = choice.finish_reason;
        if (chunk.usage) {
          completionTokens = chunk.usage.completion_tokens ?? completionTokens;
          promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
          cachedTokens = chunk.usage.prompt_tokens_details?.cached_tokens ?? chunk.usage.cache_read_input_tokens ?? cachedTokens;
        }
      }
    }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
  // Some backends omit the usage chunk on tool-call streams (and on odd truncations).
  // Fall back to a chars/4 estimate so decode tok/s never silently disappears; the
  // estimate is flagged so the UI can say so.
  if (completionTokens == null) {
    const estimated = estimateTokens(content) + estimateTokens(toolChars);
    if (estimated > 0) {
      completionTokens = estimated;
      usageEstimated = true;
    }
  }
  if (promptTokens == null) {
    for (const message of messages) promptChars += typeof message?.content === "string" ? message.content.length : 0;
    if (promptChars > 0) {
      promptTokens = Math.max(1, Math.round(promptChars / 4));
      usageEstimated = true;
    }
  }
  return {
    started,
    firstTokenAt,
    lastTokenAt,
    tokenChunks,
    content,
    toolCalls: Object.values(calls),
    finishReason,
    completionTokens,
    promptTokens,
    cachedTokens,
    usageEstimated,
    wallMs: now() - started,
  };
}

async function resolveModelId(apiUrl) {
  try {
    const response = await fetch(`${apiUrl}/v1/models`, { signal: AbortSignal.timeout(5000) });
    const json = await response.json();
    const id = json?.data?.[0]?.id;
    if (typeof id === "string" && id) return id;
  } catch {
    /* fall back to the recipe's API model id */
  }
  return "GLM-5.3-Flash-EXL3";
}

const DECODE_PROSE_PROMPT = "Write a detailed story about a lighthouse keeper, the fishing fleet, and a storm. Steady, plain prose, no headings.";
const DECODE_CODE_PROMPT = "Write a Python implementation of an LRU cache with get, set and expiry, well commented. Code only.";

/** Decode suite: a 32-token warmup, then concurrency waves 1 then 2. */
async function runDecode({ apiUrl, model, spec, signal, progress, requestLog }) {
  const maxTokens = Math.min(Math.max(spec.decodeTokens ?? DECODE_DEFAULT_TOKENS, 64), DECODE_MAX_TOKENS);
  const waves = [...(spec.concurrency ?? [1, 2])];
  await chatStream({ apiUrl, model, messages: [{ role: "user", content: DECODE_PROSE_PROMPT }], maxTokens: DECODE_WARMUP_TOKENS, signal });
  progress(`warmup done (${DECODE_WARMUP_TOKENS} tokens)`);
  /** @type {object[]} */
  const waves_out = [];
  for (const concurrency of waves) {
    const prompts = Array.from({ length: concurrency }, (_, index) => (index % 2 === 0 ? DECODE_PROSE_PROMPT : DECODE_CODE_PROMPT));
    const startedAt = now();
    // One failing stream must not take the whole wave down; its row carries the error.
    const results = await Promise.all(prompts.map(async (content) => {
      try {
        return await chatStream({ apiUrl, model, messages: [{ role: "user", content }], maxTokens, signal });
      } catch (error) {
        return { error: error?.message ?? String(error) };
      }
    }));
    for (const result of results) requestLog(result);
    const rows = results.map((result) => {
      const windowMs = result.lastTokenAt && result.firstTokenAt ? result.lastTokenAt - result.firstTokenAt : null;
      const tokens = result.completionTokens != null ? Math.max(0, result.completionTokens - 1) : null;
      const measurable = (result.tokenChunks ?? 2) > 1;
      return {
        ttftMs: result.firstTokenAt ? result.firstTokenAt - result.started : null,
        decodeTokens: tokens,
        windowMs,
        decodeTps: measurable && windowMs > 0 && tokens != null ? round2((tokens / windowMs) * 1000) : null,
        completionTokens: result.completionTokens,
        usageEstimated: Boolean(result.usageEstimated),
        singleChunk: Boolean(result.firstTokenAt) && !measurable,
        error: result.firstTokenAt ? null : (result.error ?? "no tokens streamed"),
      };
    });
    const totalTokens = rows.reduce((sum, row) => sum + (row.decodeTokens ?? 0), 0);
    // Concurrent decode window: first token of the fastest stream → last token of the
    // slowest (the sparkDash aggregateDecodeTps denominator), not the sum of windows.
    const firsts = results.map((result) => result.firstTokenAt).filter(Boolean);
    const lasts = results.map((result) => result.lastTokenAt).filter(Boolean);
    const windowMs = firsts.length && lasts.length ? Math.max(...lasts) - Math.min(...firsts) : null;
    const wall = now() - startedAt;
    const streamTps = rows.map((row) => row.decodeTps).filter((value) => value != null);
    const wave = {
      concurrency,
      maxTokens,
      perRequest: rows,
      /** Σ decode tokens over the concurrent first→last token window. */
      decodeTps: windowMs > 0 ? round2((totalTokens / windowMs) * 1000) : null,
      /** Plain mean of the streams' own tok/s. */
      meanStreamTps: streamTps.length ? round2(streamTps.reduce((a, b) => a + b, 0) / streamTps.length) : null,
      /** Σ tokens over the whole wave wall time (includes TTFT and stragglers). */
      windowDecodeTps: wall > 0 ? round2((totalTokens / wall) * 1000) : null,
      ttft: stats(rows.map((row) => row.ttftMs)),
    };
    waves_out.push(wave);
    progress(`decode c=${concurrency}: ${wave.decodeTps} tok/s`);
  }
  return { waves: waves_out };
}

function fillerPrompt(tokenSize, seed) {
  const words = Math.round(tokenSize * WORDS_PER_TOKEN);
  let body = "";
  while (body.length < words * 8) body += seed % 2 ? CODE_FILLER : FILLER;
  return `Report ID BX-${seed.toString(36)}-${tokenSize}. Summarize the reference text in one short sentence.\n\n${body.slice(0, words * 8 + 200)}`;
}

/** Prefill suite: cold prompts of several sizes; TTFT minus a calibrated overhead. */
async function runPrefill({ apiUrl, model, spec, signal, progress, requestLog }) {
  const sizes = Array.isArray(spec.prefillSizes) && spec.prefillSizes.length ? spec.prefillSizes : PREFILL_SIZES;
  // Fixed overhead = median TTFT of three tiny prompts.
  const overheads = [];
  for (let index = 0; index < 3; index += 1) {
    const result = await chatStream({ apiUrl, model, messages: [{ role: "user", content: fillerPrompt(48, now() + index) }], maxTokens: 8, signal });
    requestLog(result);
    if (result.firstTokenAt) overheads.push(result.firstTokenAt - result.started);
  }
  overheads.sort((a, b) => a - b);
  const overhead = overheads.length ? overheads[Math.floor(overheads.length / 2)] : 0;
  progress(`overhead calibration: ${overhead} ms`);
  /** @type {object[]} */
  const rows = [];
  let seed = now() % 1e6;
  for (const size of sizes) {
    const prompt = fillerPrompt(size, seed += 7);
    const startedAt = now();
    const result = await chatStream({ apiUrl, model, messages: [{ role: "user", content: prompt }], maxTokens: 8, signal });
    requestLog(result);
    const ttftMs = result.firstTokenAt ? result.firstTokenAt - result.started : null;
    const computed = result.promptTokens != null && result.cachedTokens != null ? result.promptTokens - result.cachedTokens : result.promptTokens ?? null;
    const effectiveMs = ttftMs == null ? null : Math.max(ttftMs - overhead, ttftMs * 0.5);
    const estimated = Boolean(result.usageEstimated);
    rows.push({
      requestedTokens: size,
      promptTokens: result.promptTokens,
      cachedTokens: result.cachedTokens,
      ttftMs,
      prefillTps: effectiveMs && computed ? round2((computed / effectiveMs) * 1000) : null,
      lowConfidence: estimated || Boolean(effectiveMs && ttftMs && overhead > ttftMs * 0.3),
      usageEstimated: estimated,
      wallMs: now() - startedAt,
      error: ttftMs == null ? "no first token" : null,
    });
    progress(`prefill ${size}: ${rows[rows.length - 1].prefillTps} tok/s`);
  }
  return { overheadMs: overhead, sizes: rows };
}

const AGENT_SYSTEM = `You are pi, a careful terminal assistant. Use the provided tools to answer. Call exactly one tool per step, then wait for its result. When the task is complete, answer in one short sentence with no tool call.`;
const AGENT_TOOLS = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a file from the workspace.",
      parameters: { type: "object", properties: { path: { type: "string", description: "relative path" } }, required: ["path"] },
    },
  },
  {
    type: "function",
    function: {
      name: "run_cmd",
      description: "Run a shell command in the workspace.",
      parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] },
    },
  },
];
/** Each turn: the tool we expect the model to call (null = expect a plain answer). */
const AGENT_TURNS = [
  { user: "Find the disk usage of the workspace and tell me the largest directory.", expect: "run_cmd" },
  { toolName: "run_cmd", toolOutput: "du: 4.0K ./etc\n2.1G ./models\n18M ./src\n310M ./cache", expect: "read_file" },
  { toolName: "read_file", toolOutput: "# Config\nwindow = 1048576\nslots = 4\nkv = fp8\n", expect: null },
];

async function runAgentConversation({ apiUrl, model, signal, progress, requestLog, label }) {
  const messages = [{ role: "system", content: AGENT_SYSTEM }];
  /** @type {object[]} */
  const turns = [];
  for (let index = 0; index < AGENT_TURNS.length; index += 1) {
    const turn = AGENT_TURNS[index];
    if (turn.user) messages.push({ role: "user", content: turn.user });
    // Scripted history: the previous assistant call and its tool result.
    if (turn.toolName) {
      messages.push({ role: "assistant", tool_calls: [{ id: `call_${index}`, type: "function", function: { name: turn.toolName, arguments: "{}" } }] });
      messages.push({ role: "tool", tool_call_id: `call_${index}`, content: turn.toolOutput });
    }
    const startedAt = now();
    const result = await chatStream({ apiUrl, model, messages, tools: AGENT_TOOLS, maxTokens: 256, signal });
    requestLog(result);
    const call = result.toolCalls[0];
    const called = call?.name ?? null;
    const argsOk = call ? (() => { try { JSON.parse(call.arguments || "{}"); return true; } catch { return false; } })() : false;
    const windowMs = result.lastTokenAt && result.firstTokenAt ? result.lastTokenAt - result.firstTokenAt : null;
    // Same rule as decode: the window starts at the first token, so all but that
    // first token are decode — tool-call or not. TensorFold delivers a whole tool
    // call in one SSE chunk, so a pure tool-call turn has no window to measure;
    // usage does arrive, the turn is just over in one batch.
    const tokens = result.completionTokens != null && result.completionTokens > 0 ? result.completionTokens - 1 : null;
    const measurable = (result.tokenChunks ?? 2) > 1;
    let decodeTps = null;
    let decodeNote = null;
    if (windowMs == null) decodeNote = "no tokens streamed";
    else if (!measurable) decodeNote = `all ${result.completionTokens ?? "?"} tokens arrived in one chunk — no decode window to measure`;
    else if (tokens == null) decodeNote = "no completion tokens reported or streamable";
    else decodeTps = round2((tokens / windowMs) * 1000);
    if (result.usageEstimated && decodeNote == null) decodeNote = "completion tokens estimated from streamed characters (server sent no usage)";
    turns.push({
      index,
      expected: turn.expect,
      called,
      argsOk,
      accurate: turn.expect == null ? called == null : called === turn.expect && argsOk,
      ttftMs: result.firstTokenAt ? result.firstTokenAt - result.started : null,
      decodeTps,
      decodeNote,
      promptTokens: result.promptTokens,
      cachedTokens: result.cachedTokens,
      finishReason: result.finishReason,
    });
    if (turn.expect != null && !called) {
      progress(`${label}: turn ${index} did not call a tool (expected ${turn.expect})`);
      break;
    }
  }
  return turns;
}

/** Agent suite: pi-shaped tool-call loop, single agent then two in parallel. */
async function runAgent({ apiUrl, model, spec, signal, progress, requestLog }) {
  const waves = [...(spec.concurrency ?? [1, 2])];
  /** @type {object[]} */
  const waves_out = [];
  for (const concurrency of waves) {
    const startedAt = now();
    const conversations = await Promise.all(
      Array.from({ length: concurrency }, (_, index) => runAgentConversation({ apiUrl, model, signal, requestLog, label: `c${concurrency}#${index}` })),
    );
    const turns = conversations.flat();
    const totalWindowMs = turns.reduce((sum, turn) => sum + (turn.ttftMs ?? 0), 0);
    const promptTokens = turns.reduce((sum, turn) => sum + (turn.promptTokens ?? 0), 0);
    const cachedTokens = turns.reduce((sum, turn) => sum + (turn.cachedTokens ?? 0), 0);
    const accurate = turns.filter((turn) => turn.accurate).length;
    const turnTps = turns.map((turn) => turn.decodeTps).filter((value) => value != null);
    const wave = {
      concurrency,
      turns,
      toolcallAccuracy: turns.length ? round2((accurate / turns.length) * 100) : null,
      promptCacheHitPercent: promptTokens ? round2((cachedTokens / promptTokens) * 100) : null,
      /** Mean of the turns' own decode tok/s; null when no turn had a measurable window. */
      meanTurnDecodeTps: turnTps.length ? round2(turnTps.reduce((a, b) => a + b, 0) / turnTps.length) : null,
      ttft: stats(turns.map((turn) => turn.ttftMs)),
      wallMs: now() - startedAt,
    };
    waves_out.push(wave);
    progress(`agent c=${concurrency}: accuracy ${wave.toolcallAccuracy}%, cache hit ${wave.promptCacheHitPercent}%`);
  }
  return { waves: waves_out };
}

export function createBench({ apiUrl, dataHome }) {
  const historyPath = path.join(dataHome, "spark-scope", HISTORY_FILE);
  /** @type {null | { id: string, suite: string, state: string, startedAt: number, endedAt: number | null, progress: string[], result: object | null, error: string | null, controller: AbortController }} */
  let active = null;

  async function loadHistory() {
    try {
      const parsed = JSON.parse(await readFile(historyPath, "utf8"));
      return Array.isArray(parsed.runs) ? parsed.runs : [];
    } catch {
      return [];
    }
  }

  async function saveRun(run) {
    const runs = await loadHistory();
    runs.unshift(run);
    const trimmed = JSON.stringify({ runs: runs.slice(0, HISTORY_LIMIT) }, null, 1);
    await writeFile(`${historyPath}.tmp`, trimmed);
    await rename(`${historyPath}.tmp`, historyPath);
  }

  async function run(spec) {
    const suite = spec?.suite;
    if (!SUITES.has(suite)) throw new Error(`unknown suite "${suite}"`);
    const concurrency = spec.concurrency ?? [1, 2];
    if (!Array.isArray(concurrency) || !concurrency.every((count) => ALLOWED_CONCURRENCY.has(count))) {
      throw new Error("concurrency must be a list of 1 and/or 2");
    }
    if (active && active.state === "running") throw Object.assign(new Error("a bench is already running"), { code: "busy" });
    const job = {
      id: randomUUID(),
      suite,
      spec: { ...spec, concurrency },
      state: "running",
      startedAt: now(),
      endedAt: null,
      progress: [],
      result: null,
      error: null,
      controller: new AbortController(),
    };
    active = job;
    void (async () => {
      const requestLog = [];
      const log = (result) => {
        if (result.firstTokenAt) requestLog.push({ ttftMs: result.firstTokenAt - result.started });
      };
      const record = (result) => log(result);
      const progress_ = (message) => {
        job.progress.push(`${new Date().toISOString().slice(11, 19)} ${message}`);
        if (job.progress.length > 40) job.progress.shift();
      };
      try {
        const model = await resolveModelId(apiUrl);
        progress_(`model ${model} @ ${apiUrl}`);
        const runner = suite === "decode" ? runDecode : suite === "prefill" ? runPrefill : runAgent;
        const result = await runner({ apiUrl, model, spec: job.spec, signal: job.controller.signal, progress: progress_, requestLog: record });
        const ttfts = requestLog.map((entry) => entry.ttftMs).filter((value) => value > 0);
        job.result = { model, requests: requestLog.length, benchTtft: stats(ttfts), ...result };
        job.state = "done";
      } catch (error) {
        job.error = error?.message ?? String(error);
        job.state = job.controller.signal.aborted ? "cancelled" : "error";
      } finally {
        job.endedAt = now();
        const record = { ...job, controller: undefined };
        delete record.controller;
        await saveRun(record).catch(() => {});
      }
    })();
    return job;
  }

  return {
    async summary() {
      const running = active?.state === "running" ? { id: active.id, suite: active.suite, startedAt: active.startedAt, progress: active.progress } : null;
      const last = active && active.state !== "running"
        ? { id: active.id, suite: active.suite, state: active.state, startedAt: active.startedAt, endedAt: active.endedAt, progress: active.progress, result: active.result, error: active.error }
        : null;
      const history = await loadHistory();
      return { active: running, last, history: history.slice(0, 10) };
    },
    run,
    cancel() {
      if (active?.state === "running") {
        active.controller.abort(new Error("cancelled"));
        return true;
      }
      return false;
    },
  };
}
