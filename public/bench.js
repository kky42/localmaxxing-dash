// Bench view: run decode / prefill / agent suites and show results.
// Self-contained module; talks to /api/bench only. Load lands on the inference
// server — at most 2 concurrent requests, enforced server-side too.
const $ = (selector) => document.querySelector(selector);
const fmt = (value, unit = "") => (value == null ? "—" : `${typeof value === "number" ? Math.round(value * 100) / 100 : value}${unit}`);
let pollTimer = null;
let lastState = null;

async function api(path, body) {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(json.error || `HTTP ${response.status}`);
  return json;
}

function el(id) {
  return document.getElementById(id);
}

function renderJob(job) {
  const box = el("bench-active");
  if (!job) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  el("bench-active-title").textContent = `${job.suite} bench running (started ${new Date(job.startedAt).toLocaleTimeString()})`;
  el("bench-active-progress").textContent = (job.progress ?? []).slice(-6).join("\n") || "starting…";
}

function renderResult(run) {
  const wrap = el("bench-last");
  if (!run) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  el("bench-last-title").textContent = `${run.suite} — ${run.state}${run.error ? `: ${run.error}` : ""}`;
  const body = el("bench-last-body");
  body.textContent = "";
  const table = document.createElement("table");
  table.className = "model-table bench-table";
  const rows = [];
  const addRow = (label, value, note = "") => rows.push(`<tr><td>${label}</td><td><b>${value}</b></td><td class="bench-note">${note}</td></tr>`);
  const result = run.result ?? {};
  if (run.suite === "decode") {
    for (const wave of result.waves ?? []) {
      addRow(`Decode c=${wave.concurrency}`, `${fmt(wave.decodeTps)} tok/s`, "Σ decode tokens / concurrent first→last token window (bench requests only)");
      addRow(`· per-stream c=${wave.concurrency}`, `${fmt(wave.meanStreamTps)} tok/s`, "mean of each stream's own (completion − 1) / window");
      addRow(`· wall c=${wave.concurrency}`, `${fmt(wave.windowDecodeTps)} tok/s`, "Σ tokens / whole wave wall time (includes TTFT)");
      addRow(`· TTFT c=${wave.concurrency}`, `${fmt(wave.ttft?.median)} / ${fmt(wave.ttft?.p95)} ms`, "median / p95, bench requests only");
    }
  } else if (run.suite === "prefill") {
    addRow("Fixed overhead", `${fmt(result.overheadMs)} ms`, "median TTFT of tiny prompts");
    for (const row of result.sizes ?? []) {
      const notes = [];
      if (row.lowConfidence) notes.push("low confidence");
      if (row.usageEstimated) notes.push("prompt tokens estimated — the server sent no usage");
      notes.push(`TTFT ${fmt(row.ttftMs)} ms`);
      if (row.cachedTokens) notes.push(`cached ${row.cachedTokens}`);
      addRow(`Prefill ${row.promptTokens ?? row.requestedTokens} tok`, `${fmt(row.prefillTps)} tok/s`, notes.join(", ") || "cold prompt");
    }
  } else if (run.suite === "agent") {
    for (const wave of result.waves ?? []) {
      addRow(`Agent c=${wave.concurrency}`, `${fmt(wave.toolcallAccuracy)}% accurate`, `${wave.turns?.length ?? 0} turns, pi-shaped tool loop`);
      addRow(`· Cache hit c=${wave.concurrency}`, `${fmt(wave.promptCacheHitPercent)}%`, "cached / prompt tokens across turns");
      addRow(`· Decode c=${wave.concurrency}`, `${fmt(wave.meanTurnDecodeTps)} tok/s`, "mean of measurable turns; a turn whose tokens arrive in one chunk has no window and shows “—”");
      addRow(`· TTFT c=${wave.concurrency}`, `${fmt(wave.ttft?.median)} / ${fmt(wave.ttft?.p95)} ms`, "median / p95 across turns");
      for (const turn of wave.turns ?? []) {
        const note = turn.decodeNote ?? "";
        addRow(`· turn ${turn.index + 1}`, `${turn.called ?? "answer"}${turn.called ? (turn.argsOk ? " ✓" : " ✗ bad args") : ""} — ${fmt(turn.decodeTps)} tok/s`, `${turn.accurate ? "as expected" : `expected ${turn.expected ?? "a plain answer"}`}${note ? `; ${note}` : ""}`);
      }
    }
  }
  if (result.benchTtft) addRow("Bench TTFT", `${fmt(result.benchTtft.median)} / ${fmt(result.benchTtft.p95)} ms`, `median / p95 over ${result.benchTtft.n} bench requests (not the home page's live p95)`);
  table.innerHTML = `<thead><tr><th>Measurement</th><th>Value</th><th></th></tr></thead><tbody>${rows.join("")}</tbody>`;
  body.append(table);
  if (run.error && (run.state === "error" || run.state === "cancelled")) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = `The run ended ${run.state}: ${run.error}`;
    body.append(note);
  }
}

function renderHistory(history) {
  const list = el("bench-history");
  list.textContent = "";
  for (const run of history ?? []) {
    const when = new Date(run.startedAt).toLocaleString();
    const item = document.createElement("tr");
    const error = run.error ? String(run.error) : "";
    item.innerHTML = `<td>${when}</td><td>${run.suite}</td><td${error ? ' class="bench-fail"' : ""}>${run.state}</td><td>${run.result?.model ?? ""}</td><td class="bench-note" title="${error.replace(/"/g, "&quot;")}">${error}</td>`;
    list.append(item);
  }
}

async function refresh() {
  try {
    const state = await api("/api/bench");
    lastState = state;
    renderJob(state.active);
    renderResult(state.last ?? (state.history?.[0] ? state.history[0] : null));
    renderHistory(state.history);
    document.querySelectorAll(".bench-run").forEach((button) => {
      button.disabled = Boolean(state.active);
      button.textContent = state.active ? "Running…" : button.dataset.label;
    });
    el("bench-cancel").disabled = !state.active;
    if (state.active && !pollTimer) startPolling(2000);
  } catch (error) {
    el("bench-active").hidden = false;
    el("bench-active-title").textContent = `Bench unavailable: ${error.message}`;
  }
}

function startPolling(interval = 2000) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (document.getElementById("bench").hidden) return;
    void refresh();
  }, interval);
}

async function startRun(suite) {
  const spec = { suite, concurrency: [1, 2] };
  if (suite === "decode") spec.decodeTokens = Number(el("bench-decode-tokens").value) || 384;
  if (suite === "prefill") spec.prefillSizes = (el("bench-prefill-sizes").value || "8192,32768,65536").split(",").map((value) => Number(value.trim())).filter(Number.isFinite);
  if (!confirm(`Run the ${suite} bench against the model server?\nLoad is capped at 2 concurrent requests. Other agents may be running.`)) return;
  try {
    await api("/api/bench/run", spec);
  } catch (error) {
    alert(`Bench could not start: ${error.message}`);
  }
  await refresh();
  startPolling(2000);
}

function wire() {
  document.querySelectorAll(".bench-run").forEach((button) => {
    button.dataset.label = button.textContent;
    button.addEventListener("click", () => void startRun(button.dataset.suite));
  });
  el("bench-cancel").addEventListener("click", async () => {
    try {
      await api("/api/bench/cancel", {});
    } catch {
      /* already idle */
    }
    await refresh();
  });
}

wire();
void refresh();
if (!pollTimer) startPolling(2000);
