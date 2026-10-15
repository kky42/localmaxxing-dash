// Bench view: run decode / prefill / agent suites and show results.
// Self-contained module; talks to /api/bench only. Load lands on the inference
// server — at most 2 concurrent requests, enforced server-side too. The flow
// follows the sparkDash bench pages: chip configuration with a summary line, a
// live big-number readout while a run is in flight, hero-number results with
// the detail table underneath, and friendly history rows.

const $ = (selector) => document.querySelector(selector);
const el = (id) => document.getElementById(id);

const SUITES = {
  decode: {
    label: 'Decode',
    run: 'Run decode bench',
    note: 'Streaming decode tok/s at rising concurrency. 32-token warmup, temperature 0, thinking off; tok/s = (completion − 1) / first→last token window.',
  },
  prefill: {
    label: 'Prefill',
    run: 'Run prefill bench',
    note: 'Cold prompts of each size: prefill tok/s and TTFT, calibrated against the median TTFT of tiny prompts.',
  },
  agent: {
    label: 'Agent',
    run: 'Run agent bench',
    note: 'A pi-shaped 3-turn tool loop with scripted tool results: tool-call accuracy, prompt-cache hit, decode tok/s and TTFT per turn.',
  },
};
const SUITE_ORDER = ['decode', 'prefill', 'agent'];
const TOKEN_PRESETS = [128, 384, 768, 1536];
const SIZE_PRESETS = [8192, 32768, 65536, 131072];

const state = {
  suite: 'decode',
  config: {
    decode: { concurrency: [1], tokens: 384 },
    prefill: { sizes: [...SIZE_PRESETS] },
    agent: { concurrency: [1] },
  },
  viewId: null, // history run opened by hand; null = follow the latest of the selected suite
  api: null,
  locked: false, // a run is in flight; chips disabled
  starting: false,
  lastJobId: null,
  samples: [],
  peak: null,
  live: null, // { startedAt, steps:{done,total} } for the 1s ticker
};

let pollTimer = null;
let tickTimer = null;

/* ── helpers ─────────────────────────────────────────────────────────── */

async function api(path, body) {
  const response = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(json.error || `HTTP ${response.status}`);
  return json;
}

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function fmt(value) {
  if (value == null) return '—';
  if (typeof value !== 'number') return String(value);
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString('en-US');
  return String(Math.round(value * 100) / 100);
}

const humanSize = (n) => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : String(n));

function fmtDuration(ms) {
  if (ms == null) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rest = s % 60;
  if (m < 60) return `${m}m ${String(rest).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

function relTime(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

function pillClass(stateName) {
  return { done: 'done', error: 'error', cancelled: 'cancelled', running: 'running' }[stateName] ?? 'cancelled';
}

function showError(message) {
  const node = el('bench-error');
  node.textContent = message;
  node.hidden = false;
}

function hideError() {
  const node = el('bench-error');
  node.hidden = true;
  node.textContent = '';
}

/* ── configuration card ──────────────────────────────────────────────── */

const field = (label, hint, body) =>
  `<div class="bench-field"><div class="bench-field-head"><span class="bench-field-label">${label}</span><small>${hint}</small></div>${body}</div>`;
const chipsWrap = (inner) => `<div class="bench-chips">${inner}</div>`;

function chip(label, kind, value, on) {
  return `<button type="button" class="bench-chip" data-kind="${kind}" data-value="${value}" aria-pressed="${on}"${state.locked ? ' disabled' : ''}>${label}</button>`;
}

function renderConfig() {
  const c = state.config[state.suite];
  const locked = state.locked ? ' disabled' : '';
  const parts = [];
  if (state.suite === 'decode') {
    parts.push(field(
      'Concurrency',
      'Levels run sequentially; each opens that many parallel streams. The server caps load at 8.',
      chipsWrap([1, 2, 4, 8].map((n) => chip(String(n), 'conc', n, c.concurrency.includes(n))).join('')),
    ));
    parts.push(field(
      'Tokens per stream',
      'Length of each generation — pick a preset or type your own.',
      chipsWrap(TOKEN_PRESETS.map((t) => chip(String(t), 'tokens', t, c.tokens === t)).join('')) +
      `<div class="bench-custom"><input id="bench-tokens-custom" type="number" inputmode="numeric" min="64" max="2048" step="16" value="${c.tokens}"${locked} aria-label="Custom tokens per stream"><span>tokens</span></div>`,
    ));
  } else if (state.suite === 'prefill') {
    parts.push(field(
      'Context sizes',
      'One cold prompt per size. Tap to include or drop a size.',
      chipsWrap(SIZE_PRESETS.map((s) => chip(humanSize(s), 'size', s, c.sizes.includes(s))).join('')) +
      `<div class="bench-custom"><input id="bench-size-custom" type="number" inputmode="numeric" min="64" placeholder="custom"${locked} aria-label="Custom context size in tokens"><span>tokens — press Enter to add</span></div>`,
    ));
  } else {
    parts.push(field(
      'Concurrency',
      'One agent, then two in parallel. Each agent plays the same 3-turn scripted loop.',
      chipsWrap([1, 2, 4, 8].map((n) => chip(String(n), 'conc', n, c.concurrency.includes(n))).join('')),
    ));
  }
  el('bench-suite-panel').innerHTML = parts.join('');
  el('bench-suite-note').textContent = SUITES[state.suite].note;
  updateSummary();
  wireConfig();
}

function updateSummary() {
  const c = state.config[state.suite];
  let text;
  if (state.suite === 'decode') text = `Decode · concurrency ${c.concurrency.join(' + ')} · ${c.tokens} tokens/stream`;
  else if (state.suite === 'prefill') text = `Prefill · ${c.sizes.map(humanSize).join(', ')} tokens`;
  else text = `Agent · ${c.concurrency.join(' + ')} concurrent · 3 scripted turns`;
  el('bench-summary').textContent = text;
}

function wireConfig() {
  const panel = el('bench-suite-panel');
  panel.querySelectorAll('.bench-chip').forEach((button) => {
    button.addEventListener('click', () => {
      const kind = button.dataset.kind;
      const value = Number(button.dataset.value);
      const c = state.config[state.suite];
      if (kind === 'conc') {
        const has = c.concurrency.includes(value);
        if (has && c.concurrency.length === 1) return; // keep at least one level
        c.concurrency = has ? c.concurrency.filter((n) => n !== value) : [...c.concurrency, value].sort((a, b) => a - b);
      } else if (kind === 'tokens') {
        c.tokens = value;
      } else if (kind === 'size') {
        const has = c.sizes.includes(value);
        if (has && c.sizes.length === 1) return;
        c.sizes = has ? c.sizes.filter((n) => n !== value) : [...c.sizes, value].sort((a, b) => a - b);
      }
      renderConfig();
    });
  });
  const tokensInput = el('bench-tokens-custom');
  if (tokensInput) {
    tokensInput.addEventListener('change', () => {
      const value = Number(tokensInput.value);
      if (Number.isFinite(value) && value >= 64) {
        state.config.decode.tokens = Math.min(Math.round(value), 2048);
        renderConfig();
      }
    });
  }
  const sizeInput = el('bench-size-custom');
  if (sizeInput) {
    sizeInput.addEventListener('change', () => {
      const value = Number(sizeInput.value);
      if (Number.isFinite(value) && value >= 64) {
        const rounded = Math.round(value);
        if (!state.config.prefill.sizes.includes(rounded)) state.config.prefill.sizes.push(rounded);
        state.config.prefill.sizes.sort((a, b) => a - b);
        renderConfig();
      }
    });
  }
}

function syncTabs() {
  document.querySelectorAll('#bench-suite-tabs button').forEach((button) => {
    button.setAttribute('aria-selected', String(button.dataset.suite === state.suite));
  });
}

/* ── live view (a run is in flight) ──────────────────────────────────── */

function parseLiveTps(progress) {
  for (let i = progress.length - 1; i >= 0; i -= 1) {
    const match = progress[i].match(/([\d.]+)\s*tok\/s/);
    if (match) return Number(match[1]);
  }
  return null;
}

function progressSteps(job) {
  const lines = job.progress ?? [];
  const waves = (job.spec?.concurrency ?? [1, 2]).length;
  if (job.suite === 'decode') {
    return { done: lines.filter((l) => /warmup done|decode c=/.test(l)).length, total: 1 + waves };
  }
  if (job.suite === 'prefill') {
    return { done: lines.filter((l) => /overhead calibration|prefill \d/.test(l)).length, total: 1 + (job.spec?.prefillSizes?.length || 3) };
  }
  return { done: lines.filter((l) => /agent c=/.test(l)).length, total: waves };
}

function sparkline(samples) {
  if (samples.length < 2) return '<svg class="bench-spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true"></svg>';
  const max = Math.max(...samples) * 1.08;
  const points = samples.map((v, i) => `${((i / (samples.length - 1)) * 100).toFixed(2)},${(35 - (v / max) * 33).toFixed(2)}`).join(' ');
  return `<svg class="bench-spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true"><polyline points="${points}"/></svg>`;
}

function renderLive(job) {
  if (job.id !== state.lastJobId) {
    state.lastJobId = job.id;
    state.samples = [];
    state.peak = null;
  }
  const progress = job.progress ?? [];
  const tps = parseLiveTps(progress);
  if (tps != null) {
    state.samples.push(tps);
    if (state.samples.length > 60) state.samples.shift();
    state.peak = state.peak == null ? tps : Math.max(state.peak, tps);
  }
  const steps = progressSteps(job);
  state.live = { startedAt: job.startedAt, steps };
  const pct = steps.total ? Math.min(100, Math.round((steps.done / steps.total) * 100)) : 0;
  const eta = steps.done > 0 ? fmtDuration(((Date.now() - job.startedAt) * (steps.total - steps.done)) / steps.done) : null;
  const readout = tps != null
    ? `<div class="bench-live-num num">${fmt(tps)}<small>tok/s</small></div>
       <div class="bench-live-side"><span class="bench-peak">peak ${fmt(state.peak)} tok/s</span>${sparkline(state.samples)}</div>`
    : `<div class="bench-live-num num">…</div><div class="bench-live-side"><span class="bench-peak">warming up</span></div>`;
  el('bench-live').innerHTML = `
    <div class="bench-live-head">
      <span class="bench-live-label"><i class="bench-dot"></i>${esc(SUITES[job.suite]?.label ?? job.suite)} bench running</span>
      <span class="bench-live-meta"><b data-elapsed>${fmtDuration(Date.now() - job.startedAt)}</b> elapsed${eta ? ` · <span data-eta>~${eta} left</span>` : ''}</span>
    </div>
    <div class="bench-live-body">${readout}</div>
    <div class="bench-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" aria-label="Bench progress"><i style="width:${pct}%"></i></div>
    <div class="bench-progress-row"><span>step ${steps.done}/${steps.total}</span><span>${esc(progress.slice(-1)[0] ?? 'starting…')}</span></div>
    <pre class="bench-live-log">${esc(progress.slice(-5).join('\n'))}</pre>
    <button type="button" class="plain bench-cancel" id="bench-cancel">Cancel run</button>`;
  el('bench-cancel').addEventListener('click', () => void cancelRun());
}

function startTicker() {
  clearInterval(tickTimer);
  tickTimer = setInterval(() => {
    if (!state.live || el('bench-live').hidden) return;
    const elapsed = Date.now() - state.live.startedAt;
    document.querySelectorAll('[data-elapsed]').forEach((node) => { node.textContent = fmtDuration(elapsed); });
    const etaNode = document.querySelector('[data-eta]');
    if (etaNode) {
      const { done, total } = state.live.steps;
      if (done > 0) etaNode.textContent = `~${fmtDuration((elapsed * (total - done)) / done)} left`;
    }
  }, 1000);
}

/* ── results ─────────────────────────────────────────────────────────── */

const table = (rows) => `<table class="model-table bench-table"><tbody>${rows.join('')}</tbody></table>`;
const row = (label, value, note) => `<tr><td>${label}</td><td><b>${value}</b></td><td>${note}</td></tr>`;

function resultHead(run) {
  const result = run.result ?? {};
  const duration = run.endedAt && run.startedAt ? fmtDuration(run.endedAt - run.startedAt) : null;
  const meta = [result.model, duration, result.requests ? `${result.requests} requests` : null].filter(Boolean).map(esc).join(' · ');
  return `<div class="bench-result-head"><span class="bench-pill ${pillClass(run.state)}">${esc(run.state)}</span><span class="bench-result-meta">${meta}</span></div>` +
    (run.error && run.state !== 'done' ? `<p class="bench-inline-error">${esc(run.error)}</p>` : '');
}

function hero({ badge, label, value, unit, tier = '', bar = 0, facts = '' }) {
  return `<div class="bench-hero">
    <div class="bench-hero-top">${badge ? `<span class="bench-wave">${badge}</span>` : ''}<span class="bench-hero-label">${label}</span></div>
    <span class="bench-hero-value num ${tier}">${value}${unit ? `<small>${unit}</small>` : ''}</span>
    ${bar ? `<div class="bench-hero-bar"><i style="width:${bar}%"></i></div>` : ''}
    ${facts ? `<div class="bench-hero-facts">${facts}</div>` : ''}
  </div>`;
}

function decodeHeroes(result) {
  const waves = result.waves ?? [];
  const max = Math.max(0, ...waves.map((w) => w.decodeTps ?? 0));
  return waves.map((w) => hero({
    badge: `×${w.concurrency}`,
    label: 'Aggregate decode',
    value: fmt(w.decodeTps),
    unit: 'tok/s',
    bar: max > 0 && w.decodeTps != null ? Math.max(2, Math.round((w.decodeTps / max) * 100)) : 0,
    facts: `stream ${fmt(w.meanStreamTps)} tok/s · TTFT ${fmt(w.ttft?.median)}/${fmt(w.ttft?.p95)} ms`,
  })).join('');
}

function prefillHeroes(result) {
  const sizes = result.sizes ?? [];
  const max = Math.max(0, ...sizes.map((s) => s.prefillTps ?? 0));
  return sizes.map((s) => hero({
    badge: humanSize(s.requestedTokens),
    label: 'Prefill tok/s',
    value: fmt(s.prefillTps),
    unit: 'tok/s',
    bar: max > 0 && s.prefillTps != null ? Math.max(2, Math.round((s.prefillTps / max) * 100)) : 0,
    facts: `TTFT ${fmt(s.ttftMs)} ms${s.cachedTokens ? ` · cached ${fmt(s.cachedTokens)}` : ''}`,
  })).join('');
}

function agentHeroes(waves) {
  return waves.map((w) => {
    const acc = w.toolcallAccuracy;
    const tier = acc == null ? '' : acc >= 90 ? 'is-good' : acc >= 70 ? 'is-warn' : 'is-bad';
    return hero({
      badge: `×${w.concurrency}`,
      label: 'Tool-call accuracy',
      value: fmt(acc),
      unit: '%',
      tier,
      facts: `<span>cache hit ${fmt(w.promptCacheHitPercent)}%</span><span>decode ${fmt(w.meanTurnDecodeTps)} tok/s</span><span>TTFT ${fmt(w.ttft?.median)} ms</span>`,
    });
  }).join('');
}

function decodeTable(result) {
  const rows = [];
  for (const w of result.waves ?? []) {
    rows.push(row(`Decode ×${w.concurrency}`, `${fmt(w.decodeTps)} tok/s`, 'Σ decode tokens / concurrent first→last token window'));
    rows.push(row(`· per stream ×${w.concurrency}`, `${fmt(w.meanStreamTps)} tok/s`, "mean of each stream's own rate"));
    rows.push(row(`· window ×${w.concurrency}`, `${fmt(w.windowDecodeTps)} tok/s`, 'Σ tokens / whole wave wall time (includes TTFT)'));
    (w.perRequest ?? []).forEach((r, i) => {
      if (r.error) rows.push(row(`· stream ${i + 1} ×${w.concurrency}`, 'failed', esc(r.error)));
      else rows.push(row(`· stream ${i + 1} ×${w.concurrency}`, `${fmt(r.decodeTps)} tok/s`, `${fmt(r.decodeTokens)} tokens${r.singleChunk ? ' · single chunk' : ''}${r.usageEstimated ? ' · usage estimated' : ''}`));
    });
  }
  if (result.benchTtft) rows.push(row('Bench TTFT', `${fmt(result.benchTtft.median)} / ${fmt(result.benchTtft.p95)} ms`, `median / p95 over ${result.benchTtft.n} bench requests (not the home page's live p95)`));
  return rows.length ? table(rows) : '';
}

function prefillTable(result) {
  const rows = [row('Overhead', `${fmt(result.overheadMs)} ms`, 'median TTFT of tiny prompts, subtracted from each TTFT')];
  for (const s of result.sizes ?? []) {
    const notes = [s.error ?? `TTFT ${fmt(s.ttftMs)} ms`];
    if (s.cachedTokens != null) notes.push(`cached ${fmt(s.cachedTokens)}`);
    if (s.lowConfidence) notes.push('low confidence');
    if (s.usageEstimated) notes.push('tokens estimated');
    rows.push(row(`Prompt ${humanSize(s.requestedTokens)}`, `${fmt(s.prefillTps)} tok/s`, esc(notes.join(' · '))));
  }
  if (result.benchTtft) rows.push(row('Bench TTFT', `${fmt(result.benchTtft.median)} / ${fmt(result.benchTtft.p95)} ms`, `median / p95 over ${result.benchTtft.n} bench requests`));
  return table(rows);
}

function agentTable(waves) {
  const rows = [];
  for (const w of waves ?? []) {
    for (const t of w.turns ?? []) {
      const called = t.called ? `${t.called} ${t.argsOk ? '✓' : '✗ bad args'}` : 'answer';
      const note = [t.accurate ? 'as expected' : `expected ${t.expected ?? 'a plain answer'}`, t.decodeNote].filter(Boolean).join('; ');
      rows.push(row(`×${w.concurrency} turn ${t.index + 1}`, `${esc(called)} · ${fmt(t.decodeTps)} tok/s`, esc(note)));
    }
  }
  return rows.length ? table(rows) : '';
}

const LEGENDS = {
  decode: '<strong>Aggregate</strong> — Σ decode tokens over the concurrent first→last token window. <strong>Stream</strong> — per-stream mean. Measured from the bench\'s own requests only.',
  prefill: '<strong>Prefill tok/s</strong> excludes the calibrated tiny-prompt overhead; <strong>TTFT</strong> is measured from the bench\'s own requests only.',
  agent: '<strong>Accuracy</strong> — turns that called the expected tool (or produced a plain answer) with valid JSON arguments. <strong>Cache hit</strong> — cached / prompt tokens across turns.',
};

function renderResult(run) {
  const result = run.result ?? {};
  let body = resultHead(run);
  if (result) {
    if (run.suite === 'decode') body += `<div class="bench-heroes">${decodeHeroes(result)}</div>${decodeTable(result)}`;
    else if (run.suite === 'prefill') body += `<div class="bench-heroes">${prefillHeroes(result)}</div>${prefillTable(result)}`;
    else body += `<div class="bench-heroes">${agentHeroes(result.waves ?? [])}</div>${agentTable(result.waves ?? [])}`;
  }
  if (LEGENDS[run.suite]) body += `<p class="bench-legend">${LEGENDS[run.suite]}</p>`;
  el('bench-result').innerHTML = body;
}

/* ── history ─────────────────────────────────────────────────────────── */

function headline(run) {
  const r = run.result;
  if (!r) return null;
  const best = (list, get) => list.reduce((a, x) => ((get(x) ?? -1) > (get(a) ?? -1) ? x : a), null);
  if (run.suite === 'decode') {
    const w = best(r.waves ?? [], (x) => x.decodeTps);
    return w?.decodeTps != null ? `${fmt(w.decodeTps)} tok/s` : null;
  }
  if (run.suite === 'prefill') {
    const s = best((r.sizes ?? []).filter((x) => x.prefillTps != null), (x) => x.prefillTps);
    return s ? `${fmt(s.prefillTps)} tok/s` : null;
  }
  if (run.suite === 'agent') {
    const w = (r.waves ?? [])[0];
    return w ? `${fmt(w.toolcallAccuracy)}% acc` : null;
  }
  return null;
}

function renderHistory(history) {
  const runs = history ?? [];
  el('bench-history-box').hidden = runs.length === 0;
  el('bench-history-count').textContent = runs.length ? `${runs.length} run${runs.length === 1 ? '' : 's'}` : '';
  const list = el('bench-history');
  list.textContent = '';
  for (const run of runs) {
    const item = document.createElement('div');
    item.className = 'bench-runrow' + (run.id === state.viewId ? ' is-viewing' : '');
    item.innerHTML = `
      <button type="button" class="bench-runrow-main">
        <span class="bench-runrow-time">${relTime(run.startedAt)}</span>
        <span class="bench-runrow-suite"><i class="bench-suitedot ${esc(run.suite)}"></i>${esc(SUITES[run.suite]?.label ?? run.suite)}</span>
        <span class="bench-pill ${pillClass(run.state)}">${esc(run.state)}</span>
        <span class="bench-runrow-headline">${esc(headline(run) ?? '')}</span>
      </button>
      ${run.error && run.state !== 'done' ? `<div class="bench-runrow-error">${esc(run.error)}</div>` : ''}`;
    item.querySelector('.bench-runrow-main').addEventListener('click', () => {
      state.viewId = run.id;
      if (run.suite !== state.suite) {
        state.suite = run.suite;
        syncTabs();
        renderConfig();
      }
      renderMain();
    });
    list.append(item);
  }
}

/* ── orchestration ───────────────────────────────────────────────────── */

function pickRun() {
  const data = state.api;
  if (!data) return null;
  if (state.viewId) {
    const found = (data.history ?? []).find((r) => r.id === state.viewId);
    if (found) return found;
  }
  const last = data.last && data.last.suite === state.suite && data.last.state !== 'running' ? data.last : null;
  return last ?? (data.history ?? []).find((r) => r.suite === state.suite) ?? null;
}

function renderMain() {
  const active = state.api?.active ?? null;
  const liveBox = el('bench-live');
  const resultBox = el('bench-result');
  const emptyBox = el('bench-empty');
  if (active) {
    renderLive(active);
    liveBox.hidden = false;
    resultBox.hidden = true;
    emptyBox.hidden = true;
    renderHistory(state.api?.history);
    return;
  }
  liveBox.hidden = true;
  state.live = null;
  const run = pickRun();
  if (run) {
    renderResult(run);
    resultBox.hidden = false;
    emptyBox.hidden = true;
  } else {
    resultBox.hidden = true;
    emptyBox.hidden = false;
  }
  renderHistory(state.api?.history);
}

function updateRunButton() {
  const button = el('bench-run');
  const running = Boolean(state.api?.active);
  button.disabled = running || state.starting;
  button.textContent = running ? 'Running…' : state.starting ? 'Starting…' : SUITES[state.suite].run;
}

async function refresh() {
  let data;
  try {
    data = await api('/api/bench');
  } catch (error) {
    showError(`Bench unavailable: ${error.message}`);
    return;
  }
  state.api = data;
  hideError();
  const running = Boolean(data.active);
  if (running !== state.locked) {
    state.locked = running;
    renderConfig();
  }
  updateRunButton();
  renderMain();
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (document.getElementById('bench').hidden) return;
    void refresh();
  }, 2000);
}

async function startRun() {
  if (state.locked || state.starting) return;
  const spec = { suite: state.suite };
  const c = state.config[state.suite];
  if (state.suite !== 'prefill') spec.concurrency = [...c.concurrency];
  if (state.suite === 'decode') spec.decodeTokens = c.tokens;
  if (state.suite === 'prefill') spec.prefillSizes = [...c.sizes];
  state.starting = true;
  updateRunButton();
  try {
    await api('/api/bench/run', spec);
    state.viewId = null;
  } catch (error) {
    showError(`Bench could not start: ${error.message}`);
  }
  state.starting = false;
  await refresh();
  startPolling();
}

async function cancelRun() {
  try {
    await api('/api/bench/cancel', {});
  } catch {
    /* already idle */
  }
  await refresh();
}

/* ── wire up ─────────────────────────────────────────────────────────── */

document.querySelectorAll('#bench-suite-tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    if (button.dataset.suite === state.suite) return;
    state.suite = button.dataset.suite;
    state.viewId = null; // following the latest of the newly picked suite
    syncTabs();
    renderConfig();
    renderMain();
  });
});
el('bench-run').addEventListener('click', () => void startRun());

renderConfig();
startTicker();
void refresh();
startPolling();
