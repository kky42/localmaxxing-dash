// Bench view: run decode / prefill / agent suites and show results.
// Self-contained module; talks to /api/bench only. Load lands on the inference
// server — at most 8 concurrent requests, enforced server-side too. The flow
// follows the sparkDash bench pages: chip configuration with a summary line, a
// live big-number readout while a run is in flight, hero-number results with
// the detail table underneath, and friendly history rows.

const $ = (selector) => document.querySelector(selector);
const el = (id) => document.getElementById(id);

const SUITES = {
  decode: {
    label: 'Decode',
    run: 'Run decode bench',
    note: 'Streaming generation speed at rising concurrency, prose and/or code. Every stream runs the full length; temperature 0, thinking off, 32-token warmup.',
  },
  prefill: {
    label: 'Prefill',
    run: 'Run prefill bench',
    note: 'Cold prompts of each size: prefill tok/s and TTFT, calibrated against the median TTFT of tiny prompts.',
  },
  agent: {
    label: 'Agent',
    run: 'Run agent bench',
    note: 'A pi-shaped tool loop: read a config, run du on the directory it names, answer with the size. Scores each step; the model\'s own correct calls carry forward.',
  },
};
const SUITE_ORDER = ['decode', 'prefill', 'agent'];
const DECODE_TYPES = ['prose', 'code'];
const TYPE_LABEL = { prose: 'Prose', code: 'Code' };
const TOKEN_PRESETS = [128, 384, 768, 1536];
const SIZE_PRESETS = [8192, 32768, 65536, 131072];

const state = {
  suite: 'decode',
  config: {
    decode: { concurrency: [1], tokens: 384, types: ['prose', 'code'] },
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
      'Type',
      'Prose: the lab hash-map explanation. Code: a different Python task per stream. Pick one or both.',
      chipsWrap(DECODE_TYPES.map((t) => chip(TYPE_LABEL[t], 'type', t, c.types.includes(t))).join('')),
    ));
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
      'Levels run in turn; each runs that many agents in parallel, all on the same 3-step task.',
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
  if (state.suite === 'decode') text = `Decode · ${c.types.map((t) => TYPE_LABEL[t].toLowerCase()).join(' + ')} · concurrency ${c.concurrency.join(' + ')} · ${c.tokens} tokens/stream`;
  else if (state.suite === 'prefill') text = `Prefill · ${c.sizes.map(humanSize).join(', ')} tokens`;
  else text = `Agent · ${c.concurrency.join(' + ')} concurrent · 3-step task`;
  el('bench-summary').textContent = text;
}

function wireConfig() {
  const panel = el('bench-suite-panel');
  panel.querySelectorAll('.bench-chip').forEach((button) => {
    button.addEventListener('click', () => {
      const kind = button.dataset.kind;
      const value = Number(button.dataset.value);
      const c = state.config[state.suite];
      if (kind === 'type') {
        const type = button.dataset.value;
        const has = c.types.includes(type);
        if (has && c.types.length === 1) return; // keep at least one type
        c.types = DECODE_TYPES.filter((t) => (t === type ? !has : c.types.includes(t)));
      } else if (kind === 'conc') {
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
  // The server sends the run's spec with the live job; fall back to this page's config for an older server.
  const spec = job.spec ?? { ...state.config[job.suite], prefillSizes: state.config.prefill.sizes, decodeTypes: state.config.decode.types };
  const lines = job.progress ?? [];
  const waves = (spec.concurrency ?? [1]).length;
  if (job.suite === 'decode') {
    const types = (spec.decodeTypes ?? DECODE_TYPES).length;
    return { done: lines.filter((l) => /warmup done|decode \w+ c=/.test(l)).length, total: 1 + waves * types };
  }
  if (job.suite === 'prefill') {
    return { done: lines.filter((l) => /overhead calibration|prefill \d/.test(l)).length, total: 1 + (spec.prefillSizes?.length || SIZE_PRESETS.length) };
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


function resultHead(run) {
  const result = run.result ?? {};
  const duration = run.endedAt && run.startedAt ? fmtDuration(run.endedAt - run.startedAt) : null;
  const meta = [result.model, duration, result.requests ? `${result.requests} requests` : null].filter(Boolean).map(esc).join(' · ');
  return `<div class="bench-result-head"><span class="bench-pill ${pillClass(run.state)}">${esc(run.state)}</span><span class="bench-result-meta">${meta}</span></div>` +
    (run.error && run.state !== 'done' ? `<p class="bench-inline-error">${esc(run.error)}</p>` : '');
}

function hero({ badge, label, value, unit, tier = '', facts = '' }) {
  return `<div class="bench-hero">
    <div class="bench-hero-top">${badge ? `<span class="bench-wave">${badge}</span>` : ''}<span class="bench-hero-label">${label}</span></div>
    <span class="bench-hero-value num ${tier}">${value}${unit ? `<small>${unit}</small>` : ''}</span>
    ${facts ? `<div class="bench-hero-facts">${facts}</div>` : ''}
  </div>`;
}

/** A compact grid: header row, then one row per entry. Cells are pre-escaped HTML; columns in textCols stay left-aligned. */
function grid(head, rows, caption = '', textCols = []) {
  const cls = (i) => (textCols.includes(i) ? ' class="txt"' : '');
  return `${caption ? `<h4 class="bench-grid-title">${caption}</h4>` : ''}<table class="model-table bench-table bench-matrix"><thead><tr>${head.map((h, i) => `<th${cls(i)}>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td${cls(i)}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

/** tok/s with one decimal (thousands without). */
const tps = (value) => (value == null ? '—' : value >= 1000 ? fmt(value) : (Math.round(value * 10) / 10).toFixed(1));
const ms = (value) => (value == null ? '—' : `${fmt(Math.round(value))} ms`);

/** Decode: one small table per type — aggregate, per stream, TTFT for each concurrency level. */
function decodeView(result) {
  const waves = result.waves ?? [];
  // Older runs mixed prose/code streams in one wave with no type.
  const types = [...new Set(waves.map((w) => w.type ?? 'mixed'))];
  return types.map((type) => {
    const rows = waves.filter((w) => (w.type ?? 'mixed') === type).map((w) => [
      `×${w.concurrency}`,
      `<b>${tps(w.decodeTps)}</b>`,
      tps(w.meanStreamTps),
      ms(w.ttft?.median),
      w.failed ? `<span class="bench-bad">${w.failed} failed</span>` : '',
    ]);
    return grid(['Concurrency', 'Aggregate tok/s', 'Per stream tok/s', 'TTFT', ''], rows, TYPE_LABEL[type] ?? 'Prose + code (mixed)');
  }).join('');
}

/** Prefill: size, prompt tokens, speed, TTFT. */
function prefillView(result) {
  const rows = (result.sizes ?? []).map((s) => [
    humanSize(s.requestedTokens),
    fmt(s.promptTokens),
    s.error ? `<span class="bench-bad">${esc(s.error)}</span>` : `<b>${tps(s.prefillTps)}</b>${s.lowConfidence ? ' <small title="TTFT is close to the fixed overhead, or tokens were estimated">≈</small>' : ''}`,
    ms(s.ttftMs),
    s.cachedTokens ? `${fmt(s.cachedTokens)} cached` : '',
  ]);
  return grid(['Prompt', 'Tokens', 'Prefill tok/s', 'TTFT', ''], rows);
}

/** Agent: a hero per level, then one row per step with who missed and why. */
function agentView(result) {
  const waves = result.waves ?? [];
  const heroes = waves.map((w) => {
    const acc = w.toolcallAccuracy;
    const tier = acc == null ? '' : acc >= 90 ? 'is-good' : acc >= 70 ? 'is-warn' : 'is-bad';
    const agents = w.concurrency;
    const done = w.tasksCompleted;
    return hero({
      badge: `×${w.concurrency}`,
      label: 'Step accuracy',
      value: fmt(acc),
      unit: '%',
      tier,
      facts: `<span>${done != null ? `${done}/${agents} task${agents === 1 ? '' : 's'} done` : ''}</span><span>cache hit ${fmt(w.promptCacheHitPercent)}%</span><span>TTFT ${ms(w.ttft?.median)}</span>`,
    });
  }).join('');
  const tables = waves.map((w) => {
    if (!w.steps) {
      // Runs from before the 3-step task: one row per turn.
      return grid(['Turn', 'Called', 'Result'], (w.turns ?? []).map((t) => [`${t.index + 1}`, esc(t.called ?? 'answer'), t.accurate ? 'ok' : `<span class="bench-bad">expected ${esc(t.expected ?? 'a plain answer')}</span>`]), `×${w.concurrency} (old bench)`);
    }
    const rows = w.steps.map((st) => [
      `${st.index + 1}`,
      esc(st.expected),
      `<b class="${st.correct === st.total ? '' : 'bench-bad'}">${st.correct}/${st.total}</b>`,
      ms(st.ttft?.median),
      st.misses.length ? `<small>${esc(st.misses.slice(0, 2).join(' · '))}${st.misses.length > 2 ? ` · +${st.misses.length - 2} more` : ''}</small>` : '',
    ]);
    return grid(['Step', 'Expected', 'Correct', 'TTFT', ''], rows, waves.length > 1 ? `×${w.concurrency}` : '', [1]);
  }).join('');
  return `<div class="bench-heroes">${heroes}</div>${tables}`;
}

const LEGENDS = {
  decode: '<strong>Aggregate</strong> — all streams\' tokens over the shared first→last token window. <strong>Per stream</strong> — what one user sees. <strong>TTFT</strong> — median time to first token.',
  prefill: '<strong>Prefill tok/s</strong> — cold prompt tokens over TTFT, minus the fixed overhead of a tiny prompt; ≈ marks a low-confidence point.',
  agent: '<strong>Step accuracy</strong> — steps that called the right tool with sensible arguments (or gave the right answer). <strong>Cache hit</strong> — cached / prompt tokens across turns.',
};

function renderResult(run) {
  const result = run.result ?? {};
  let body = resultHead(run);
  if (result) {
    if (run.suite === 'decode') body += decodeView(result);
    else if (run.suite === 'prefill') body += prefillView(result);
    else body += agentView(result);
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
    // e.g. "prose ×1 53.6 · ×4 103 tok/s": the lowest and highest level of the first type.
    const waves = r.waves ?? [];
    const type = waves[0]?.type;
    const ofType = waves.filter((w) => w.type === type && w.decodeTps != null);
    if (!ofType.length) return null;
    const lo = ofType[0];
    const hi = ofType[ofType.length - 1];
    const levels = lo === hi ? `×${lo.concurrency} ${fmt(lo.decodeTps)}` : `×${lo.concurrency} ${fmt(lo.decodeTps)} · ×${hi.concurrency} ${fmt(hi.decodeTps)}`;
    return `${type ? `${type} ` : ''}${levels} tok/s`;
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
  if (state.suite === 'decode') {
    spec.decodeTokens = c.tokens;
    spec.decodeTypes = [...c.types];
  }
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
