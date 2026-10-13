// AcademIQ web client — Ask, Compare, Health.
// Plain ES module, no framework. Talks to app/server.py.

import { Atlas, METHOD_META, methodGlyph } from "./atlas.js";

const view = document.getElementById("view");

const METHOD_LABELS = {
  "TF-IDF": "TF-IDF",
  "MinHash+LSH": "MinHash + LSH",
  "SimHash": "SimHash",
  "Hybrid (All Methods)": "Hybrid — all three (RRF)",
  "Hybrid (LSH Only)": "Hybrid — LSH only (RRF)",
  "Cascade": "Cascade",
};
const COMPARE = ["TF-IDF", "MinHash+LSH", "SimHash", "Hybrid (All Methods)"];

const EXAMPLES = {
  ug: [
    "What is the minimum CGPA to avoid academic probation?",
    "What is the attendance policy?",
    "What happens if a student fails a course?",
    "What is the policy on plagiarism?",
  ],
  pg: [
    "What is the minimum CGPA required for a master's degree?",
    "What is the maximum duration of a PhD programme?",
    "What are the thesis submission requirements?",
    "What is the attendance policy?",
  ],
};

const STOPWORDS = new Set(("is the a an of for in on what how can do does are was that this to and it if or be at by " +
  "we my i me any which who when where will would should there their its with from about").split(" "));

// ── State ─────────────────────────────────────────────────────────────────

const store = {
  get(k, d) { try { const v = localStorage.getItem(`academiq.${k}`); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`academiq.${k}`, JSON.stringify(v)); } catch { /* private mode */ } },
};

const state = {
  status: null,
  book: store.get("book", "ug"),
  view: "ask",
  sky: {},                                   // book → atlas payload | { error }
  ask: { ug: { history: [], current: null }, pg: { history: [], current: null } },
  compare: { ug: null, pg: null },
  sort: { key: "Hybrid (All Methods)", dir: "ascending" },
  prefs: {
    method: store.get("method", "Hybrid (All Methods)"),
    top_k: store.get("top_k", 5),
    use_llm: store.get("use_llm", false),
  },
  pending: false,
};

let atlas = null;
let healthTimer = null;

// ── Utilities ─────────────────────────────────────────────────────────────

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ms = (v) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${(+v).toFixed(v < 10 ? 2 : 1)} ms`);
const score = (v) => (v == null ? "—" : (+v).toFixed(3));
const bookLabel = (b) => (b === "pg" ? "Postgraduate" : "Undergraduate");
const qno = (n) => `Q-${String(n).padStart(3, "0")}`;
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : detail?.detail || "Request failed");
    this.status = status;
    this.detail = detail;
  }
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : undefined);
  } catch {
    throw new ApiError(0, "Can't reach the AcademIQ server. Check that it is still running, then try again.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data.detail ?? data);
  return data;
}

function highlight(text, query) {
  const words = [...new Set(query.toLowerCase().match(/[a-z0-9.]+/g) || [])]
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w))
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const safe = esc(text);
  if (!words.length) return safe;
  return safe.replace(new RegExp(`\\b(${words.join("|")})\\w*`, "gi"), "<mark>$&</mark>");
}

// Escape text and set only its numbers (values, units, ids stay legible) in the mono face
function numify(v) {
  return esc(v).replace(/(?<![\w\-])([−+]?\d[\d.,]*(?:\u2009?\s?(?:(?:ms|s|min)\b|%))?)/g, '<span class="num">$1</span>');
}

let frameObservers = [];

/** Diamond nodes at every junction of a hairline frame grid. */
function frameNodes(container, childSel) {
  if (container._placeNodes) { container._placeNodes(); return; }
  const place = () => {
    container.querySelectorAll(":scope > .node").forEach((n) => n.remove());
    const cr = container.getBoundingClientRect();
    const seen = new Set();
    container.querySelectorAll(childSel).forEach((child) => {
      const r = child.getBoundingClientRect();
      for (const [x, y] of [[r.left, r.top], [r.right, r.top], [r.left, r.bottom], [r.right, r.bottom]]) {
        const px = Math.round(x - cr.left), py = Math.round(y - cr.top);
        const key = `${Math.round(px / 3)}:${Math.round(py / 3)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const n = document.createElement("i");
        n.className = "node";
        n.style.left = `${px}px`;
        n.style.top = `${py}px`;
        container.appendChild(n);
      }
    });
  };
  container.classList.add("framed");
  container._placeNodes = place;
  place();
  const ro = new ResizeObserver(place);
  ro.observe(container);
  frameObservers.push(ro);
}

function icon(id, cls = "icon") {
  return `<svg class="${cls}" aria-hidden="true"><use href="#i-${id}"/></svg>`;
}

function label(text, aside = "") {
  return `<h2 class="label">${esc(text)}${aside ? `<span class="aside">${numify(aside)}</span>` : ""}</h2>`;
}

function eph(rows) {
  return `<dl class="eph">${rows.map((r) => `
    <div class="${r.cls || ""}"><dt>${r.swatch ? `<i class="swatch" style="background:${r.swatch}"></i>` : ""}${esc(r.k)}</dt>
    <dd>${r.html ?? numify(r.v)}</dd></div>`).join("")}</dl>`;
}

// ── Shell ─────────────────────────────────────────────────────────────────

function syncShell() {
  document.querySelectorAll(".tab").forEach((t) => {
    if (t.dataset.view === state.view) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
  });
  document.querySelectorAll(".book button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.book === state.book));
    const info = state.status?.handbooks?.[b.dataset.book];
    const count = b.querySelector("small");
    if (info) {
      b.dataset.ready = String(info.ready);
      count.textContent = info.ready ? info.n_chunks : "no index";
    }
  });
  document.title = `AcademIQ — ${{ ask: "Ask", compare: "Compare methods", health: "System health" }[state.view]}`;
}

async function loadSky(book) {
  if (state.sky[book]) return state.sky[book];
  try {
    state.sky[book] = await api(`/api/atlas?handbook=${book}`);
  } catch (e) {
    state.sky[book] = { error: e };
  }
  return state.sky[book];
}

function mountAtlas(container, { onHover, onSelect }) {
  atlas?.destroy();
  atlas = new Atlas(container, { onHover, onSelect });
  const sky = state.sky[state.book];
  if (sky && !sky.error) atlas.setSky(sky);
  else atlas.setSky(null, sky?.error ? `The ${bookLabel(state.book).toLowerCase()} sky is empty — no index yet.` : "");
  return atlas;
}

// Output names follow config.yaml paths: UG uses the unprefixed files, PG the pg_ ones
function indexCommands(book) {
  const p = book === "pg" ? "pg_" : "";
  const d = "data/processed";
  return [
    `python -m src.ingestion.pdf_extraction --input data/raw/${book} --output ${d}/${p}raw_pages.jsonl`,
    `python -m src.ingestion.chunker --input ${d}/${p}raw_pages.jsonl --output ${d}/${p}chunks.jsonl`,
    `python -m src.retrieval.tfidf_retriever --chunks ${d}/${p}chunks.jsonl --index ${d}/${p}tfidf_index.pkl`,
    `python -m src.retrieval.minhash_lsh --chunks ${d}/${p}chunks.jsonl --index ${d}/${p}minhash_index.pkl`,
    `python -m src.retrieval.simhash --chunks ${d}/${p}chunks.jsonl --index ${d}/${p}simhash_index.pkl`,
    `python -m src.ranking.recommender --chunks ${d}/${p}chunks.jsonl --index ${d}/${p}reranker_index.pkl`,
  ].join("\n");
}

function missingNotice(book, err) {
  const detail = err?.detail?.detail || err?.message || "";
  return `
    <div class="notice">
      <h3>${icon("alert")}The ${bookLabel(book)} handbook isn't indexed yet</h3>
      <p>AcademIQ answers only from built indexes, and the ${bookLabel(book).toLowerCase()} set is missing
        ${detail ? `(<code>${esc(detail.replace(/^.*not found:\s*/i, ""))}</code>)` : ""}.
        Put the PDF in <code>data/raw/${book}/</code>, run these from the project root, then restart the server.</p>
      <pre>${indexCommands(book)}</pre>
    </div>`;
}

function errorNotice(err, retryLabel = "Try again") {
  return `
    <div class="notice" role="alert">
      <h3>${icon("alert")}That didn't go through</h3>
      <p>${esc(err.message)}</p>
      <p><button type="button" class="link" data-retry>${esc(retryLabel)}</button></p>
    </div>`;
}

// ── Ask ───────────────────────────────────────────────────────────────────

function methodOptions(selected, list = state.status?.methods || Object.keys(METHOD_LABELS)) {
  return list.map((m) => `<option value="${esc(m)}"${m === selected ? " selected" : ""}>${esc(METHOD_LABELS[m] || m)}</option>`).join("");
}

function topkOptions(selected) {
  const { min = 3, max = 15 } = state.status?.top_k || {};
  let out = "";
  for (let k = min; k <= max; k++) out += `<option value="${k}"${k === +selected ? " selected" : ""}>${k}</option>`;
  return out;
}

function renderAsk() {
  const llm = state.status?.llm;
  view.innerHTML = `
    <div class="split">
      <section class="col" aria-label="Ask the handbook">
        <form class="query" id="ask-form" autocomplete="off">
          <label class="query-field">
            <span class="sr-only">Your question</span>
            ${icon("search")}
            <input id="ask-input" name="q" maxlength="500" placeholder="Ask about a ${bookLabel(state.book).toLowerCase()} policy…" required>
          </label>
          <button class="btn" id="ask-btn" type="submit">${icon("star")}Ask</button>
        </form>
        <div class="controls">
          <label class="control">Method
            <select id="pref-method">${methodOptions(state.prefs.method)}</select></label>
          <label class="control">Top-k
            <select id="pref-k">${topkOptions(state.prefs.top_k)}</select></label>
          <label class="control">Answer
            <select id="pref-llm">
              <option value="0"${!state.prefs.use_llm ? " selected" : ""}>Extractive</option>
              <option value="1"${state.prefs.use_llm ? " selected" : ""}${llm?.available ? "" : " disabled"}>
                ${llm?.available ? `LLM (${esc(llm.model)})` : "LLM — no API key"}</option>
            </select></label>
        </div>
        <div id="ask-body"></div>
      </section>
      <section id="ask-atlas" aria-label="Handbook star chart"></section>
    </div>`;

  mountAtlas(document.getElementById("ask-atlas"), {
    onHover: (i) => lightByIndex(i),
    onSelect: (i) => {
      const li = view.querySelector(`.source[data-i="${i}"]`);
      li?.scrollIntoView({ behavior: "smooth", block: "center" });
    },
  });

  const form = document.getElementById("ask-form");
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(document.getElementById("ask-input").value); });
  document.getElementById("pref-method").addEventListener("change", (e) => { state.prefs.method = e.target.value; store.set("method", e.target.value); });
  document.getElementById("pref-k").addEventListener("change", (e) => { state.prefs.top_k = +e.target.value; store.set("top_k", +e.target.value); });
  document.getElementById("pref-llm").addEventListener("change", (e) => { state.prefs.use_llm = e.target.value === "1"; store.set("use_llm", state.prefs.use_llm); });

  renderAskBody();
  const current = state.ask[state.book].current;
  if (current) atlas.setQuery(askQueryForAtlas(current));
}

function renderAskBody() {
  const body = document.getElementById("ask-body");
  if (!body) return;
  const book = state.book;
  const sky = state.sky[book];
  const { history, current, error, pendingQuery } = state.ask[book];

  if (sky?.error) {
    body.innerHTML = missingNotice(book, sky.error);
    setAskEnabled(false);
    return;
  }
  setAskEnabled(!state.pending);

  if (pendingQuery) {
    body.innerHTML = `
      <div class="pending" aria-live="polite">
        <h1 class="asked">${esc(pendingQuery)}</h1>
        <p>Searching the ${bookLabel(book).toLowerCase()} handbook</p>
      </div>`;
    return;
  }

  if (error) {
    body.innerHTML = errorNotice(error);
    body.querySelector("[data-retry]")?.addEventListener("click", () => ask(error.query));
    return;
  }

  if (!current) {
    const c = state.status?.config || {};
    body.innerHTML = `
      <div class="intro">
        <p>Ask a question about NUST academic policy. Answers come only from the
          <b>${bookLabel(book)} handbook</b>, and every sentence names the section and page it came from.</p>
      </div>
      <div class="block">
        ${label("Try asking")}
        <ul class="examples">
          ${EXAMPLES[book].map((q) => `<li><button type="button" data-example="${esc(q)}">${esc(q)}${icon("arrow")}</button></li>`).join("")}
        </ul>
      </div>
      <div class="block">
        ${label("This handbook")}
        ${eph([
          { k: "Chunks indexed", v: sky?.n_chunks ?? "—" },
          { k: "Sections", v: sky?.n_sections ?? "—" },
          { k: "Chunking", v: `${c.chunk_size ?? 350} words · ${c.chunk_overlap ?? 50} overlap` },
          { k: "MinHash + LSH", v: `${c.num_perm ?? 128} perms · J ≥ ${c.lsh_threshold ?? 0.3}` },
          { k: "SimHash", v: `${c.simhash_bits ?? 64}-bit · Hamming ≤ ${c.hamming ?? 10}` },
          { k: "Fusion", v: `RRF k = ${c.rrf_k ?? 60}` },
          { k: "Answer backend", v: state.status?.llm?.available ? `${state.status.llm.backend} · ${state.status.llm.model}` : "Extractive only (no API key)" },
        ])}
      </div>`;
    body.querySelectorAll("[data-example]").forEach((b) =>
      b.addEventListener("click", () => {
        document.getElementById("ask-input").value = b.dataset.example;
        ask(b.dataset.example);
      }));
    return;
  }

  body.innerHTML = answerHTML(current) + historyHTML(history, current);
  wireAnswer(body, current);
}

function setAskEnabled(on) {
  const btn = document.getElementById("ask-btn");
  const input = document.getElementById("ask-input");
  if (!btn) return;
  btn.disabled = !on;
  btn.innerHTML = state.pending ? `${icon("star")}Asking` : `${icon("star")}Ask`;
  input.disabled = !!state.sky[state.book]?.error;
}

function answerHTML(r) {
  const t = r.timings;
  const cited = new Set(r.answer.cited);
  const llmFellBack = r.llm_requested && r.answer_mode !== "llm";
  const meta = [
    `<span class="num">${qno(r.qno)}</span>`,
    esc(METHOD_LABELS[r.method] || r.method),
    `top-k ${numify(r.top_k)}`,
    r.answer_mode === "llm" ? `LLM · ${esc(r.model)}` : llmFellBack ? "Extractive (no API key)" : "Extractive",
    r.cache_hit ? `<span class="hit">Cache hit</span>` : "Cache miss",
    r.precision_k != null ? `Precision@k ${numify(r.precision_k.toFixed(2))}` : null,
  ].filter(Boolean);

  // Group consecutive sentences that cite the same sources into one paragraph
  // The top-scored sentence leads as the answer line; the rest follow as prose
  const [lead, ...rest] = r.answer.sentences;
  const paras = [];
  for (const s of rest) {
    const key = s.refs.join(",");
    const last = paras[paras.length - 1];
    if (last && last.key === key && last.items.length < 3) last.items.push(s);
    else paras.push({ key, items: [s] });
  }
  const refBtns = (refs) => refs.map((n) =>
    `<button type="button" class="ref" data-rank="${n}" aria-label="Source ${n}">${n}</button>`).join("");

  let answer;
  if (r.answer.not_found || !r.chunks.length || !lead) {
    answer = `
      <div class="notice">
        <h3>${icon("alert")}Not in the handbook</h3>
        <p>${r.chunks.length
          ? "The closest sections don't answer this question, so AcademIQ won't guess. Try different wording, or read the nearest sections below."
          : "No section of the handbook matched this question. Try different wording or another retrieval method."}</p>
      </div>`;
  } else {
    answer = `<p class="lead"><span class="sent" data-refs="${lead.refs.join(" ")}">${esc(lead.text)}</span>${refBtns(lead.refs)}</p>` +
      (paras.length ? `<div class="answer">${paras.map((p) =>
        `<p>${p.items.map((s) => `<span class="sent" data-refs="${s.refs.join(" ")}">${esc(s.text)}</span>`).join(" ")}${refBtns(p.items[p.items.length - 1].refs)}</p>`).join("")}</div>` : "");
  }

  const stages = t.stages || {};
  const stageRows = [];
  if (r.cache_hit) {
    stageRows.push({ k: "Retrieval + re-rank", v: "from cache", cls: "note", swatch: "var(--st-1)" });
  } else {
    stageRows.push({ k: "Retrieval", v: ms(t.retrieval_ms), swatch: "var(--st-1)" });
    for (const [key, name] of [["tfidf_ms", "TF-IDF"], ["minhash_ms", "MinHash + LSH"], ["simhash_ms", "SimHash"]]) {
      if (key in stages) stageRows.push({ k: name, v: ms(stages[key]), cls: "sub" });
    }
    stageRows.push({ k: "Re-rank", v: ms(t.rerank_ms), swatch: "var(--st-2)" });
  }
  stageRows.push({ k: "Generate", v: ms(t.generation_ms), swatch: "var(--st-3)" });
  stageRows.push({ k: "Total", v: ms(t.total_ms), cls: "total" });

  const seg = (v, c) => `<span style="flex-grow:${Math.max(v, 0.0001)};background:${c}"></span>`;
  const bar = `<div class="stagebar" aria-hidden="true">${seg(t.retrieval_ms, "var(--st-1)")}${seg(t.rerank_ms, "var(--st-2)")}${seg(t.generation_ms, "var(--st-3)")}</div>`;

  const sources = r.chunks.map((c) => `
    <li class="source${cited.has(c.rank) ? " is-cited" : ""}" data-rank="${c.rank}" data-i="${c.i ?? ""}" tabindex="0">
      <span class="rank" aria-label="Rank ${c.rank}">${c.rank}</span>
      <div class="src-head"><span class="src-title">${esc(c.section)}</span><span class="src-page">p. ${esc(c.page ?? "?")}</span></div>
      <div class="src-meta">Score <span class="num">${score(c.score)}</span> · <span class="num">${esc(c.id)}</span>${cited.has(c.rank) ? "" : " · not cited"}</div>
      <p class="src-text">${highlight(c.text, r.query)}</p>
      <button type="button" class="link src-toggle" aria-expanded="false">Show full chunk</button>
    </li>`).join("");

  return `
    <article aria-live="polite">
      <h1 class="asked">${esc(r.query)}</h1>
      <div class="meta">${meta.map((m) => `<span>${m}</span>`).join("")}</div>
      ${answer}
      <div class="block">${label("Stage timings")}${eph(stageRows)}${bar}</div>
      <div class="block">${label("Sources", `${r.chunks.length} retrieved · ${r.answer.cited.length} cited`)}
        <ol class="sources">${sources}</ol></div>
    </article>`;
}

function historyHTML(history, current) {
  const earlier = history.filter((h) => h !== current);
  if (!earlier.length) return "";
  return `
    <div class="block">
      ${label("Earlier this session")}
      <ul class="history">
        ${[...history].reverse().map((h) => `
          <li><button type="button" data-qno="${h.qno}"${h === current ? ' aria-current="true"' : ""}>
            <span class="num">${qno(h.qno)}</span><span class="q">${esc(h.query)}</span><span class="t num">${ms(h.timings.total_ms)}</span>
          </button></li>`).join("")}
      </ul>
    </div>`;
}

function wireAnswer(body, r) {
  const lightRank = (rank) => {
    body.querySelectorAll(".is-lit").forEach((n) => n.classList.remove("is-lit"));
    if (rank == null) { atlas?.highlight(null); return; }
    body.querySelectorAll(`.source[data-rank="${rank}"], .ref[data-rank="${rank}"]`).forEach((n) => n.classList.add("is-lit"));
    body.querySelectorAll(".sent").forEach((s) => { if (s.dataset.refs.split(" ").includes(String(rank))) s.classList.add("is-lit"); });
    const chunk = r.chunks.find((c) => c.rank === rank);
    atlas?.highlight(chunk?.i ?? null);
  };
  body._lightRank = lightRank;

  body.querySelectorAll(".ref").forEach((b) => {
    const rank = +b.dataset.rank;
    b.addEventListener("pointerenter", () => lightRank(rank));
    b.addEventListener("pointerleave", () => lightRank(null));
    b.addEventListener("focus", () => lightRank(rank));
    b.addEventListener("blur", () => lightRank(null));
    b.addEventListener("click", () => body.querySelector(`.source[data-rank="${rank}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  });
  body.querySelectorAll(".source").forEach((li) => {
    const rank = +li.dataset.rank;
    li.addEventListener("pointerenter", () => lightRank(rank));
    li.addEventListener("pointerleave", () => lightRank(null));
    li.addEventListener("focus", () => lightRank(rank));
    li.addEventListener("blur", () => lightRank(null));
    const toggle = li.querySelector(".src-toggle");
    toggle.addEventListener("click", () => {
      const open = li.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Show less" : "Show full chunk";
    });
  });
  body.querySelectorAll("[data-qno]").forEach((b) => b.addEventListener("click", () => {
    const h = state.ask[state.book].history.find((x) => x.qno === +b.dataset.qno);
    if (!h) return;
    state.ask[state.book].current = h;
    renderAskBody();
    atlas?.setQuery(askQueryForAtlas(h));
    view.scrollIntoView({ behavior: "smooth" });
  }));
}

function lightByIndex(i) {
  const body = document.getElementById("ask-body");
  if (!body?._lightRank) { atlas?.highlight(i); return; }
  const current = state.ask[state.book].current;
  const chunk = current?.chunks.find((c) => c.i === i);
  if (chunk) body._lightRank(chunk.rank);
  else { body._lightRank(null); atlas?.highlight(i); }
}

function askQueryForAtlas(r) {
  const cited = new Set(r.answer.cited);
  return {
    point: r.query_point,
    field: r.field,
    rings: r.chunks.map((c) => ({ i: c.i, rank: c.rank, cited: cited.has(c.rank), page: c.page })),
  };
}

async function ask(raw) {
  const query = (raw || "").trim();
  if (!query || state.pending) return;
  const book = state.book;
  const slot = state.ask[book];
  state.pending = true;
  slot.error = null;
  slot.pendingQuery = query;
  renderAskBody();
  try {
    const r = await api("/api/ask", {
      handbook: book, query, method: state.prefs.method, top_k: state.prefs.top_k, use_llm: state.prefs.use_llm,
    });
    slot.history.push(r);
    slot.current = r;
  } catch (e) {
    if (e.status === 409) state.sky[book] = { error: e };
    e.query = query;
    slot.error = e;
  } finally {
    state.pending = false;
    slot.pendingQuery = null;
  }
  if (state.view !== "ask" || state.book !== book) return;
  renderAskBody();
  if (slot.current && !slot.error) atlas?.setQuery(askQueryForAtlas(slot.current));
}

// ── Compare ───────────────────────────────────────────────────────────────

const PLATES = (c = {}) => ({
  "TF-IDF": ["Exact cosine similarity over the full vocabulary", "1–2-grams · sublinear tf"],
  "MinHash+LSH": ["Weighted MinHash, banded into LSH buckets", `${c.num_perm ?? 128} perms · J ≥ ${c.lsh_threshold ?? 0.3}`],
  "SimHash": ["Fingerprints compared by Hamming distance", `${c.simhash_bits ?? 64}-bit · ≤ ${c.hamming ?? 10} bits`],
  "Hybrid (All Methods)": ["Reciprocal Rank Fusion of all three", `k ${c.rrf_k ?? 60} · weights ${Object.values(c.weights || { a: 1, b: 0.8, c: 0.6 }).join(" / ")}`],
});

function renderCompare() {
  const result = state.compare[state.book];
  view.innerHTML = `
    <div class="split wide">
      <section class="col" aria-label="Compare retrieval methods">
        <form class="query" id="cmp-form" autocomplete="off">
          <label class="query-field">
            <span class="sr-only">Question to compare</span>
            ${icon("search")}
            <input id="cmp-input" maxlength="500" placeholder="One question, four retrievers…" value="${esc(result?.query || "")}" required>
          </label>
          <button class="btn" id="cmp-btn" type="submit">${icon("star")}Compare</button>
        </form>
        <div class="controls">
          <label class="control">Top-k
            <select id="cmp-k">${topkOptions(result?.top_k || state.prefs.top_k)}</select></label>
          <span class="control">Changing top-k re-runs the comparison</span>
        </div>
        <div id="cmp-body"></div>
      </section>
      <section id="cmp-atlas" aria-label="Handbook star chart with each method's picks"></section>
    </div>`;

  mountAtlas(document.getElementById("cmp-atlas"), {
    onHover: (i) => {
      atlas.highlight(i);
      view.querySelectorAll(".agree tr.is-lit").forEach((r) => r.classList.remove("is-lit"));
      if (i != null) view.querySelector(`.agree tr[data-i="${i}"]`)?.classList.add("is-lit");
    },
    onSelect: (i) => view.querySelector(`.agree tr[data-i="${i}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" }),
  });

  document.getElementById("cmp-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runCompare(document.getElementById("cmp-input").value, +document.getElementById("cmp-k").value);
  });
  document.getElementById("cmp-k").addEventListener("change", (e) => {
    const q = document.getElementById("cmp-input").value;
    if (q.trim()) runCompare(q, +e.target.value);
  });

  renderCompareBody();
  if (result && !result.error) atlas.setQuery(compareQueryForAtlas(result));
}

function compareQueryForAtlas(r) {
  return {
    point: r.query_point,
    field: r.field,
    marks: r.columns.map((col) => ({ method: col.method, items: col.chunks.map((c) => c.i) })),
  };
}

function renderCompareBody() {
  const body = document.getElementById("cmp-body");
  if (!body) return;
  const sky = state.sky[state.book];
  const r = state.compare[state.book];
  const btn = document.getElementById("cmp-btn");
  btn.disabled = !!sky?.error || state.pending;
  btn.innerHTML = `${icon("star")}${state.pending ? "Comparing" : "Compare"}`;

  if (sky?.error) { body.innerHTML = missingNotice(state.book, sky.error); return; }
  if (r?.pendingQuery) {
    body.innerHTML = `<div class="pending"><h1 class="question">${esc(r.pendingQuery)}</h1><p>Running four retrievers</p></div>`;
    return;
  }
  if (r?.error) {
    body.innerHTML = errorNotice(r.error);
    body.querySelector("[data-retry]")?.addEventListener("click", () => runCompare(r.error.query, r.error.k));
    return;
  }
  if (!r) {
    body.innerHTML = `
      <div class="intro"><p>Run one question through <b>TF-IDF, MinHash + LSH, SimHash and Hybrid RRF</b> side by side:
        what each retrieved, in what order, and what it cost. Answers aren't generated here.</p></div>
      <div class="block">${label("Try comparing")}
        <ul class="examples">${EXAMPLES[state.book].map((q) =>
          `<li><button type="button" data-example="${esc(q)}">${esc(q)}${icon("arrow")}</button></li>`).join("")}</ul></div>`;
    body.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
      document.getElementById("cmp-input").value = b.dataset.example;
      runCompare(b.dataset.example, +document.getElementById("cmp-k").value);
    }));
    return;
  }

  const plates = PLATES(state.status?.config);
  const platesHTML = r.columns.map((col) => `
    <div class="plate">
      <div class="plate-head">${methodGlyph(col.method)}<h3>${esc(METHOD_META[col.method].name)}</h3></div>
      <p class="mech">${esc(plates[col.method][0])}</p>
      <p class="param">${esc(plates[col.method][1])}</p>
      ${eph(col.cache_hit
        ? [{ k: "Served", v: "from cache" , cls: "note" }, { k: "Wall time", v: ms(col.total_ms), cls: "total" }]
        : [{ k: "Retrieval", v: ms(col.retrieval_ms) }, { k: "Re-rank", v: ms(col.rerank_ms) }, { k: "Total", v: ms(col.total_ms), cls: "total" }])}
    </div>`).join("");

  // Union of every retrieved chunk, with each method's rank
  const rows = new Map();
  r.columns.forEach((col) => col.chunks.forEach((c) => {
    if (!rows.has(c.id)) rows.set(c.id, { ...c, ranks: {} });
    rows.get(c.id).ranks[col.method] = c.rank;
  }));
  const { key, dir } = state.sort;
  const list = [...rows.values()].sort((a, b) => {
    const ra = a.ranks[key] ?? 99, rb = b.ranks[key] ?? 99;
    const best = (x) => Math.min(...Object.values(x.ranks));
    return (dir === "ascending" ? ra - rb : rb - ra) || best(a) - best(b);
  });
  const agreeAll = list.filter((x) => Object.keys(x.ranks).length === COMPARE.length).length;

  body.innerHTML = `
    <h1 class="question">${esc(r.query)}</h1>
    <div class="meta"><span>${bookLabel(state.book)} handbook</span><span>top-k ${numify(r.top_k)}</span>
      <span>${numify(rows.size)} distinct chunks</span><span>${numify(agreeAll)} found by all four</span></div>
    <div class="block">${label("Methods")}<div class="plates">${platesHTML}</div></div>
    <div class="block">${label("Rank agreement")}
      <div class="table-wrap"><table class="agree">
        <thead><tr><th scope="col">Section</th>${COMPARE.map((m) => `
          <th scope="col" class="r"${m === key ? ` aria-sort="${dir}"` : ""}>
            <button type="button" data-sort="${esc(m)}">${methodGlyph(m, 12)}${esc(METHOD_META[m].short)}
            <svg class="sort" aria-hidden="true" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-sort"/></svg></button></th>`).join("")}</tr></thead>
        <tbody>${list.map((x) => `
          <tr data-i="${x.i ?? ""}">
            <td class="sec">${esc(x.section)}<small>p. ${esc(x.page ?? "?")} · ${esc(x.id)}</small></td>
            ${COMPARE.map((m) => {
              const rk = x.ranks[m];
              return `<td class="r${rk == null ? " none" : rk === 1 ? " best" : ""}">${rk ?? "—"}</td>`;
            }).join("")}
          </tr>`).join("")}</tbody>
      </table></div>
      <p class="foot">— means the chunk wasn't in that method's top-k. Select a column heading to sort by it.</p>
    </div>`;

  frameNodes(body.querySelector(".plates"), ".plate");
  body.querySelectorAll("[data-sort]").forEach((b) => b.addEventListener("click", () => {
    const m = b.dataset.sort;
    state.sort = { key: m, dir: state.sort.key === m && state.sort.dir === "ascending" ? "descending" : "ascending" };
    renderCompareBody();
  }));
  body.querySelectorAll(".agree tbody tr").forEach((tr) => {
    const i = tr.dataset.i === "" ? null : +tr.dataset.i;
    tr.addEventListener("pointerenter", () => atlas?.highlight(i));
    tr.addEventListener("pointerleave", () => atlas?.highlight(null));
  });
}

async function runCompare(raw, k) {
  const query = (raw || "").trim();
  if (!query || state.pending) return;
  const book = state.book;
  const prev = state.compare[book];
  state.pending = true;
  state.compare[book] = { ...(prev && !prev.error ? prev : {}), pendingQuery: prev?.query === query ? null : query };
  if (state.compare[book].pendingQuery) renderCompareBody();
  try {
    state.compare[book] = await api("/api/compare", { handbook: book, query, top_k: k });
  } catch (e) {
    if (e.status === 409) state.sky[book] = { error: e };
    e.query = query; e.k = k;
    state.compare[book] = { error: e };
  } finally {
    state.pending = false;
  }
  if (state.view !== "compare" || state.book !== book) return;
  renderCompareBody();
  const r = state.compare[book];
  if (!r.error) atlas?.setQuery(compareQueryForAtlas(r));
}

// ── Health ────────────────────────────────────────────────────────────────

function renderHealthShell() {
  atlas?.destroy();
  atlas = null;
  view.innerHTML = `
    <section class="health" aria-label="System health">
      <div class="health-head">
        <h1>System health</h1>
        <p id="health-stamp">This server session</p>
      </div>
      <div class="board" id="board"></div>
    </section>`;
  refreshHealth();
}

async function refreshHealth() {
  const board = document.getElementById("board");
  if (!board) return;
  try {
    const h = await api("/api/health");
    renderBoard(board, h);
    document.getElementById("health-stamp").textContent =
      `This server session · updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
  } catch (e) {
    board.innerHTML = `<div class="panel span-12">${errorNotice(e, "Refresh")}</div>`;
    board.querySelector("[data-retry]")?.addEventListener("click", refreshHealth);
  }
}

function bars(rows, max, fmt) {
  return `<div class="bars">${rows.map((r) => `
    <div class="bar-row" title="${esc(r.name)}: ${esc(fmt(r.value))}">
      <span class="bar-name">${r.glyph || ""}${esc(r.name)}</span>
      <span class="bar-track"><span class="bar-fill" style="display:block;width:${max ? (r.value / max) * 100 : 0}%;background:var(--amber)"></span></span>
      <span class="bar-val">${esc(fmt(r.value))}</span>
    </div>`).join("")}</div>`;
}

function renderBoard(board, h) {
  const s = h.summary, c = h.cache, pk = s.precision_k || {};
  const none = `<p class="empty">No queries yet. Ask something and this fills in.</p>`;
  const glyph = (m) => (METHOD_META[m] ? methodGlyph(m, 14) : "");
  const short = (m) => METHOD_META[m]?.name || { "Hybrid (LSH Only)": "Hybrid, LSH only" }[m] || m;
  const usage = Object.entries(s.method_breakdown || {}).sort((a, b) => b[1] - a[1])
    .map(([m, v]) => ({ name: short(m), value: v, glyph: glyph(m) }));
  const prec = Object.entries(h.precision_by_method || {})
    .map(([m, v]) => ({ name: short(m), value: v.avg, glyph: glyph(m) }));

  board.innerHTML = `
    <div class="panel span-8">${label("Latency per query", "total, ms")}
      ${h.history.length ? `<div class="chart" id="lat-chart"></div>
        <div class="chart-key"><span><i style="background:var(--verm)"></i>Computed</span><span><i style="border:2px solid var(--amber)"></i>Cache hit</span></div>` : none}</div>
    <div class="panel span-4">${label("Latency")}
      ${eph([
        { k: "Queries", v: s.total_queries },
        { k: "P50 total", v: ms(s.p50_total_ms) },
        { k: "P99 total", v: ms(s.p99_total_ms), cls: "total" },
        { k: "P50 retrieval", v: ms(s.p50_retrieval) },
        { k: "P99 retrieval", v: ms(s.p99_retrieval) },
        { k: "Avg chunks", v: s.total_queries ? (+s.avg_chunks).toFixed(1) : "—" },
      ])}</div>
    <div class="panel span-4">${label("Query cache")}
      ${eph([
        { k: "Hit rate", v: `${(c.hit_rate * 100).toFixed(0)}%`, cls: "total" },
        { k: "Hits / misses", v: `${c.hits} / ${c.misses}` },
        { k: "Entries", v: `${c.size} / ${c.max_size}` },
        { k: "Expiry", v: `${Math.round(c.ttl_seconds / 60)} min` },
      ])}
      <div class="actions"><button type="button" class="btn-quiet" id="clear-cache">${icon("clear")}Clear cache</button></div></div>
    <div class="panel span-4">${label("Precision@k")}
      ${pk.n ? eph([
        { k: "Average", v: pk.avg.toFixed(2), cls: "total" },
        { k: "P50", v: pk.p50.toFixed(2) },
        { k: "P99", v: pk.p99.toFixed(2) },
        { k: "Queries scored", v: pk.n },
      ]) + `<div class="block" style="margin-top:20px">${bars(prec, 1, (v) => v.toFixed(2))}</div>` : none}
      <p class="foot">Share of retrieved chunks scoring ≥ 0.05 — a proxy from the retrievers' own scores. Cache hits aren't scored.</p></div>
    <div class="panel span-4">${label("Method usage", "queries")}
      ${usage.length ? bars(usage, Math.max(...usage.map((u) => u.value)), (v) => String(v)) : none}</div>
    <div class="panel span-12">${label("Recent queries")}
      ${h.recent.length ? `<div class="table-wrap"><table class="log">
        <thead><tr><th scope="col">No.</th><th scope="col">Question</th><th scope="col">Handbook</th><th scope="col">Method</th>
          <th scope="col" class="n">Retrieval</th><th scope="col" class="n">Re-rank</th><th scope="col" class="n">Generate</th><th scope="col" class="n">Total</th><th scope="col">Cache</th></tr></thead>
        <tbody>${h.recent.map((q) => `
          <tr><td class="qno">${qno(q.qno)}</td><td class="q" title="${esc(q.query)}">${esc(q.query)}</td>
            <td>${q.handbook.toUpperCase()}</td><td>${esc(METHOD_LABELS[q.method] || q.method)}</td>
            <td class="n">${ms(q.retrieval_ms)}</td><td class="n">${ms(q.rerank_ms)}</td><td class="n">${ms(q.generation_ms)}</td>
            <td class="n">${ms(q.total_ms)}</td><td>${q.cache_hit ? '<span class="tag">hit</span>' : "—"}</td></tr>`).join("")}</tbody>
      </table></div>` : none}</div>`;

  board.querySelector("#clear-cache").addEventListener("click", async (e) => {
    e.currentTarget.disabled = true;
    try { await api("/api/cache/clear", {}); } finally { refreshHealth(); }
  });
  frameNodes(board, ":scope > .panel");
  const chart = board.querySelector("#lat-chart");
  if (chart) lineChart(chart, h.history);
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function lineChart(root, data) {
  const NS = "http://www.w3.org/2000/svg";
  const w = root.clientWidth, h = root.clientHeight;
  const m = { l: 52, r: 12, t: 10, b: 26 };
  const ymax = niceMax(Math.max(...data.map((d) => d.total_ms)));
  const x = (i) => m.l + (data.length === 1 ? (w - m.l - m.r) / 2 : (i / (data.length - 1)) * (w - m.l - m.r));
  const y = (v) => m.t + (1 - v / ymax) * (h - m.t - m.b);

  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Total latency for the last ${data.length} queries, up to ${ms(ymax)}">`;
  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    svg += `<line class="${k ? "gl" : "base"}" x1="${m.l}" x2="${w - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
    svg += `<text class="tick" x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${ms(v).replace(" ", " ")}</text>`;
  }
  const first = data[0], last = data[data.length - 1];
  svg += `<text class="tick" x="${x(0)}" y="${h - 6}" text-anchor="start">${qno(first.qno)}</text>`;
  if (data.length > 1) svg += `<text class="tick" x="${x(data.length - 1)}" y="${h - 6}" text-anchor="end">${qno(last.qno)}</text>`;
  if (data.length > 1) svg += `<polyline class="line" points="${data.map((d, i) => `${x(i)},${y(d.total_ms)}`).join(" ")}"/>`;
  svg += `<line class="cross" id="cross" x1="0" x2="0" y1="${m.t}" y2="${h - m.b}" visibility="hidden"/>`;
  svg += data.map((d, i) => `<circle class="pt${d.cache_hit ? " cached" : ""}" cx="${x(i)}" cy="${y(d.total_ms)}" r="4.5"/>`).join("");
  svg += `<rect id="hover" x="${m.l}" y="0" width="${w - m.l - m.r}" height="${h}" fill="transparent"/></svg>`;
  root.innerHTML = svg + `<div class="chart-tip" hidden></div>`;

  const tip = root.querySelector(".chart-tip");
  const cross = root.querySelector("#cross");
  root.querySelector("#hover").addEventListener("pointermove", (e) => {
    const bx = e.clientX - root.getBoundingClientRect().left;
    let best = 0;
    data.forEach((_, i) => { if (Math.abs(x(i) - bx) < Math.abs(x(best) - bx)) best = i; });
    const d = data[best];
    cross.setAttribute("x1", x(best)); cross.setAttribute("x2", x(best)); cross.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.innerHTML = `<b>${esc(d.query)}</b>
      <div><span class="num">${qno(d.qno)}</span><span>${esc(METHOD_META[d.method]?.short || d.method)}${d.cache_hit ? " · cache hit" : ""}</span></div>
      <div><span>Retrieval</span><span class="num">${ms(d.retrieval_ms)}</span></div>
      <div><span>Re-rank</span><span class="num">${ms(d.rerank_ms)}</span></div>
      <div><span>Generate</span><span class="num">${ms(d.generation_ms)}</span></div>
      <div><span>Total</span><span class="num">${ms(d.total_ms)}</span></div>`;
    const left = x(best) + 14 + 220 > w ? x(best) - 14 - 220 : x(best) + 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(0, y(d.total_ms) - 40)}px`;
  });
  root.querySelector("#hover").addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

// ── Routing ───────────────────────────────────────────────────────────────

async function render() {
  clearInterval(healthTimer);
  frameObservers.forEach((ro) => ro.disconnect());
  frameObservers = [];
  syncShell();
  if (state.view === "health") {
    renderHealthShell();
    healthTimer = setInterval(() => { if (!document.hidden) refreshHealth(); }, 5000);
    return;
  }
  await loadSky(state.book);
  if (state.view === "compare") renderCompare(); else renderAsk();
}

function route() {
  const v = location.hash.replace("#", "");
  state.view = ["ask", "compare", "health"].includes(v) ? v : "ask";
  render();
}

document.querySelectorAll(".book button").forEach((b) => b.addEventListener("click", () => {
  if (state.book === b.dataset.book) return;
  state.book = b.dataset.book;
  store.set("book", state.book);
  render();
}));
window.addEventListener("hashchange", route);

(async function boot() {
  try {
    state.status = await api("/api/status");
    if (!state.status.handbooks[state.book]?.ready) {
      const ready = Object.entries(state.status.handbooks).find(([, v]) => v.ready);
      if (ready) state.book = ready[0];
    }
    if (!state.status.methods.includes(state.prefs.method)) state.prefs.method = state.status.default_method;
    if (!state.status.llm.available) state.prefs.use_llm = false;
  } catch (e) {
    view.innerHTML = `<div class="col" style="max-width:640px">${errorNotice(e, "Reload")}</div>`;
    view.querySelector("[data-retry]")?.addEventListener("click", () => location.reload());
    return;
  }
  route();
})();
