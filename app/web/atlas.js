// The handbook as a star chart.
// Every chunk is a star placed by its TF-IDF coordinates (SVD → PCA, computed
// server-side). A query becomes a reticle in the same space; its similarity to
// each chunk sets that star's magnitude; retrieved chunks are ringed by rank.

const NS = "http://www.w3.org/2000/svg";
const PAD = { l: 64, t: 30, r: 30, b: 40 };
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const METHOD_META = {
  "TF-IDF":               { short: "TF-IDF",  name: "TF-IDF",        shape: "circle",   color: "var(--s-tfidf)",   angle: 225 },
  "MinHash+LSH":          { short: "MinHash", name: "MinHash + LSH", shape: "square",   color: "var(--s-minhash)", angle: 315 },
  "SimHash":              { short: "SimHash", name: "SimHash",       shape: "triangle", color: "var(--s-simhash)", angle: 45 },
  "Hybrid (All Methods)": { short: "Hybrid",  name: "Hybrid RRF",    shape: "diamond",  color: "var(--s-hybrid)",  angle: 135 },
};

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function shapeNode(shape, cx, cy, s, attrs = {}) {
  if (shape === "circle") return el("circle", { cx, cy, r: s, ...attrs });
  const pts = {
    square:   [[-s, -s], [s, -s], [s, s], [-s, s]].map(([x, y]) => [x * 0.86, y * 0.86]),
    triangle: [[0, -s * 1.15], [s * 1.05, s * 0.75], [-s * 1.05, s * 0.75]],
    diamond:  [[0, -s * 1.2], [s * 1.2, 0], [0, s * 1.2], [-s * 1.2, 0]],
  }[shape];
  return el("polygon", { points: pts.map(([x, y]) => `${cx + x},${cy + y}`).join(" "), ...attrs });
}

/** Small inline SVG of a method's mark, for legends, plates and tables. */
export function methodGlyph(method, size = 18) {
  const m = METHOD_META[method];
  const svg = el("svg", { viewBox: "-10 -10 20 20", width: size, height: size, "aria-hidden": "true" });
  svg.appendChild(shapeNode(m.shape, 0, 0, 6.5, { fill: "none", stroke: m.color, "stroke-width": 1.8 }));
  return svg.outerHTML;
}

function cleanSection(name) {
  return String(name || "")
    .replace(/^chapter\s+\d+\s*[:.\-–]\s*/i, "")
    .replace(/^\d+(\.\d+)*\.?\s+/, "")
    .replace(/[.:]\s*$/, "")
    .trim();
}

const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtScore = (v) => (Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3).replace(/^(-?)0/, "$1"));
const fmtTick = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
// Rough advance widths, enough for collision boxes
const textW = (s, px = 7.2) => String(s).length * px;

export class Atlas {
  constructor(root, { onHover = () => {}, onSelect = () => {} } = {}) {
    this.root = root;
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.sky = null;
    this.query = null;
    this.lit = null;
    this.lastPoint = null;

    root.classList.add("atlas");
    root.innerHTML = "";
    this.svg = el("svg", { role: "img" }, root);
    this.band = Object.assign(document.createElement("div"), { className: "atlas-band" });
    this.tip = Object.assign(document.createElement("div"), { className: "atlas-tip", role: "status" });
    this.empty = Object.assign(document.createElement("div"), { className: "atlas-empty", hidden: true });
    root.append(this.band, this.tip, this.empty);

    this.ro = new ResizeObserver(() => this.render());
    this.ro.observe(root);
  }

  destroy() { this.ro.disconnect(); }

  // ── Public API ────────────────────────────────────────────────────────

  setSky(sky, emptyMessage = "") {
    this.sky = sky;
    this.query = null;
    this.lastPoint = null;
    this._prevR = null;
    this.empty.hidden = !!sky;
    this.empty.textContent = emptyMessage;
    this.render();
  }

  /**
   * query = { point:[x,y], field:[sim per star], rings:[{i, rank, cited, page}],
   *           marks:[{method, items:[i,...]}] }
   */
  setQuery(query) {
    this.query = query;
    this.lit = null;
    this.render({ animate: true });
  }

  highlight(i) {
    if (this.lit === i) return;
    this.lit = i;
    this._drawFocus();
  }

  // ── Geometry ──────────────────────────────────────────────────────────

  _px(x, y) {
    return [PAD.l + x * (this.w - PAD.l - PAD.r), PAD.t + y * (this.h - PAD.t - PAD.b)];
  }

  _t(i) {
    const f = this.query?.field;
    return f ? Math.max(0, f[i] || 0) / (this._fieldMax || 1e-6) : null;
  }

  _radius(i) {
    const t = this._t(i);
    return t == null ? 1.9 : 1.2 + 7.4 * Math.pow(t, 0.8);
  }

  _opacity(i) {
    const t = this._t(i);
    return t == null ? 0.62 : 0.28 + 0.72 * t;
  }

  // ── Render ────────────────────────────────────────────────────────────

  render({ animate = false } = {}) {
    this._fieldMax = this.query?.field ? Math.max(...this.query.field, 1e-6) : 0;
    this._renderBand();

    const box = this.root.getBoundingClientRect();
    const bandH = this.band.hidden ? 0 : this.band.offsetHeight;
    this.w = Math.max(320, box.width);
    this.h = Math.max(280, box.height - bandH);
    const svg = this.svg;
    svg.style.height = `${this.h}px`;
    svg.setAttribute("viewBox", `0 0 ${this.w} ${this.h}`);
    svg.innerHTML = "";
    this.obstacles = [];

    const defs = el("defs", {}, svg);
    const grad = el("radialGradient", { id: "star-glow" }, defs);
    el("stop", { offset: "0", "stop-color": "#dddcd3", "stop-opacity": "0.55" }, grad);
    el("stop", { offset: "0.35", "stop-color": "#dddcd3", "stop-opacity": "0.12" }, grad);
    el("stop", { offset: "1", "stop-color": "#dddcd3", "stop-opacity": "0" }, grad);

    this._drawGraticule();
    if (!this.sky) return;

    const stars = this.sky.stars;
    this.layers = {};
    for (const name of ["glow", "const", "stars", "marks", "rings", "labels", "focus", "reticle", "hits"]) {
      this.layers[name] = el("g", { class: `layer-${name}` }, svg);
    }
    this._groupSections(stars);
    this._drawConstellations(stars);

    const anim = animate && !reduceMotion();
    stars.forEach((s, i) => {
      const [x, y] = this._px(s.x, s.y);
      const r = this._radius(i);
      const prev = this._prevR?.[i];
      const t = this._t(i);
      if (t != null && t > 0.45) {
        el("circle", { cx: x, cy: y, r: r * 3.4, fill: "url(#star-glow)", opacity: ((t - 0.45) / 0.55).toFixed(2) }, this.layers.glow);
      }
      const c = el("circle", { class: "star", cx: x, cy: y, r: anim && prev != null ? prev : r,
                               "data-i": i, opacity: this._opacity(i) }, this.layers.stars);
      if (anim && prev != null && prev !== r) requestAnimationFrame(() => requestAnimationFrame(() => c.setAttribute("r", r)));
      const hit = el("circle", { class: "hit", cx: x, cy: y, r: Math.max(9, r + 4), "data-i": i }, this.layers.hits);
      hit.addEventListener("pointerenter", () => { this._showTip(i); this.onHover(i); });
      hit.addEventListener("pointerleave", () => { this._hideTip(); this.onHover(null); });
      hit.addEventListener("click", () => this.onSelect(i));
    });
    this._prevR = stars.map((_, i) => this._radius(i));
    // Bright stars keep their own space; names settle around them
    stars.forEach((s, i) => {
      const r = this._prevR[i];
      if (r < 3.5) return;
      const [x, y] = this._px(s.x, s.y);
      this.obstacles.push({ x: x - r - 2, y: y - r - 2, w: 2 * r + 4, h: 2 * r + 4 });
    });

    // Marks and the reticle claim their space first; section names fit around them
    if (this.query?.marks) this._drawMarks();
    if (this.query?.rings) this._drawRings(anim);
    this._drawReticle(anim);
    this._drawSectionLabels(stars);
    this._drawFocus();

    const n = stars.length;
    const ringed = this.query?.rings?.length || 0;
    svg.setAttribute("aria-label",
      `Star chart of ${n} ${this.sky.label.toLowerCase()} handbook chunks across ${this.sky.n_sections} sections` +
      (ringed ? `; ${ringed} retrieved chunks are ringed.` : "."));
  }

  _drawGraticule() {
    const g = el("g", {}, this.svg);
    const cols = 6, rows = 5;
    const axes = this.sky?.axes;
    // Normalised chart coords → component value (inverse of the server's mapping)
    const val = (t, [lo, hi], flip) => lo + ((flip ? 1 - (t - 0.06) / 0.88 : (t - 0.06) / 0.88)) * (hi - lo);

    for (let c = 0; c <= cols; c++) {
      const t = c / cols;
      const x = PAD.l + t * (this.w - PAD.l - PAD.r);
      if (c > 0 && c < cols) el("line", { class: "grat", x1: x, y1: 0, x2: x, y2: this.h - PAD.b + 6 }, g);
      if (axes && c > 0 && c < cols) {
        const lab = el("text", { class: "tick", x, y: this.h - PAD.b + 20, "text-anchor": "middle" }, g);
        lab.textContent = fmtTick(val(t, axes.x, false));
      }
    }
    for (let r = 0; r <= rows; r++) {
      const t = r / rows;
      const y = PAD.t + t * (this.h - PAD.t - PAD.b);
      if (r > 0 && r < rows) el("line", { class: "grat", x1: PAD.l - 6, y1: y, x2: this.w, y2: y }, g);
      if (axes && r > 0 && r < rows) {
        const lab = el("text", { class: "tick", x: PAD.l - 12, y: y + 4, "text-anchor": "end" }, g);
        lab.textContent = fmtTick(val(t, axes.y, true));
      }
    }
    const R = Math.max(this.w, this.h) * 0.62;
    el("circle", { class: "sphere", cx: this.w * 0.5, cy: this.h * 0.5, r: R }, g);
    el("circle", { class: "sphere", cx: this.w * 0.5, cy: this.h * 0.5, r: R * 0.72, "stroke-dasharray": "2 6" }, g);
    if (!this.sky) return;
    const t1 = el("text", { class: "axis", x: this.w - PAD.r, y: this.h - 6, "text-anchor": "end" }, g);
    t1.textContent = "Component 1";
    const t2 = el("text", { class: "axis", x: 16, y: PAD.t, transform: `rotate(-90 16 ${PAD.t})`, "text-anchor": "end" }, g);
    t2.textContent = "Component 2";
  }

  _groupSections(stars) {
    const groups = new Map();
    stars.forEach((s, i) => {
      if (!groups.has(s.section)) groups.set(s.section, []);
      groups.get(s.section).push(i);
    });
    for (const idx of groups.values()) idx.sort((a, b) => (stars[a].page ?? 0) - (stars[b].page ?? 0) || a - b);
    this._groups = groups;
  }

  _drawConstellations(stars) {
    for (const idx of this._groups.values()) {
      if (idx.length < 2) continue;
      const pts = idx.map((i) => this._px(stars[i].x, stars[i].y).join(",")).join(" ");
      el("polyline", { class: "const", points: pts }, this.layers.const);
    }
  }

  /** Place a label box at the first candidate that stays on the chart and clear of obstacles. */
  _place(candidates, { claim = true, ignore = null } = {}) {
    for (const box of candidates) {
      if (box.x < 4 || box.y < 2 || box.x + box.w > this.w - 4 || box.y + box.h > this.h - PAD.b + 4) continue;
      // `ignore` exempts only a star's own footprint; everything placed is claimed untagged
      if (this.obstacles.some((o) => (ignore == null || o.owner !== ignore) && overlaps(o, box))) continue;
      if (claim) this.obstacles.push(box);
      return box;
    }
    return null;
  }

  _drawSectionLabels(stars) {
    this._secLabels = [];
    const groups = [...this._groups.entries()].filter(([, idx]) => idx.length >= 3)
      .sort((a, b) => b[1].length - a[1].length).slice(0, 10);
    for (const [section, idx] of groups) {
      const name = cleanSection(section);
      if (!name) continue;
      const pts = idx.map((i) => this._px(stars[i].x, stars[i].y));
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
      const top = Math.min(...pts.map((p) => p[1]));
      const bottom = Math.max(...pts.map((p) => p[1]));
      const pages = idx.map((i) => stars[i].page).filter((p) => p != null);
      const range = pages.length ? (Math.min(...pages) === Math.max(...pages)
        ? `p. ${pages[0]}` : `pp. ${Math.min(...pages)}–${Math.max(...pages)}`) : "";
      const w = Math.max(textW(name, 7.6), textW(range, 6.6)) + 6, h = 34;
      const box = this._place([
        { x: cx - w / 2, y: top - h - 8, w, h },
        { x: cx - w / 2, y: bottom + 12, w, h },
        { x: Math.max(...pts.map((p) => p[0])) + 14, y: (top + bottom) / 2 - h / 2, w, h },
      ]);
      if (!box) continue;
      const tx = box.x + box.w / 2;
      const g = el("g", {}, this.layers.labels);
      const t = el("text", { class: "sec", x: tx, y: box.y + 14, "text-anchor": "middle" }, g);
      t.textContent = name;
      if (range) {
        const r = el("text", { class: "sec-pages", x: tx, y: box.y + 29, "text-anchor": "middle" }, g);
        r.textContent = range;
      }
      this._secLabels.push({ box, node: g });
    }
  }

  _drawRings(anim) {
    const stars = this.sky.stars;
    const rings = this.query.rings.filter((r) => r.i != null);
    // Claim each ring's footprint before any numeral is placed
    for (const ring of rings) {
      const s = stars[ring.i];
      const [x, y] = this._px(s.x, s.y);
      const rr = this._radius(ring.i) + 6;
      this.obstacles.push({ x: x - rr, y: y - rr, w: rr * 2, h: rr * 2 });
    }
    for (const ring of rings) {
      const s = stars[ring.i];
      const [x, y] = this._px(s.x, s.y);
      const rr = this._radius(ring.i) + 6;
      const c = el("circle", { class: `ring ${ring.cited ? "" : "uncited"}`, cx: x, cy: y, r: rr }, this.layers.rings);
      if (anim && ring.cited) {
        const C = 2 * Math.PI * rr;
        c.style.strokeDasharray = C;
        c.style.strokeDashoffset = C;
        c.style.transitionDelay = `${120 + ring.rank * 90}ms`;
        c.classList.add("draw");
        requestAnimationFrame(() => requestAnimationFrame(() => { c.style.strokeDashoffset = 0; }));
      } else if (anim) {
        c.style.opacity = 0;
        c.style.transition = `opacity .5s ${120 + ring.rank * 90}ms`;
        requestAnimationFrame(() => requestAnimationFrame(() => { c.style.opacity = 1; }));
      }
      const label = String(ring.rank);
      const w = textW(label, 8) + 4, h = 14, d = rr * 0.72 + 3;
      const box = this._place([
        { x: x + d, y: y - d - h, w, h },
        { x: x - d - w, y: y - d - h, w, h },
        { x: x + d, y: y + d, w, h },
        { x: x - d - w, y: y + d, w, h },
        { x: x + rr + 4, y: y - h / 2, w, h },
        { x: x - rr - 4 - w, y: y - h / 2, w, h },
      ]) || { x: x + d, y: y - d - h, w, h };
      const t = el("text", { class: `ring-label ${ring.cited ? "" : "uncited"}`, x: box.x + 2, y: box.y + 11 }, this.layers.rings);
      t.textContent = label;
    }
  }

  _drawMarks() {
    // Each method's mark prefers its own bearing around the star (TF-IDF up-left,
    // MinHash up-right, SimHash down-right, Hybrid down-left), so a chunk picked
    // by all four reads as a compass. Stars a few pixels apart share one fan, and
    // a mark that would land on a neighbour rotates or steps out on a short tick.
    const stars = this.sky.stars;
    const order = Object.keys(METHOD_META);
    const fans = [];
    for (const m of this.query.marks) {
      for (const i of m.items) {
        if (i == null) continue;
        const [x, y] = this._px(stars[i].x, stars[i].y);
        let fan = fans.find((f) => Math.hypot(f.x - x, f.y - y) < 9);
        if (!fan) fans.push(fan = { id: `fan${fans.length}`, x, y, r: 0, methods: new Set(), stars: [] });
        fan.r = Math.max(fan.r, this._radius(i));
        fan.methods.add(m.method);
        if (!fan.stars.some((s) => s.i === i)) fan.stars.push({ i, x, y, r: this._radius(i) });
      }
    }
    for (const f of fans) {
      for (const s of f.stars) this.obstacles.push({ x: s.x - s.r - 2, y: s.y - s.r - 2, w: 2 * s.r + 4, h: 2 * s.r + 4, owner: f.id });
    }
    fans.sort((a, b) => b.methods.size - a.methods.size);

    for (const f of fans) {
      for (const method of order.filter((m) => f.methods.has(m))) {
        const meta = METHOD_META[method];
        const base = f.r + 9;
        let spot = null;
        search:
        for (const dist of [base, base + 7, base + 15, base + 24, base + 34]) {
          for (const turn of [0, 25, -25, 50, -50, 80, -80, 120, -120, 180]) {
            const a = ((meta.angle + turn) * Math.PI) / 180;
            const mx = f.x + Math.cos(a) * dist, my = f.y + Math.sin(a) * dist;
            // Drawn extent: a 4.6 diamond reaches ±5.5 plus stroke; pad to ±8
            const box = this._place([{ x: mx - 8, y: my - 8, w: 16, h: 16 }], { ignore: f.id });
            if (box) { spot = { mx, my, a, dist }; break search; }
          }
        }
        if (!spot) {
          const a = (meta.angle * Math.PI) / 180;
          spot = { mx: f.x + Math.cos(a) * base, my: f.y + Math.sin(a) * base, a, dist: base };
        }
        if (spot.dist > base) {
          el("line", { class: "mark-tick", stroke: meta.color,
                       x1: f.x + Math.cos(spot.a) * (f.r + 2), y1: f.y + Math.sin(spot.a) * (f.r + 2),
                       x2: f.x + Math.cos(spot.a) * (spot.dist - 5), y2: f.y + Math.sin(spot.a) * (spot.dist - 5) }, this.layers.marks);
        }
        this.layers.marks.appendChild(shapeNode(meta.shape, spot.mx, spot.my, 4.6, { class: "mark", stroke: meta.color }));
      }
    }
  }

  _drawReticle(anim) {
    const p = this.query?.point;
    const g = el("g", { class: "reticle", opacity: p ? 1 : 0 }, this.layers.reticle);
    el("circle", { r: 11 }, g);
    el("circle", { class: "dot", r: 2.5 }, g);
    for (const [x1, y1, x2, y2] of [[0, -15, 0, -21], [0, 15, 0, 21], [-15, 0, -21, 0], [15, 0, 21, 0]]) {
      el("line", { x1, y1, x2, y2 }, g);
    }
    if (!p) return;
    const [x, y] = this._px(p[0], p[1]);
    this.obstacles.push({ x: x - 22, y: y - 22, w: 44, h: 44 });
    const w = 74, h = 16;
    const box = this._place([
      { x: x + 18, y: y - 30, w, h }, { x: x - 18 - w, y: y - 30, w, h },
      { x: x + 18, y: y + 16, w, h }, { x: x - 18 - w, y: y + 16, w, h },
    ]) || { x: x + 18, y: y - 30, w, h };
    const t = el("text", { x: box.x - x, y: box.y - y + 12 }, g);
    t.textContent = "your question";

    const from = this.lastPoint && anim ? this.lastPoint : [x, y];
    g.style.transform = `translate(${from[0]}px, ${from[1]}px)`;
    if (from[0] !== x || from[1] !== y) {
      requestAnimationFrame(() => requestAnimationFrame(() => { g.style.transform = `translate(${x}px, ${y}px)`; }));
    }
    this.lastPoint = [x, y];
  }

  _drawFocus() {
    if (!this.layers) return;
    const layer = this.layers.focus;
    layer.innerHTML = "";
    this.svg.querySelectorAll(".star.is-lit").forEach((n) => n.classList.remove("is-lit"));
    const i = this.lit;
    this.root.classList.toggle("has-focus", i != null && !!this.sky);
    for (const l of this._secLabels || []) l.node.classList.remove("is-hidden");
    if (i == null || !this.sky) return;
    this.svg.querySelector(`.star[data-i="${i}"]`)?.classList.add("is-lit");
    const s = this.sky.stars[i];
    const [x, y] = this._px(s.x, s.y);
    const rr = this._radius(i) + 12;
    el("circle", { class: "focus-ring", cx: x, cy: y, r: rr }, layer);

    // Leader line: from the ring out to a label shelf clear of other ink
    const name = cleanSection(s.section);
    const label = `${name} · p.${s.page ?? "?"}`;
    const lw = textW(label, 7.4) + 8, lh = 18;
    const shelves = [];
    for (const rise of [30, 54, -30, -54, 78]) {
      for (const dir of [1, -1]) {
        const ang = (rise > 0 ? -1 : 1) * Math.PI / 4;
        const sx = x + dir * Math.cos(ang) * rr, sy = y + Math.sin(ang) * rr;
        const ex = sx + dir * Math.abs(rise), ey = sy + (rise > 0 ? -1 : 1) * Math.abs(rise);
        const lx = dir > 0 ? ex + 24 : ex - 24 - lw;
        shelves.push({ dir, sx, sy, ex, ey, box: { x: lx, y: ey - lh / 2, w: lw, h: lh } });
      }
    }
    const fits = (b) => b.x >= 4 && b.y >= 2 && b.x + b.w <= this.w - 4 && b.y + b.h <= this.h - PAD.b;
    const shelf =
      shelves.find((c) => fits(c.box) && !this.obstacles.some((o) => o.owner == null && overlaps(o, c.box)
        && !this._secLabels?.some((l) => l.box === o))) ||
      shelves.find((c) => fits(c.box)) || shelves[0];
    const { dir, sx, sy, ex, ey, box } = shelf;
    for (const l of this._secLabels || []) l.node.classList.toggle("is-hidden", overlaps(l.box, box));
    el("polyline", { class: "leader", points: `${sx},${sy} ${ex},${ey} ${ex + dir * 18},${ey}` }, layer);
    const t = el("text", { class: "focus-label", x: ex + dir * 24, y: ey + 5, "text-anchor": dir > 0 ? "start" : "end" }, layer);
    t.textContent = label;
  }

  _showTip(i) {
    const s = this.sky.stars[i];
    const [x, y] = this._px(s.x, s.y);
    const ring = this.query?.rings?.find((r) => r.i === i);
    const sim = this.query?.field ? this.query.field[i] : null;
    const picks = this.query?.marks?.filter((m) => m.items.includes(i)).map((m) => METHOD_META[m.method].short);
    this.tip.innerHTML =
      `<b>${esc(s.section)}</b>` +
      `<span>Page <em class="num">${esc(s.page ?? "?")}</em> · <span class="num">${esc(s.id)}</span></span>` +
      (sim != null ? `<span>Similarity to your question <em class="num">${fmtScore(sim)}</em></span>` : "") +
      (ring ? `<span>Rank <em class="num">${ring.rank}</em> · ${ring.cited ? "cited in the answer" : "retrieved, not cited"}</span>` : "") +
      (picks?.length ? `<span>Picked by ${esc(picks.join(", "))}</span>` : "");
    const tipW = 280;
    const left = x + 16 + tipW > this.w ? x - 16 - tipW : x + 16;
    this.tip.style.left = `${Math.max(8, left)}px`;
    this.tip.style.top = `${Math.max(8, Math.min(this.h - 120, y - 12))}px`;
    this.tip.classList.add("is-on");
  }

  _hideTip() { this.tip.classList.remove("is-on"); }

  // ── Key band (below the chart, so it never covers a star) ─────────────

  _renderBand() {
    const band = this.band;
    if (!this.sky) { band.hidden = true; return; }
    band.hidden = false;
    const q = this.query;
    const caption =
      `<div class="band-cap"><span class="num">${this.sky.n_chunks}</span> chunks · <span class="num">${this.sky.n_sections}</span> sections` +
      `<br>TF-IDF space, projected to two components (SVD, then PCA)</div>`;
    const reticle =
      `<span class="key"><svg viewBox="-10 -10 20 20"><circle r="6" fill="none" stroke="var(--verm)" stroke-width="1.5"/>` +
      `<circle r="1.8" fill="var(--verm)"/></svg>Your question</span>`;

    let keys;
    if (!q) {
      keys = `<div class="band-group"><h3 class="band-label">Reading the sky</h3>` +
        `<p>Each star is one chunk, placed by its vocabulary. Dotted lines join chunks of the same section. Ask a question to light the ones that match.</p></div>`;
    } else if (q.marks) {
      keys = `<div class="band-group"><h3 class="band-label">Methods</h3>` +
        q.marks.map((m) => `<span class="key">${methodGlyph(m.method, 16)}${METHOD_META[m.method].name}</span>`).join("") +
        reticle + `</div>`;
    } else {
      const max = this._fieldMax || 0;
      const steps = [1, 0.75, 0.5, 0.25, 0.05];
      keys =
        `<div class="band-group"><h3 class="band-label">Magnitude</h3>` +
        steps.map((t) => {
          const d = 2 * (1.2 + 7.4 * Math.pow(t, 0.8));
          return `<span class="mag"><i style="width:${d}px;height:${d}px;opacity:${0.28 + 0.72 * t}"></i><span class="num">${fmtScore(max * t)}</span></span>`;
        }).join("") +
        `<span class="band-note">TF-IDF similarity to your question</span></div>` +
        `<div class="band-group">` +
        `<span class="key"><svg viewBox="-10 -10 20 20"><circle r="7" fill="none" stroke="var(--verm)" stroke-width="1.5"/></svg>Cited source</span>` +
        `<span class="key"><svg viewBox="-10 -10 20 20"><circle r="7" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="3 3"/></svg>Retrieved, not cited</span>` +
        reticle + `</div>`;
    }
    band.innerHTML = `<i class="node" style="left:0;top:0"></i>${keys}${caption}`;
  }
}
