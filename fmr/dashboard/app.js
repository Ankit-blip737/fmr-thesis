/* FMR dashboard — SPA, thesis-defense grade. Zero deps; renders window.FMR_DATA.
   Reuses the existing data bundle; adds tab routing, dark mode, hover tooltips,
   entrance animation, honest small-n treatment, and new tabs. */
(function () {
  "use strict";
  const DATA = window.FMR_DATA || { sources: {} };
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (h) => { const t = document.createElement("template"); t.innerHTML = h.trim(); return t.content.firstChild; };
  const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
  const pct = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (100 * x).toFixed(1) + "%";
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const SMALL_N = 50;   // below this, real curves are drawn provisionally (dashed/muted)

  // Colors read from CSS vars so charts follow light/dark theme.
  function cssv(name, fb) { const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v || fb; }
  function COLORS() {
    return {
      primary: cssv("--primary", "#0e7490"), secondary: cssv("--secondary", "#6d28d9"),
      grounded: cssv("--grounded", "#15a34a"), ungrounded: cssv("--ungrounded", "#dc2626"),
      abstain: cssv("--abstain", "#e08600"), a: cssv("--a", "#4f46e5"), b: cssv("--b", "#0e7490"),
      c: cssv("--c", "#db2777"), ink: cssv("--ink", "#0f1e2e"), faint: cssv("--ink-faint", "#8195a8"),
      line: cssv("--line", "#e3e9f0"), surface: cssv("--surface", "#fff"),
    };
  }

  /* ---------------------- chart tooltip ---------------------- */
  const tip = () => $("#chart-tip");
  function bindTips(container) {
    $$("[data-tip]", container).forEach(node => {
      node.addEventListener("mouseenter", (e) => {
        const t = tip(); t.innerHTML = node.getAttribute("data-tip"); t.style.opacity = "1";
      });
      node.addEventListener("mousemove", (e) => {
        const t = tip(); t.style.left = (e.clientX + 14) + "px"; t.style.top = (e.clientY - 10) + "px";
      });
      node.addEventListener("mouseleave", () => { tip().style.opacity = "0"; });
    });
  }

  /* ---------------------- SVG primitives ---------------------- */
  function frame(w, h, pad) {
    const p = Object.assign({ t: 16, r: 16, b: 34, l: 44 }, pad || {});
    return { w, h, p, iw: w - p.l - p.r, ih: h - p.t - p.b,
      open: `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">`, close: `</svg>` };
  }
  let _uid = 0;
  function axes(f, { xTicks = [], yTicks = [], xlab = "", ylab = "" }, C) {
    let s = "";
    yTicks.forEach(t => { const y = f.p.t + f.ih - t.f * f.ih;
      s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="${C.line}" stroke-dasharray="1 5" stroke-linecap="round" opacity="0.9"/>`;
      s += `<text x="${f.p.l - 6}" y="${y + 3}" text-anchor="end" font-size="10" fill="${C.faint}">${t.label}</text>`; });
    xTicks.forEach(t => { const x = f.p.l + t.f * f.iw;
      s += `<text x="${x}" y="${f.p.t + f.ih + 15}" text-anchor="middle" font-size="10" fill="${C.faint}">${t.label}</text>`; });
    if (ylab) s += `<text x="12" y="${f.p.t + f.ih / 2}" transform="rotate(-90 12 ${f.p.t + f.ih / 2})" text-anchor="middle" font-size="10.5" fill="${C.faint}">${ylab}</text>`;
    if (xlab) s += `<text x="${f.p.l + f.iw / 2}" y="${f.h - 2}" text-anchor="middle" font-size="10.5" fill="${C.faint}">${xlab}</text>`;
    return s;
  }
  function yScale(vals, lo, hi) {
    let mn = lo !== undefined ? lo : Math.min(...vals), mx = hi !== undefined ? hi : Math.max(...vals);
    if (mn === mx) { mx += 1; mn -= 1; } return { lo: mn, hi: mx, f: v => (v - mn) / (mx - mn) };
  }
  function yticks(sc, n = 4) { const o = []; for (let i = 0; i <= n; i++) { const v = sc.lo + (sc.hi - sc.lo) * i / n; o.push({ f: i / n, label: v.toFixed(2) }); } return o; }

  // series: [{name,color,points:[[x,y]],dash,width,provisional}]
  function lineChart(series, opt = {}) {
    const C = COLORS(), f = frame(opt.w || 480, opt.h || 260);
    const all = series.flatMap(s => s.points.map(p => p[1]));
    const sc = yScale(all, opt.yMin, opt.yMax);
    const xs = series.flatMap(s => s.points.map(p => p[0]));
    const xlo = Math.min(...xs), xhi = Math.max(...xs);
    const fx = x => f.p.l + (xhi === xlo ? .5 : (x - xlo) / (xhi - xlo)) * f.iw;
    const fy = y => f.p.t + f.ih - sc.f(y) * f.ih;
    const xTicks = (opt.xTicks || [...new Set(xs)]).map(x => ({ f: xhi === xlo ? .5 : (x - xlo) / (xhi - xlo), label: opt.xFmt ? opt.xFmt(x) : x }));
    const gid = "g" + (++_uid);
    let s = f.open + `<defs><filter id="glow${gid}" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="${(series[0]||{}).color||C.primary}" flood-opacity="0.35"/></filter></defs>`;
    s += axes(f, { xTicks, yTicks: yticks(sc), xlab: opt.xlab, ylab: opt.ylab }, C);
    if (opt.hline !== undefined) { const y = fy(opt.hline); s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="${opt.hlineColor || C.ink}" stroke-dasharray="3 4" opacity="0.7"/>`; }
    series.forEach((se, si) => {
      const d = se.points.map((p, i) => (i ? "L" : "M") + fx(p[0]) + " " + fy(p[1])).join(" ");
      const op = se.provisional ? 0.7 : 1;
      const len = 1600, primary = !se.provisional && (se.width || 2) >= 2.2;
      const glow = primary ? `filter="url(#glow${gid})"` : "";
      const dashAttr = se.provisional ? `stroke-dasharray="7 4"` : `stroke-dasharray="${len}" stroke-dashoffset="${len}"`;
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" opacity="${op}" stroke-linecap="round" stroke-linejoin="round" ${glow} ${dashAttr}>`;
      if (!se.provisional && opt.animate !== false) s += `<animate attributeName="stroke-dashoffset" from="${len}" to="0" dur="0.8s" fill="freeze"/>`;
      s += `</path>`;
      if (se.dots !== false) se.points.forEach(p => {
        const tt = `<div class='tt-h'>${esc(se.name)}</div>${opt.xlab ? opt.xlab : 'x'} ${opt.xFmt ? opt.xFmt(p[0]) : (+p[0]).toFixed(2)} · ${(opt.ylab || 'y')} ${(+p[1]).toFixed(3)}`;
        // ring dot: colored ring with a surface-colored core (crisper, more premium)
        s += `<circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="${se.provisional ? 2.6 : 3.4}" fill="${C.surface}" stroke="${se.color}" stroke-width="${se.provisional ? 1.6 : 2}" opacity="${op}"/>`;
        s += `<circle class="pt" data-tip="${esc(tt)}" cx="${fx(p[0])}" cy="${fy(p[1])}" r="10" fill="transparent" style="cursor:pointer"/>`;
      });
    });
    (opt.marks || []).forEach(m => { s += `<path class="pt" data-tip="${esc(m.tip || '')}" d="M${fx(m.x)} ${fy(m.y)} l-6 -11 l12 0 z" fill="${m.color || C.ungrounded}"/>`; });
    return s + f.close;
  }

  function barChart(categories, series, opt = {}) {
    const C = COLORS(), f = frame(opt.w || 480, opt.h || 260, opt.pad);
    const all = series.flatMap(s => s.values.filter(v => v != null && !Number.isNaN(v)));
    const sc = yScale(all.concat([opt.baseline0 === false ? all[0] || 0 : 0]), opt.yMin, opt.yMax);
    const n = categories.length, g = series.length, bandW = f.iw / n, barW = Math.min(46, (bandW * .72) / g);
    const fy = y => f.p.t + f.ih - sc.f(y) * f.ih;
    const xTicks = categories.map((c, i) => ({ f: (i + .5) / n, label: c }));
    // one vertical gradient per unique bar color (top brighter → base color)
    const gid = "b" + (++_uid), colorsUsed = new Set();
    series.forEach((se, j) => categories.forEach((_, i) => colorsUsed.add((se.colors && se.colors[i]) || se.color)));
    const gradOf = new Map(); let defs = "";
    [...colorsUsed].forEach((col, k) => { const id = `${gid}_${k}`; gradOf.set(col, id);
      defs += `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${col}" stop-opacity="1"/><stop offset="1" stop-color="${col}" stop-opacity="0.62"/></linearGradient>`; });
    let s = f.open + `<defs>${defs}</defs>` + axes(f, { xTicks, yTicks: yticks(sc), ylab: opt.ylab }, C);
    if (opt.hline !== undefined) { const y = fy(opt.hline); s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="${opt.hlineColor || C.faint}" stroke-dasharray="4 3" opacity="0.75"/>`; }
    categories.forEach((c, i) => { const cx = f.p.l + bandW * (i + .5);
      series.forEach((se, j) => { const v = se.values[i]; if (v == null || Number.isNaN(v)) return;
        const x = cx - (g * barW) / 2 + j * barW, y = fy(Math.max(v, sc.lo)), h = Math.max(0, fy(sc.lo) - y);
        const col = (se.colors && se.colors[i]) || se.color;
        const tt = `<div class='tt-h'>${esc(se.name)}</div>${esc(c)}: ${fmt(v, opt.labelDigits ?? 3)}`;
        s += `<rect class="pt" data-tip="${esc(tt)}" x="${x}" y="${y}" width="${barW - 2}" height="0" rx="4" fill="url(#${gradOf.get(col)})" style="cursor:pointer">`;
        if (opt.animate !== false) s += `<animate attributeName="height" from="0" to="${h}" dur="0.6s" fill="freeze"/><animate attributeName="y" from="${fy(sc.lo)}" to="${y}" dur="0.6s" fill="freeze"/>`;
        else s += `<set attributeName="height" to="${h}"/>`;
        s += `</rect>`;
        if (opt.valueLabels) s += `<text x="${x + barW / 2 - 1}" y="${y - 4}" text-anchor="middle" font-size="9.5" fill="${C.faint}">${fmt(v, opt.labelDigits ?? 2)}</text>`;
      });
    });
    return s + f.close;
  }

  function histogram(groups, opt = {}) {
    const C = COLORS(), f = frame(opt.w || 480, opt.h || 260);
    const bins = opt.bins || 22, bw = 1 / bins;
    const counts = groups.map(gr => { const arr = new Array(bins).fill(0);
      gr.values.forEach(v => { let k = Math.floor(v / bw); if (k < 0) k = 0; if (k >= bins) k = bins - 1; arr[k]++; });
      const tot = gr.values.length || 1; return arr.map(x => x / tot); });
    const maxY = Math.max(.001, ...counts.flat()), sc = yScale([0, maxY], 0, maxY), fy = y => f.p.t + f.ih - sc.f(y) * f.ih;
    let s = f.open + axes(f, { xTicks: [0, .25, .5, .75, 1].map(x => ({ f: x, label: x })), yTicks: [], xlab: opt.xlab, ylab: "density" }, C);
    groups.forEach((gr, gi) => { let d = `M${f.p.l} ${fy(0)}`;
      counts[gi].forEach((cc, k) => { const x0 = f.p.l + (k / bins) * f.iw, x1 = f.p.l + ((k + 1) / bins) * f.iw; d += ` L${x0} ${fy(cc)} L${x1} ${fy(cc)}`; });
      d += ` L${f.p.l + f.iw} ${fy(0)} Z`;
      s += `<path d="${d}" fill="${gr.color}" fill-opacity="0.4" stroke="${gr.color}" stroke-width="1.3"/>`; });
    return s + f.close;
  }
  function legend(items) { return `<div class="legend">` + items.map(i => `<span><span class="sw ${i.dash ? 'dash' : ''}" style="${i.dash ? 'color:' + i.color : 'background:' + i.color}"></span>${esc(i.label)}</span>`).join("") + `</div>`; }

  /* ---------------------- small-n honesty ---------------------- */
  function srcN(src) { const fr = src.fmr_results || {}; return fr.n_test != null ? fr.n_test : (src.records ? src.records.length : null); }
  function sampleBadge(src, n) {
    if (src.kind !== "real") return `<span class="samplesize mock" title="synthetic machinery validation">mock</span>`;
    const cls = (n != null && n < SMALL_N) ? "small" : "real";
    return `<span class="samplesize ${cls}" title="${n} real test samples">n=${n ?? "?"}${n != null && n < SMALL_N ? " · small" : ""}</span>`;
  }
  function isProvisional(src, n) { return src.kind === "real" && n != null && n < SMALL_N; }
  function tagInto(id, src) { const n = srcN(src); const t = $("#" + id); if (t) t.outerHTML = `<span id="${id}">${sampleBadge(src, n)}</span>`; }
  function markMock(container, src) { if (container) container.classList.toggle("mockmark", src.kind !== "real"); }

  /* ---------------------- data helpers ---------------------- */
  const STATE = { source: DATA.default_source, filter: "all", caseIdx: null, sepSig: "fs", search: "", alpha: 0.10, view: "overview" };
  function activeSource() { return DATA.sources[STATE.source] || {}; }
  function mockSource() { return DATA.sources["mock"] || {}; }

  /* ==================================================================
     TAB: Overview
     ================================================================== */
  // compact inline stroke icons (24x24, currentColor)
  const ICON = {
    image: '<path d="M3 5h18v14H3z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="10" r="1.8" fill="currentColor"/><path d="M21 17l-5-5-4 4-2-2-4 4" fill="none" stroke="currentColor" stroke-width="2"/>',
    signals: '<path d="M4 20V10M10 20V4M16 20v-8M22 20V7" stroke="currentColor" stroke-width="2.4" fill="none" stroke-linecap="round"/>',
    gauge: '<path d="M12 13l5-4" stroke="currentColor" stroke-width="2.2" fill="none" stroke-linecap="round"/><path d="M4 18a8 8 0 1 1 16 0" stroke="currentColor" stroke-width="2.2" fill="none" stroke-linecap="round"/>',
    wrench: '<path d="M14.5 6a3.5 3.5 0 0 0-4.6 4.3l-6 6 2.8 2.8 6-6A3.5 3.5 0 0 0 18 8.6l-2.3 2.3-1.8-1.8L16.2 6.8A3.5 3.5 0 0 0 14.5 6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>',
    shield: '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M9 12l2 2 4-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    branch: '<path d="M7 4v6a4 4 0 0 0 4 4h6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="7" cy="4" r="2.2" fill="currentColor"/><circle cx="18" cy="14" r="2.2" fill="currentColor"/>',
    trend: '<path d="M3 17l6-6 4 4 8-8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M15 7h6v6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    target: '<circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="2"/>',
    filter: '<path d="M3 5h18l-7 8v5l-4 2v-7z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
    db: '<ellipse cx="12" cy="6" rx="7" ry="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" fill="none" stroke="currentColor" stroke-width="2"/>',
    arrow: '<path d="M5 12h13M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  };
  const svgIc = (name, sz = 24) => `<svg viewBox="0 0 24 24" width="${sz}" height="${sz}" aria-hidden="true">${ICON[name] || ""}</svg>`;

  function miniFlow() {
    const C = COLORS();
    const nodes = [
      { t: "Image + Q", ic: "image", c: C.faint }, { t: "3 Signals", ic: "signals", c: C.a },
      { t: "Fused FS", ic: "gauge", c: C.b }, { t: "Correct", ic: "wrench", c: C.abstain },
      { t: "Abstain gate", ic: "shield", c: C.grounded }, { t: "Answer / Defer", ic: "branch", c: C.primary }];
    let s = `<div class="mflow">`;
    nodes.forEach((nd, i) => {
      s += `<div class="mflow-node" style="--mc:${nd.c}"><span class="m-ic">${svgIc(nd.ic, 18)}</span><span class="m-t">${nd.t}</span></div>`;
      if (i < nodes.length - 1) s += `<div class="mflow-arrow">${svgIc("arrow", 20)}</div>`;
    });
    return s + `</div>`;
  }
  function countUp(node, target, d = 3, suffix = "") {
    if (target == null || Number.isNaN(target)) { node.textContent = "—"; return; }
    const start = performance.now(), dur = 700;
    function step(t) { const k = Math.min(1, (t - start) / dur); const e = 1 - Math.pow(1 - k, 3);
      node.textContent = (target * e).toFixed(d) + suffix; if (k < 1) requestAnimationFrame(step); }
    requestAnimationFrame(step);
  }
  function renderOverview(src) {
    const fr = src.fmr_results || {}, bt = src.blind_test || {}, val = fr.validation || {}, ab = fr.abstention || {};
    const gate = (ab.fs && ab.fs.test) || {}, rep = bt.replication || {}, n = srcN(src);
    const head = (rep.primary_evidence === "blind_gap" || rep.drift_available === false)
      ? { k: "Image-reliance gap", v: rep.blind_gap_reasoning, d: 3, s: rep.blind_gap_supports ? `< non-reasoning ${fmt(rep.blind_gap_nonreasoning, 3)} ✓` : (rep.both_image_independent ? "both image-independent" : "vs non-reasoning") }
      : { k: "Grounding drift slope", v: rep.drift_slope, d: 3, s: rep.replicated ? "decays along chain ✓" : "no decay" };
    const C = COLORS();
    head.ic = "trend"; head.accent = C.secondary; head.headline = true;
    const cards = [
      head,
      { k: "Fused-FS separation", ic: "target", accent: C.grounded, v: val.auroc_fs != null ? val.auroc_fs : null, d: 3, s: val.auroc_fs != null ? "AUROC vs grounding" : "no grounding labels (real)" },
      { k: "Answered @ gate", ic: "filter", accent: C.primary, v: gate.coverage != null ? gate.coverage : null, d: 2, pctv: true, s: gate.coverage ? `err ${pct(gate.retained_error)} · α=${fmt(ab.alpha, 2)}` : `abstain-all · α=${fmt(ab.alpha, 2)}` },
      { k: "Test cases", ic: "db", accent: C.abstain, v: n, d: 0, s: (src.kind === "real" ? "real" : "synthetic") + " dataset" },
    ];
    $("#stat-cards").innerHTML = cards.map(c => `<div class="card stat hoverable ${c.headline ? "headline" : ""}" style="--accent:${c.accent}">
      <div class="k"><span class="stat-ic">${svgIc(c.ic, 15)}</span>${c.k}</div>
      <div class="v" data-target="${c.v ?? ""}" data-d="${c.d}" data-pct="${c.pctv ? 1 : 0}">—</div>
      <div class="s">${c.s}</div></div>`).join("");
    $$("#stat-cards .v").forEach(node => { const tg = node.getAttribute("data-target");
      if (tg === "" || tg === "null") { node.textContent = "—"; return; }
      const isPct = node.getAttribute("data-pct") === "1"; countUp(node, isPct ? +tg * 100 : +tg, isPct ? 1 : +node.getAttribute("data-d"), isPct ? "%" : ""); });
    $("#mini-flow").innerHTML = miniFlow();
    $("#provenance").textContent = `Source: ${src.label || "—"} · dataset "${fr.dataset || bt.dataset || "?"}" · bundle generated ${DATA.generated_at || "?"}`;

    const cav = $("#data-caveat");
    if (src.kind === "real") {
      const integ = src.adapter_integration || {}, notes = [];
      if (n != null && n < SMALL_N) notes.push(`small real test set (n=${n}) — numbers are noisy`);
      if (integ.schema && integ.schema.signal_b_constant) notes.push("Signal B constant (real attention extraction pending) — fusion uses A+C");
      if (ab.fs && !ab.fs.feasible) notes.push(`conformal gate infeasible at α=${fmt(ab.alpha, 2)} on this set`);
      cav.className = "caveat real"; cav.innerHTML = `<span class="icon">🟢</span><div><b>REAL model data</b> (${esc(src.label)}). ${notes.length ? "Caveats: " + notes.join("; ") + "." : ""}</div>`;
    } else {
      cav.className = "caveat mock"; cav.innerHTML = `<span class="icon">🟡</span><div><b>MOCK / synthetic data</b> — validates the machinery on a known grounding latent. Not a real-model result; empirical claims come from the real sources.</div>`;
    }
  }

  /* ==================================================================
     TAB: Methodology
     ================================================================== */
  const FLOW = [
    { id: "in", t: "Input", d: "Medical image + clinical question.", cls: "" },
    { id: "gen", t: "Base VLM", d: "A frozen reasoning VLM generates a chain-of-thought + answer. FMR never fine-tunes it.", cls: "" },
    { id: "fmm", t: "① Measure (FMM)", d: "Three independent signals — A counterfactual sensitivity, B attention grounding, C self-consistency — fuse into a Faithfulness Score.", cls: "pillar1" },
    { id: "corr", t: "② Correct", d: "When FS is low, training-free Visual Contrastive Decoding + clue-tracing + verify/revise re-anchor the reasoning to the image.", cls: "pillar2" },
    { id: "gate", t: "③ Abstain gate", d: "Split-conformal calibration on the (post-correction) FS gives a distribution-free bound: answer only if FS ≥ τ, else defer.", cls: "pillar3" },
    { id: "out", t: "Output", d: "A grounded answer, or ABSTAIN → defer to a clinician.", cls: "" },
  ];
  function renderMethodology() {
    let s = "";
    FLOW.forEach((nd, i) => { s += `<div class="flow-node ${nd.cls}" data-fid="${nd.id}"><div class="n-t">${nd.t}</div><div class="n-d">${nd.d.slice(0, 42)}…</div></div>`;
      if (i < FLOW.length - 1) s += `<div class="flow-arrow">→</div>`; });
    $("#flow-diagram").innerHTML = s;
    $$("#flow-diagram .flow-node").forEach(node => node.onclick = () => {
      $$("#flow-diagram .flow-node").forEach(x => x.classList.remove("sel")); node.classList.add("sel");
      const nd = FLOW.find(f => f.id === node.getAttribute("data-fid"));
      $("#flow-detail").innerHTML = `<b>${nd.t}</b> — ${nd.d}`;
    });
    const pillars = [
      { n: "Pillar 1 — Measurement", c: "--a", d: "Multi-signal Faithfulness Score. Never trusts one signal: counterfactual + attention + consistency fuse, validated against grounding labels where boxes exist." },
      { n: "Pillar 2 — Correction", c: "--abstain", d: "Training-free, selective. Only fires when FS is low; re-anchors reasoning to the image (VCD + clue tracing + verify/revise). Frozen base is the fallback." },
      { n: "Pillar 3 — Abstention", c: "--grounded", d: "The safety contribution. Split-conformal gate on the deployed FS controls selective risk with a distribution-free guarantee; defers what it can't verify." },
    ];
    $("#pillars").innerHTML = pillars.map(p => `<div class="card hoverable" style="border-top:3px solid var(${p.c})"><div class="card-h">${p.n}</div><p class="lead" style="font-size:13px">${p.d}</p></div>`).join("");
  }

  /* ==================================================================
     TAB: Diagnosis
     ================================================================== */
  function renderDiagnosis(src) {
    const C = COLORS(), bt = src.blind_test || {}, rep = bt.replication || {}, n = srcN(src);
    tagInto("drift-src", src); tagInto("blind-src", src);
    const vEl = $("#replication-verdict");
    if (rep.tested) {
      vEl.className = "verdict " + (rep.replicated ? "yes" : "no");
      let sub = rep.note || "";
      if (rep.primary_evidence === "blind_gap" && rep.blind_gap_reasoning != null)
        sub = `${rep.reasoning_model}: image-reliance gap ${fmt(rep.blind_gap_reasoning, 3)} vs non-reasoning ${fmt(rep.blind_gap_nonreasoning, 3)}` + (rep.accuracy_confound ? " — ⚠ confounded by lower reasoning-model accuracy" : "") + (rep.drift_available ? "" : "; per-step drift pending attention instrumentation");
      else if (rep.primary_evidence === "drift") sub = `${rep.reasoning_model} · drift slope ${fmt(rep.drift_slope, 4)} (decays along the chain)`;
      const head = rep.replicated ? (rep.primary_evidence === "drift" ? "Headline REPLICATED" : "Headline SUPPORTED") : "Headline NOT supported — reporting the actual effect";
      vEl.innerHTML = `<span class="icon">${rep.replicated ? "✓" : "✕"}</span><div><div>${head} <span style="font-weight:500;opacity:.7">(${rep.primary_evidence} lens)</span></div><div class="sub">${esc(sub)}</div></div>`;
    } else { vEl.className = ""; vEl.innerHTML = ""; }

    const models = bt.models || {}, series = [], prov = isProvisional(src, n);
    Object.values(models).forEach(m => { const curve = m.iou_vs_step_index || {}, ks = Object.keys(curve).map(Number).sort((a, b) => a - b);
      if (!ks.length) return; series.push({ name: m.name, color: m.is_reasoning ? C.primary : C.faint, dash: !m.is_reasoning, points: ks.map(k => [k + 1, curve[String(k)]]) }); });
    markMock($("#chart-drift"), src);
    $("#chart-drift").innerHTML = series.length
      ? lineChart(series, { yMin: 0, ylab: "mean IoU", xlab: "reasoning step", xFmt: x => "s" + x }) + legend(series.map(s => ({ label: s.name, color: s.color })))
      : `<p class="empty">No per-step grounding curve for this source. ${src.kind === "real" ? "Real per-step attention→region extraction is pending (Signal B currently constant); this panel activates once it lands." : ""}</p>`;

    const variants = ["original", "blank", "mismatch"], mk = Object.keys(models);
    markMock($("#chart-blind"), src);
    $("#chart-blind").innerHTML = mk.length
      ? barChart(variants, mk.map((k, i) => ({ name: models[k].name, color: [C.primary, C.abstain][i] || C.faint, values: variants.map(v => models[k].accuracy ? models[k].accuracy[v] : null) })), { yMin: 0, yMax: 1, ylab: "accuracy", valueLabels: true }) + legend(mk.map((k, i) => ({ label: models[k].name, color: [C.primary, C.abstain][i] || C.faint })))
      : `<p class="empty">No blind-test data.</p>`;
    renderChainLen(src);
  }

  /* ==================================================================
     TAB: Measurement
     ================================================================== */
  const TIP_AUROC = "AUROC: probability the score ranks a random grounded case above a random ungrounded one. 0.5 = chance, 1.0 = perfect.";
  const TIP_IOU = "IoU: overlap between the model's attended region and the ground-truth evidence box (0–1).";
  const TIPS = { A: "Signal A — counterfactual sensitivity: does the answer change when the image is removed/swapped?", B: "Signal B — attention grounding: does the reasoning attend to the true evidence region?", C: "Signal C — self-consistency: do resampled reasoning chains agree?" };
  function renderMeasurement(src) {
    const C = COLORS(), val = (src.fmr_results || {}).validation || {};
    tagInto("auroc-src", src);
    const keys = [["signal_a", "A", C.a], ["signal_b", "B", C.b], ["signal_c", "C", C.c], ["fs", "Fused FS", C.grounded], ["confidence", "Confidence", C.abstain]];
    const cats = keys.map(k => k[1]), vals = keys.map(k => val["auroc_" + k[0]]);
    markMock($("#chart-auroc"), src);
    $("#chart-auroc").innerHTML = vals.some(v => v != null)
      ? barChart(cats, keys.map((k, i) => ({ name: k[1], color: k[2], values: cats.map((_, j) => j === i ? val["auroc_" + k[0]] : null) })), { yMin: 0.5, yMax: 1, ylab: "AUROC", valueLabels: true, hline: 0.5 })
      : `<p class="empty">No grounding labels for this source — AUROC needs the hidden latent (synthetic) or GT boxes. On real data (no boxes) this stays n/a; see the Case Explorer for per-case signals.</p>`;
    renderSeparation(src);
    renderFSAcc(src); renderCorrectWrong(src);
  }
  function renderSeparation(src) {
    const C = COLORS(), recs = src.records || [], labelled = recs.filter(r => r.grounded_latent === 0 || r.grounded_latent === 1);
    const box = $("#chart-sep"), toggle = $("#sep-toggle");
    if (!labelled.length) { toggle.innerHTML = ""; box.innerHTML = `<p class="empty">Grounding labels unavailable for this source — separation view needs them (synthetic only).</p>`; return; }
    const sigs = [["signal_a", "Signal A"], ["signal_b", "Signal B"], ["signal_c", "Signal C"], ["fs", "Fused FS"]];
    const cur = STATE.sepSig || "fs";
    toggle.innerHTML = sigs.map(s => `<button data-sig="${s[0]}" class="${s[0] === cur ? "active" : ""}">${s[1]}</button>`).join("");
    const draw = sg => { const g = labelled.filter(r => r.grounded_latent === 1).map(r => r[sg]), u = labelled.filter(r => r.grounded_latent === 0).map(r => r[sg]);
      markMock(box, src);
      box.innerHTML = histogram([{ name: "ungrounded", color: C.ungrounded, values: u }, { name: "grounded", color: C.grounded, values: g }], { xlab: (sigs.find(x => x[0] === sg) || [])[1] }) + legend([{ label: "grounded", color: C.grounded }, { label: "ungrounded", color: C.ungrounded }]); };
    toggle.querySelectorAll("button").forEach(b => b.onclick = () => { STATE.sepSig = b.dataset.sig; toggle.querySelectorAll("button").forEach(x => x.classList.remove("active")); b.classList.add("active"); draw(STATE.sepSig); bindTips(box); });
    draw(cur);
  }

  /* ==================================================================
     TAB: Safety
     ================================================================== */
  function renderSafety(src) {
    const C = COLORS(), ab = (src.fmr_results || {}).abstention || {}, n = srcN(src), prov = isProvisional(src, n);
    tagInto("rc-src", src); tagInto("mod-src", src);
    const defs = [["fs", "Faithfulness Score", C.grounded, 2.6], ["confidence", "Answer confidence", C.abstain, 1.4],
      ["signal_a_only", "Signal A", C.a, 1], ["signal_b_only", "Signal B", C.b, 1], ["signal_c_only", "Signal C", C.c, 1]];
    const series = [];
    defs.forEach(d => { const o = ab[d[0]]; if (o && o.risk_coverage && o.risk_coverage.coverage) {
      series.push({ name: `${d[1]}${o.degenerate ? " (constant)" : ""} · AURC ${fmt(o.aurc, 3)}`, color: d[2], width: d[3], dots: true, provisional: prov || o.degenerate,
        points: o.risk_coverage.coverage.map((c, i) => [c, o.risk_coverage.risk[i]]) }); } });
    const gate = ab.fs && ab.fs.test;
    const marks = (gate && gate.n_retained && gate.coverage != null && gate.retained_error != null) ? [{ x: gate.coverage, y: gate.retained_error, color: C.ungrounded, tip: `calibrated gate · cov ${pct(gate.coverage)} · err ${pct(gate.retained_error)}` }] : [];
    markMock($("#chart-rc"), src);
    $("#chart-rc").innerHTML = series.length
      ? lineChart(series, { yMin: 0, yMax: 1, xlab: "coverage", ylab: "risk (error on answered)", hline: ab.alpha, hlineColor: C.ink, marks, xTicks: [0, .25, .5, .75, 1], animate: !prov }) + legend(series.map(s => ({ label: s.name, color: s.color, dash: prov })).concat([{ label: `target α=${fmt(ab.alpha, 2)}`, color: C.ink }]))
      : `<p class="empty">No risk–coverage data.</p>`;
    renderAlphaSlider(src, ab);

    // guarantee card
    const g = ab.fs || {}, t = g.test || {}, hold = g.feasible ? (t.retained_error == null || t.retained_error <= (ab.alpha ?? 1)) : true;
    const gc = $("#guarantee-card"); gc.className = "card guarantee " + (g.feasible ? (hold ? "hold" : "fail") : "fail");
    gc.innerHTML = `<h3>Distribution-free guarantee ${sampleBadge(src, n)}</h3>
      <div class="row"><span>Target error α</span><b>${fmt(ab.alpha, 2)}</b></div>
      <div class="row"><span>Threshold τ (FS ≥)</span><b>${g.threshold == null ? "—" : (g.threshold === Infinity || g.threshold > 1e6 ? "∞" : fmt(g.threshold, 3))}</b></div>
      <div class="row"><span>Coverage (answered)</span><b>${pct(t.coverage)}</b></div>
      <div class="row"><span>Error on answered</span><b>${pct(t.retained_error)}</b></div>
      <div class="row"><span>Feasible / holds</span><b>${g.feasible ? "yes" : "no"} / ${hold ? "✓" : "✕"}</b></div>
      ${!g.feasible ? `<div class="gate-flag abstain">⚠ Gate INFEASIBLE at α=${fmt(ab.alpha, 2)} → the safe output is <b>abstain-all</b> (need more calibration data). Shown, not hidden.</div>` : ""}
      ${ab.provisional_pre_correction ? `<p class="note" style="margin-top:8px">Provisional: pre-correction FS.</p>` : ""}`;

    const pm = (src.fmr_results || {}).per_modality || {}, mk = Object.keys(pm);
    const label = m => m === "unknown" ? "unspec.*" : m;
    markMock($("#chart-modality"), src);
    $("#chart-modality").innerHTML = mk.length
      ? barChart(mk.map(label), [{ name: "accuracy", color: C.primary, values: mk.map(m => pm[m].accuracy) }, { name: "mean FS", color: C.grounded, values: mk.map(m => pm[m].mean_fs) }], { yMin: 0, yMax: 1 }) + legend([{ label: "accuracy", color: C.primary }, { label: "mean FS", color: C.grounded }]) + (mk.includes("unknown") ? `<p class="note">*"unspec." = modality metadata absent in the VQA-RAD mirror (not a bucket); SLAKE carries real ct/mri/xray labels.</p>` : "")
      : `<p class="empty">No per-modality breakdown for this source.</p>`;
    renderBaselines(src);
    renderAURC(src); renderReliability(src);
  }
  function renderAlphaSlider(src, ab) {
    const wrap = $("#alpha-slider-wrap"); const rc = (ab.fs || {}).risk_coverage;
    if (!rc || !rc.coverage) { wrap.innerHTML = ""; return; }
    wrap.innerHTML = `<label style="font-size:12px;color:var(--ink-soft)">Hypothetical α: <b id="alpha-val">${STATE.alpha.toFixed(2)}</b></label>
      <input type="range" id="alpha-range" min="0.02" max="0.40" step="0.01" value="${STATE.alpha}" style="width:100%">
      <div id="alpha-out" class="note" style="margin-top:4px"></div>`;
    const compute = a => { // largest coverage whose risk <= a, from the (tie-aware) curve
      let best = null; for (let i = 0; i < rc.coverage.length; i++) if (rc.risk[i] <= a) best = i;
      if (best == null) return { feasible: false };
      return { feasible: true, coverage: rc.coverage[best], risk: rc.risk[best] }; };
    const upd = () => { const a = +$("#alpha-range").value; STATE.alpha = a; $("#alpha-val").textContent = a.toFixed(2);
      const r = compute(a); $("#alpha-out").innerHTML = r.feasible
        ? `→ answer <b>${pct(r.coverage)}</b> of cases at empirical risk <b>${pct(r.risk)}</b> (client-side from the risk-coverage data; not a certified bound).`
        : `→ no coverage achieves risk ≤ ${a.toFixed(2)} on this set → <b>abstain-all</b>.`; };
    $("#alpha-range").oninput = upd; upd();
  }
  function renderBaselines(src) {
    const C = COLORS(); const m = ((DATA.abstention_baselines || {}).sources || {})[STATE.source];
    tagInto("baselines-src", src);
    const chart = $("#chart-baselines"), table = $("#baselines-table");
    if (!m || !m.triggers) { chart.innerHTML = `<p class="empty">No baseline comparison for this source.</p>`; table.innerHTML = ""; return; }
    const names = { fs_ours: ["FS (ours)", C.grounded, 2.6], confidence: ["confidence", C.abstain, 1.3], self_consistency: ["self-consistency", C.c, 1.3], radflag: ["RadFlag", C.secondary, 1.1], signal_a: ["Signal A", C.a, 1], signal_b: ["Signal B", C.b, 1] };
    const prov = isProvisional(src, srcN(src)), series = [];
    Object.entries(m.triggers).forEach(([k, v]) => { if (!v.curve || !v.curve.coverage.length || !names[k]) return;
      series.push({ name: `${names[k][0]}${v.degenerate ? " (constant)" : ""} · ${fmt(v.aurc, 3)}`, color: names[k][1], width: names[k][2], dots: false, provisional: prov || v.degenerate,
        points: v.curve.coverage.map((c, i) => [c, v.curve.risk[i]]) }); });
    markMock(chart, src);
    chart.innerHTML = series.length ? lineChart(series, { yMin: 0, yMax: 1, xlab: "coverage", ylab: "risk", xTicks: [0, .25, .5, .75, 1], animate: !prov }) + legend(series.map(s => ({ label: s.name, color: s.color, dash: s.provisional }))) : `<p class="empty">No curve data.</p>`;
    const mce = m.matched_coverage_error || {}, covs = Object.keys(mce), trigs = Object.keys(m.triggers), degen = m.degenerate_triggers || [];
    if (covs.length) {
      let h = `<table class="btab"><thead><tr><th>coverage</th>` + trigs.map(t => `<th>${names[t] ? names[t][0] : t}${degen.includes(t) ? "*" : ""}</th>`).join("") + `</tr></thead><tbody>`;
      covs.forEach(c => { const row = mce[c], live = trigs.filter(t => !degen.includes(t)), best = Math.min(...live.map(t => row[t]));
        h += `<tr><td>${(+c).toFixed(2)}</td>` + trigs.map(t => `<td class="${degen.includes(t) ? "degen" : (row[t] === best ? "best" : "")}">${fmt(row[t], 3)}</td>`).join("") + `</tr>`; });
      h += `</tbody></table><p class="cap">n=${m.n}, base acc ${fmt(m.base_accuracy, 3)}. Green = lowest error among <b>discriminating</b> triggers. *constant/degenerate signals (no discrimination at this n) — excluded from ranking.</p>`;
      table.innerHTML = h;
    } else table.innerHTML = "";
  }

  /* ==================================================================
     TAB: Case Explorer
     ================================================================== */
  function decisionFor(src, r) { const t = src.fs_threshold; if (t == null) return null; return r.fs >= t ? "answer" : "abstain"; }
  function renderExplorer(src) {
    const recs = src.records || [];
    const filters = [["all", "All"], ["answer", "Answered"], ["abstain", "Abstained"], ["grounded", "Grounded"], ["ungrounded", "Ungrounded"]];
    $("#case-filters").innerHTML = filters.map(f => `<button data-f="${f[0]}" class="${STATE.filter === f[0] ? "active" : ""}">${f[1]}</button>`).join("");
    $("#case-filters").querySelectorAll("button").forEach(b => b.onclick = () => { STATE.filter = b.dataset.f; STATE.caseIdx = null; renderExplorer(src); });
    const sb = $("#case-search"); if (sb && sb.value !== STATE.search) sb.value = STATE.search;
    if (sb && !sb._bound) { sb._bound = true; sb.oninput = () => { STATE.search = sb.value; STATE.caseIdx = null; renderExplorer(src); }; }
    const q = (STATE.search || "").toLowerCase();
    const pass = r => { const d = decisionFor(src, r);
      if (STATE.filter === "answer" && d !== "answer") return false;
      if (STATE.filter === "abstain" && d !== "abstain") return false;
      if (STATE.filter === "grounded" && r.grounded_latent !== 1) return false;
      if (STATE.filter === "ungrounded" && r.grounded_latent !== 0) return false;
      if (q && !((r.question || "") + " " + (r.answer || "") + " " + (r.gt_answer || "")).toLowerCase().includes(q)) return false;
      return true; };
    const list = recs.filter(pass), listEl = $("#case-list");
    if (!recs.length) { listEl.innerHTML = `<p class="empty">No per-case records for this source.</p>`; $("#case-detail").innerHTML = ""; return; }
    if (!list.length) { listEl.innerHTML = `<p class="empty">No cases match.</p>`; $("#case-detail").innerHTML = ""; return; }
    listEl.innerHTML = list.map(r => { const d = decisionFor(src, r), gl = r.grounded_latent;
      return `<button class="case-item ${STATE.caseIdx === r.sample_id ? "active" : ""}" data-id="${r.sample_id}">
        ${gl === 1 ? '<span class="dot g"></span>' : gl === 0 ? '<span class="dot u"></span>' : ''}
        <span class="cq">${esc(r.question || r.sample_id)}</span>
        <span class="ans">${esc(String(r.answer))}${r.correct != null ? (r.correct ? " ✓" : " ✗") : ""}</span>
        ${d ? `<span class="pill ${d}">${d}</span>` : ""}</button>`; }).join("");
    listEl.querySelectorAll(".case-item").forEach(b => b.onclick = () => { STATE.caseIdx = b.dataset.id; renderExplorer(src); });
    const chosen = list.find(r => r.sample_id === STATE.caseIdx) || list[0]; STATE.caseIdx = chosen.sample_id; renderCase(src, chosen);
    listEl.querySelectorAll(".case-item").forEach(b => b.classList.toggle("active", b.dataset.id === chosen.sample_id));
  }
  function renderCase(src, r) {
    const C = COLORS(), d = decisionFor(src, r);
    const sig = (name, val, color) => `<div class="sigbar"><span>${name}</span><span class="track"><span class="fill" style="width:${Math.round((val || 0) * 100)}%;background:${color}"></span></span><b>${fmt(val, 3)}</b></div>`;
    const steps = r.steps_text || [], stepFs = r.fs_per_step || [];
    const stepsHtml = steps.length ? `<h4 style="margin:16px 0 6px;font-size:13px;color:var(--ink-soft)">Reasoning chain (per-step faithfulness)</h4><ul class="steps">${steps.map((tx, i) => { const v = stepFs[i], col = v == null ? C.faint : (v > .5 ? C.grounded : v > .33 ? C.abstain : C.ungrounded); return `<li><span>${esc(tx)}</span><span class="stepfs"><div style="width:${Math.round((v || 0) * 100)}%;background:${col}"></div></span></li>`; }).join("")}</ul>` : "";
    $("#case-detail").innerHTML = `
      <p class="q">${esc(r.question || r.sample_id)}</p>
      <p class="meta">${r.modality === "unknown" ? "modality unspecified" : (r.modality || "?")} · id ${r.sample_id}</p>
      <div class="answers-row">
        <div class="ans-box model ${r.correct ? "ok" : "bad"}"><div class="lab">Model answer</div><div class="val">${esc(String(r.answer))} <span class="pill ${r.correct ? "correct" : "wrong"}">${r.correct ? "correct" : "wrong"}</span></div></div>
        <div class="ans-box truth"><div class="lab">Ground truth</div><div class="val">${esc(String(r.gt_answer))}</div></div>
        ${r.grounded_latent != null ? `<div class="ans-box"><div class="lab">Latent</div><div class="val"><span class="pill ${r.grounded_latent ? "correct" : "wrong"}">${r.grounded_latent ? "grounded" : "ungrounded"}</span></div></div>` : ""}
      </div>
      ${d ? `<div class="decision ${d}">${d === "answer" ? "✓ ANSWER" : "⚠ ABSTAIN → defer to clinician"}</div>` : ""}
      <div class="sigbars">
        ${sig("Signal A · counterfactual", r.signal_a, C.a)}
        ${sig("Signal B · grounding", r.signal_b, C.b)}
        ${sig("Signal C · consistency", r.signal_c, C.c)}
        ${sig("Fused Faithfulness Score", r.fs, C.grounded)}
        ${sig("Model confidence", r.confidence, C.abstain)}
      </div>
      ${src.fs_threshold != null ? `<p class="note">Gate τ = ${fmt(src.fs_threshold, 3)}: answer iff FS ≥ τ.</p>` : `<p class="note">No calibrated threshold in this source.</p>`}
      ${stepsHtml}`;
  }

  /* ==================================================================
     TAB: Robustness  (ablations are synthetic-fixture → always mock, badged)
     ================================================================== */
  function renderRobustness(src) {
    const C = COLORS(); const mock = mockSource(); const usingMock = true;  // ablations are inherently synthetic
    const fb = mock.full_benchmark || {}, abl = mock.ablations || {};
    $("#robustness-sub").innerHTML = `Sensitivity analyses for the headline components. <b>These ablations run on the synthetic fixture</b> (they need a known grounding latent), so they are shown from the Mock source regardless of the selected data source ${sampleBadge({ kind: "mock" })} — real ablations are not applicable.`;
    ["chart-incr", "chart-weights", "chart-grid", "chart-power"].forEach(id => markMock($("#" + id), { kind: "mock" }));

    const models = fb.models || {}, stages = ["auroc_fs_A", "auroc_fs_AB", "auroc_fs_ABC"], mk = Object.keys(models).filter(k => models[k].validation && models[k].validation.auroc_fs_A != null);
    $("#chart-incr").innerHTML = mk.length ? lineChart(mk.map((k, i) => ({ name: models[k].model, color: [C.primary, C.c][i] || C.faint, points: stages.map((s, j) => [j, models[k].validation[s]]) })), { ylab: "AUROC", xTicks: [0, 1, 2], xFmt: x => ["A", "A+B", "A+B+C"][x] }) + legend(mk.map((k, i) => ({ label: models[k].model, color: [C.primary, C.c][i] || C.faint }))) : `<p class="empty insufficient">Insufficient data — run the full benchmark.</p>`;
    const ws = abl.weight_sensitivity;
    $("#chart-weights").innerHTML = ws ? barChart(["min", "mean", "max"], [{ name: "AUROC", color: C.primary, values: [ws.auroc_min, ws.auroc_mean, ws.auroc_max] }], { yMin: 0.5, yMax: 1, valueLabels: true, hline: ws.default_weights_auroc, hlineColor: C.grounded }) + legend([{ label: `default weights = ${fmt(ws.default_weights_auroc, 3)}`, color: C.grounded }]) : `<p class="empty insufficient">Insufficient data — run ablations.</p>`;
    const gs = abl.grid_sensitivity;
    if (gs) { const gk = Object.keys(gs).map(Number).sort((a, b) => a - b);
      $("#chart-grid").innerHTML = lineChart([{ name: "Signal B", color: C.b, points: gk.map(g => [g, gs[String(g)].auroc_signal_b]) }, { name: "Fused FS", color: C.grounded, points: gk.map(g => [g, gs[String(g)].auroc_fs]) }], { yMin: 0.5, yMax: 1, ylab: "AUROC", xlab: "grid resolution", xTicks: gk }) + legend([{ label: "Signal B", color: C.b }, { label: "Fused FS", color: C.grounded }]); }
    else $("#chart-grid").innerHTML = `<p class="empty insufficient">Insufficient data — run ablations.</p>`;
    const pw = abl.abstention_power && abl.abstention_power.min_cal_size_for_alpha;
    if (pw) { const ak = Object.keys(pw).map(Number).sort((a, b) => a - b);
      $("#chart-power").innerHTML = lineChart([{ name: "min cal size", color: C.abstain, points: ak.map(a => [a, pw[String(a)] == null ? 0 : pw[String(a)]]) }], { yMin: 0, ylab: "min calibration N", xlab: "target error α", xTicks: ak }); }
    else $("#chart-power").innerHTML = `<p class="empty insufficient">Insufficient data — run ablations.</p>`;
  }

  /* ==================================================================
     TAB: Limitations & Timeline
     ================================================================== */
  function renderLimitations() {
    const lims = [
      { ok: false, h: "Small real-data samples (n=20–50)", p: "Real runs are Colab smoke sets. Abstention/verifier numbers are noisy and several triggers show no separation at this n. Charts carry sample-size badges and dashed/muted styling below n=50." },
      { ok: false, h: "Signal B constant on real data", p: "Per-step attention→region extraction is still stubbed for real HF models, so Signal B is a constant 0.5 there — the fused FS uses A+C on real data until real attention grounding lands." },
      { ok: false, h: "Blind-gap headline is confounded", p: "On VQA-RAD the reasoning model relies on the image less than the non-reasoning one (supports the hypothesis) — but it is also less accurate overall, a confound we flag rather than hide. On PathVQA both models are image-independent (no grounding effect)." },
      { ok: false, h: "Conformal gate infeasible at small n", p: "Controlling 5% error at 95% confidence needs ≳500–1000 calibration points; on n=20–50 the gate honestly reports abstain-all rather than a false guarantee." },
      { ok: false, h: "Verifier weak-label ceiling", p: "The learned verifier beats the heuristic (0.768→0.816 AUROC) but is capped by 0.77 weak-label fidelity vs a 0.951 oracle — recoverable only with real grounding supervision." },
      { ok: false, h: "No per-QA boxes in the HF mirrors", p: "VQA-RAD/SLAKE mirrors lack per-question bounding boxes, so Signal-B IoU is validated on synthetic data; real Signal-B validation awaits the SLAKE segmentation masks." },
      { ok: true, h: "Machinery is validated", p: "On the synthetic fixture with a known latent, every component behaves as designed: signals separate grounded from ungrounded, fusion beats singles, and the guarantee holds empirically across many draws." },
    ];
    $("#limitations-list").innerHTML = lims.map(l => `<div class="card lim ${l.ok ? "ok" : ""}"><h3>${l.ok ? "✓ " : "⚠ "}${l.h}</h3><p>${l.p}</p></div>`).join("");
  }
  function renderTimeline() {
    const tl = (DATA.timeline && DATA.timeline.length) ? DATA.timeline : [
      { date: "Stage 0–1", tag: "A", body: "Repo scaffolding, data loaders (VQA-RAD/SLAKE/PathVQA verified on live HF mirrors), disjoint splits." },
      { date: "Stage 2", tag: "A", body: "Baselines + blind test; headline replication verdict machinery (drift + blind-gap lenses)." },
      { date: "Stage 3", tag: "A", body: "FMM: Signals A/B/C + fused FS; validated on synthetic (fused AUROC 0.87 > any single)." },
      { date: "Stage 4", tag: "B", body: "Training-free correction (VCD + clue-tracing + verify/revise); real cross-model sweep on Qwen2.5-VL." },
      { date: "Stage 5", tag: "A", body: "Conformal abstention (SGR guarantee); honest calibration-power finding (α=0.05 needs ~1k cal points)." },
      { date: "Stage 6", tag: "A", body: "Full benchmark + ablations + model-agnosticism; correction wired in (calibration ordering fix)." },
      { date: "Stretch", tag: "B", body: "Learned verifier beats heuristic on real signals; weak-label ceiling reported honestly. LLM-judge validated (κ)." },
      { date: "Real runs", tag: "A", body: "Colab: MedVLM-R1 vs Qwen2.5-VL on VQA-RAD/PathVQA/SLAKE; results pushed back and wired into this dashboard." },
      { date: "Integration", tag: "A", body: "Branches unified; tie-aware risk-coverage fix (constant signals no longer fabricate AURCs); full UI overhaul." },
    ];
    $("#timeline-list").innerHTML = tl.map(t => `<div class="tl-item ${t.tag === "B" ? "b" : ""}"><div class="tl-date">${esc(t.date)} · [${t.tag}]</div><div class="tl-body">${esc(t.body)}</div></div>`).join("");
  }

  /* ==================================================================
     TAB: Live Demo  (dynamic — calls the Colab+ngrok FastAPI backend)
     ================================================================== */
  const LIVE = { file: null, bound: false };
  const apiBase = () => (($("#live-api") || {}).value || "").trim().replace(/\/+$/, "");
  // ngrok's free interstitial returns HTML unless this header is present.
  const NGROK_HDR = { "ngrok-skip-browser-warning": "true" };

  function liveSpinner() {
    return `<div class="live-loading"><div class="spinner"></div>
      <div class="ll-t">Loading…</div>
      <div class="ll-s">GPU calculating 5 consistency passes — the first run also loads the model, so allow up to a minute or two on a free T4.</div></div>`;
  }
  function liveError(msg) {
    return `<div class="caveat" style="border-color:var(--ungrounded);background:color-mix(in srgb,var(--ungrounded) 7%,var(--surface))">
      <span class="icon">⚠</span><div>${msg}</div></div>`;
  }
  function liveResult(d) {
    const C = COLORS(), ans = d.decision === "answer";
    const sig = (name, val, color) => `<div class="sigbar"><span>${esc(name)}</span><span class="track"><span class="fill" style="width:${Math.round((val || 0) * 100)}%;background:${color}"></span></span><b>${fmt(val, 3)}</b></div>`;
    const s = d.signals || {}, g = d.gate || {};
    const tauPos = Math.round((Number.isFinite(g.threshold) ? g.threshold : 1) * 100);
    const steps = d.reasoning_steps || [], sfs = d.fs_per_step || [];
    const stepsHtml = steps.length ? `<h4 class="live-h">Reasoning chain (per-step faithfulness)</h4><ul class="steps">${steps.map((tx, i) => {
      const v = sfs[i], col = v == null ? C.faint : (v > .5 ? C.grounded : v > .33 ? C.abstain : C.ungrounded);
      return `<li><span>${esc(tx)}</span><span class="stepfs"><div style="width:${Math.round((v || 0) * 100)}%;background:${col}"></div></span></li>`;
    }).join("")}</ul>` : "";
    return `
      <div class="live-score" data-tip="Fused Faithfulness Score (the FMR score). The gate answers only if FS ≥ τ.">
        <div class="ls-num" style="color:${ans ? C.grounded : C.abstain}">${fmt(d.fmr_score, 3)}</div>
        <div class="ls-lab">FMR score</div>
        <div class="ls-track"><div class="ls-fill" style="width:${Math.round((d.fmr_score || 0) * 100)}%;background:${ans ? C.grounded : C.abstain}"></div>
          <div class="ls-tau" style="left:${tauPos}%" title="gate threshold τ"></div></div>
      </div>
      <div class="decision ${ans ? "answer" : "abstain"}">${ans ? "✓ ANSWER" : "⚠ ABSTAIN → defer to clinician"}</div>
      <div class="live-answer"><span class="lab">Model answer</span><b>${esc(String(d.model_answer))}</b></div>
      <div class="sigbars" style="margin-top:14px">
        ${sig("Signal A · counterfactual", s.counterfactual_A, C.a)}
        ${sig("Signal B · grounding", s.grounding_B, C.b)}
        ${sig("Signal C · consistency", s.consistency_C, C.c)}
        ${sig("Fused FMR score", d.fmr_score, C.grounded)}
        ${sig("Model confidence", d.confidence, C.abstain)}
      </div>
      <p class="note live-gate">Gate τ = ${Number.isFinite(g.threshold) ? fmt(g.threshold, 3) : "∞"} · α=${fmt(g.alpha, 2)} ·
        ${g.certified ? "certified split-conformal bound" : esc(g.mode || "uncertified operating point")} ·
        ${d.n_consistency} consistency passes · ${fmt(d.elapsed_seconds, 1)}s on ${esc(d.model || "GPU")}.</p>
      ${stepsHtml}`;
  }

  function updateApiStatus() {
    const s = $("#live-api-status"), v = apiBase();
    if (!s) return;
    if (!v) { s.textContent = "Not connected — paste the URL the Colab cell printed."; return; }
    s.innerHTML = `Will POST to <b>${esc(v)}/analyze</b>. <button class="btn" id="live-ping" style="padding:2px 9px;margin-left:6px">Test /health</button>`;
    const b = $("#live-ping"); if (b) b.onclick = pingHealth;
  }
  async function pingHealth() {
    const s = $("#live-api-status"), v = apiBase();
    if (!v) return;
    s.innerHTML = "Pinging…";
    try {
      const r = await fetch(v + "/health", { headers: NGROK_HDR });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      s.innerHTML = `✅ Connected — model <b>${esc(j.model || "?")}</b>, weights ${j.model_loaded ? "loaded" : "will load on first request"}.`;
    } catch (e) { s.innerHTML = `❌ Could not reach <b>${esc(v)}/health</b> — is the Colab cell still running?`; }
  }
  async function runLive() {
    const out = $("#live-result"), v = apiBase();
    const q = (($("#live-q") || {}).value || "").trim();
    if (!v) { updateApiStatus(); out.innerHTML = liveError("Enter the ngrok API URL first (step 1)."); return; }
    if (!LIVE.file) { out.innerHTML = liveError("Choose an image first (step 2)."); return; }
    if (!q) { out.innerHTML = liveError("Type a question first (step 2)."); return; }
    const btn = $("#live-run"); if (btn) { btn.disabled = true; btn.textContent = "Analyzing on GPU…"; }
    out.innerHTML = liveSpinner();
    const fd = new FormData();
    fd.append("file", LIVE.file);
    fd.append("question", q);
    const ch = (($("#live-choices") || {}).value || "").trim(); if (ch) fd.append("choices", ch);
    try {
      const res = await fetch(v + "/analyze", { method: "POST", body: fd, headers: NGROK_HDR });
      if (!res.ok) { const t = await res.text().catch(() => ""); throw new Error(`HTTP ${res.status} ${res.statusText}. ${esc(t.slice(0, 200))}`); }
      const data = await res.json();
      out.innerHTML = liveResult(data);
      bindTips(out);
    } catch (e) {
      out.innerHTML = liveError(`${esc(String(e && e.message || e))}<br><span class="note">Check the URL, that the Colab cell is still running, and that the runtime has a GPU.</span>`);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "Analyze on GPU →"; }
    }
  }
  function renderLive() {
    const api = $("#live-api"); if (!api) return;
    if (!LIVE.bound) {
      LIVE.bound = true;
      api.value = localStorage.getItem("fmr-api-url") || "";
      api.addEventListener("input", () => { localStorage.setItem("fmr-api-url", apiBase()); updateApiStatus(); });
      const file = $("#live-file"), drop = $("#live-drop"), prev = $("#live-preview"), inner = $("#live-drop-inner");
      file.addEventListener("change", () => {
        const f = file.files && file.files[0]; if (!f) return;
        LIVE.file = f; prev.src = URL.createObjectURL(f); prev.hidden = false;
        if (inner) inner.style.display = "none"; drop.classList.add("has");
      });
      $("#live-run").addEventListener("click", runLive);
      $("#live-q").addEventListener("keydown", e => { if (e.key === "Enter") runLive(); });
      $("#live-choices").addEventListener("keydown", e => { if (e.key === "Enter") runLive(); });
    }
    updateApiStatus();
  }

  /* ==================================================================
     Derived-data charts (real-data capable via correct / fs / confidence /
     n_steps). These deliberately light up on synthetic and read flat/null on
     the small real sets — shown honestly with sample badges + empty states,
     never smoothed into a trend the data doesn't support.
     ================================================================== */
  function _eqCountBins(n, nb) {   // -> array of [startIdx, endIdx) over a length-n sorted array
    const out = [];
    for (let i = 0; i < nb; i++) { const a = Math.floor(i * n / nb), b = Math.floor((i + 1) * n / nb); if (b > a) out.push([a, b]); }
    return out;
  }
  function accByFSBins(recs) {
    const rs = recs.filter(r => r.fs != null && r.correct != null).sort((a, b) => a.fs - b.fs);
    if (rs.length < 6) return null;
    const nb = Math.max(3, Math.min(6, Math.floor(rs.length / 12)));
    return _eqCountBins(rs.length, nb).map(([a, b]) => { const ch = rs.slice(a, b);
      return { label: `${ch[0].fs.toFixed(2)}–${ch[ch.length - 1].fs.toFixed(2)}`,
        acc: ch.reduce((s, r) => s + r.correct, 0) / ch.length, n: ch.length }; });
  }
  function renderFSAcc(src) {
    const C = COLORS(), recs = src.records || []; tagInto("fsacc-src", src);
    const box = $("#chart-fsacc"); if (!box) return; markMock(box, src);
    const bins = accByFSBins(recs);
    if (!bins) { box.innerHTML = `<p class="empty">Need ≥6 scored cases with correctness labels for this view.</p>`; return; }
    const base = recs.reduce((s, r) => s + (r.correct || 0), 0) / recs.length;
    box.innerHTML = barChart(bins.map(b => b.label), [{ name: "accuracy", color: C.grounded, values: bins.map(b => b.acc) }],
      { yMin: 0, yMax: 1, ylab: "answer accuracy", xlab: "Faithfulness Score bin (low → high)", valueLabels: true, hline: base, hlineColor: C.faint })
      + legend([{ label: "accuracy", color: C.grounded }, { label: `base rate ${pct(base)}`, color: C.faint }])
      + `<p class="note">Per-bin n: ${bins.map(b => b.n).join(" · ")}. Rising bars ⇒ FS ranks correct answers above wrong ones.</p>`;
  }
  function renderCorrectWrong(src) {
    const C = COLORS(), recs = src.records || []; tagInto("fscw-src", src);
    const box = $("#chart-fscw"); if (!box) return; markMock(box, src);
    const g = recs.filter(r => r.correct === 1).map(r => r.fs).filter(v => v != null);
    const w = recs.filter(r => r.correct === 0).map(r => r.fs).filter(v => v != null);
    if (g.length < 3 || w.length < 3) { box.innerHTML = `<p class="empty">Need ≥3 correct and ≥3 wrong scored cases for this view.</p>`; return; }
    box.innerHTML = histogram([{ name: "wrong", color: C.ungrounded, values: w }, { name: "correct", color: C.grounded, values: g }], { xlab: "Faithfulness Score" })
      + legend([{ label: `correct (n=${g.length})`, color: C.grounded }, { label: `wrong (n=${w.length})`, color: C.ungrounded }]);
  }
  function renderAURC(src) {
    const C = COLORS(), ab = (src.fmr_results || {}).abstention || {}; tagInto("aurc-src", src);
    const box = $("#chart-aurc"); if (!box) return; markMock(box, src);
    const defs = [["fs", "Fused FS", C.grounded], ["confidence", "Confidence", C.abstain], ["signal_a_only", "Signal A", C.a], ["signal_b_only", "Signal B", C.b], ["signal_c_only", "Signal C", C.c]];
    const cats = [], vals = [], cols = [];
    defs.forEach(d => { const o = ab[d[0]]; if (o && o.aurc != null) { cats.push(d[1] + (o.degenerate ? "*" : "")); vals.push(o.aurc); cols.push(o.degenerate ? C.faint : d[2]); } });
    if (!cats.length) { box.innerHTML = `<p class="empty">No abstention curves for this source.</p>`; return; }
    box.innerHTML = barChart(cats, [{ name: "AURC", color: C.primary, colors: cols, values: vals }],
      { yMin: 0, yMax: Math.min(1, Math.max(...vals) + 0.06), ylab: "AURC (lower = better)", valueLabels: true, labelDigits: 3 })
      + `<p class="note">Area under risk–coverage; lower is better. *constant/degenerate signal — no discrimination at this n.</p>`;
  }
  function reliabilityBins(recs) {
    const rs = recs.filter(r => r.confidence != null && r.correct != null).sort((a, b) => a.confidence - b.confidence);
    if (rs.length < 8) return null;
    const nb = Math.max(3, Math.min(6, Math.floor(rs.length / 10)));
    return _eqCountBins(rs.length, nb).map(([a, b]) => { const ch = rs.slice(a, b);
      return { conf: ch.reduce((s, r) => s + r.confidence, 0) / ch.length, acc: ch.reduce((s, r) => s + r.correct, 0) / ch.length, n: ch.length }; });
  }
  function renderReliability(src) {
    const C = COLORS(), recs = src.records || []; tagInto("rel-src", src);
    const box = $("#chart-reliability"); if (!box) return; markMock(box, src);
    const b = reliabilityBins(recs);
    if (!b) { box.innerHTML = `<p class="empty">Need ≥8 scored cases with confidence + correctness for this view.</p>`; return; }
    const prov = isProvisional(src, srcN(src));
    box.innerHTML = lineChart([
      { name: "perfectly calibrated", color: C.faint, points: [[0, 0], [1, 1]], dots: false, provisional: true, width: 1.4 },
      { name: "observed", color: C.primary, points: b.map(x => [x.conf, x.acc]), width: 2.4, provisional: prov },
    ], { yMin: 0, yMax: 1, xTicks: [0, .25, .5, .75, 1], xlab: "model confidence", ylab: "empirical accuracy", animate: !prov })
      + legend([{ label: "observed", color: C.primary }, { label: "perfect calibration", color: C.faint, dash: true }]);
  }
  function fsByChainLen(src) {
    const recs = (src.records || []).filter(r => r.n_steps != null && r.fs != null);
    const uniq = [...new Set(recs.map(r => r.n_steps))].sort((a, b) => a - b);
    if (uniq.length < 2 || recs.length < 8) return null;
    const lo = uniq[0], hi = uniq[uniq.length - 1], nb = Math.min(5, uniq.length), w = (hi - lo + 1) / nb, out = [];
    for (let i = 0; i < nb; i++) {
      const a = lo + i * w, b = (i === nb - 1) ? hi + 1 : lo + (i + 1) * w;
      const ch = recs.filter(r => r.n_steps >= a && r.n_steps < b);
      if (ch.length) out.push({ label: (nb === uniq.length ? `${Math.round(a)}` : `${Math.round(a)}–${Math.round(b - 1)}`),
        fs: ch.reduce((s, r) => s + r.fs, 0) / ch.length, n: ch.length });
    }
    return out.length > 1 ? out : null;
  }
  function renderChainLen(src) {
    const C = COLORS(); tagInto("chainlen-src", src);
    const box = $("#chart-chainlen"); if (!box) return; markMock(box, src);
    const d = fsByChainLen(src);
    if (!d) { box.innerHTML = `<p class="empty">Chain length is (near-)constant for this source — synthetic uses a fixed chain, so this activates on real data.</p>`; return; }
    box.innerHTML = barChart(d.map(x => x.label), [{ name: "mean FS", color: C.b, values: d.map(x => x.fs) }],
      { yMin: 0, yMax: 1, ylab: "mean Faithfulness Score", xlab: "reasoning-chain length (steps)", valueLabels: true })
      + `<p class="note">Per-bin n: ${d.map(x => x.n).join(" · ")}. Thesis: longer chains drift less-grounded (small real n ⇒ noisy).</p>`;
  }

  /* ==================================================================
     Render dispatch + router
     ================================================================== */
  const RENDER = { overview: renderOverview, methodology: () => renderMethodology(), diagnosis: renderDiagnosis,
    measurement: renderMeasurement, safety: renderSafety, explorer: renderExplorer, robustness: renderRobustness,
    limitations: () => renderLimitations(), timeline: () => renderTimeline(), live: () => renderLive() };
  function renderView(v) { const src = activeSource(); try { RENDER[v] && RENDER[v](src); } catch (e) { console.error("render " + v, e); }
    bindTips($("#view-" + v)); }
  function renderAll() {
    const src = activeSource(), badge = $("#source-badge");
    badge.textContent = src.kind === "real" ? "REAL" : "MOCK"; badge.className = "badge " + (src.kind === "real" ? "real" : "mock");
    renderView(STATE.view);   // active view now; others render lazily on switch
  }

  function switchTab(v, push = true) {
    if (!document.getElementById("view-" + v)) v = "overview";
    STATE.view = v;
    $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.view === v));
    $$(".view").forEach(sec => { const on = sec.dataset.view === v; sec.classList.toggle("active", on); if (on) { sec.classList.remove("anim"); void sec.offsetWidth; sec.classList.add("anim"); } });
    moveIndicator(); renderView(v);
    if (push && location.hash !== "#" + v) history.replaceState(null, "", "#" + v);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function moveIndicator() { const tab = $(".tab.active"), ind = $("#tab-indicator"); if (!tab || !ind) return;
    ind.style.left = tab.offsetLeft + "px"; ind.style.width = tab.offsetWidth + "px"; }

  function initTabs() {
    $$(".tab").forEach(t => t.onclick = () => switchTab(t.dataset.view));
    window.addEventListener("keydown", e => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const order = $$(".tab").map(t => t.dataset.view), i = order.indexOf(STATE.view);
      if (e.key === "ArrowRight") switchTab(order[Math.min(order.length - 1, i + 1)]);
      else if (e.key === "ArrowLeft") switchTab(order[Math.max(0, i - 1)]);
      else if (/^[1-9]$/.test(e.key) && +e.key <= order.length) switchTab(order[+e.key - 1]); });
    window.addEventListener("resize", moveIndicator);
    const h = location.hash.slice(1); switchTab(h || "overview", false);
  }

  function initHeader() {
    // theme
    const saved = localStorage.getItem("fmr-theme") || "light";
    document.documentElement.setAttribute("data-theme", saved);
    $("#theme-btn").textContent = saved === "dark" ? "☀️" : "🌙";
    $("#theme-btn").onclick = () => { const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", cur); localStorage.setItem("fmr-theme", cur);
      $("#theme-btn").textContent = cur === "dark" ? "☀️" : "🌙"; renderView(STATE.view); moveIndicator(); };
    // slim header on scroll
    window.addEventListener("scroll", () => $("#topbar").classList.toggle("slim", window.scrollY > 40), { passive: true });
    // export (print)
    $("#export-btn").onclick = () => window.print();
    // compare mode
    $("#compare-btn").onclick = openCompare;
  }

  function openCompare() {
    const modal = $("#compare-modal"); const mock = DATA.sources["mock"], real = Object.entries(DATA.sources).find(([k, s]) => s.kind === "real" && (s.fmr_results || s.records));
    if (!real) { alert("No real source available to compare."); return; }
    const headline = s => { const fr = s.fmr_results || {}, bt = s.blind_test || {}, rep = bt.replication || {}, ab = fr.abstention || {}, g = (ab.fs || {}).test || {};
      return { auroc: (fr.validation || {}).auroc_fs, drift: rep.drift_slope, blind: rep.blind_gap_reasoning, cov: g.coverage, err: g.retained_error, n: srcN(s) }; };
    const m = headline(mock), r = headline(real[1]);
    const row = (lab, mv, rv, f) => `<tr><td>${lab}</td><td>${f(mv)}</td><td>${f(rv)}</td></tr>`;
    modal.innerHTML = `<div style="position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100;display:grid;place-items:center" id="cmp-bg">
      <div class="card" style="max-width:560px;width:92%"><div class="card-h" style="justify-content:space-between">Mock vs Real — headline metrics <button class="btn" id="cmp-close">✕</button></div>
      <table class="btab"><thead><tr><th>metric</th><th>Mock (synthetic) ${sampleBadge({ kind: "mock" })}</th><th>Real ${sampleBadge(real[1], r.n)}</th></tr></thead><tbody>
      ${row("Fused-FS AUROC", m.auroc, r.auroc, v => v == null ? "n/a" : fmt(v, 3))}
      ${row("Blind-gap (reasoning)", m.blind, r.blind, v => v == null ? "n/a" : fmt(v, 3))}
      ${row("Gate coverage", m.cov, r.cov, pct)}
      ${row("Gate retained error", m.err, r.err, pct)}
      </tbody></table>
      <p class="cap">Mock validates machinery on a known latent; Real is the empirical finding (small n). This is the machinery-vs-empirical view side by side.</p></div></div>`;
    $("#cmp-close").onclick = () => modal.innerHTML = "";
    $("#cmp-bg").onclick = e => { if (e.target.id === "cmp-bg") modal.innerHTML = ""; };
  }

  function initSourcePicker() {
    const sel = $("#source"), keys = Object.keys(DATA.sources);
    if (!keys.length) { $("#app").innerHTML = `<p class="empty">No data. Run the pipeline + make_dashboard.py.</p>`; return false; }
    sel.innerHTML = keys.map(k => `<option value="${k}" ${k === STATE.source ? "selected" : ""}>${esc(DATA.sources[k].label || k)}</option>`).join("");
    sel.onchange = () => { STATE.source = sel.value; STATE.caseIdx = null; renderAll(); };
    return true;
  }

  if (initSourcePicker()) { initHeader(); renderAll(); initTabs(); }
})();
