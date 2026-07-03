/* FMR dashboard — renders window.FMR_DATA into charts + an interactive explorer.
   Zero dependencies: all charts are hand-rolled SVG so it runs from file://. */
(function () {
  "use strict";
  const DATA = window.FMR_DATA || { sources: {} };
  const $ = (s, r = document) => r.querySelector(s);
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
  const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
  const pct = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (100 * x).toFixed(1) + "%";
  const COL = { primary: "#0e7490", grounded: "#16a34a", ungrounded: "#dc2626", abstain: "#f59e0b",
                grey: "#94a3b8", a: "#6366f1", b: "#0e7490", c: "#db2777", fs: "#16a34a", conf: "#f59e0b" };

  /* ------------------------------------------------------------------ charts */
  function frame(w, h, pad) {
    const p = Object.assign({ t: 18, r: 16, b: 34, l: 42 }, pad || {});
    return { w, h, p, iw: w - p.l - p.r, ih: h - p.t - p.b,
      open: `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet">`, close: `</svg>` };
  }
  function axes(f, { xTicks = [], yTicks = [], xlab = "", ylab = "" }) {
    let s = "";
    yTicks.forEach(t => {
      const y = f.p.t + f.ih - t.f * f.ih;
      s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="#eef2f7"/>`;
      s += `<text x="${f.p.l - 6}" y="${y + 3}" text-anchor="end" font-size="10">${t.label}</text>`;
    });
    xTicks.forEach(t => {
      const x = f.p.l + t.f * f.iw;
      s += `<text x="${x}" y="${f.p.t + f.ih + 15}" text-anchor="middle" font-size="10">${t.label}</text>`;
    });
    if (ylab) s += `<text x="12" y="${f.p.t + f.ih / 2}" transform="rotate(-90 12 ${f.p.t + f.ih / 2})" text-anchor="middle" font-size="10.5" fill="#64748b">${ylab}</text>`;
    if (xlab) s += `<text x="${f.p.l + f.iw / 2}" y="${f.h - 2}" text-anchor="middle" font-size="10.5" fill="#64748b">${xlab}</text>`;
    return s;
  }
  function yScale(vals, forceMin, forceMax) {
    let lo = forceMin !== undefined ? forceMin : Math.min(...vals);
    let hi = forceMax !== undefined ? forceMax : Math.max(...vals);
    if (lo === hi) { hi += 1; lo -= 1; }
    return { lo, hi, f: (v) => (v - lo) / (hi - lo) };
  }
  function yticks(sc, n = 4) {
    const out = [];
    for (let i = 0; i <= n; i++) { const v = sc.lo + (sc.hi - sc.lo) * i / n; out.push({ f: (v - sc.lo) / (sc.hi - sc.lo), label: v.toFixed(2) }); }
    return out;
  }

  function lineChart(series, opt = {}) {
    const f = frame(opt.w || 480, opt.h || 260);
    const all = series.flatMap(s => s.points.map(p => p[1]));
    const sc = yScale(all, opt.yMin, opt.yMax);
    const xs = series[0] ? series[0].points.map(p => p[0]) : [0, 1];
    const xlo = Math.min(...series.flatMap(s => s.points.map(p => p[0])));
    const xhi = Math.max(...series.flatMap(s => s.points.map(p => p[0])));
    const fx = (x) => f.p.l + (xhi === xlo ? 0.5 : (x - xlo) / (xhi - xlo)) * f.iw;
    const fy = (y) => f.p.t + f.ih - sc.f(y) * f.ih;
    const xTicks = (opt.xTicks || xs).map(x => ({ f: xhi === xlo ? 0.5 : (x - xlo) / (xhi - xlo), label: opt.xFmt ? opt.xFmt(x) : x }));
    let s = f.open + axes(f, { xTicks, yTicks: yticks(sc), xlab: opt.xlab, ylab: opt.ylab });
    series.forEach(se => {
      const d = se.points.map((p, i) => (i ? "L" : "M") + fx(p[0]) + " " + fy(p[1])).join(" ");
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="5 4"` : ""}/>`;
      if (se.dots !== false) se.points.forEach(p => s += `<circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="${se.r || 3}" fill="${se.color}"/>`);
    });
    (opt.marks || []).forEach(m => {
      s += `<path d="M${fx(m.x)} ${fy(m.y)} l-6 -10 l12 0 z" fill="${m.color || COL.ungrounded}"/>`;
    });
    if (opt.hline !== undefined) { const y = fy(opt.hline); s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="${opt.hlineColor || '#111'}" stroke-dasharray="3 3"/>`; }
    return s + f.close;
  }

  function barChart(categories, series, opt = {}) {
    const f = frame(opt.w || 480, opt.h || 260, opt.pad);
    const all = series.flatMap(s => s.values.filter(v => v !== null && !Number.isNaN(v)));
    const sc = yScale(all.concat(opt.baseline0 === false ? [] : [0]), opt.yMin, opt.yMax);
    const n = categories.length, g = series.length;
    const bandW = f.iw / n, barW = Math.min(46, (bandW * 0.7) / g);
    const fy = (y) => f.p.t + f.ih - sc.f(y) * f.ih;
    const xTicks = categories.map((c, i) => ({ f: (i + 0.5) / n, label: c }));
    let s = f.open + axes(f, { xTicks, yTicks: yticks(sc), ylab: opt.ylab });
    categories.forEach((c, i) => {
      const cx = f.p.l + bandW * (i + 0.5);
      series.forEach((se, j) => {
        const v = se.values[i]; if (v === null || v === undefined || Number.isNaN(v)) return;
        const x = cx - (g * barW) / 2 + j * barW;
        const y = fy(Math.max(v, sc.lo)), h = Math.max(0, fy(sc.lo) - y);
        s += `<rect x="${x}" y="${y}" width="${barW - 2}" height="${h}" rx="3" fill="${se.color}"/>`;
        if (opt.valueLabels) s += `<text x="${x + barW / 2 - 1}" y="${y - 3}" text-anchor="middle" font-size="9.5" fill="#475569">${fmt(v, opt.labelDigits ?? 2)}</text>`;
      });
    });
    if (opt.hline !== undefined) { const y = fy(opt.hline); s += `<line x1="${f.p.l}" y1="${y}" x2="${f.p.l + f.iw}" y2="${y}" stroke="${opt.hlineColor || '#111'}" stroke-dasharray="4 3"/>`; }
    return s + f.close;
  }

  function histogram(groups, opt = {}) {
    const f = frame(opt.w || 480, opt.h || 260);
    const bins = opt.bins || 22, lo = 0, hi = 1, bw = (hi - lo) / bins;
    const counts = groups.map(gr => {
      const arr = new Array(bins).fill(0);
      gr.values.forEach(v => { let k = Math.floor((v - lo) / bw); if (k < 0) k = 0; if (k >= bins) k = bins - 1; arr[k]++; });
      const tot = gr.values.length || 1; return arr.map(c => c / tot);
    });
    const maxY = Math.max(0.001, ...counts.flat());
    const sc = yScale([0, maxY], 0, maxY);
    const fy = (y) => f.p.t + f.ih - sc.f(y) * f.ih;
    const xTicks = [0, 0.25, 0.5, 0.75, 1].map(x => ({ f: x, label: x }));
    let s = f.open + axes(f, { xTicks, yTicks: yticks(sc, 3).map(t => ({ f: t.f, label: "" })), xlab: opt.xlab, ylab: "density" });
    groups.forEach((gr, gi) => {
      let d = `M${f.p.l} ${fy(0)}`;
      counts[gi].forEach((c, k) => { const x0 = f.p.l + (k / bins) * f.iw, x1 = f.p.l + ((k + 1) / bins) * f.iw; d += ` L${x0} ${fy(c)} L${x1} ${fy(c)}`; });
      d += ` L${f.p.l + f.iw} ${fy(0)} Z`;
      s += `<path d="${d}" fill="${gr.color}" fill-opacity="0.4" stroke="${gr.color}" stroke-width="1.3"/>`;
    });
    return s + f.close;
  }

  function legend(items) {
    return `<div class="legend">` + items.map(i => `<span><span class="sw" style="background:${i.color}"></span>${i.label}</span>`).join("") + `</div>`;
  }

  /* ------------------------------------------------------------------ sections */
  function activeSource() { return DATA.sources[STATE.source] || {}; }

  function renderOverview(src) {
    const fr = src.fmr_results || {}, bt = src.blind_test || {};
    const val = fr.validation || {}, ab = fr.abstention || {};
    const gate = (ab.fs && ab.fs.test) || {};
    const rep = bt.replication || {};
    // Card 1 — headline: drift slope when available, else the blind-gap delta.
    const headCard = (rep.primary_evidence === "blind_gap" || rep.drift_available === false)
      ? { k: "Image-reliance gap (reasoning)", v: fmt(rep.blind_gap_reasoning, 3),
          s: rep.blind_gap_supports ? `< non-reasoning ${fmt(rep.blind_gap_nonreasoning,3)} ✓` : "vs non-reasoning" }
      : { k: "Grounding drift slope", v: fmt(rep.drift_slope, 3), s: rep.replicated ? "decays along chain ✓" : "no decay" };
    const cards = [
      headCard,
      { k: "Fused-FS separation", v: val.auroc_fs != null ? fmt(val.auroc_fs, 3) : "n/a",
        s: val.auroc_fs != null ? "AUROC vs grounding" : "no grounding labels (real)" },
      { k: "Answered @ gate", v: gate.coverage != null ? pct(gate.coverage) : "—",
        s: gate.coverage ? `error ${pct(gate.retained_error)} (α=${fmt(ab.alpha,2)})` : `abstain-all (α=${fmt(ab.alpha,2)})` },
      { k: "Test cases", v: fr.n_test ?? "—", s: (src.kind === "real" ? "real" : "synthetic") + " dataset" },
    ];
    $("#stat-cards").innerHTML = cards.map(c => `<div class="card"><div class="k">${c.k}</div><div class="v">${c.v}</div><div class="s">${c.s}</div></div>`).join("");
    $("#provenance").textContent = `Source: ${src.label || "—"} · dataset "${fr.dataset || bt.dataset || "?"}" · bundle generated ${DATA.generated_at || "?"}`;
  }

  function renderDiagnosis(src) {
    const bt = src.blind_test || {}; const rep = bt.replication || {};
    const vEl = $("#replication-verdict");
    if (rep.tested) {
      vEl.className = "verdict " + (rep.replicated ? "yes" : "no");
      let sub = rep.note || "";
      if (rep.primary_evidence === "blind_gap" && rep.blind_gap_reasoning != null) {
        sub = `${rep.reasoning_model}: image-reliance gap ${fmt(rep.blind_gap_reasoning, 3)} vs `
            + `non-reasoning ${fmt(rep.blind_gap_nonreasoning, 3)}`
            + (rep.accuracy_confound ? " — ⚠ confounded by lower reasoning-model accuracy" : "")
            + (rep.drift_available ? "" : "; per-step drift pending attention instrumentation");
      } else if (rep.primary_evidence === "drift") {
        sub = `${rep.reasoning_model} · grounding-drift slope ${fmt(rep.drift_slope, 4)} (decays along the chain)`;
      }
      const head = rep.replicated
        ? (rep.primary_evidence === "drift" ? "Headline REPLICATED" : "Headline SUPPORTED")
        : "Headline NOT supported — reporting the actual effect";
      vEl.innerHTML = `<span class="icon">${rep.replicated ? "✓" : "✕"}</span>
        <div><div>${head} <span style="font-weight:500;opacity:.7">(${rep.primary_evidence} lens)</span></div>
        <div class="sub">${sub}</div></div>`;
    } else { vEl.className = ""; vEl.innerHTML = ""; }

    const models = bt.models || {};
    const series = [];
    Object.values(models).forEach(m => {
      const curve = m.iou_vs_step_index || {}; const keys = Object.keys(curve).map(Number).sort((a, b) => a - b);
      if (!keys.length) return;
      series.push({ name: m.name, color: m.is_reasoning ? COL.primary : COL.grey, dash: !m.is_reasoning,
        points: keys.map(k => [k + 1, curve[String(k)]]) });
    });
    $("#chart-drift").innerHTML = series.length
      ? lineChart(series, { yMin: 0, ylab: "mean IoU vs GT", xlab: "reasoning step", xFmt: x => "s" + x }) +
        legend(series.map(s => ({ label: s.name, color: s.color })))
      : `<p class="empty">No step-grounding curve in this source.</p>`;

    const variants = ["original", "blank", "mismatch"];
    const mkeys = Object.keys(models);
    $("#chart-blind").innerHTML = mkeys.length
      ? barChart(variants, mkeys.map((k, i) => ({ name: models[k].name, color: [COL.primary, COL.abstain][i] || COL.grey,
          values: variants.map(v => models[k].accuracy ? models[k].accuracy[v] : null) })), { yMin: 0, yMax: 1, ylab: "accuracy", valueLabels: true }) +
        legend(mkeys.map((k, i) => ({ label: models[k].name, color: [COL.primary, COL.abstain][i] || COL.grey })))
      : `<p class="empty">No blind-test data.</p>`;
  }

  function renderMeasurement(src) {
    const val = (src.fmr_results || {}).validation || {};
    const keys = [["signal_a", "A", COL.a], ["signal_b", "B", COL.b], ["signal_c", "C", COL.c], ["fs", "Fused FS", COL.fs], ["confidence", "Confidence", COL.conf]];
    const cats = keys.map(k => k[1]);
    const vals = keys.map(k => val["auroc_" + k[0]]);
    $("#chart-auroc").innerHTML = vals.some(v => v != null)
      ? barChart(cats, [{ name: "AUROC", values: vals, color: COL.primary, colors: keys.map(k => k[2]) }].map(s => ({ ...s })),
          { yMin: 0.5, yMax: 1, ylab: "AUROC", valueLabels: true, hline: 0.5 }).replace(/fill="#0e7490"/g, "fill="+JSON.stringify(COL.primary))
      : `<p class="empty">No validation labels for this source (grounding labels unavailable).</p>`;
    // color each AUROC bar distinctly by re-rendering with per-cat series
    if (vals.some(v => v != null)) {
      $("#chart-auroc").innerHTML = barChart(cats, keys.map((k, i) => ({ name: k[1], color: k[2],
        values: cats.map((_, j) => j === i ? val["auroc_" + k[0]] : null) })), { yMin: 0.5, yMax: 1, ylab: "AUROC", valueLabels: true, hline: 0.5, hlineColor: "#94a3b8" });
    }
    renderSeparation(src);
  }

  function renderSeparation(src) {
    const recs = src.records || [];
    const labelled = recs.filter(r => r.grounded_latent === 0 || r.grounded_latent === 1);
    const box = $("#chart-sep"), toggle = $("#sep-toggle");
    if (!labelled.length) { toggle.innerHTML = ""; box.innerHTML = `<p class="empty">Grounding labels unavailable for this source — separation view needs them.</p>`; return; }
    const sigs = [["signal_a", "Signal A"], ["signal_b", "Signal B"], ["signal_c", "Signal C"], ["fs", "Fused FS"]];
    toggle.innerHTML = sigs.map((s, i) => `<button data-sig="${s[0]}" class="${i === (STATE.sepSig ? sigs.findIndex(x=>x[0]===STATE.sepSig) : 3) === i ? "active" : ""}">${s[1]}</button>`).join("");
    const sig = STATE.sepSig || "fs";
    const draw = (sg) => {
      const g = labelled.filter(r => r.grounded_latent === 1).map(r => r[sg]);
      const u = labelled.filter(r => r.grounded_latent === 0).map(r => r[sg]);
      box.innerHTML = histogram([{ name: "ungrounded", color: COL.ungrounded, values: u }, { name: "grounded", color: COL.grounded, values: g }],
        { xlab: (sigs.find(x => x[0] === sg) || [])[1] }) + legend([{ label: "grounded", color: COL.grounded }, { label: "ungrounded", color: COL.ungrounded }]);
    };
    toggle.querySelectorAll("button").forEach(b => b.onclick = () => {
      STATE.sepSig = b.dataset.sig; toggle.querySelectorAll("button").forEach(x => x.classList.remove("active")); b.classList.add("active"); draw(STATE.sepSig);
    });
    draw(sig);
  }

  function renderSafety(src) {
    const ab = (src.fmr_results || {}).abstention || {};
    const defs = [["fs", "Faithfulness Score", COL.fs, 2.6], ["confidence", "Answer confidence", COL.conf, 1.4],
                  ["signal_a_only", "Signal A", COL.a, 1], ["signal_b_only", "Signal B", COL.b, 1], ["signal_c_only", "Signal C", COL.c, 1]];
    const series = [];
    defs.forEach(d => { const o = ab[d[0]]; if (o && o.risk_coverage && o.risk_coverage.coverage) {
      series.push({ name: `${d[1]} (AURC ${fmt(o.aurc,3)})`, color: d[2], width: d[3], dots: false,
        points: o.risk_coverage.coverage.map((c, i) => [c, o.risk_coverage.risk[i]]) }); } });
    const gate = ab.fs && ab.fs.test;
    const marks = (gate && gate.n_retained && gate.coverage != null && gate.retained_error != null)
      ? [{ x: gate.coverage, y: gate.retained_error, color: COL.ungrounded }] : [];
    $("#chart-rc").innerHTML = series.length
      ? lineChart(series, { yMin: 0, xlab: "coverage (fraction answered)", ylab: "risk (error on answered)", hline: ab.alpha, hlineColor: "#111", marks,
          xTicks: [0, 0.25, 0.5, 0.75, 1] }) + legend(series.map(s => ({ label: s.name, color: s.color })).concat([{ label: `target α=${fmt(ab.alpha,2)}`, color: "#111" }]))
      : `<p class="empty">No risk–coverage data.</p>`;

    // guarantee card
    const g = ab.fs || {};
    const t = g.test || {};
    const hold = g.feasible ? (t.retained_error == null || t.retained_error <= (ab.alpha ?? 1)) : true;
    const gc = $("#guarantee-card");
    gc.className = "card guarantee " + (g.feasible ? (hold ? "hold" : "fail") : "fail");
    gc.innerHTML = `<h3>Distribution-free guarantee</h3>
      <div class="row"><span>Target error α</span><b>${fmt(ab.alpha,2)}</b></div>
      <div class="row"><span>Threshold τ (FS ≥)</span><b>${g.threshold === null || g.threshold === undefined ? "—" : (g.threshold === Infinity || g.threshold > 1e6 ? "∞ (abstain all)" : fmt(g.threshold,3))}</b></div>
      <div class="row"><span>Coverage (answered)</span><b>${pct(t.coverage)}</b></div>
      <div class="row"><span>Error on answered</span><b>${pct(t.retained_error)}</b></div>
      <div class="row"><span>Feasible / guarantee holds</span><b>${g.feasible ? "yes" : "no"} / ${hold ? "✓" : "✕"}</b></div>
      ${ab.provisional_pre_correction ? `<p class="note" style="margin-top:8px">Provisional: pre-correction FS.</p>` : ""}`;

    // per-modality
    const pm = (src.fmr_results || {}).per_modality || {};
    const mkeys = Object.keys(pm);
    $("#chart-modality").innerHTML = mkeys.length
      ? barChart(mkeys, [{ name: "accuracy", color: COL.primary, values: mkeys.map(m => pm[m].accuracy) },
          { name: "mean FS", color: COL.grounded, values: mkeys.map(m => pm[m].mean_fs) }], { yMin: 0, yMax: 1, valueLabels: false }) +
        legend([{ label: "accuracy", color: COL.primary }, { label: "mean FS", color: COL.grounded }])
      : `<p class="empty">No per-modality breakdown.</p>`;
  }

  function decisionFor(src, r) {
    const t = src.fs_threshold;
    if (t === null || t === undefined) return null;
    return r.fs >= t ? "answer" : "abstain";
  }

  function renderExplorer(src) {
    const recs = src.records || [];
    const filters = [["all", "All"], ["answer", "Answered"], ["abstain", "Abstained"], ["grounded", "Grounded"], ["ungrounded", "Ungrounded"]];
    $("#case-filters").innerHTML = filters.map(f => `<button data-f="${f[0]}" class="${STATE.filter === f[0] ? "active" : ""}">${f[1]}</button>`).join("");
    $("#case-filters").querySelectorAll("button").forEach(b => b.onclick = () => { STATE.filter = b.dataset.f; STATE.caseIdx = null; renderExplorer(src); });

    const pass = (r) => {
      const d = decisionFor(src, r);
      switch (STATE.filter) {
        case "answer": return d === "answer";
        case "abstain": return d === "abstain";
        case "grounded": return r.grounded_latent === 1;
        case "ungrounded": return r.grounded_latent === 0;
        default: return true;
      }
    };
    const list = recs.filter(pass);
    const listEl = $("#case-list");
    if (!list.length) { listEl.innerHTML = `<p class="empty">No cases match.</p>`; $("#case-detail").innerHTML = ""; return; }
    listEl.innerHTML = list.map((r, i) => {
      const d = decisionFor(src, r);
      const gl = r.grounded_latent;
      return `<button class="case-item ${STATE.caseIdx === r.sample_id ? "active" : ""}" data-id="${r.sample_id}">
        ${gl === 1 ? '<span class="dot g" title="grounded"></span>' : gl === 0 ? '<span class="dot u" title="ungrounded"></span>' : ''}
        <span class="cq">${escapeHtml(r.question || r.sample_id)}</span>
        ${d ? `<span class="pill ${d}">${d}</span>` : ""}
      </button>`;
    }).join("");
    listEl.querySelectorAll(".case-item").forEach(b => b.onclick = () => { STATE.caseIdx = b.dataset.id; renderExplorer(src); });

    const chosen = list.find(r => r.sample_id === STATE.caseIdx) || list[0];
    STATE.caseIdx = chosen.sample_id;
    renderCase(src, chosen);
    listEl.querySelectorAll(".case-item").forEach(b => b.classList.toggle("active", b.dataset.id === chosen.sample_id));
  }

  function renderCase(src, r) {
    const d = decisionFor(src, r);
    const sig = (name, val, color) => `<div class="sigbar"><span>${name}</span>
      <span class="track"><span class="fill" style="width:${Math.round((val||0)*100)}%;background:${color}"></span></span><b>${fmt(val,3)}</b></div>`;
    const steps = (r.steps_text || []);
    const stepFs = r.fs_per_step || [];
    const stepsHtml = steps.length ? `<h4 style="margin:16px 0 6px;font-size:13px;color:#475569">Reasoning chain (per-step faithfulness)</h4>
      <ul class="steps">${steps.map((tx, i) => { const v = stepFs[i]; const col = v == null ? COL.grey : (v > 0.5 ? COL.grounded : v > 0.33 ? COL.abstain : COL.ungrounded);
        return `<li><span>${escapeHtml(tx)}</span><span class="stepfs"><div style="width:${Math.round((v||0)*100)}%;background:${col}"></div></span></li>`; }).join("")}</ul>` : "";
    $("#case-detail").innerHTML = `
      <p class="q">${escapeHtml(r.question || r.sample_id)}</p>
      <p class="meta">${r.modality || "?"} · id ${r.sample_id} · model answer <b>${escapeHtml(String(r.answer))}</b> vs truth <b>${escapeHtml(String(r.gt_answer))}</b>
        <span class="pill ${r.correct ? "correct" : "wrong"}">${r.correct ? "correct" : "wrong"}</span>
        ${r.grounded_latent === 1 ? '<span class="pill correct">grounded</span>' : r.grounded_latent === 0 ? '<span class="pill wrong">ungrounded</span>' : ''}</p>
      ${d ? `<div class="decision ${d}">${d === "answer" ? "✓ ANSWER" : "⚠ ABSTAIN → defer to clinician"}</div>` : ""}
      <div class="sigbars">
        ${sig("Signal A · counterfactual", r.signal_a, COL.a)}
        ${sig("Signal B · grounding", r.signal_b, COL.b)}
        ${sig("Signal C · consistency", r.signal_c, COL.c)}
        ${sig("Fused Faithfulness Score", r.fs, COL.fs)}
        ${sig("Model confidence", r.confidence, COL.conf)}
      </div>
      ${src.fs_threshold != null ? `<p class="note">Gate threshold τ = ${fmt(src.fs_threshold,3)}: answer iff FS ≥ τ.</p>` : `<p class="note">No calibrated threshold in this source.</p>`}
      ${stepsHtml}`;
  }

  function renderRobustness(src) {
    const fb = src.full_benchmark || {}; const abl = src.ablations || {};
    // incremental fusion across models
    const models = fb.models || {};
    const stages = ["auroc_fs_A", "auroc_fs_AB", "auroc_fs_ABC"];
    const mkeys = Object.keys(models).filter(k => models[k].validation && models[k].validation.auroc_fs_A != null);
    $("#chart-incr").innerHTML = mkeys.length
      ? lineChart(mkeys.map((k, i) => ({ name: models[k].model, color: [COL.primary, COL.c][i] || COL.grey,
          points: stages.map((s, j) => [j, models[k].validation[s]]) })), { ylab: "AUROC", xTicks: [0, 1, 2], xFmt: x => ["A", "A+B", "A+B+C"][x] }) +
        legend(mkeys.map((k, i) => ({ label: models[k].model, color: [COL.primary, COL.c][i] || COL.grey })))
      : `<p class="empty">Run the full benchmark to populate this.</p>`;

    // weight sensitivity
    const ws = abl.weight_sensitivity;
    $("#chart-weights").innerHTML = ws
      ? barChart(["min", "mean", "max"], [{ name: "AUROC", color: COL.primary, values: [ws.auroc_min, ws.auroc_mean, ws.auroc_max] }],
          { yMin: 0.5, yMax: 1, valueLabels: true, hline: ws.default_weights_auroc, hlineColor: COL.grounded }) +
        legend([{ label: `default weights = ${fmt(ws.default_weights_auroc,3)}`, color: COL.grounded }])
      : `<p class="empty">Run ablations to populate this.</p>`;

    // grid sensitivity
    const gs = abl.grid_sensitivity;
    if (gs) {
      const gk = Object.keys(gs).map(Number).sort((a, b) => a - b);
      $("#chart-grid").innerHTML = lineChart([
        { name: "Signal B", color: COL.b, points: gk.map(g => [g, gs[String(g)].auroc_signal_b]) },
        { name: "Fused FS", color: COL.fs, points: gk.map(g => [g, gs[String(g)].auroc_fs]) }],
        { yMin: 0.5, yMax: 1, ylab: "AUROC", xlab: "grid resolution", xTicks: gk }) +
        legend([{ label: "Signal B", color: COL.b }, { label: "Fused FS", color: COL.fs }]);
    } else $("#chart-grid").innerHTML = `<p class="empty">Run ablations to populate this.</p>`;

    // conformal power
    const pw = abl.abstention_power && abl.abstention_power.min_cal_size_for_alpha;
    if (pw) {
      const ak = Object.keys(pw).map(Number).sort((a, b) => a - b);
      $("#chart-power").innerHTML = lineChart([{ name: "min cal size", color: COL.abstain,
        points: ak.map(a => [a, pw[String(a)] == null ? 0 : pw[String(a)]]) }], { yMin: 0, ylab: "min calibration N", xlab: "target error α", xTicks: ak, xFmt: x => x });
    } else $("#chart-power").innerHTML = `<p class="empty">Run ablations to populate this.</p>`;
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  /* ------------------------------------------------------------------ shell */
  const STATE = { source: DATA.default_source, filter: "all", caseIdx: null, sepSig: "fs" };

  function renderAll() {
    const src = activeSource();
    const badge = $("#source-badge");
    badge.textContent = src.kind === "real" ? "REAL DATA" : "MOCK / SYNTHETIC";
    badge.className = "badge " + (src.kind === "real" ? "real" : "mock");
    renderOverview(src); renderDiagnosis(src); renderMeasurement(src);
    renderSafety(src); renderExplorer(src); renderRobustness(src);
  }

  function initSourcePicker() {
    const sel = $("#source");
    const keys = Object.keys(DATA.sources);
    if (!keys.length) { $("#app").innerHTML = `<p class="empty">No data yet. Run the pipeline and <code>python fmr/scripts/make_dashboard.py</code>.</p>`; return false; }
    sel.innerHTML = keys.map(k => `<option value="${k}" ${k === STATE.source ? "selected" : ""}>${DATA.sources[k].label || k}</option>`).join("");
    sel.onchange = () => { STATE.source = sel.value; STATE.caseIdx = null; renderAll(); };
    return true;
  }

  function initTabs() {
    const links = Array.from(document.querySelectorAll(".tabs a"));
    const secs = links.map(a => document.querySelector(a.getAttribute("href")));
    const obs = new IntersectionObserver((ents) => {
      ents.forEach(e => { if (e.isIntersecting) { const id = "#" + e.target.id;
        links.forEach(l => l.classList.toggle("active", l.getAttribute("href") === id)); } });
    }, { rootMargin: "-140px 0px -70% 0px" });
    secs.forEach(s => s && obs.observe(s));
  }

  if (initSourcePicker()) { renderAll(); initTabs(); }
})();
