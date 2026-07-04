FULL UI/UX OVERHAUL
═══════════════════════════════════════════════════════════

NAVIGATION ARCHITECTURE (the core structural change requested)
- Replace the current long-scroll-with-anchor-links behavior with true single-page-app 
  tab switching: each tab (Overview, The Diagnosis, Measurement, Safety Layer, Case 
  Explorer, Robustness, plus new tabs below) is its own view. Clicking a tab slides/
  cross-fades the content out and the new tab's content in — smooth, ~200-300ms 
  transition, no jarring cuts. Keep the URL hash in sync (#safety etc.) so links/refresh 
  still work, but the visual behavior should feel like switching screens, not scrolling.
- Add keyboard navigation (arrow keys or number keys to switch tabs) and make the active 
  tab state visually unambiguous (not just a color change — consider an animated 
  underline/indicator that slides between tab positions).

VISUAL DESIGN SYSTEM
- Move beyond the current flat teal-header-plus-white-cards look toward something that 
  reads as a genuine thesis-defense-quality product: consider a cohesive color system 
  with a real accent hierarchy (not just teal everywhere), proper type scale (distinct 
  sizes for page titles vs section headers vs body vs captions — currently everything 
  reads fairly uniform), and intentional whitespace/rhythm between sections rather than 
  uniform card padding throughout.
- Add a dark mode toggle — genuinely useful for a late-night defense-prep session, and 
  signals polish.
- Micro-interactions: hover states on cards/chart elements, smooth number count-up 
  animations when stat cards first render, subtle chart entrance animations (bars 
  growing in, lines drawing in) rather than appearing instantly.
- Redesign the header: current version is functional but plain. Consider a slimmer 
  sticky header on scroll, a more distinctive logo treatment, and moving the data-source 
  switcher + REAL/MOCK badge into a more prominent, visually distinct control area.

HANDLING SMALL-N AND PROVISIONAL DATA HONESTLY (important — don't lose this in the 
redesign)
- n=20 real-data results need visual treatment that communicates uncertainty, not just a 
  text caveat: consider confidence bands/error bars where statistically meaningful, a 
  visible sample-size badge directly on every chart that uses the real source (not just 
  in a banner at the top), and a distinct visual style (e.g., dashed lines, muted 
  saturation) for any curve/table built from fewer than some threshold (~50?) samples.
- Never let smoothing, animation, or polish imply a trend/precision that 20 data points 
  don't support. A jagged real risk-coverage curve should look jagged, not smoothed into 
  false confidence.
- Keep mock vs. real visually distinguishable at a glance everywhere, not just via the 
  header badge — e.g. a persistent small watermark/pattern on mock-sourced charts.

TAB-BY-TAB CONTENT ENHANCEMENTS
- Overview: add a compact "how it works" mini-diagram (input → 3 signals → fusion → 
  correction → abstention gate → output) so a first-time viewer (examiner) orients 
  immediately, before reading prose.
- The Diagnosis: once real attention grounding lands (separate backend task, already in 
  progress), the step-grounding curve panel should come alive here — leave the layout 
  ready for it now rather than redesigning again later.
- Measurement: add tooltips/info-icons on "AUROC," "IoU," "Signal A/B/C" explaining each 
  in one sentence for a non-specialist committee member reading quickly.
- Safety Layer: make the risk-coverage chart interactive — hovering a point shows exact 
  coverage/error/method; add a slider to let the viewer pick a hypothetical α and see the 
  resulting threshold/coverage update live (compute this client-side from existing risk-
  coverage data, don't fabricate).
- Case Explorer: add a search/filter box (currently only category pills), and show the 
  ground-truth answer alongside model answer more prominently for at-a-glance scanning 
  across many cases.
- Robustness: once Part 1's fixes land, these panels should show real content — design 
  them now assuming they'll be populated, with a clear "insufficient data" state that's 
  visually distinct from "not yet run."

NEW TABS/SECTIONS TO ADD
- "Methodology" or "How It Works": the architecture diagram from the proposal (the three 
  pillars flow), rendered as an actual interactive diagram, not just prose — this is 
  likely the first thing an examiner wants oriented before diving into results.
- "Limitations & Honesty": promote the honesty notes currently buried as small text at 
  the bottom of Robustness into their own first-class tab. Given how central honest 
  reporting is to this whole project, it deserves visibility, not a footnote.
- "Project Timeline" (optional, nice-to-have): a chronological view built from 
  RESULTS_LOG.md entries showing the project's actual progression — useful defense 
  narrative material ("here's what we tried and found, in order").

INTERACTIVITY BASELINE
- Every chart should respond to hover with exact values (many currently look static).
- The data-source switcher should support a side-by-side comparison mode (Mock vs Real) 
  for at least the headline metrics, so the viewer can see machinery-validation vs. 
  empirical-finding at a glance rather than switching back and forth.
- Add a "copy/export this view" or "generate PDF snapshot" action — useful for pulling 
  specific charts into the actual thesis document or defense slides later.

TECHNICAL CONSTRAINTS
- Reuse existing data files and wiring (app.js, the dashboard data bundle) — this is a 
  restructure and enhancement, not a rewrite from scratch, unless a specific piece of the 
  current architecture genuinely can't support the above.
- You hit a browser-caching issue last session (stale app.js) — verify with a hard 
  cache-bypass reload before considering any visual change confirmed working.
- Test both Mock and Real data sources against every tab after changes — several panels 
  currently behave differently depending on source (e.g., Robustness empty on Real); 
  make sure the new design handles both gracefully, not just whichever source you happen 
  to test with.
- Commit incrementally (per tab or per major section) rather than one giant commit, so 
  if something regresses it's easy to isolate.