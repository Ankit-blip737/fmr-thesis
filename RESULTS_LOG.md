# RESULTS_LOG.md — running experimental log (append-only)

Every entry: `[A]`/`[B]` tag · date · what ran · actual numbers · verdict.
Never rewrite or delete an existing entry.

---

### [B] 2026-07-03 — Stage 4 correction module: unit tests + full mock verification

**Shared-foundation fix first:** the initial commit could not `import fmr` at all
(circular import `types → data.regions → data/__init__ → synthetic → types`).
Fixed minimally in `types.py` (`Region` import moved under `TYPE_CHECKING`); logged
in DECISIONS.md [B]. This unblocks *everything*, Instance A's code included.

**Unit tests:** `fmr/tests/test_correction.py` — **12 passed** (1.05s). Covers VCD
math (prior cancels / evidence amplified / no-op when distributions match /
plausibility mask blocks implausible promotion), clue-tracing grounded-vs-blind
separation and near-GT region recovery, verify-revise step dropping + no-mutation +
conservative answer policy, rescore output-shape parity with Signal A, and pipeline
selective application / prior-rescue / abstention routing / second-VLM factory.

**Full run:** `scripts/run_correction.py --model both --n 400` (synthetic, seed 7).
Two mock backends, two deliberately different failure modes.

**MockVLM `mock-reasoner` (image-blind pathology):** 265/400 flagged (FS<0.5).
- *Fixable (grounded latent, n=61):* acc 1.00→1.00; **FS_after mean 0.54; 91.8%
  recover above the abstention gate** — i.e. grounded chains flagged only for
  reasoning-drift are re-grounded so they need not be deferred. This is the
  "correction shrinks the abstain set" story.
- *Image-blind (latent, n=204):* acc 0.539→0.529 (flat), FS_after 0.044, **96.1%
  stay below the gate → routed to abstention.** Correction correctly refuses to
  fabricate confidence when there is no image signal.
- *Regression (correction force-applied to 135 unflagged):* **0 correct answers
  broken**, acc 1.00→1.00.

**PriorHeavyMockVLM `mock-prior-heavy` (prior-dominated pathology — VCD's target):**
393/400 flagged.
- *Fixable (grounded latent, n=189):* **acc 0.534→0.868 (+33.3 pts)** — VCD flips
  prior-dominated wrong answers to the image-grounded truth. Of the 66 answers it
  changed, the rescue is decisive (see audit rows syn-00001/2/4/7: prior answer →
  GT answer, FS 0.02→~0.60).
- *Image-blind (latent, n=204):* acc 0.539→0.534 (flat), FS_after 0.036, **98.5%
  stay below the gate → abstention.**
- *Regression (forced on 7 unflagged):* **0 correct answers broken.**

**Verdict:** PASS. Both the improvement claim (prior-dominated cases rescued,
+33 pts accuracy on the fixable subset) and the two non-regression guarantees
(image-blind cases left to abstention; well-grounded cases never corrupted) hold.
The fixable/must-abstain split is exactly the framing required by external-review
fix #1: correction is the layer that *shrinks* what abstention must defer, not a
co-equal contribution. Artifacts: `fmr/results/correction_{summary,audit}_*.{json,jsonl}`.

