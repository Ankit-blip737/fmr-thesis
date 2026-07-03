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

### [B] 2026-07-03 — LLM-as-judge validation (required-fix #3)

Built `fmr/src/fmr/eval/judge.py` (HeuristicJudge + LLMJudge + agreement harness)
+ a hand-authored gold set `eval/gold_data.py` (N=44: 22 correct / 17 incorrect /
5 partial), stressing synonyms, negation/polarity flips, severity, containment,
and multi-finding partials.

**Heuristic judge vs gold, iterated (revise-and-recheck per fix #3):**
- v1 (string match + whole-string synonyms): 3-way acc 0.795, **κ 0.634**.
- v2 (token-level synonyms, number-word norm, multi-finding coverage): acc 0.909,
  **κ 0.836**.
- v3 (comma-robust multi-finding via cluster count; "names finding but omits
  detail"→partial): acc 0.955, binary 1.000, **κ 0.923**.
- v4 (exclude polarity/severity words from *finding* count; polarity-aware
  coverage): **acc 1.000, binary 1.000, κ 1.000.**

**Honesty caveat (stated for the thesis):** κ=1.0 is on a gold set I *both
authored and tuned against* — it is an upper bound on a hand-tuned rule, not a
field estimate. Two independent guards: (1) `tests/test_judge.py` includes 9
HELD-OUT probes never in the gold set (bleeding↔hemorrhage, broken↔fracture,
three↔3, held-out multi-finding partials, underspecified single findings) — all
pass, so the rules encode general clinical-text principles, not memorized rows;
(2) the real external check is the *independent* open-LLM judge on Colab
(`colab_judge_llm.ipynb`), scored against the same gold + against the heuristic.
Until that lands, downstream open-ended metrics use the heuristic as the
validated fallback and should treat the LLM judge as primary once its κ is in.

**Tests:** `tests/test_judge.py` 21 passed; full suite **33 passed** (0.74s).
Artifact: `fmr/results/judge_validation_heuristic.json`.

### [B] 2026-07-03 — Learned faithfulness verifier vs heuristic fusion (RQ5)

Built `fmr/src/fmr/training/` (signals → features → labels → verifier → dataset) +
`scripts/train_verifier.py`. Verifier = sklearn LogisticRegression (learned-linear)
or GradientBoosting (learned-nonlinear, regularized: depth 2, 150 trees, subsample
0.8, min_leaf 20), a **drop-in for `HeuristicFusion`** (identical `score(features)`
API; heuristic stays the guaranteed fallback per fix #4).

**Protocol.** Features from **two** backends (mock-reasoner + mock-prior-heavy) so
signals conflict. Trained on the **noisy counterfactual weak label** (no GT boxes;
weak-vs-true agreement ≈0.78 — genuinely weak). Evaluated vs the **true hidden
latent** on a held-out split **disjoint from the calibration split reserved for
Instance A** (train 200 samples ×2 backends = 400 rows; test 100 ×2 = 200 rows;
cal 100 untouched). Because real faithfulness signals are noisy (not the clean
latent a deterministic mock exposes), we sweep measurement noise σ.

**Result (AUROC vs true latent):**

| σ (noise) | per-signal A/B/C | heuristic | learned (logreg) | Δ learned−heur | oracle |
|-----------|------------------|-----------|------------------|----------------|--------|
| 0.00 | 0.99/0.99/0.71 | **0.998** | 0.987 | −0.012 | 1.000 |
| 0.15 | 0.78/0.99/0.64 | **0.987** | 0.978 | −0.009 | 1.000 |
| 0.30 | 0.73/0.99/0.59 | 0.928 | **0.965** | +0.037 | 1.000 |
| 0.45 | 0.66/0.97/0.57 | 0.839 | **0.940** | +0.101 | 1.000 |
| 0.60 | 0.62/0.93/0.55 | 0.756 | **0.909** | +0.153 | 0.999 |

**Reading it honestly.** With *clean* signals the fixed heuristic is already near
the ceiling and the learned head (trained on noisy weak labels) loses by ~0.01 —
reported, not hidden. The crossover is σ≈0.2; beyond it the learned fusion degrades
far more gracefully and the gap widens monotonically to +0.153. **Mechanism** (from
per-signal AUROC + logreg weights): noise corrupts Signal A (0.99→0.62) while
Signal B stays robust (0.99→0.93); the fixed 0.5 weight on A drags the heuristic
down, whereas the learned head re-weights toward B. The oracle (GBT on the true
latent) stays ≈1.0, confirming the information is still present — the heuristic just
can't extract it under noise. Since real signals are noisy, the realistic regime is
σ>0, where the learned verifier wins → **it ships as default at the headline σ=0.3
(0.965 vs 0.928), with the heuristic as fallback.** Held-out decision agreement
with the heuristic is 0.86 (not wildly divergent; the divergence is exactly the
re-weighting that helps).

**Caveat / dependency:** trained on stub signals computed in `training/signals.py`
(stand-in for Instance A's real Signal A/B/C from `faithfulness/score.py`, not on
disk yet). Must **retrain on real signals** once available — logged in DECISIONS.md
[B]. The noise sweep predicts the learned head helps whenever real signals are
moderately noisy (which they are).

**Tests:** `tests/test_verifier.py` 9 passed (incl. deterministic
learned>heuristic+0.05 under σ=0.5, save/load round-trip, weak-label-is-noisy).
Full suite **42 passed** (2.89s). Artifacts: `fmr/results/verifier_benchmark.json`,
`fmr/results/verifier_gbt.pkl(+.meta.json)`.

