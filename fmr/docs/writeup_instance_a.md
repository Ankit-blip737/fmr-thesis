# FMR — Draft thesis sections (Instance A components)

Companion to `writeup_instance_b.md` (which covers Pillar 2 correction §X, its
contribution to abstention §Y, LLM-judge validation §Z, and real-model
integration §W). This document drafts the sections owned by the measurement /
diagnosis / abstention side: the diagnostic headline (§D), the Faithfulness
Measurement Module (§M), conformal faithfulness-aware abstention (§T), and an
honest analysis of the learned verifier's weak-label ceiling (§V).

All numbers are reproducible from `fmr/outputs/` (mock/synthetic machinery) and
`fmr/outputs/real/<dataset>/` (real-model runs on Colab). Synthetic figures
validate the *machinery* and are labelled as such throughout; empirical claims
about model behaviour come from the real-model runs.

---

## D. The Diagnosis — Do Medical Reasoning VLMs Look at the Image?

### D.1 Protocol
Two lenses, both model-agnostic and requiring no ground-truth regions:

1. **Blind test.** Re-run each model on the original image, a mid-grey **blank**,
   and a **mismatched** image. `blind_gap = acc(original) − acc(blank)` measures
   how much the answer depends on the image. A small gap means the model answers
   largely from language priors.
2. **Grounding-drift.** For a reasoning model, track per-step grounding (attended
   region ↔ evidence) as a function of step index. A negative slope is the
   headline "more reasoning → less grounded" effect.

The headline is evaluated by an explicit, auto-computed **replication verdict**
(`run_blind_test._replication_verdict`) that reports whichever lens has evidence,
flags confounds, and — critically — refuses to declare support when the data
does not warrant it.

### D.2 Findings (real models, VQA-RAD)
On VQA-RAD (MedVLM-R1 reasoning vs Qwen2.5-VL-3B non-reasoning):

| Model | acc(orig) | acc(blank) | blind_gap |
|---|---|---|---|
| MedVLM-R1 (reasoning) | 0.40 | 0.33 | **0.067** |
| Qwen2.5-VL-3B (non-reasoning) | 0.573 | 0.44 | **0.133** |

Two honest observations:
- **Both models are substantially prior-driven** — blanking the image leaves
  accuracy at 0.33–0.44, far above chance. Medical VQA answerable-without-image
  is a real phenomenon, reproduced here.
- The **reasoning model relies on the image less** (gap 0.067 < 0.133),
  *consistent* with the headline hypothesis via the blind-gap lens — **but
  confounded** by MedVLM-R1's lower overall accuracy (a weaker model can show a
  smaller gap for the wrong reason). We report this as *supported-with-caveat*,
  not proven.

### D.3 A reported null (honesty)
On **PathVQA**, both models show ~zero/negative blind-gaps: they are effectively
**image-independent** on this split. The verdict machinery detects this
(`both_image_independent`, gaps ≤ 0.03) and explicitly reports **"not a grounding
effect"** rather than exploiting the numerically-smaller reasoning gap to claim
support. This is the review-mandated Plan B in action: the framing follows the
data.

### D.4 The sharper test, and its instrumentation dependency
The per-step **grounding-drift** curve is the sharper form of the headline. It
requires per-step attended regions, which on real HF models need attention-
rollout extraction (a documented, still-exploratory plug point). On the
**synthetic** machinery — where regions are known exactly — the drift replicates
cleanly (mean IoU 0.32 → 0.22 across a 4-step chain, slope −0.03), validating
that the measurement and verdict logic are correct and will register the effect
when real attention extraction lands. Until then, the real-data headline rests on
the blind-gap lens (§D.2), stated with its confound.

---

## M. Faithfulness Measurement Module (Pillar 1)

### M.1 Three complementary signals
Robustness principle: **never trust a single signal.**

- **Signal A — counterfactual sensitivity.** Answer-flip rate + JS-divergence of
  the answer distribution between the original and blank/mismatched images. A
  faithful answer changes when the image is removed/replaced.
- **Signal B — attention grounding.** Runtime: spatial coherence of the attended
  regions across the chain. Validation/weak-labels: IoU against ground-truth
  regions **where boxes exist** (synthetic; SLAKE/VQA-RAD once masks are wired).
  On box-free data B is reported as unvalidated/exploratory (review fix #3).
- **Signal C — self-consistency.** Agreement of answers across N sampled chains
  (RadFlag-style), rescaled so chance agreement maps to 0.

These fuse into a single **Faithfulness Score (FS)** ∈ [0,1]; the raw per-signal
scores are exposed (`compute_faithfulness`) so the learned verifier (§V) can
train on them.

### M.2 Validation (machinery, synthetic)
On the synthetic fixture with a known binary grounding latent, each signal is
informative and the fused FS separates grounded from ungrounded reasoning; class-
mean gaps are correctly signed for A, B, C, FS and GT-IoU (grounded > ungrounded
throughout). Manual inspection of examples confirms signal directionality (e.g.
lucky-prior-correct ungrounded answers correctly receive low B/IoU).

### M.3 Robustness & sensitivity (deepest-treatment ablations)
- **Fusion-weight sensitivity.** Sweeping the full (a,b,c) simplex, fused-FS
  AUROC varies in a **tight band** (std ≈ 0.03) — the score does not hinge on a
  lucky weighting. The envelope maximum is the concrete number the learned
  verifier must beat.
- **Signal-B grid sensitivity.** The grounding signal is present across region-
  grid resolutions (rises then plateaus) — **not a grid artifact.**
- **Seed stability.** Signal AUROCs are stable across independent synthetic draws
  (std ≤ 0.02).

---

## T. Trust — Conformal Faithfulness-Aware Abstention (Pillar 3)

### T.1 The guarantee
We control **selective risk** — the error rate *conditional on answering*. Naive
split-conformal quantiles bound marginal coverage, not this quantity, so we use
the **Selection-with-Guaranteed-Risk** procedure (Geifman & El-Yaniv, NeurIPS
2017): an exact Clopper-Pearson upper confidence bound on retained error,
Bonferroni-corrected over a bounded quantile grid of candidate thresholds. With
probability ≥ 1−δ over the calibration draw, the retained error at the chosen
threshold τ is ≤ α. If no threshold certifies, the safe output is **abstain-all**
(reported honestly, never hidden).

### T.2 It works, and it has a data requirement
On synthetic calibration (n=1000), the α=0.05 gate certifies with **coverage
0.58, retained error 0.019 ≤ 0.05**. The honest **power finding**: certifying a
5%-error / 95%-confidence guarantee genuinely needs ≳500–1000 calibration points;
at n=200 the α=0.05 bound is infeasible (needs α≈0.15). The thesis therefore
states the calibration-size requirement explicitly rather than reporting an
abstain-all as if it were a modelling failure. Consequence for real data: pool
calibration across datasets or relax α on small sets.

### T.3 Calibration ordering (review fix #4)
The gate is calibrated on the **post-correction** FS — the score the deployed
system actually emits — recomputed on the corrected output, not on the raw
pre-correction FS and not on the correction module's Signal-A-only rescore. Both
the deployed (post-correction) and the pre-correction contrast are reported.

### T.4 Head-to-head vs standard deferral triggers (proposal §8)
`run_abstention_baselines.py` compares FS against confidence-thresholding,
self-consistency / RadFlag-style consistency thresholding, and each single FMM
signal, on both mock and real records, by AURC, coverage@risk, and risk@coverage.
On the **mock**, the model's softmax confidence is near-oracle, so confidence and
FS are statistically tied (AURC 0.080 vs 0.083) — expected and stated. The
motivating hypothesis is that on **real** models, whose confidence is known to be
mis-calibrated/over-confident, FS becomes the better trigger; the harness
produces this comparison directly from each real run (larger real runs needed to
settle it — current smoke sets are too small for a clean verdict).

---

## V. The Learned Verifier's Weak-Label Ceiling (honest RQ5 complement)

Instance B's §W.2 reports the positive result: a learned fusion of Signals A/B/C
beats the hand-weighted heuristic on real signals — **heuristic AUROC 0.768 →
learned (GBT, weak labels) 0.816 (+0.048)**, logreg 0.801 (+0.033). This section
adds the honest ceiling that keeps the claim defensible.

The verifier is trained on **weak** grounding labels derived from counterfactual
answer-flip behaviour (no ground-truth boxes), which agree with the true latent
only **~0.77** of the time. That weak-label fidelity is a ceiling: a GBT trained
on the **true** labels (oracle) reaches AUROC **0.951**, far above the weak-label
model's 0.816. So the honest reading of RQ5 is:

- **Yes, a learned fusion genuinely helps** (+0.03–0.05 AUROC over the heuristic,
  reproducibly, on real signals) — the trained component earns its place.
- **But its ceiling is set by the weak-label quality**, not by the model class.
  The 0.816 → 0.951 gap is *recoverable only with real grounding supervision*
  (GT boxes / masks) — which is exactly the Signal-B validation limitation
  (§M.1, review fix #3). This ties the trained-enhancement story back to the
  measurement limitation rather than over-claiming.

This is a stronger scientific statement than "we trained something and it helped":
it quantifies *how much* the training helps, *why* it is bounded, and *what* would
lift the bound — with a training-free heuristic as the guaranteed fallback
throughout.

---

*Cross-references: §X/§Y/§Z/§W in `writeup_instance_b.md`. All results reproducible
via `fmr/scripts/{run_baselines,run_blind_test,run_fmr,run_fmr_full,run_ablations,
run_abstention_baselines}.py` and the dashboard (`fmr/dashboard/index.html`).*
