# DECISIONS.md — autonomous design decisions (append-only)

Format: `[A]`/`[B]` tag · date · decision · reasoning. Never rewrite or delete an
entry; append corrections as new entries.

---

- **[B] 2026-07-03 — Correction operates at the answer-distribution level through
  `BaseVLM.generate`.** True VCD contrasts next-token logits *during* decoding; the
  `BaseVLM` contract exposes per-variant answer distributions (`answer_logits`),
  which for the closed-vocabulary Med-VQA setting *is* the next-token distribution
  over answers. `fmr.correction.vcd` therefore implements the exact VCD math
  ((1+α)·log p_orig − α·log p_distorted + adaptive plausibility mask, Leng et al.
  CVPR 2024) at answer granularity, model-agnostically. The per-token variant for
  real HF models lives in the GPU handoff notebook as a thin adapter that
  implements the same `generate` contract — nothing downstream changes.

- **[B] 2026-07-03 — Signal A (`counterfactual_signal`) is the correction trigger
  until the fused FS exists.** Instance A's `faithfulness/score.py` (fused FS +
  per-signal API) is not on disk yet. The correction pipeline takes an optional
  externally-computed `fs`; when absent it falls back to Signal A alone.
  **Dependency:** once `score.py` lands, callers should pass the fused FS.

- **[B] 2026-07-03 — Post-correction rescoring lives in
  `correction/rescore.py`, same output shape as `counterfactual_signal`.**
  Required-fix #2 (calibration ordering): conformal calibration must be computed on
  faithfulness *after* correction. `CorrectionResult.fs_after` carries that value;
  `rescore.post_correction_sensitivity` mirrors Signal A's dict shape
  (`counterfactual` / `flip_rate` / `js_divergence`) so Instance A's calibration
  can consume it without adaptation. A "ready to wire" note follows once verified.

- **[B] 2026-07-03 — Second mock VLM (`PriorHeavyMockVLM`) subclasses `MockVLM`,
  overriding only `_pref` + defaults.** Read-only reuse of Instance A's class (no
  edits to `mock_vlm.py`). Profile: a language-prior boost in *every* variant plus
  a weaker latent image-evidence boost when grounded — i.e. the model *sees* the
  finding but the prior dominates. This is the failure mode VCD provably fixes
  (prior term cancels in the contrast, evidence term is amplified), and it is
  distinct from `MockVLM`'s failure mode (fully image-blind when ungrounded),
  giving the cross-model comparison two different pathologies. If Instance A
  refactors `MockVLM` internals, this subclass may need a touch-up — flagged here.

- **[B] 2026-07-03 — Shared-foundation bugfix in `types.py` (circular import).**
  The initial commit has a package-fatal cycle: `fmr/__init__` → `types` →
  `data.regions` → `data/__init__` → `synthetic` → `types` (partially
  initialized) — *any* `import fmr` raised ImportError. Minimal fix applied in
  `types.py`: the `Region` import is now under `TYPE_CHECKING` (it is only used
  in PEP-563 lazy annotations there). No behavior change; nothing in Instance
  A's files touched. [A]: if you refactor `data/__init__` (e.g. stop importing
  `synthetic` eagerly), this guard can stay — it is harmless either way.

- **[B] 2026-07-03 — Framing (required-fix #1): correction is supporting
  infrastructure for abstention, not a co-equal pillar.** Code comments and the
  results write-up present correction as "fix what is fixable so fewer cases need
  deferring; leave truly image-blind cases to the calibrated abstention gate."
  The mock experiments are designed to show exactly that split: prior-dominated
  cases get rescued, image-blind cases stay low-FS and flow to abstention.

- **[B] 2026-07-03 — ✅ HANDOFF TO [A]: correction module ready for conformal
  calibration (required-fix #2 satisfied).** `fmr/src/fmr/correction/` is
  committed (a0abb9e), unit-tested (12 passing), and verified on 400 synthetic
  samples across two pathologies. **Instance A: build the split-conformal gate on
  the POST-correction faithfulness score, not the raw pre-correction score.**
  Concretely:
  * For each calibration/test sample, run
    `correct_sample(vlm, sample, fs=<your fused FS>, config=CorrectionConfig())`
    → `CorrectionResult.fs_after` is the deployed-system faithfulness score to
    calibrate on. When correction is not triggered, `fs_after == fs_before`, so
    passing everything through is safe and correct.
  * If you only need the counterfactual component recomputed for an already
    corrected output, `correction.post_correction_sensitivity(vlm, sample,
    corrected_output)` returns the same dict shape as your
    `counterfactual.counterfactual_signal` (`counterfactual`/`flip_rate`/
    `js_divergence`), so it drops straight into your fusion.
  * Pass your fused FS into `correct_sample(..., fs=FS)` so the correction
    trigger uses the real fused score instead of my Signal-A fallback. Until
    then it degrades gracefully to Signal A alone.
  This ordering is what makes the coverage guarantee measure the distribution the
  system actually deploys. Nothing else in your gate changes.

- **[B] 2026-07-03 — Judge verdicts are 3-way (correct/partial/incorrect), and
  the judge is validated before use (required-fix #3).** `eval/judge.py` returns
  a `JudgeVerdict(label, score, source)` where score = 1.0/0.5/0.0. The heuristic
  judge is the offline-validated fallback (κ=1.0 on a hand-authored N=44 gold set;
  see the honesty caveat in RESULTS_LOG — it's a tuned upper bound, cross-checked
  by held-out probes + an independent Colab LLM judge). **Handoff to [A]:** for
  Stage-6 open-ended scoring, import `from fmr.eval import build_judge`; use the
  LLM judge as primary once `colab_judge_llm.ipynb` reports its κ, else the
  heuristic. Don't treat any judge score as ground truth before its κ is logged.

- **[B] 2026-07-03 — Judge gold set lives in-package (`eval/gold_data.py`), not
  under `data/`.** `data/` is Instance A's territory; the judge gold is Instance
  B's eval asset, so it ships as an importable module (usable from tests, the
  validation script, and the Colab notebook without a filesystem path).

- **[B] 2026-07-03 — Learned verifier trains on STUB signals (`training/signals.py`);
  needs retraining on Instance A's real Signals A/B/C.** Instance A owns the
  per-signal sources; `faithfulness/score.py` (raw A/B/C API) is not on disk yet.
  `training/signals.py` computes A/B/C-shaped features from committed code
  (counterfactual signal + region IoU proxy + self-consistency sampling) as a
  faithful stand-in. **Rewire point:** `training/dataset.build_feature_frame` is
  the *only* place that calls the signal provider — swap it to Instance A's API
  and everything else (labels, verifier, eval, tests) is unchanged. **[A]: to
  make this trivial, expose per-step raw Signals A/B/C (not just fused FS) from
  `score.py` returning a dict with keys `sig_a_counterfactual`, `sig_b_grounding`,
  `sig_c_consistency` (+ any auxiliaries); I'll map the rest.**

- **[B] 2026-07-03 — Verifier measurement-noise model + noise sweep (design
  choice, not rigging).** The deterministic mock exposes near-clean signals
  (each ~0.99 AUROC), which is unrealistic — real counterfactual/attention
  signals are noisy. `training/signals.py` adds optional deterministic Gaussian
  measurement noise (σ, default 0 so tests stay exact) and the benchmark sweeps
  σ. Verdict is honest: heuristic wins at σ=0 (reported), learned wins for σ≳0.2
  (the realistic regime). This tests RQ5 rather than asserting it. The learned
  head ships as default only where it clears the margin; else the heuristic
  stands (fix #4).

- **[B] 2026-07-03 — ✅ HANDOFF TO [A]: judge validated & usable in Stage 6;
  verifier is optional and reversible.** (1) `from fmr.eval import build_judge`
  for open-ended scoring — heuristic is validated (κ on gold logged); use the LLM
  judge as primary once `colab_judge_llm.ipynb` reports its κ. (2) The learned
  verifier is a drop-in for the FS fusion via `LearnedVerifier.score(features)`;
  if you'd rather keep the training-free path, do nothing — `HeuristicFusion`
  is the default and the pipeline never hard-depends on the trained head.
