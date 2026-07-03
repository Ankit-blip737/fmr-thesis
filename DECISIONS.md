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
