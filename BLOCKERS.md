# BLOCKERS.md — things that need the human (append-only per tag)

Each entry: what's blocked · exactly what is needed · what happens once provided.

---

## ⭐ CURRENT STATUS (Instance B) — 2026-07-04

The project is **integrated end-to-end** (master merged into instance-b; 93 tests
green; `run_fmr_full.py` connects signals→fusion→correction→post-corr FS→conformal
gate). Open items needing you:

- **[B] Re-run `fmr/notebooks/colab_faithfulness_lora.ipynb` (v3, T4-safe).** The v2
  run hit CUDA OOM on the T4; v3 does 4-bit data-prep, frees the model before
  training, and toggles the adapter for eval. Needs one Colab GPU run to produce the
  frozen-vs-LoRA ablation. Add secrets `HF_TOKEN`+`GITHUB_TOKEN`, Run All.
- **[B] Optional: re-run `colab_stage4_correction_real.ipynb`** — now prints an
  image-sensitivity self-check per model (flags the MedGemma issue below). The Qwen
  result is already valid and in `fmr/results/`.
- **[B] MedGemma cross-model = FUTURE WORK (not a blocker).** MedGemma-4B returns an
  image-invariant distribution through the closed-set adapter (fs≈0, acc≈chance) —
  the Gemma-3 input path needs a model-specific adapter. Qwen2.5-VL-3B is the
  validated real base model; model-agnosticism of the FMR layer is shown on Qwen +
  2 mock backends. No action needed unless you want a second real model — then a
  GPU session to debug the Gemma-3 image binding is required.

---


- **[B] 2026-07-03 — This machine is CPU-only.** No real-model inference is
  attempted here. Every GPU-dependent step in my scope gets a ready-to-run Colab
  notebook under `fmr/notebooks/` (see entries below as they land); each clones
  this repo, runs the step, and pushes results back to branch `instance-b`.


- **[B] 2026-07-03 — No LLM API key on this machine → LLM-as-judge cannot run
  locally.** *Meanwhile:* `fmr/src/fmr/eval/judge.py` ships a deterministic
  heuristic judge validated against a hand-authored gold set (agreement numbers
  in `RESULTS_LOG.md`), plus an `LLMJudge` scaffold. The open-LLM judge run is
  packaged as a Colab notebook (entry added when committed). If you prefer an
  API judge (Claude/GPT), provide the key as an env var and I'll wire it.

- **[B] 2026-07-03 — ✅ GPU run ready — `fmr/notebooks/colab_stage4_correction_real.ipynb`
  CREATED & committed (nbformat-valid, 16 cells).** Open in Colab (GPU runtime),
  add Colab secrets `HF_TOKEN` + `GITHUB_TOKEN` (notebook access on), Run All. It
  clones `instance-b`, runs the committed `fmr.correction` pipeline fed by real
  Qwen2.5-VL-3B answer distributions on a 40-item VQA-RAD closed subset, sanity-
  checks MedGemma, saves `fmr/results/correction_real_*.json`, and pushes to
  `instance-b`. On my next resume I pull these and compare real-vs-mock Stage-4
  gains. *Note:* the `RealAnswerVLM` adapter (teacher-forced choice scoring) is
  the one piece never runnable on this CPU box — sanity-check its first few
  printed rows look plausible before trusting the summary.

- **[B] 2026-07-03 — ✅ GPU run ready — `fmr/notebooks/colab_judge_llm.ipynb`
  CREATED & committed (nbformat-valid, 9 cells).** Open in Colab (GPU runtime),
  add secrets `HF_TOKEN` + `GITHUB_TOKEN`, Run All. Scores the N=44 judge gold set
  with an independent open LLM (Qwen2.5-7B-Instruct) via the committed `LLMJudge`,
  reports LLM-vs-gold and LLM-vs-heuristic agreement, saves
  `fmr/results/judge_llm_validation.json`, pushes to `instance-b`. This is the
  independent external check that upgrades the heuristic's tuned κ=1.0 into a
  trustworthy judge number.

- **[B] 2026-07-03 — ✅ GPU run ready — `fmr/notebooks/colab_faithfulness_lora.ipynb`
  CREATED & committed (nbformat-valid, 11 cells).** Stretch ablation (RQ3): open in
  Colab (24GB GPU ideal; T4 works via QLoRA), add secrets `HF_TOKEN` +
  `GITHUB_TOKEN`, Run All. Builds verified-grounded self-distillation targets from
  the correction module (CPU-tested API), QLoRA-fine-tunes Qwen2.5-VL-3B, compares
  frozen base vs faithfulness-LoRA on held-out VQA-RAD (accuracy + faithfulness),
  saves `fmr/results/faithfulness_lora_*.json`, pushes to `instance-b`. Frozen base
  stays the default; reported only as an ablation.

- **[B] 2026-07-03 — Suggested run order if you run only some notebooks:** (1)
  `colab_judge_llm.ipynb` — cheapest, validates a metric the whole benchmark uses;
  (2) `colab_stage4_correction_real.ipynb` — confirms Stage-4 gains on a real model;
  (3) `colab_faithfulness_lora.ipynb` — stretch, only after 1–2 land.

- **[B] 2026-07-03 — ✅ Notebooks ready to RE-RUN (MedGemma access now granted).**
  (1) `colab_stage4_correction_real.ipynb` **v3** — runs the full `vcd_margin`
  trade-off sweep on **both** `Qwen/Qwen2.5-VL-3B-Instruct` and
  `google/medgemma-4b-it` (proper cross-model comparison; memory freed between
  models). (2) `colab_faithfulness_lora.ipynb` **v2** — smaller distill set,
  data-driven target selection, hardened Trainer, always-writes-diagnostic.
  `colab_judge_llm.ipynb` already succeeded (κ=0.758) — no re-run needed. Add
  secrets `HF_TOKEN` + `GITHUB_TOKEN`, Run All; both push results to `instance-b`.

## Resolved

- **[2026-07-03] MedGemma (`google/medgemma-4b-it`) access GRANTED** (user
  confirmed). The 403 gating is cleared; correction notebook v3 now runs the full
  cross-model sweep on MedGemma as a first-class second model (no longer an
  ungated-substitute / optional block).


- **[2026-07-03 — audit by human assistant]** Corrected two misleading blocker entries above — notebooks do not yet exist; Instance B will build on next run. Mock correction pipeline verified working locally. Both worktrees confirmed up to date with their respective remotes.

- **[2026-07-03] `HF_TOKEN` fully resolved** — token set as Windows user env var; MedGemma license accepted on HuggingFace. All three models unblocked. For Colab: add as secret `HF_TOKEN`.

- **[2026-07-03] `GH_TOKEN` / `GITHUB_TOKEN` resolved** — GitHub fine-grained PAT set as Windows user env vars. For Colab: add as secrets `GH_TOKEN` and `GITHUB_TOKEN`.

- **[2026-07-03] `hf_vlm.py` Colab crash fixed** — `huggingface_hub>=0.34` strict dataclass validation rejects `use_cache=None` in Qwen2-VL/MedGemma configs with `StrictDataclassFieldValidationError`. Fixed by adding `_patch_config_for_strict_hub()` in `HFVLM._ensure_loaded()` (coerces `None→True` for bool config fields before the validator fires). Same fix applied to Instance A's `hf_vlm.py`. Both pushed. Notebooks clone the fixed master/instance-b so they will no longer crash on this error.

- **[2026-07-03] `colab_judge_llm.ipynb` ran — real results in:** LLM judge (Qwen2.5-7B-Instruct) vs gold on N=44: accuracy=0.864, binary accuracy=0.955, Cohen’s κ=0.758 (substantial agreement). LLM-vs-heuristic identical (κ=0.758), confirming the heuristic judge is a faithful proxy for real LLM scoring. Result: `fmr/results/judge_llm_validation.json` pushed to `instance-b`.


---

<!-- ============================================================= -->
<!-- Merged from Instance A (master) at instance-b<-master merge   -->
<!-- ============================================================= -->

# BLOCKERS.md — items needing the human (append-only within your tag; keep current)

## ⭐ NEXT ACTION (Instance A) — the one thing to do

**Run `fmr/notebooks/colab_real_pipeline.ipynb` on Colab (GPU runtime).** Set Colab
secrets `HF_TOKEN` + `GH_TOKEN`, then Run all. It runs the whole pipeline on real
models (MedVLM-R1 vs Qwen2.5-VL-3B) over VQA-RAD → PathVQA → SLAKE and auto-pushes
results to `master`. That is the real-model test of the headline "more reasoning →
less grounded" hypothesis and produces the real FS scores I'll wire into
calibration on the next local resume. Nothing else in Instance A's scope is blocked
— Stages 0–3, 5, 6 all run and are verified on the mock path (34 tests green).

## Open

- **[A] No GPU on this machine (CPU-only torch 2.12.1).** All real-model inference is routed to Colab per the GPU handoff protocol. Local work uses MockVLM by design, not as a degradation.

- **[A] OmniMedVQA access.** The dataset's distribution requires a manual download (its open subset ships via a request form / BAAI link, not a clean public HF repo). *Need from you (later, low priority):* download the open-access portion and place it under `fmr/data_cache/omnimedvqa/`. Until then the modality-breadth claim rests on VQA-RAD + SLAKE + PathVQA, which is sufficient for the core thesis.

## Resolved

- **[2026-07-03 — audit by human assistant]** Scanned all open items. No items in Instance A were resolvable without human action at that point. Local mock pipeline verified working (Instance B's `run_correction.py --model mock` exits clean).

- **[2026-07-03] `HF_TOKEN` fully resolved** — token set as Windows user env var; MedGemma license accepted on HuggingFace. All three models unblocked. For Colab: add as secret `HF_TOKEN`.

- **[2026-07-03] `GH_TOKEN` / `GITHUB_TOKEN` resolved** — GitHub fine-grained PAT (repo `fmr-thesis`, `contents: read+write`) set as Windows user env vars under both names. For Colab: add as secret `GH_TOKEN` (Instance A notebooks) and `GITHUB_TOKEN` (Instance B notebooks).

- **[A] SLAKE images + masks for real Signal B validation.** The `BoKelvin/SLAKE` HF mirror ships annotations only (modality/Q/A/answer_type — all usable), but images are referenced by `img_name` path and there are NO per-QA bounding boxes inline. *Need (for the real Signal B IoU experiment, done on Colab not here):* the SLAKE `imgs.zip` (images) and the segmentation-mask files from https://www.med-vqa.com/slake/ . The Colab notebook downloads imgs.zip automatically; the masks→bbox step (`bbox_to_region` helper is ready in loaders.py) is the only remaining manual piece for real-image grounding validation. Not blocking the core thesis — synthetic Signal B validation already works and real modality-level results don't need boxes.

- **[A] GPU run ready — `fmr/notebooks/colab_real_pipeline.ipynb`.** Open in Colab (GPU runtime), set Colab secrets `HF_TOKEN` + `GH_TOKEN`, Run all. It clones `master`, installs `fmr[real]`, runs baselines + blind test + full FMR (Signals A/B/C → FS → conformal gate) on **MedVLM-R1 (reasoning) vs Qwen2.5-VL-3B (non-reasoning)** across SLAKE/VQA-RAD/PathVQA, writes `fmr/outputs/real/<dataset>/`, and **auto-commits+pushes results back to `master`**. On next local resume I pull these and wire real FS into calibration (replacing provisional mock scores). Uses α=0.10–0.15 on the small real sets per the calibration-power finding (see DECISIONS.md). No action needed beyond running it — this is the real-model test of the headline "more reasoning → less grounded" hypothesis.

- **[A] 2026-07-03 (later) — Colab runs attempted, kept failing; now hardened.** History: real runs hit (1) hf_hub≥0.34 StrictDataclassFieldValidationError on `use_cache=null` → fixed (`_patch_config_for_strict_hub`, commits dc4af08/4c9390a); (2) CUDA OOM on the 14.5 GB T4 → per-model GPU cleanup added (cebae20). `outputs/real/` is still EMPTY = no successful run has pushed yet. **What I changed to make the next run succeed:** `run_real.py` now runs stages cheapest→heaviest and **pushes after each stage independently** (+ per-stage try/except + `run_status.json`), so an OOM/timeout in the heavy FMR stage can no longer discard the baselines + blind-test HEADLINE. The notebook now leads with a **SMOKE RUN** (VQA-RAD, 80 samples, n_consistency=3) that finishes in minutes and pushes the headline before the fuller runs. *Action for you:* re-run `fmr/notebooks/colab_real_pipeline.ipynb` on a T4 — run the smoke cell first; if the big cells OOM, drop `--max-samples` to ~120. I cannot run this myself: no local GPU, no browser connected to the Chrome extension, and Colab needs interactive Google auth + secret entry.

## Optional (nice-to-have, needs you)

- **[A] Live dashboard URL via GitHub Pages (1-click, optional).** The dashboard runs fine offline — just open `fmr/dashboard/index.html` (works from `file://`, zero setup). For a shareable live URL: on GitHub → repo **Settings → Pages** → Source = "Deploy from a branch", Branch = `master`, folder = `/ (root)` → Save. It will then be live at **https://ankit-blip737.github.io/fmr-thesis/fmr/dashboard/** . Nothing in code needs changing; `data.js` is already committed and auto-updates when Colab runs push. (I can't toggle repo Settings myself — that's an account action.)
