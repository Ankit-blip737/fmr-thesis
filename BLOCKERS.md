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
