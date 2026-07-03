# BLOCKERS.md — items needing the human (append-only within your tag; keep current)

## Open

- **[A] No GPU on this machine (CPU-only torch 2.12.1).** All real-model inference is routed to Colab per the GPU handoff protocol. Local work uses MockVLM by design, not as a degradation.

- **[A] OmniMedVQA access.** The dataset's distribution requires a manual download (its open subset ships via a request form / BAAI link, not a clean public HF repo). *Need from you (later, low priority):* download the open-access portion and place it under `fmr/data_cache/omnimedvqa/`. Until then the modality-breadth claim rests on VQA-RAD + SLAKE + PathVQA, which is sufficient for the core thesis.

## Resolved

- **[2026-07-03 — audit by human assistant]** Scanned all open items. No items in Instance A were resolvable without human action at that point. Local mock pipeline verified working (Instance B's `run_correction.py --model mock` exits clean).

- **[2026-07-03] `HF_TOKEN` fully resolved** — token set as Windows user env var; MedGemma license accepted on HuggingFace. All three models unblocked. For Colab: add as secret `HF_TOKEN`.

- **[2026-07-03] `GH_TOKEN` / `GITHUB_TOKEN` resolved** — GitHub fine-grained PAT (repo `fmr-thesis`, `contents: read+write`) set as Windows user env vars under both names. For Colab: add as secret `GH_TOKEN` (Instance A notebooks) and `GITHUB_TOKEN` (Instance B notebooks).

- **[A] SLAKE images + masks for real Signal B validation.** The `BoKelvin/SLAKE` HF mirror ships annotations only (modality/Q/A/answer_type — all usable), but images are referenced by `img_name` path and there are NO per-QA bounding boxes inline. *Need (for the real Signal B IoU experiment, done on Colab not here):* the SLAKE `imgs.zip` (images) and the segmentation-mask files from https://www.med-vqa.com/slake/ . The Colab notebook downloads imgs.zip automatically; the masks→bbox step (`bbox_to_region` helper is ready in loaders.py) is the only remaining manual piece for real-image grounding validation. Not blocking the core thesis — synthetic Signal B validation already works and real modality-level results don't need boxes.

- **[A] GPU run ready — `fmr/notebooks/colab_real_pipeline.ipynb`.** Open in Colab (GPU runtime), set Colab secrets `HF_TOKEN` + `GH_TOKEN`, Run all. It clones `master`, installs `fmr[real]`, runs baselines + blind test + full FMR (Signals A/B/C → FS → conformal gate) on **MedVLM-R1 (reasoning) vs Qwen2.5-VL-3B (non-reasoning)** across SLAKE/VQA-RAD/PathVQA, writes `fmr/outputs/real/<dataset>/`, and **auto-commits+pushes results back to `master`**. On next local resume I pull these and wire real FS into calibration (replacing provisional mock scores). Uses α=0.10–0.15 on the small real sets per the calibration-power finding (see DECISIONS.md). No action needed beyond running it — this is the real-model test of the headline "more reasoning → less grounded" hypothesis.
