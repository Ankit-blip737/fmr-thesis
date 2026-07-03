# BLOCKERS.md — items needing the human (append-only within your tag; keep current)

## Open

- **[A] No GPU on this machine (CPU-only torch 2.12.1).** All real-model inference is routed to Colab per the GPU handoff protocol. Local work uses MockVLM by design, not as a degradation.

- **[A] OmniMedVQA access.** The dataset's distribution requires a manual download (its open subset ships via a request form / BAAI link, not a clean public HF repo). *Need from you (later, low priority):* download the open-access portion and place it under `fmr/data_cache/omnimedvqa/`. Until then the modality-breadth claim rests on VQA-RAD + SLAKE + PathVQA, which is sufficient for the core thesis.

## Resolved

- **[2026-07-03 — audit by human assistant]** Scanned all open items. No items in Instance A were resolvable without human action at that point. Local mock pipeline verified working (Instance B's `run_correction.py --model mock` exits clean).

- **[2026-07-03] `HF_TOKEN` fully resolved** — token set as Windows user env var; MedGemma license accepted on HuggingFace. All three models unblocked. For Colab: add as secret `HF_TOKEN`.

- **[2026-07-03] `GH_TOKEN` / `GITHUB_TOKEN` resolved** — GitHub fine-grained PAT (repo `fmr-thesis`, `contents: read+write`) set as Windows user env vars under both names. For Colab: add as secret `GH_TOKEN` (Instance A notebooks) and `GITHUB_TOKEN` (Instance B notebooks).
