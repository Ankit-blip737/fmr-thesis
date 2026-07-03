# BLOCKERS.md — things that need the human (append-only per tag)

Each entry: what's blocked · exactly what is needed · what happens once provided.

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

- **[B] 2026-07-03 — GPU notebook `colab_stage4_correction_real.ipynb` NOT YET
  CREATED.** Instance B must create this notebook in the next run before it can
  be opened in Colab. Tokens `HF_TOKEN` + `GITHUB_TOKEN` are now ready —
  add both as Colab secrets when running.

- **[B] 2026-07-03 — GPU/LLM notebook `colab_judge_llm.ipynb` NOT YET CREATED.**
  Instance B must create this notebook in the next run. Tokens `HF_TOKEN` +
  `GITHUB_TOKEN` are now ready — add both as Colab secrets when running.

## Resolved

- **[2026-07-03 — audit by human assistant]** Corrected two misleading blocker entries above — notebooks do not yet exist; Instance B will build on next run. Mock correction pipeline verified working locally. Both worktrees confirmed up to date with their respective remotes.

- **[2026-07-03] `HF_TOKEN` fully resolved** — token set as Windows user env var; MedGemma license accepted on HuggingFace. All three models unblocked. For Colab: add as secret `HF_TOKEN`.

- **[2026-07-03] `GH_TOKEN` / `GITHUB_TOKEN` resolved** — GitHub fine-grained PAT (repo `fmr-thesis`, `contents: read+write`) set as Windows user env vars under both names. For Colab: add as secret `GH_TOKEN` (Instance A notebooks) and `GITHUB_TOKEN` (Instance B notebooks).

