# FMR — Faithful Medical Reasoning

A model-agnostic, largely **training-free** layer that **measures** whether a
medical VLM's chain-of-thought is grounded in the image, **corrects** it when it
drifts, and **abstains** (defers to a clinician) with a calibrated guarantee when
it cannot be verified.

> Two headline claims drive the project: **(a)** medical reasoning VLMs lose
> visual groundedness as the reasoning chain grows (the diagnostic finding), and
> **(b)** a calibrated, faithfulness-triggered abstention mechanism is a genuinely
> new safety layer for that failure mode. Correction is supporting infrastructure
> for (b), not a co-equal third contribution.

## Design: runs anywhere, verifies on hardware

The whole pipeline runs **fully offline** on a built-in `MockVLM` (no GPU, no
downloads) so every component — signals, fusion, conformal gate, ablations — is
exercised and unit-tested on a CPU. The `MockVLM` is driven by a hidden per-sample
*grounding* latent, so the faithfulness signals are genuinely predictive of it;
this validates the **machinery**, not clinical performance. Real open medical VLMs
(MedVLM-R1, Qwen2.5-VL, …) plug in behind the same `BaseVLM` interface and are run
on a GPU via the Colab notebook, which pushes results back to this branch.

## Architecture (three pillars)

```
Base VLM ──► FMM (Pillar 1) ──► Correction (Pillar 2) ──► Abstention (Pillar 3)
             Signals A/B/C        VCD + clue trace           split-conformal
             → fused FS           + verify/revise            selective-risk gate
```

| Layer | Module (owner) | What it does |
|---|---|---|
| Data | `data/loaders.py`, `data/synthetic.py` | Unified `Sample` loader (VQA-RAD/SLAKE/PathVQA + synthetic); disjoint train/cal/test splits |
| Base model | `models/{base_vlm,mock_vlm,hf_vlm}.py` | Model-agnostic `generate` contract; offline mock + real HF backend |
| **Signal A** | `faithfulness/counterfactual.py` | Counterfactual sensitivity (answer flips when image is removed/swapped) |
| **Signal B** | `faithfulness/attention.py` | Attention-region coherence + IoU-vs-GT weak labels |
| **Signal C** | `faithfulness/consistency.py` | Self-consistency across sampled chains |
| **Fusion** | `faithfulness/score.py` | Heuristic fused FS + per-signal record schema + verifier feature adapter |
| Correction | `correction/` *(Instance B)* | Selective, training-free re-anchoring; integrated via a guarded import |
| **Abstention** | `abstention/conformal.py` | Selective-risk guarantee (Clopper-Pearson UCB, distribution-free) |
| Verifier | `training/` *(Instance B)* | Learned fusion head; trains on the labels/features this side exposes |

## Quickstart (offline, CPU)

```bash
pip install -e fmr            # core deps only; no torch/transformers needed
pytest fmr/tests             # 31 tests

python fmr/scripts/run_baselines.py     # accuracy per model/modality
python fmr/scripts/run_blind_test.py    # image-blind test + grounding-drift curve
python fmr/scripts/run_fmr.py           # signals → FS → conformal gate (Stages 3+5)
python fmr/scripts/run_fmr_full.py      # Stage 6: ablations + model-agnosticism + correction
python fmr/scripts/run_ablations.py     # Stage 3/5 robustness (weights/grid/seed/power)
python fmr/scripts/make_figures.py      # regenerate all figures from the JSON artifacts
```

Outputs (JSON + figures) land in `fmr/outputs/`. Every figure is reproducible from
the saved JSON alone.

## Real models (GPU / Colab)

Open `fmr/notebooks/colab_real_pipeline.ipynb` in Colab (GPU runtime), set the
Colab secrets `HF_TOKEN` and `GH_TOKEN`, and *Run all*. It clones this branch,
installs `fmr[real]`, runs the pipeline on **MedVLM-R1 (reasoning) vs
Qwen2.5-VL-3B (non-reasoning)** across SLAKE/VQA-RAD/PathVQA, and pushes
`fmr/outputs/real/<dataset>/` back automatically. See `BLOCKERS.md` for tokens.

## Config

`configs/data.yaml` (dataset + split policy), `configs/models.yaml` (backends;
mock + real), `configs/experiment.yaml` (signal weights, consistency samples,
abstention α/δ). The `run_real.py` orchestrator overrides these for GPU runs.

## Honesty notes (carried into the thesis)

- Numbers from the mock path are **machinery validation on synthetic data**,
  labeled as such everywhere; empirical claims come from the Colab real-model run.
- **Signal B** IoU grounding is validated only where ground-truth boxes exist
  (synthetic; real SLAKE/VQA-RAD once masks are wired). On box-free data it is
  reported as unvalidated/exploratory.
- The abstention **guarantee** is finite-sample and needs enough calibration data
  (α=0.05 ≈ 500–1000 points); the power curve quantifies this rather than hiding
  an abstain-all.
- FMR is a research artifact, not a deployed clinical tool — the contribution *is*
  the safety/deferral mechanism.

See `RESULTS_LOG.md` (verified numbers), `DECISIONS.md` (design choices), and
`BLOCKERS.md` (anything needing a human) at the repo root.
