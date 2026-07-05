# FMR — Faithful Medical Reasoning

> **A layer that makes a medical AI's reasoning *checkable*: it measures whether the AI is
> actually looking at the scan, fixes the reasoning when it drifts off the image, and — when it
> still can't be sure — refuses to answer and hands the case to a doctor, with a mathematical
> guarantee on how often it's allowed to be wrong.**

This README explains the entire project from scratch — the problem, the ideas, the code, how to
run it, what we found, and what its limits are. No prior knowledge of vision-language models is
assumed; jargon is defined the first time it appears and collected in a [Glossary](#12-glossary).

---

## Table of Contents
1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [Background: the concepts you need](#2-background-the-concepts-you-need)
3. [The problem this project attacks](#3-the-problem-this-project-attacks)
4. [The core idea & the two claims](#4-the-core-idea--the-two-claims)
5. [Architecture: the three pillars](#5-architecture-the-three-pillars)
6. [How the pieces are built (the signals, the gate, the verifier)](#6-how-the-pieces-are-built)
7. [Repository map — where everything lives](#7-repository-map)
8. [How to run it](#8-how-to-run-it)
9. [Data & models](#9-data--models)
10. [Results (reported honestly)](#10-results-reported-honestly)
11. [The interactive dashboard](#11-the-interactive-dashboard)
12. [Glossary](#12-glossary)
13. [Honesty, limitations & status](#13-honesty-limitations--status)

---

## 1. The one-paragraph version

Modern medical AIs are **reasoning vision-language models (VLMs)** — they take a medical image
(X-ray, CT, MRI, pathology slide) plus a question, and produce a *chain of reasoning* ("the left
lung shows an opacity, therefore…") ending in an answer. The reasoning is meant to make them
*trustworthy*. But there is a documented, dangerous problem: **the reasoning often isn't actually
based on the image** — the model leans on language patterns it memorised, and the *more* it
reasons the *less* it may look at the scan. A confident, medical-sounding rationale that ignores
the scan is worse than no rationale at all. **FMR** is a wrapper that sits around any such model
and does three things: **Measure** how visually grounded each reasoning step is, **Correct** the
reasoning when it has drifted off the image, and **Abstain** (defer to a clinician) when
faithfulness can't be verified — with a *distribution-free statistical guarantee* on the error
rate of the cases it does answer.

---

## 2. Background: the concepts you need

**Vision-Language Model (VLM).** A neural network that takes an image + text and produces text.
A *medical* VLM is one tuned on medical images and questions (e.g. "Is there an effusion?").

**Reasoning VLM / chain-of-thought (CoT).** A VLM that first writes out intermediate reasoning
steps before the final answer, e.g. *"Step 1: there is opacity in the lower lobe. Step 2: this is
consistent with consolidation. Answer: yes."* The hope is that the steps reveal *why* the model
answered — making it auditable.

**Faithfulness (vs. plausibility).** A rationale is *plausible* if it reads well. It is
**faithful** if it reflects what the model actually used to decide. These come apart: a model can
write a fluent, clinically-styled rationale that mentions findings *not present in the scan*, or
reach the right answer for the wrong (non-visual) reason. Faithfulness is what we care about for
safety.

**Grounding.** The degree to which a reasoning step depends on the *image* (as opposed to
language priors). "Grounded" = the claim is supported by image evidence; "ungrounded" =
image-independent guessing.

**The "blind test."** Re-run the model with the real image, then with a **blank** (grey) image,
then with a **mismatched** image. If accuracy barely drops when you remove the image, the model
was answering from language priors, not the scan. `blind_gap = accuracy(real) − accuracy(blank)`.

**AUROC.** A number in [0,1] measuring how well a score separates two classes (here: grounded vs
ungrounded reasoning). 0.5 = no better than a coin flip; 1.0 = perfect separation.

**IoU (Intersection-over-Union).** Overlap between two boxes/regions, 0–1. Used to check whether
the region the model "attended to" matches the ground-truth evidence region.

**Conformal prediction / selective risk.** A statistical technique that gives a *distribution-free
guarantee*: choose a rule such that, with high probability, the error rate on the predictions you
*keep* (don't defer) is below a target α — no assumptions about the data distribution needed.

**Abstention / selective prediction.** Letting the model say "I don't know, defer to a human"
instead of forcing an answer. The research question: what's the best *trigger* for deferring?

---

## 3. The problem this project attacks

Three facts, all documented in recent literature and reproduced here:

1. **Accuracy hides the problem.** Standard benchmarks only score the final answer. They cannot
   tell you whether the reasoning used the image — and many medical-VQA questions are answerable
   *without* the image by exploiting dataset/language priors.
2. **Reasoning ≠ looking.** Multimodal reasoning models can shift attention *away* from image
   evidence as the chain grows, and sometimes hallucinate *more* than their non-reasoning
   versions.
3. **In medicine this is safety-critical.** An authoritative but ungrounded rationale causes
   "automation bias" — a clinician trusts a confident explanation that is actually disconnected
   from the scan.

Prior work either (a) *measures* faithfulness (evaluation only, usually general-domain), (b)
*grounds* reasoning (general-domain methods, no clinical deferral), or (c) *debiases* short
answers (not multi-step chains). **No prior system unifies measurement + correction + clinical
abstention into one deployable, model-agnostic layer.** That gap is FMR.

---

## 4. The core idea & the two claims

FMR is a **model-agnostic** layer: it treats the base VLM as a black box behind a small
interface, so it works with *any* open medical reasoning VLM without retraining it. It is
**largely training-free** (the core needs no model training — cheap, reproducible, always works),
with **one optional trained component** (a small "learned verifier") that can be swapped in and
always has a training-free fallback.

Two headline claims organise everything:

- **Claim A (the Diagnosis).** *Medical reasoning VLMs lose visual groundedness as the reasoning
  chain grows.* This is the empirical finding the project sets out to test.
- **Claim B (the Safety mechanism).** *A calibrated, faithfulness-triggered abstention gate is a
  genuinely new safety layer for that failure mode* — deferring based on *grounding*, not just on
  the model's own confidence (which is known to be miscalibrated). Correction is *supporting
  infrastructure* for B, not a co-equal third contribution.

---

## 5. Architecture: the three pillars

```
   ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐
   │ Image + Q│ → │  Frozen VLM  │ → │  P1: MEASURE       │ → │  P2: CORRECT       │
   └──────────┘   │ (CoT+answer) │   │  Signals A,B,C     │   │ (only if FS low)   │
                  └──────────────┘   │  → Faithfulness FS │   │ VCD + clue-trace   │
                                     └───────────────────┘   │ + verify/revise    │
                                                              └─────────┬──────────┘
                                                                        ↓
                                              ┌───────────────────────────────────┐
                                              │  P3: ABSTAIN (conformal gate)      │
                                              │  answer iff FS ≥ τ, else DEFER     │
                                              └───────────────────────────────────┘
                                                 ↓ answer + grounded rationale
                                                 ↓ or  ABSTAIN → clinician
```

- **Pillar 1 — Measurement (`faithfulness/`).** Produce a single **Faithfulness Score (FS)** ∈
  [0,1] per case by fusing three *independent* signals (below). Independence is the point: no
  single faithfulness signal is trustworthy alone, so we never rely on one.
- **Pillar 2 — Correction (`correction/`).** Applied *only when FS is low*. All components are
  decode-time and need no training: Visual Contrastive Decoding (VCD), question→vision clue
  tracing, and step verify-and-revise. The frozen model is always the fallback.
- **Pillar 3 — Abstention (`abstention/`).** A split-conformal gate calibrated on the *deployed*
  (post-correction) FS gives the distribution-free guarantee: among answered cases, error ≤ α
  with probability ≥ 1−δ. Everything else is deferred.

---

## 6. How the pieces are built

### The three faithfulness signals
- **Signal A — Counterfactual sensitivity** (`faithfulness/counterfactual.py`). Run the model on
  the real image, a blank, and a mismatched image. A *faithful* answer should **change** when the
  image is removed/replaced. Quantified by answer-flip rate + Jensen–Shannon divergence of the
  answer distribution. High = the answer genuinely depends on the image.
- **Signal B — Attention grounding** (`faithfulness/attention.py`). Does the reasoning attend to
  the true evidence region? Where ground-truth boxes exist (synthetic; SLAKE/VQA-RAD once masks
  are wired) this is validated by IoU. At inference time without boxes it scores the *spatial
  coherence* of the attended regions across steps. (On real HF models the attention→region
  extraction is a documented plug-point; until it lands, Signal B is neutral there and the fused
  FS uses A+C.)
- **Signal C — Self-consistency** (`faithfulness/consistency.py`). Sample N reasoning chains at
  non-zero temperature; if the answers scatter, confidence/faithfulness is low (RadFlag-style).

These fuse into the **FS** (`faithfulness/score.py`) — a weighted, interpretable combination,
exposed alongside all raw per-signal scores so a learned model can consume them.

### The correction module (`correction/`)
- **VCD (`vcd.py`)** — contrast the answer distribution under the real vs. a distorted image to
  suppress language-prior-driven tokens and amplify image-grounded ones.
- **Clue tracing (`clue_tracing.py`)** — trace each reasoning clue back to supporting visual
  evidence; suppress unsupported continuations.
- **Verify/revise (`verify_revise.py`)** — drop or flag unsupported steps and regenerate the
  answer from verified claims. It is **selective** (only fires when FS is low) because
  indiscriminate correction can hurt already-grounded cases — the accuracy/faithfulness trade-off
  is measured, not assumed.

### The conformal gate (`abstention/conformal.py`)
We control **selective risk** (error *conditional on answering*), which ordinary conformal
quantiles don't bound. Implementation: an exact Clopper–Pearson upper confidence bound on
retained error, Bonferroni-corrected over a bounded grid of candidate thresholds. Pick the
threshold with maximum coverage whose certified error ≤ α. If none qualifies, the safe output is
**abstain-all** (reported, never hidden). *Design note:* the risk-coverage metrics are
**tie-aware** — a constant/degenerate score (e.g. a signal that's the same for every case) is
correctly read as "no discrimination" instead of getting a fabricated score from arbitrary
tie-ordering.

### The learned verifier (`training/`, optional)
A small gradient-boosted head that *learns* to fuse the three signals instead of hand-weighting
them. Trained on **weak labels** derived from counterfactual behaviour (no manual annotation).
It beats the heuristic FS on real signals, but is honestly capped by weak-label quality (see
[Results](#10-results-reported-honestly)). The heuristic FS is always the fallback.

---

## 7. Repository map

```
.
├── README.md                     ← you are here
├── Faithful_Medical_Reasoning_Thesis_Proposal_1.md   ← the full research plan
├── RESULTS_LOG.md                ← append-only verification log (every number, dated)
├── DECISIONS.md                  ← every design decision, with reasoning
├── BLOCKERS.md                   ← anything needing a human (tokens, GPU, downloads)
└── fmr/
    ├── README.md                 ← concise technical readme
    ├── pyproject.toml            ← installable package (pip install -e fmr)
    ├── src/fmr/
    │   ├── types.py              ← core dataclasses (Sample, Step, VLMOutput)
    │   ├── data/                 ← loaders (VQA-RAD/SLAKE/PathVQA/OmniMedVQA) + synthetic + regions
    │   ├── models/               ← base_vlm (interface), mock_vlm (offline), hf_vlm (real), second_vlm
    │   ├── faithfulness/         ← Pillar 1: decompose, counterfactual(A), attention(B),
    │   │                            consistency(C), score(fusion), features_for_verifier
    │   ├── correction/           ← Pillar 2: vcd, clue_tracing, verify_revise, rescore, pipeline
    │   ├── abstention/           ← Pillar 3: conformal (split-conformal / SGR gate)
    │   ├── training/             ← optional learned verifier + faithfulness-LoRA + labels/adapter
    │   └── eval/                 ← LLM-as-judge for open-ended answers + gold set
    ├── scripts/                  ← run_baselines, run_blind_test, run_fmr, run_fmr_full,
    │                                run_ablations, run_abstention_baselines, run_real,
    │                                make_figures, make_dashboard, train_verifier, …
    ├── configs/                  ← data.yaml, models.yaml, experiment.yaml, correction.yaml, verifier.yaml
    ├── tests/                    ← 101 unit/integration tests
    ├── notebooks/                ← Colab notebooks (real-model runs, auto-push results)
    ├── dashboard/                ← interactive results dashboard (index.html + app.js + style.css + data.js)
    ├── docs/                     ← project_summary.tex + write-up sections
    ├── outputs/                  ← generated results (mock in root, real/ per dataset) + figures
    └── results/                  ← correction/verifier/judge result JSONs
```

**The three root logs are the audit trail.** `RESULTS_LOG.md` records every verified number with
a date; `DECISIONS.md` records why each design choice was made; `BLOCKERS.md` records anything
that needs a human (e.g. a GPU run). They are append-only and tagged `[A]`/`[B]` for the two
parallel build tracks.

---

## 8. How to run it

### A) Offline, on any CPU (validates the machinery — no GPU, no downloads)
```bash
pip install -e fmr                     # core deps only (numpy, scipy, sklearn, matplotlib, pyyaml)
python -m pytest fmr/tests -q          # 101 tests

python fmr/scripts/run_baselines.py        # accuracy per model / modality
python fmr/scripts/run_blind_test.py       # blind test + headline replication verdict
python fmr/scripts/run_fmr.py              # Signals A/B/C → FS → conformal gate
python fmr/scripts/run_fmr_full.py         # full benchmark: correction + ablations + model-agnosticism
python fmr/scripts/run_ablations.py        # robustness (weight/grid/seed/conformal-power)
python fmr/scripts/run_abstention_baselines.py   # FS vs confidence / self-consistency / RadFlag
python fmr/scripts/make_figures.py         # regenerate all figures from the JSON artifacts
```
Everything here runs against a built-in **`MockVLM`** — a deterministic stand-in driven by a
hidden per-sample *grounding latent*, so the signals are genuinely predictive of it. This
validates the *machinery* (that the math and code are correct); it is **not** a claim about real
clinical performance, and every mock number is labeled synthetic.

### B) Real models, on a GPU (Google Colab)
This machine is CPU-only, so real-model inference runs on Colab:
1. Open `fmr/notebooks/colab_real_pipeline.ipynb` in Colab (T4 GPU).
2. Set the Colab secrets `HF_TOKEN` (Hugging Face) and `GH_TOKEN` (GitHub, to push results back).
3. Run all. It clones the repo, installs `fmr[real]`, runs baselines → blind test → full FMR on
   **MedVLM-R1 vs Qwen2.5-VL-3B** across VQA-RAD / PathVQA / SLAKE, writes
   `fmr/outputs/real/<dataset>/`, and **auto-commits the results back to the branch**. On the next
   local resume, `make_dashboard.py` regenerates the dashboard so it shows the real numbers.

### C) The dashboard
```bash
python fmr/scripts/make_dashboard.py   # bundle all outputs → fmr/dashboard/data.js
# then open fmr/dashboard/index.html  (works from file:// — no server needed)
```

---

## 9. Data & models

**Datasets (all fully open, no gated access):**
| Dataset | Modalities | Role |
|---|---|---|
| **VQA-RAD** | X-ray, CT | Core radiology eval |
| **SLAKE** (English) | CT, MRI, X-ray | Core eval + **real modality labels** (drives the per-modality view) |
| **PathVQA** | Pathology | Modality breadth (transfer beyond radiology) |
| **Synthetic** | all four (simulated) | Machinery validation with a known grounding latent |
| *OmniMedVQA* (optional) | 12 modalities | Breadth / held-out modality (parser ready; gated download) |

**Models:** *reasoning* = **MedVLM-R1** (RL-tuned medical reasoner), *non-reasoning* =
**Qwen2.5-VL-3B-Instruct** (its general-purpose cousin, same prompts) — a clean reasoning-vs-
non-reasoning contrast. A second mock base model demonstrates model-agnosticism.

---

## 10. Results (reported honestly)

### Machinery validation (synthetic — proves the code is correct)
- Per-signal separation AUROC: **A = 1.00, B = 0.93, C = 0.92**; the **fused FS = 0.999** matches
  or beats any single signal (the multi-signal design pays off).
- Conformal gate certifies **α = 0.05** at 1000 calibration points: coverage 0.58, retained error
  0.019 — the guarantee holds empirically.
- Robustness: FS is stable across fusion weights and random seeds; Signal B is present across grid
  resolutions (not an artifact).

### Real models (small Colab "smoke" sets — treated as provisional)
| Dataset | MedVLM-R1 acc | Qwen2.5-VL acc | Diagnosis verdict (blind-gap lens) |
|---|---|---|---|
| VQA-RAD (n≈20–75) | 0.40 | 0.57 | **Supported** — reasoning model relies on the image *less* … but **confounded** by its lower accuracy |
| SLAKE (n≈75) | 0.27 | 0.63 | **Supported** (same confound) |
| PathVQA (n≈50) | 0.24 | 0.28 | **No effect** — *both* models are image-independent here |

- The sharper **per-step grounding-drift** curve (Claim A's strongest form) awaits real
  attention→region extraction; on real data Signal B is currently constant, so the fused FS uses
  A+C there. The headline currently rests on the **blind-gap lens**, stated *with* its confound.
- **Abstention head-to-head** (risk-coverage AURC, lower = better): FS beats confidence on VQA-RAD
  (**0.749 vs 0.768**) but *not* on PathVQA/SLAKE, where confidence is stronger; at these small n
  no trigger separates cleanly and the gate honestly reports abstain-all. Constant signals
  (self-consistency/RadFlag/Signal B on some sets) are flagged and excluded from ranking.
- **Learned verifier:** beats the heuristic on real signals (**AUROC 0.768 → 0.816**) but is
  **capped by weak-label fidelity** (0.77) vs. a true-label oracle (0.951) — the gap is
  recoverable only with real grounding supervision.

**Reading these honestly:** the synthetic numbers prove the pipeline is *correct*; the real
numbers are *early and small-n*, with confounds surfaced rather than hidden. This mirror-honest
reporting is a deliberate feature of the project.

---

## 11. The interactive dashboard

A **zero-dependency** single-page app (no server, no build step, no CDN) at `fmr/dashboard/` —
open `index.html` and it runs. Nine tabs: **Overview, How It Works** (interactive 3-pillar
diagram), **The Diagnosis, Measurement, Safety Layer** (live risk-coverage with an α-slider),
**Case Explorer** (search + per-case reasoning chain + ANSWER/ABSTAIN decision), **Robustness,
Limitations & Honesty, Timeline.**

Honesty is built into the visuals: a **Mock/Real source switcher** with a persistent badge, a
**sample-size badge on every chart**, a watermark on mock-sourced charts, **dashed/muted styling
for provisional (n<50) curves** (so 20 points never look like a confident trend), the **gate
infeasibility shown, not hidden**, dark mode, hover tooltips, and a Mock-vs-Real compare view.

---

## 12. Glossary
- **VLM** — vision-language model (image+text → text).
- **CoT** — chain-of-thought; the model's written reasoning steps.
- **Faithfulness** — does the reasoning reflect what the model actually used (vs. just sounding good)?
- **Grounding** — how much a step depends on the image vs. language priors.
- **FS** — Faithfulness Score, the fused [0,1] output of Pillar 1.
- **Signal A/B/C** — counterfactual sensitivity / attention grounding / self-consistency.
- **Blind gap** — accuracy(real image) − accuracy(blank image); small = prior-driven.
- **AUROC** — how well a score separates two classes (0.5 chance, 1.0 perfect).
- **IoU** — region overlap (0–1).
- **Coverage / retained error** — fraction of cases answered / error rate among those answered.
- **Conformal / SGR gate** — the distribution-free rule bounding retained error at α.
- **Abstain / defer** — decline to answer, hand to a clinician.
- **MockVLM** — deterministic offline stand-in for machinery validation (synthetic, labeled).

---

## 13. Honesty, limitations & status

**Status:** the full pipeline (Pillars 1–3 + learned verifier + LLM-judge) is implemented,
integrated end-to-end, and covered by **101 passing tests**. Real-model results for VQA-RAD,
SLAKE, and PathVQA are on the branch and wired into the dashboard.

**Limitations (surfaced, not hidden):**
- Real runs are small-n smoke sets — noisy; treated as provisional.
- The blind-gap headline is confounded by the reasoning model's lower accuracy.
- Real Signal B (attention→region on HF models) is pending; fused FS uses A+C on real data meanwhile.
- The conformal guarantee needs ≳500–1000 calibration points; on small real sets it honestly
  reports abstain-all rather than a false guarantee.
- The public dataset mirrors lack per-question bounding boxes, so Signal-B IoU is validated on
  synthetic data.

**FMR is a research artifact, not a deployed clinical tool.** Its entire purpose is to make
medical reasoning models *verifiable* and *safe to defer* — the safety mechanism *is* the
contribution. Any real clinical use would require prospective validation.

---

*See `fmr/docs/project_summary.tex` for a compilable one-page academic summary, and
`RESULTS_LOG.md` / `DECISIONS.md` / `BLOCKERS.md` for the full audit trail.*
