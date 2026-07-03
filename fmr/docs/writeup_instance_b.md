# FMR — Draft thesis sections (Instance B components)

> Draft prose for the thesis, grounded in the numbers logged in `RESULTS_LOG.md`
> (tag `[B]`). All figures below are from the offline synthetic pipeline
> (`MockVLM` + `PriorHeavyMockVLM`); real-model numbers replace them once the
> Colab runs land (`fmr/notebooks/colab_*.ipynb`). Every claim here is
> reproducible from `fmr/scripts/` against `fmr/results/`. These sections cover
> the correction module (Pillar 2), correction's contribution to abstention
> (supporting Pillar 3), and the LLM-as-judge validation.

---

## X. Training-Free Correction of Ungrounded Reasoning (Pillar 2)

### X.1 Role and scope

Correction is deliberately positioned as *supporting infrastructure for the
abstention layer*, not as a co-equal contribution. The thesis's headline claims
are (i) that medical reasoning VLMs lose visual groundedness as their reasoning
lengthens, and (ii) that calibrated, faithfulness-triggered abstention is a
genuinely new safety layer. Correction exists to make that second claim stronger:
by repairing the cases that are *fixable*, it reduces how many cases must be
deferred to a clinician, while leaving the genuinely unrecoverable cases — those
where the model never used the image — to the abstention gate. Every design and
evaluation choice below follows from that framing.

The module is applied **selectively**: it runs only when a sample's faithfulness
score falls below a trigger threshold. On well-grounded reasoning it is a no-op,
which is what prevents the classic failure mode of decode-time interventions
(hurting the cases that were already correct).

### X.2 Method

Correction composes three training-free, decode-time primitives.

**Visual Contrastive Decoding (VCD).** Following Leng et al. (CVPR 2024), tokens
driven by the language prior score highly whether or not the true image is
present, whereas tokens driven by image evidence score highly only with the true
image. Contrasting the two answer distributions therefore cancels the prior and
amplifies the evidence:

  log p_vcd = (1 + α) · log p(y | image) − α · log p(y | distorted image),

subject to an adaptive plausibility constraint that keeps only answers with
p(y | image) ≥ β · max p(· | image), so the contrast can never promote an answer
the original model found implausible. In the closed-vocabulary medical-VQA
setting the answer distribution *is* the next-token distribution over answer
choices, so this operates model-agnostically through the base-model interface;
the per-token variant for open-ended decoding is a direct instantiation of the
same formula inside a logits processor.

**Question-to-vision clue tracing.** We locate the image region that actually
carries the evidence for the question — the "clue" — independently of any single
reasoning step. Probing several reasoning chains and taking the weighted medoid
of every attended region (early steps up-weighted, since grounding drifts as the
chain grows) yields both an evidence region and a confidence equal to the medoid's
mean agreement: grounded chains cluster tightly around the evidence, ungrounded
chains scatter.

**Step verify-and-revise.** Each reasoning step is scored for visual support
against the traced clue; unsupported steps are dropped and the rationale is
rebuilt from the supported steps only. The answer adopts the VCD-corrected value
only when the contrast changed it with a clear margin — a low-margin flip is
treated as contrast noise — so the revision is conservative by construction.

The (possibly corrected) output is then re-scored for faithfulness. This
post-correction score, not the raw pre-correction score, is what the abstention
gate must calibrate on, because it is the score of the system that is actually
deployed (base model + correction).

### X.3 Component ablation

To attribute the gain to its source we decompose correction on the flagged
(low-faithfulness) samples, adding one primitive at a time (Section 9 protocol).
Two synthetic backends are used: an *image-blind* model whose ungrounded answers
carry no recoverable image signal, and a *prior-dominated* model that does attend
to the evidence but lets a language prior out-vote it — the exact regime VCD is
designed to repair.

*Prior-dominated backend (flagged = 393):*

| Configuration | Answer acc. | Step-support rate | Steps kept | Correct broken |
|---|---|---|---|---|
| Baseline (no correction) | 0.537 | 0.260 | 100% | 0 |
| + VCD | **0.695** | 0.260 | 100% | 0 |
| + Verify-revise (no clue) | 0.537 | **0.529** | 29% | 0 |
| + Verify-revise + clue | 0.537 | **0.551** | 30% | 0 |
| Full | **0.695** | **0.551** | 30% | 0 |

Three findings follow. **VCD owns the answer-accuracy gain** (+15.8 points on the
prior-dominated pathology; approximately zero, −0.7 points, on the image-blind
pathology where there is nothing to recover), and it never edits the rationale.
**Verify-revise owns rationale faithfulness**, roughly doubling the fraction of
retained steps that are genuinely image-grounded (0.260 → 0.529) by discarding
ungrounded steps down to about 30% of the chain. **Clue-tracing improves the
support source** over a no-trace self-consistency baseline (0.529 → 0.551) on the
fixable pathology; honestly, on the image-blind pathology it is neutral-to-slightly
worse, because when almost no step is grounded there is no coherent clue to trace —
which is consistent with clue-tracing helping only where evidence exists. Across
*every* configuration, **zero** correct answers were broken on a well-grounded
holdout, confirming the selective design does not harm cases that were already
right.

### X.4 End-to-end effect

Run as a selective pipeline, correction raises overall accuracy on the
prior-dominated backend from 0.545 to 0.700. On the image-blind backend, where
grounded chains are flagged only because of reasoning drift rather than a missing
image signal, 92% of those grounded chains are re-scored above the abstention gate
after correction and so no longer need deferring. On both backends, 96–99% of the
genuinely image-blind cases remain below the gate and flow to deferral. (On the
prior-dominated backend, gate-recovery is lower at 33%, because that model's
counterfactual faithfulness stays low even once the answer is fixed — those cases
gain accuracy but are still, correctly, routed to review; the accuracy gain is
realized regardless of whether the case is retained or deferred.) The downstream
consequence for abstention is quantified in Section Y.

### X.5 Limitations

The numbers above are from a synthetic pipeline whose signals are clean by
construction; they validate the machinery and the attribution logic, not the
effect size on real scans. The real-model instantiation (Qwen2.5-VL-3B on VQA-RAD,
`colab_stage4_correction_real.ipynb`) runs the identical correction code on real
answer distributions and will replace these figures. Clue-tracing's benefit is
modest and evidence-dependent; VCD is the dominant primitive.

---

## Y. Correction's Contribution to Faithfulness-Aware Abstention

> The conformal gate itself is described elsewhere (Pillar 3). This section
> isolates *what correction contributes to it*, via an illustrative
> threshold-sweep comparison — not a conformal guarantee.

A selective predictor answers a case when its faithfulness score exceeds a
threshold and defers otherwise. We compare two deferral *signals* on identical
data: a **naive** gate that thresholds the pre-correction score and returns the raw
answer, and an **FMR** gate that thresholds the post-correction score and returns
the corrected answer.

On the prior-dominated backend the difference is large. Correction lifts overall
accuracy from 0.545 to 0.700, and the area under the risk–coverage curve (lower is
better) falls from 0.352 to 0.128 — a 2.7× reduction in risk under selective
prediction. Most tellingly for the clinical framing, the fraction of cases the
system can answer while holding retained accuracy at or above 90% rises from 9.3%
to 48.5% (a 5.2× increase); at the ≥95% level it rises from 4.8% to roughly 43%.
In other words, correction lets the system safely answer five times as many cases
before deferring.

On the image-blind backend correction is, correctly, close to neutral (risk–coverage
AUC 0.065 vs 0.076): when there is no image evidence to recover, correction does
not move the frontier, and those cases remain deferred. This asymmetry is the
intended behaviour — correction shrinks the abstain set precisely where the
problem is fixable and stays inert where it is not.

This result also motivates the calibration ordering: because correction shifts the
faithfulness distribution of the deployed system, the conformal gate must be
calibrated on post-correction scores for its coverage guarantee to describe the
system that is actually run.

---

## Z. Validating the LLM-as-Judge

Open-ended medical answers are scored with an LLM-as-judge. A generic judge
scoring clinical correctness is an underspecified risk: if the judge is wrong,
every open-ended metric downstream inherits that error silently. We therefore
treat the judge as something to be *validated before trusted*, not assumed.

### Z.1 Protocol

We hand-authored a 44-item gold set (22 correct / 17 incorrect / 5 partial)
deliberately adversarial to naive string matching: clinical synonyms
(“enlarged heart” ↔ “cardiomegaly”), negation and polarity flips
(“no effusion” vs “effusion”), severity gradations, containment, and multi-finding
partial-credit cases. Judge verdicts are three-way (correct / partial / incorrect)
and agreement with the gold labels is measured by three-way accuracy, a
correct-vs-not binary accuracy, and Cohen’s κ.

### Z.2 Iterating the heuristic judge

The dependency-free heuristic judge — the fallback that runs when no LLM is
available — was revised until its disagreements were principled rather than
mechanical:

| Version | 3-way acc. | Cohen’s κ |
|---|---|---|
| v1 (string + whole-string synonyms) | 0.795 | 0.634 |
| v2 (token-level synonyms, number/plural normalization, multi-finding coverage) | 0.909 | 0.836 |
| v3 (comma-robust multi-finding; “names finding, omits detail” → partial) | 0.955 | 0.923 |
| v4 (polarity/severity words excluded from *finding* count; polarity-aware coverage) | **1.000** | **1.000** |

### Z.3 Honesty about κ = 1.0

A κ of 1.0 on a gold set that was *both authored and tuned against* is an upper
bound on a hand-tuned rule, not a field estimate, and the thesis reports it as
such. Two independent guards support the claim that the judge encodes general
clinical-text principles rather than memorized rows. First, the unit tests include
nine held-out probes never present in the gold set (e.g. “bleeding” ↔ “haemorrhage”,
“broken bone” ↔ “fracture”, “three” ↔ “3”, held-out multi-finding partials); all
are graded correctly. Second, an *independent* open LLM judge (Qwen2.5-7B-Instruct,
`colab_judge_llm.ipynb`) scores the same gold set with the rubric-constrained
prompt, reporting both judge-vs-gold and judge-vs-heuristic agreement; that number,
once available, is the external check that upgrades the tuned heuristic into a
trustworthy metric. Until then, downstream open-ended metrics use the validated
heuristic as the fallback and treat any judge score as untrusted until its κ is
logged.

### Z.4 Robustness

The LLM judge is rubric-constrained (verdict word on the first line, one-sentence
justification on the second) and falls back to the heuristic on any provider error
or unparseable completion, so a flaky judge can never silently emit garbage
verdicts — it degrades to the validated rule instead.
