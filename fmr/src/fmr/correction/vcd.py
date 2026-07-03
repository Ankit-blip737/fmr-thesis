"""Visual Contrastive Decoding (VCD) — Leng et al., CVPR 2024 (arXiv:2311.16922).

Idea: tokens driven by the *language prior* score highly whether or not the true
image is present; tokens driven by *image evidence* score highly only with the
true image. Contrasting the two distributions therefore cancels the prior and
amplifies the evidence:

    log p_vcd = (1 + alpha) * log p(y | image) - alpha * log p(y | distorted)

subject to the paper's *adaptive plausibility constraint*: only answers with
p(y | image) >= beta * max p(. | image) stay candidates, so the contrast can
never promote an answer the original model found implausible.

Granularity note: the reference implementation applies this per decoding step.
Here it is applied to the answer distribution that ``BaseVLM.generate`` already
exposes — in the closed-vocabulary Med-VQA setting that distribution *is* the
next-token distribution over answers, so the math is identical and the module
stays model-agnostic. For real HF models the per-token version is the same
formula inside a logits processor (see the GPU handoff notebook).

Expected behaviour (and the honest limitation): VCD rescues answers where image
evidence exists but is out-voted by the prior. If the model extracted *no*
image information (p_orig == p_distorted), the contrast is a no-op — those cases
are precisely what the abstention gate is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.base_vlm import BaseVLM
from ..types import Sample, VLMOutput

_EPS = 1e-12


@dataclass
class VCDResult:
    """Outcome of the answer-level VCD contrast for one sample."""

    answer: str                       # VCD-corrected answer
    original_answer: str
    changed: bool                     # did the contrast flip the answer?
    p_vcd: np.ndarray                 # corrected answer distribution
    margin: float                     # top1-top2 gap in contrasted log-space
    n_plausible: int                  # answers kept by the plausibility mask
    contrast_variants: tuple[str, ...] = ("blank", "mismatch")
    outputs: dict[str, VLMOutput] = field(default_factory=dict)  # variant -> raw output


def vcd_contrast(
    p_orig: np.ndarray,
    p_dist: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> np.ndarray:
    """Pure VCD math on two answer distributions -> corrected distribution."""
    log_o = np.log(np.asarray(p_orig, dtype=float) + _EPS)
    log_d = np.log(np.asarray(p_dist, dtype=float) + _EPS)
    combo = (1.0 + alpha) * log_o - alpha * log_d
    # Adaptive plausibility constraint (relative to the *original* distribution).
    keep = p_orig >= beta * float(np.max(p_orig))
    combo = np.where(keep, combo, -np.inf)
    combo -= np.max(combo)
    p = np.exp(combo)
    return p / p.sum()


def vcd_answer(
    vlm: BaseVLM,
    sample: Sample,
    alpha: float = 1.0,
    beta: float = 0.1,
    contrast_variants: tuple[str, ...] = ("blank", "mismatch"),
) -> VCDResult:
    """Run the VCD contrast for one sample through the ``BaseVLM`` interface.

    Contrasts the original answer distribution against each distorted variant
    and averages the contrasted log-scores (equivalent to contrasting against
    the geometric mean of the distorted distributions).
    """
    orig = vlm.generate(sample, variant="original", temperature=0.0)
    outputs: dict[str, VLMOutput] = {"original": orig}

    vocab = sample.answer_choices or [sample.answer]
    if orig.answer_logits is None:
        # Backend exposes no distribution (real-model scaffold path): no-op.
        return VCDResult(
            answer=orig.answer,
            original_answer=orig.answer,
            changed=False,
            p_vcd=np.full(len(vocab), 1.0 / len(vocab)),
            margin=0.0,
            n_plausible=len(vocab),
            contrast_variants=tuple(contrast_variants),
            outputs=outputs,
        )

    p_o = np.asarray(orig.answer_logits, dtype=float)
    log_o = np.log(p_o + _EPS)
    combos = []
    for variant in contrast_variants:
        out_v = vlm.generate(sample, variant=variant, temperature=0.0)
        outputs[variant] = out_v
        log_d = np.log(np.asarray(out_v.answer_logits, dtype=float) + _EPS)
        combos.append((1.0 + alpha) * log_o - alpha * log_d)
    combo = np.mean(combos, axis=0)

    keep = p_o >= beta * float(np.max(p_o))
    combo = np.where(keep, combo, -np.inf)

    idx = int(np.argmax(combo))
    finite = np.sort(combo[np.isfinite(combo)])[::-1]
    margin = float(finite[0] - finite[1]) if len(finite) > 1 else float("inf")

    shifted = combo - np.max(combo)
    p_vcd = np.exp(shifted)
    p_vcd = p_vcd / p_vcd.sum()

    answer = vocab[idx] if idx < len(vocab) else orig.answer
    return VCDResult(
        answer=answer,
        original_answer=orig.answer,
        changed=answer != orig.answer,
        p_vcd=p_vcd,
        margin=margin,
        n_plausible=int(keep.sum()),
        contrast_variants=tuple(contrast_variants),
        outputs=outputs,
    )
