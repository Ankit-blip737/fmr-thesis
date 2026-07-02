"""Signal A - Counterfactual sensitivity.

A faithful answer *depends on the image*: remove or swap the image and the answer
should change. We contrast the original output against a blanked image and a
mismatched image, and quantify sensitivity via (i) answer-flip rate and (ii)
Jensen-Shannon divergence of the answer-logit distribution.

Inspired by Visual Contrastive Decoding's distorted-input contrast and
perturbation-based faithfulness evaluation.
"""
from __future__ import annotations

from ..models.base_vlm import BaseVLM
from ..types import Sample, VLMOutput
from ..utils import clip01, js_divergence


def counterfactual_signal(vlm: BaseVLM, sample: Sample) -> dict:
    """Return the original output + counterfactual sensitivity features.

    ``sensitivity`` in [0, 1] is high when the model's answer genuinely responds
    to the image (faithful) and near 0 when it ignores the image (prior-driven).
    """
    orig: VLMOutput = vlm.generate(sample, variant="original", temperature=0.0)
    blank: VLMOutput = vlm.generate(sample, variant="blank", temperature=0.0)
    mism: VLMOutput = vlm.generate(sample, variant="mismatch", temperature=0.0)

    flip = 0.5 * (orig.answer != blank.answer) + 0.5 * (orig.answer != mism.answer)

    js = 0.0
    if orig.answer_logits is not None:
        js_blank = js_divergence(orig.answer_logits, blank.answer_logits)
        js_mism = js_divergence(orig.answer_logits, mism.answer_logits)
        js = 0.5 * (js_blank + js_mism)  # already in [0, 1] (log base 2)

    sensitivity = clip01(0.5 * float(flip) + 0.5 * float(js))
    return {
        "orig": orig,
        "counterfactual": sensitivity,
        "flip_rate": float(flip),
        "js_divergence": float(js),
    }
