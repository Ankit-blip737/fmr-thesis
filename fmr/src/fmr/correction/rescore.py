"""Post-correction faithfulness — the score conformal calibration MUST use.

Calibration-ordering requirement (see DECISIONS.md [B]): the abstention gate
guarantees an error rate for the *deployed* system, and the deployed system is
base-model + correction. Calibrating on pre-correction scores would certify the
wrong distribution. This module recomputes Signal A's counterfactual
sensitivity for the corrected output, with the exact same output shape as
``fmr.faithfulness.counterfactual.counterfactual_signal`` so it is a drop-in
input for the conformal gate.
"""
from __future__ import annotations

from ..models.base_vlm import BaseVLM
from ..types import Sample, VLMOutput
from ..utils import clip01, js_divergence


def post_correction_sensitivity(
    vlm: BaseVLM,
    sample: Sample,
    corrected: VLMOutput,
    reuse: dict[str, VLMOutput] | None = None,
) -> dict:
    """Counterfactual sensitivity of the corrected system's output.

    The corrected system reduces to the raw model under a distorted input (the
    VCD contrast of a distribution with itself is that distribution), so the
    counterfactual references are the plain ``blank``/``mismatch`` generations.
    ``reuse`` lets callers pass already-generated variant outputs (e.g. from
    ``VCDResult.outputs``) to avoid re-running the model.
    """
    reuse = reuse or {}
    blank = reuse.get("blank") or vlm.generate(sample, variant="blank", temperature=0.0)
    mism = reuse.get("mismatch") or vlm.generate(sample, variant="mismatch", temperature=0.0)

    flip = 0.5 * (corrected.answer != blank.answer) + 0.5 * (corrected.answer != mism.answer)

    js = 0.0
    if corrected.answer_logits is not None and blank.answer_logits is not None:
        js_blank = js_divergence(corrected.answer_logits, blank.answer_logits)
        js_mism = js_divergence(corrected.answer_logits, mism.answer_logits)
        js = 0.5 * (js_blank + js_mism)

    sensitivity = clip01(0.5 * float(flip) + 0.5 * float(js))
    return {
        "corrected": corrected,
        "counterfactual": sensitivity,
        "flip_rate": float(flip),
        "js_divergence": float(js),
    }
