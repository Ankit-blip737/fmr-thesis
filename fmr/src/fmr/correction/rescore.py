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

from typing import Callable, Optional

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


def post_correction_fs(
    vlm: BaseVLM,
    sample: Sample,
    corrected: VLMOutput,
    *,
    attention_fn: Optional[Callable[[VLMOutput], dict]] = None,
    consistency_c: Optional[float] = None,
    fuse_fn: Optional[Callable[[float, float, float], float]] = None,
    reuse: Optional[dict] = None,
) -> dict:
    """Full *fused* post-correction Faithfulness Score — what the conformal gate
    should calibrate on (fix #2), assembled via dependency injection so this file
    never imports Instance A's fusion.

    * Signal A is recomputed on the corrected output (``post_correction_sensitivity``).
    * Signal B is recomputed on the corrected *steps* via ``attention_fn`` — pass
      Instance A's ``faithfulness.attention.attention_signal`` here on merge; if
      absent, a mean grid-IoU proxy over the corrected steps is used.
    * Signal C (self-consistency) is a property of the model's sampling, largely
      unchanged by deterministic correction, so the pre-correction ``consistency_c``
      is reused; pass Instance A's ``signal_c`` for the sample.
    * ``fuse_fn(a, b, c)`` is Instance A's ``faithfulness.score.fuse`` on merge; the
      default mirrors their published weights (0.4 / 0.3 / 0.3).

    Returns a dict with ``fs`` (the fused post-correction score) plus the raw
    a/b/c, so the gate can consume ``fs`` directly and everything is auditable.
    """
    a_dict = post_correction_sensitivity(vlm, sample, corrected, reuse=reuse)
    a = float(a_dict["counterfactual"])

    if attention_fn is not None:
        b = float(attention_fn(corrected).get("attention", 0.0))
    else:
        b = _mean_grid_iou(corrected, sample)

    c = float(consistency_c) if consistency_c is not None else a

    if fuse_fn is not None:
        fs = float(fuse_fn(a, b, c))
    else:
        fs = clip01((0.4 * a + 0.3 * b + 0.3 * c) / 1.0)

    return {"corrected": corrected, "fs": fs, "signal_a": a, "signal_b": b,
            "signal_c": c, "flip_rate": a_dict["flip_rate"], "js_divergence": a_dict["js_divergence"]}


def _mean_grid_iou(output: VLMOutput, sample: Sample) -> float:
    """Fallback Signal-B proxy over corrected steps (coarse region<->GT IoU)."""
    gt = sample.gt_region
    if gt is None or not output.steps:
        return 0.0
    grid = int(sample.meta.get("grid", 4))
    from ..data.regions import Region

    def snap(r: Region) -> Region:
        cx = min(int(((r.x0 + r.x1) / 2) * grid), grid - 1)
        cy = min(int(((r.y0 + r.y1) / 2) * grid), grid - 1)
        return Region.from_grid_cell(cy, cx, grid, grid)

    ious = [snap(s.pred_region).iou(snap(gt)) for s in output.steps if s.pred_region is not None]
    return float(sum(ious) / len(ious)) if ious else 0.0
