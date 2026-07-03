"""Signal B - Attention / relevancy grounding.

Two distinct jobs live here, and keeping them separate is important for honesty:

1. **Runtime signal** (``attention_signal``) — computable at inference time with
   NO ground truth. We score the *spatial coherence* of the attended regions
   across the reasoning chain: a grounded chain keeps returning to the same
   evidence region (the finding), while an ungrounded chain's attention wanders
   over the image. For real HF models the per-step region comes from
   attention-rollout / relevancy maps (see ``models/hf_vlm.py``); for MockVLM it
   is emitted directly. Known limitation (stated in the thesis): a chain that
   consistently attends to the same *wrong* region scores high on B — which is
   exactly why FMR never trusts a single signal (Signals A and C catch that
   case through independent mechanisms).

2. **Validation / weak labels** (``iou_labels``) — where datasets provide
   ground-truth boxes (SLAKE, VQA-RAD), per-step IoU between the attended
   region and the GT region. This validates the runtime signal and produces
   the grounded/ungrounded weak labels that Instance B's learned verifier
   trains on. On PathVQA / OmniMedVQA no boxes exist, so Signal B is reported
   as *unvalidated / exploratory* there — every table must say so.
"""
from __future__ import annotations

from ..data.regions import Region
from ..types import Sample, VLMOutput
from ..utils import clip01

# Below this IoU the attended region is considered off-target (weak label = 0).
IOU_GROUNDED_THRESHOLD = 0.30


def _pairwise_mean_iou(region: Region, others: list[Region]) -> float:
    others = [r for r in others if r is not region]
    if not others:
        return 0.5  # single-step chain: no coherence evidence either way
    return sum(region.iou(o) for o in others) / len(others)


def attention_signal(output: VLMOutput) -> dict:
    """Score each step's attention grounding WITHOUT ground truth.

    Per-step score = mean IoU of this step's attended region with every other
    step's region (spatial coherence of the evidence trail). Aggregate = mean
    over steps. Neutral 0.5 for single-step chains (no coherence evidence),
    and for steps with no attended region at all.
    """
    regions = [s.pred_region for s in output.steps]
    per_step: list[float] = []
    for step in output.steps:
        if step.pred_region is None:
            per_step.append(0.5)
            continue
        others = [r for r in regions if r is not None]
        per_step.append(clip01(_pairwise_mean_iou(step.pred_region, others)))
    for step, score in zip(output.steps, per_step):
        step.attention_grounding = score
    agg = sum(per_step) / len(per_step) if per_step else 0.5
    return {"attention": float(agg), "per_step": per_step}


def iou_labels(output: VLMOutput, sample: Sample, threshold: float = IOU_GROUNDED_THRESHOLD) -> dict:
    """Per-step IoU vs the ground-truth region + derived weak labels.

    Only meaningful where ``sample.gt_region`` exists (SLAKE / VQA-RAD /
    synthetic). Returns ``ious=None`` otherwise so callers can mark the
    modality as unvalidated rather than silently reporting garbage.
    """
    if sample.gt_region is None:
        return {"ious": None, "labels": None, "mean_iou": None}
    ious: list[float] = []
    labels: list[int] = []
    for step in output.steps:
        iou = step.pred_region.iou(sample.gt_region) if step.pred_region is not None else 0.0
        label = int(iou >= threshold)
        step.grounded_label = label
        ious.append(float(iou))
        labels.append(label)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {"ious": ious, "labels": labels, "mean_iou": float(mean_iou)}
