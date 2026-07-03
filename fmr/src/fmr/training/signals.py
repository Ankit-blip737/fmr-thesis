"""Per-signal faithfulness features for the learned verifier.

This module produces, for one sample, the feature vector the verifier fuses:
the three FMM signals (A: counterfactual, B: attention/region grounding, C:
self-consistency) plus cheap auxiliaries already available in the harness (answer
margin, chain length, and — critically — how grounding *drifts across the reasoning
chain*, which is the very effect the thesis studies).

Interface parity: these are stand-ins for Instance A's per-signal outputs
(`faithfulness/score.py` is expected to expose raw Signals A/B/C, not just the
fused FS). The verifier consumes a plain feature dict, so swapping this provider
for Instance A's API is a one-line change in `dataset.build_feature_frame`.

Why the drift/aggregate features matter for the *learned* head: a fixed weighted
sum of (A, B, C) cannot express "trust B when A and B disagree" — exactly the
regime of a prior-dominated-but-image-present model, where the answer barely
flips (low A) yet the model does attend to the lesion (high B). The learned head
can; that is the hypothesis this feature set is designed to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.regions import Region
from ..faithfulness.counterfactual import counterfactual_signal
from ..models.base_vlm import BaseVLM
from ..types import Sample, VLMOutput
from ..utils import clip01, rng_from

# The three raw signals (what a learned fusion must combine).
SIGNAL_KEYS = ("sig_a_counterfactual", "sig_b_grounding", "sig_c_consistency")


@dataclass
class SampleFeatures:
    sample_id: str
    features: dict                        # name -> float (see FEATURE_KEYS in verifier)
    step_grounding: list[float] = field(default_factory=list)  # per-step Signal-B (IoU)
    orig: Optional[VLMOutput] = None
    meta: dict = field(default_factory=dict)


def _grid_iou(region: Optional[Region], gt: Optional[Region], grid: int) -> float:
    """Signal-B proxy: IoU of a step's attended region with the GT region, but
    measured through a coarse grid quantization (a stand-in for the imprecision of
    real attention-rollout region estimates — so the feature is an *imperfect*
    observation of grounding, not the latent itself)."""
    if region is None or gt is None:
        return 0.0
    def snap(r: Region) -> Region:
        cx = min(int(((r.x0 + r.x1) / 2) * grid), grid - 1)
        cy = min(int(((r.y0 + r.y1) / 2) * grid), grid - 1)
        return Region.from_grid_cell(cy, cx, grid, grid)
    return snap(region).iou(snap(gt))


def _self_consistency(vlm: BaseVLM, sample: Sample, greedy: VLMOutput, n_chains: int) -> tuple[float, float]:
    """Signal C. Returns (answer_consistency, region_consistency) in [0, 1].

    answer_consistency: fraction of N sampled chains whose answer matches greedy.
    region_consistency: mean, over matched step positions, of cross-chain region
    IoU agreement with the greedy chain (stable attention => faithful).
    """
    answers, chains = [], []
    for i in range(1, n_chains + 1):
        out = vlm.generate(sample, variant="original", temperature=0.7, draw=i)
        answers.append(out.answer)
        chains.append(out.steps)
    if not answers:
        return 1.0, 1.0
    ans_cons = float(np.mean([a == greedy.answer for a in answers]))

    ious = []
    for steps in chains:
        for gs, cs in zip(greedy.steps, steps):
            if gs.pred_region is not None and cs.pred_region is not None:
                ious.append(gs.pred_region.iou(cs.pred_region))
    reg_cons = float(np.mean(ious)) if ious else 0.0
    return ans_cons, reg_cons


def compute_sample_features(
    vlm: BaseVLM,
    sample: Sample,
    n_chains: int = 4,
    cf: Optional[dict] = None,
    noise: float = 0.0,
) -> SampleFeatures:
    """Compute the full feature dict for one sample.

    ``cf`` optionally passes a pre-computed counterfactual_signal dict (to avoid
    recomputation when the caller already has it).

    ``noise`` (>=0) injects deterministic Gaussian *measurement noise* into the
    raw signals before aggregation, clipped to [0, 1]. Real counterfactual and
    attention-grounding estimates are noisy, moderate-strength signals — not the
    clean latent the deterministic mock would otherwise expose. Sweeping ``noise``
    is how we test the thesis's actual claim: that a *learned* fusion of noisy
    signals degrades more gracefully than a fixed hand-weighting. ``noise=0``
    reproduces the clean (idealized) regime and keeps unit tests deterministic.
    """
    cf = cf or counterfactual_signal(vlm, sample)
    orig: VLMOutput = cf["orig"]
    gt = sample.gt_region
    grid = int(sample.meta.get("grid", 4))

    rng = rng_from(sample.sample_id, vlm.name, "signal-noise") if noise > 0 else None

    def _noisy(x: float, tag: int) -> float:
        if rng is None:
            return x
        return clip01(x + rng.normal(0.0, noise))

    # Signal B per step (grounding via region<->GT IoU proxy), with measurement noise.
    step_iou = [_noisy(_grid_iou(s.pred_region, gt, grid), i) for i, s in enumerate(orig.steps)]
    if not step_iou:
        step_iou = [0.0]
    b_mean = float(np.mean(step_iou))
    b_max = float(np.max(step_iou))
    b_first = float(step_iou[0])
    b_last = float(step_iou[-1])
    # Grounding drift across the chain (negative slope = grounding decays as it reasons).
    if len(step_iou) > 1:
        xs = np.arange(len(step_iou))
        b_slope = float(np.polyfit(xs, step_iou, 1)[0])
    else:
        b_slope = 0.0

    ans_cons, reg_cons = _self_consistency(vlm, sample, orig, n_chains)
    sig_c = _noisy(0.5 * ans_cons + 0.5 * reg_cons, 99)
    sig_a = _noisy(float(cf["counterfactual"]), 100)

    # Answer-logit margin (top1 - top2): a cheap confidence auxiliary.
    if orig.answer_logits is not None and len(orig.answer_logits) > 1:
        srt = np.sort(np.asarray(orig.answer_logits))[::-1]
        margin = float(srt[0] - srt[1])
    else:
        margin = 0.0

    features = {
        # raw signals A/B/C (the three the heuristic fuses)
        "sig_a_counterfactual": sig_a,
        "sig_b_grounding": b_mean,
        "sig_c_consistency": sig_c,
        # Signal A internals
        "a_flip_rate": float(cf["flip_rate"]),
        "a_js": float(cf["js_divergence"]),
        # Signal B aggregates / drift (extra info the learned head can exploit)
        "b_iou_max": b_max,
        "b_iou_first": b_first,
        "b_iou_last": b_last,
        "b_iou_slope": b_slope,
        # Signal C split
        "c_answer_consistency": ans_cons,
        "c_region_consistency": reg_cons,
        # auxiliaries
        "aux_answer_margin": margin,
        "aux_n_steps": float(len(orig.steps)),
    }
    return SampleFeatures(
        sample_id=sample.sample_id,
        features=features,
        step_grounding=step_iou,
        orig=orig,
        meta={"model": vlm.name},
    )
