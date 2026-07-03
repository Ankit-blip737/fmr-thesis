"""Aggregate Faithfulness Score (FS) — heuristic fusion of Signals A/B/C.

This module is the single entry point for scoring one sample
(``compute_faithfulness``) and defines the **record schema** that everything
downstream consumes — the conformal gate, the figures, and Instance B's learned
verifier. The contract (logged in DECISIONS.md, do not break silently):

Each record is a flat dict with at least these keys:
    sample_id, modality, answer, correct,
    signal_a            raw counterfactual sensitivity in [0, 1]
    signal_a_flip       answer-flip component of A
    signal_a_js         JS-divergence component of A
    signal_b            raw attention-coherence score in [0, 1]
    signal_b_per_step   list[float], one per reasoning step
    signal_c            raw self-consistency score in [0, 1]
    signal_c_vote       raw vote share behind C
    fs                  heuristic fused score in [0, 1]
    fs_per_step         list[float], per-step fused scores
    confidence          model's own answer confidence (max answer logit)
    n_steps             chain length
    iou_mean            mean step IoU vs GT region (None where no boxes)
    iou_per_step        list[float] or None
    weak_labels         list[int] or None  (IoU-threshold grounded labels)
    grounded_latent     hidden synthetic latent (synthetic data only; eval use)

The learned verifier trains on (signal_a, signal_b, signal_c, + auxiliaries)
against weak_labels — the raw per-signal scores are exposed exactly so that the
heuristic weighting here never becomes a bottleneck for it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.base_vlm import BaseVLM
from ..types import Sample
from ..utils import clip01
from .attention import attention_signal, iou_labels
from .consistency import consistency_signal
from .counterfactual import counterfactual_signal


def _top2_margin(logits) -> float:
    """Top-1 minus top-2 probability of the answer distribution (an easy
    auxiliary the learned verifier uses; 1.0 when there is only one choice)."""
    if logits is None:
        return 0.5
    a = np.sort(np.asarray(logits, dtype=float))[::-1]
    return float(a[0] - a[1]) if a.size >= 2 else 1.0


def _slope(vals: list[float]) -> float:
    """Least-squares slope of a per-step sequence — the grounding-DRIFT feature
    (negative slope = grounding decays along the chain, the thesis's effect)."""
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.asarray(vals, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

# Heuristic weights: A is the most direct probe of image dependence, B and C
# corroborate. These are the *fallback* the learned verifier must beat.
DEFAULT_WEIGHTS = {"a": 0.4, "b": 0.3, "c": 0.3}


def fuse(a: float, b: float, c: float, weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    total = w["a"] + w["b"] + w["c"]
    return clip01((w["a"] * a + w["b"] * b + w["c"] * c) / total)


def compute_faithfulness(
    vlm: BaseVLM,
    sample: Sample,
    weights: dict[str, float] | None = None,
    n_consistency_samples: int = 5,
    consistency_temperature: float = 0.7,
) -> dict[str, Any]:
    """Run Signals A, B, C on one sample and return the fused record."""
    # Signal A also produces the canonical original-image output we score.
    cf = counterfactual_signal(vlm, sample)
    output = cf["orig"]

    att = attention_signal(output)
    cons = consistency_signal(
        vlm, sample, n_samples=n_consistency_samples, temperature=consistency_temperature
    )
    iou = iou_labels(output, sample)

    a, b, c = cf["counterfactual"], att["attention"], cons["consistency"]
    fs = fuse(a, b, c, weights)

    # Per-step FS: B varies per step; A and C are chain-level, applied uniformly.
    fs_per_step = [fuse(a, b_k, c, weights) for b_k in att["per_step"]]
    for step, step_fs in zip(output.steps, fs_per_step):
        step.counterfactual = a
        step.consistency = c
        step.fs = step_fs

    confidence = float(output.answer_logits.max()) if output.answer_logits is not None else 0.5
    answer_margin = _top2_margin(output.answer_logits)

    return {
        "sample_id": sample.sample_id,
        "modality": sample.modality,
        # Additive fields (safe for the verifier contract) used by the dashboard's
        # per-case explorer to show the actual question + reasoning chain.
        "question": sample.question,
        "gt_answer": sample.answer,
        "steps_text": [s.text for s in output.steps],
        "answer": output.answer,
        "correct": int(output.answer == sample.answer),
        "signal_a": float(a),
        "signal_a_flip": cf["flip_rate"],
        "signal_a_js": cf["js_divergence"],
        "signal_b": float(b),
        "signal_b_per_step": att["per_step"],
        "signal_c": float(c),
        "signal_c_vote": cons["vote_share"],
        "fs": float(fs),
        "fs_per_step": fs_per_step,
        "confidence": confidence,
        "answer_margin": answer_margin,
        "n_steps": len(output.steps),
        "iou_mean": iou["mean_iou"],
        "iou_per_step": iou["ious"],
        "weak_labels": iou["labels"],
        "grounded_latent": sample.meta.get("grounded"),
        "output": output,
    }


def score_dataset(
    vlm: BaseVLM,
    samples: list[Sample],
    weights: dict[str, float] | None = None,
    n_consistency_samples: int = 5,
    consistency_temperature: float = 0.7,
) -> list[dict[str, Any]]:
    """Score a list of samples; the workhorse behind run_fmr.py."""
    return [
        compute_faithfulness(
            vlm,
            s,
            weights=weights,
            n_consistency_samples=n_consistency_samples,
            consistency_temperature=consistency_temperature,
        )
        for s in samples
    ]


# Feature keys emitted for Instance B's learned verifier — matched exactly to
# its training/signals.py schema so `build_feature_frame` can call
# `features_for_verifier` directly (the promised one-line provider swap).
VERIFIER_FEATURE_KEYS = (
    "sig_a_counterfactual", "sig_b_grounding", "sig_c_consistency",
    "a_flip_rate", "a_js", "b_iou_max", "b_iou_first", "b_iou_last",
    "b_iou_slope", "c_answer_consistency", "c_region_consistency",
    "aux_answer_margin", "aux_n_steps",
)


def features_for_verifier(record: dict[str, Any]) -> dict[str, float]:
    """Map a ``compute_faithfulness`` record to the verifier's feature dict.

    Per-step grounding uses the true IoU-vs-GT sequence where boxes exist
    (``iou_per_step``) and falls back to the runtime attention-coherence
    sequence (``signal_b_per_step``) elsewhere — so ``b_iou_*`` are real
    grounding features on SLAKE/VQA-RAD/synthetic and coherence proxies on the
    box-free datasets (reported as such, per review fix #3). ``b_iou_slope`` is
    the grounding-drift feature — the thesis's central effect made explicit for
    the learned head. ``c_region_consistency`` is not separately measured here;
    it is set to ``signal_b`` (region-stability proxy) and flagged as such.
    """
    per_step = record.get("iou_per_step") or record.get("signal_b_per_step") or []
    first = float(per_step[0]) if per_step else record["signal_b"]
    last = float(per_step[-1]) if per_step else record["signal_b"]
    mx = float(max(per_step)) if per_step else record["signal_b"]
    return {
        "sig_a_counterfactual": record["signal_a"],
        "sig_b_grounding": record["signal_b"],
        "sig_c_consistency": record["signal_c"],
        "a_flip_rate": record["signal_a_flip"],
        "a_js": record["signal_a_js"],
        "b_iou_max": mx,
        "b_iou_first": first,
        "b_iou_last": last,
        "b_iou_slope": _slope(per_step),
        "c_answer_consistency": record["signal_c"],
        "c_region_consistency": record["signal_b"],
        "aux_answer_margin": record["answer_margin"],
        "aux_n_steps": float(record["n_steps"]),
    }
