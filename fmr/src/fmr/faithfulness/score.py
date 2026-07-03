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

from ..models.base_vlm import BaseVLM
from ..types import Sample
from ..utils import clip01
from .attention import attention_signal, iou_labels
from .consistency import consistency_signal
from .counterfactual import counterfactual_signal

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

    return {
        "sample_id": sample.sample_id,
        "modality": sample.modality,
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
