"""Adapter: Instance A's `faithfulness.score` records -> verifier features.

Instance A committed `faithfulness/score.py` on master with a documented flat-dict
record schema (per-signal raw scores exposed *specifically* for this verifier).
This module maps one such record into the verifier's ``FEATURE_KEYS`` space, so on
merge the learned verifier trains on **real** Signals A/B/C instead of the
`training.signals` stub — with no change to labels, the verifier, eval, or tests.

Instance A's record keys consumed here (see their score.py docstring):
    signal_a, signal_a_flip, signal_a_js,
    signal_b, signal_b_per_step,
    signal_c, signal_c_vote,
    confidence, n_steps, iou_per_step, weak_labels, grounded_latent, output

Deliberately dict-only (no import of `faithfulness.score`) so this branch stays
importable/testable standalone; it reads keys the schema guarantees. `NaN`/missing
keys degrade to 0.0 so a partially-populated record never crashes the pipeline.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .verifier import FEATURE_KEYS


def _slope(xs: Sequence[float]) -> float:
    xs = [float(v) for v in xs]
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(xs)), xs, 1)[0])


def _answer_margin(record: dict) -> float:
    out = record.get("output")
    logits = getattr(out, "answer_logits", None)
    if logits is not None and len(logits) > 1:
        s = np.sort(np.asarray(logits, dtype=float))[::-1]
        return float(s[0] - s[1])
    # fall back to the model's own confidence if no full distribution is present
    return float(record.get("confidence", 0.0))


def features_from_record(record: dict) -> dict:
    """Map one Instance-A faithfulness record into the verifier feature dict.

    Grounding aggregates (b_iou_*) prefer the GT-based ``iou_per_step`` when the
    dataset has boxes, else fall back to ``signal_b_per_step`` (the attention
    coherence per step) so radiology-with-boxes and box-free modalities both work.
    """
    def g(k, default=0.0):
        v = record.get(k, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    per_step = record.get("iou_per_step") or record.get("signal_b_per_step") or []
    per_step = [float(x) for x in per_step] if per_step else [g("signal_b")]

    return {
        "sig_a_counterfactual": g("signal_a"),
        "sig_b_grounding": g("signal_b"),
        "sig_c_consistency": g("signal_c"),
        "a_flip_rate": g("signal_a_flip"),
        "a_js": g("signal_a_js"),
        "b_iou_max": float(np.max(per_step)),
        "b_iou_first": float(per_step[0]),
        "b_iou_last": float(per_step[-1]),
        "b_iou_slope": _slope(per_step),
        # A exposes one consistency scalar (+ vote share); split mirrors the stub.
        "c_answer_consistency": g("signal_c"),
        "c_region_consistency": g("signal_c_vote", g("signal_c")),
        "aux_answer_margin": _answer_margin(record),
        "aux_n_steps": g("n_steps", float(len(per_step))),
    }


def weak_label_from_record(record: dict, flip_threshold: float = 0.5) -> int:
    """Weak grounding label from an Instance-A record.

    Prefers A's IoU-derived ``weak_labels`` (majority-vote across steps) when
    present — the cleaner GT-based weak label — else the counterfactual flip
    component, matching `training.labels.weak_label_counterfactual`.
    """
    wl = record.get("weak_labels")
    if wl:
        return int(np.mean([float(x) for x in wl]) >= 0.5)
    return int(record.get("signal_a_flip", 0.0) >= flip_threshold)


def true_latent_from_record(record: dict) -> Optional[int]:
    v = record.get("grounded_latent")
    return None if v is None else int(v)


def frame_from_records(records: Sequence[dict]) -> dict[str, Any]:
    """Build the verifier's training arrays from Instance-A records.

    Returns a dict with the same fields `dataset.FeatureFrame` carries (X, feats,
    y_true, y_weak, sample_ids) so `train_verifier.py` can swap its feature source
    to real signals with a one-line change. ``y_true`` may contain -1 where the
    record has no latent (real non-synthetic data); callers filter those for eval.
    """
    from .verifier import vectorize

    feats = [features_from_record(r) for r in records]
    X = np.vstack([vectorize(f) for f in feats]) if feats else np.empty((0, len(FEATURE_KEYS)))
    y_true = np.array([(-1 if true_latent_from_record(r) is None else true_latent_from_record(r))
                       for r in records])
    y_weak = np.array([weak_label_from_record(r) for r in records])
    ids = [r.get("sample_id", f"rec-{i}") for i, r in enumerate(records)]
    return {"X": X, "feats": feats, "y_true": y_true, "y_weak": y_weak, "sample_ids": ids}
