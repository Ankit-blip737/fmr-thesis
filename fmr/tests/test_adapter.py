"""Tests — adapter from Instance A's faithfulness.score records to verifier features.

Uses a hand-built record matching Instance A's committed score.py schema (master),
so this validates the merge contract without importing their module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmr.training.adapter import (  # noqa: E402
    features_from_record,
    frame_from_records,
    true_latent_from_record,
    weak_label_from_record,
)
from fmr.training.verifier import FEATURE_KEYS, LearnedVerifier, vectorize  # noqa: E402
from fmr.types import VLMOutput  # noqa: E402


def _record(**over):
    """A record shaped exactly like faithfulness.score.compute_faithfulness output."""
    rec = {
        "sample_id": "syn-00001", "modality": "xray", "answer": "yes", "correct": 1,
        "signal_a": 0.8, "signal_a_flip": 1.0, "signal_a_js": 0.6,
        "signal_b": 0.7, "signal_b_per_step": [0.9, 0.6, 0.3],
        "signal_c": 0.5, "signal_c_vote": 0.75,
        "fs": 0.68, "fs_per_step": [0.8, 0.6, 0.4],
        "confidence": 0.72, "n_steps": 3,
        "iou_mean": 0.55, "iou_per_step": [0.8, 0.5, 0.35],
        "weak_labels": [1, 1, 0], "grounded_latent": 1,
        "output": VLMOutput(sample_id="syn-00001", answer="yes",
                            answer_logits=np.array([0.72, 0.28])),
    }
    rec.update(over)
    return rec


def test_features_have_all_keys_and_types():
    f = features_from_record(_record())
    assert set(f) == set(FEATURE_KEYS)
    assert all(isinstance(v, float) for v in f.values())


def test_grounding_aggregates_use_iou_when_present():
    f = features_from_record(_record())
    assert f["b_iou_max"] == pytest.approx(0.8)     # max of iou_per_step
    assert f["b_iou_first"] == pytest.approx(0.8)
    assert f["b_iou_last"] == pytest.approx(0.35)
    assert f["b_iou_slope"] < 0                       # grounding decays across steps


def test_falls_back_to_signal_b_per_step_without_boxes():
    f = features_from_record(_record(iou_per_step=None))
    assert f["b_iou_max"] == pytest.approx(0.9)      # max of signal_b_per_step
    assert f["b_iou_last"] == pytest.approx(0.3)


def test_answer_margin_from_logits():
    f = features_from_record(_record())
    assert f["aux_answer_margin"] == pytest.approx(0.72 - 0.28)


def test_weak_label_prefers_iou_labels():
    assert weak_label_from_record(_record(weak_labels=[1, 1, 0])) == 1
    assert weak_label_from_record(_record(weak_labels=[0, 0, 0])) == 0
    # no weak_labels -> counterfactual flip fallback
    assert weak_label_from_record(_record(weak_labels=None, signal_a_flip=0.9)) == 1
    assert weak_label_from_record(_record(weak_labels=None, signal_a_flip=0.1)) == 0


def test_missing_keys_degrade_to_zero_not_crash():
    f = features_from_record({"sample_id": "x"})
    assert set(f) == set(FEATURE_KEYS)
    assert f["sig_a_counterfactual"] == 0.0


def test_true_latent_none_for_real_data():
    assert true_latent_from_record(_record(grounded_latent=None)) is None
    assert true_latent_from_record(_record(grounded_latent=0)) == 0


def test_frame_trains_a_verifier_end_to_end():
    # 40 synthetic-schema records with a learnable signal->latent relationship
    rng = np.random.default_rng(0)
    recs = []
    for i in range(40):
        g = int(i % 2 == 0)
        a = 0.8 if g else 0.2
        recs.append(_record(
            sample_id=f"r{i}", signal_a=a + rng.normal(0, 0.05),
            signal_a_flip=float(g), signal_b=0.7 if g else 0.2,
            signal_b_per_step=[0.7 if g else 0.2], iou_per_step=[0.7 if g else 0.2],
            weak_labels=[g], grounded_latent=g,
        ))
    fr = frame_from_records(recs)
    assert fr["X"].shape == (40, len(FEATURE_KEYS))
    v = LearnedVerifier("logreg").fit(fr["X"], fr["y_weak"], seed=0)
    scores = v.score_batch(fr["feats"])
    # verifier trained on real-schema records separates the latent
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(fr["y_true"], scores) > 0.9
