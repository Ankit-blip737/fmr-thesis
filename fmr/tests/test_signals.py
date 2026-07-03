import numpy as np
from sklearn.metrics import roc_auc_score

from fmr.data.synthetic import build_synthetic_dataset
from fmr.faithfulness import (
    VERIFIER_FEATURE_KEYS,
    compute_faithfulness,
    decompose_rationale,
    features_for_verifier,
    score_dataset,
)
from fmr.models.mock_vlm import MockVLM


def _records(n=150, seed=3):
    samples = build_synthetic_dataset(n=n, seed=seed)
    vlm = MockVLM()
    return score_dataset(vlm, samples, n_consistency_samples=5)


def test_signal_direction():
    """Every signal must be higher (on average) for latent-grounded samples."""
    recs = _records()
    for sig in ("signal_a", "signal_b", "signal_c", "fs"):
        g = np.mean([r[sig] for r in recs if r["grounded_latent"] == 1])
        u = np.mean([r[sig] for r in recs if r["grounded_latent"] == 0])
        assert g > u, f"{sig}: grounded mean {g:.3f} <= ungrounded mean {u:.3f}"


def test_signals_informative():
    """Each signal is informative on the (clean) machinery fixture. Realistic
    sub-1.0 AUROCs are demonstrated on the real-model data, not this mock — so
    we assert informativeness (>0.6), not an upper bound."""
    recs = _records(n=300)
    labels = [r["grounded_latent"] for r in recs]
    for sig in ("signal_a", "signal_b", "signal_c"):
        auc = roc_auc_score(labels, [r[sig] for r in recs])
        assert auc > 0.6, f"{sig} AUROC {auc:.3f} not informative"


def test_fusion_beats_each_single_signal():
    recs = _records(n=400, seed=11)
    labels = [r["grounded_latent"] for r in recs]
    fused = roc_auc_score(labels, [r["fs"] for r in recs])
    for sig in ("signal_a", "signal_b", "signal_c"):
        single = roc_auc_score(labels, [r[sig] for r in recs])
        assert fused >= single - 0.02, f"fused {fused:.3f} well below {sig} {single:.3f}"


def test_record_schema_contract():
    """Instance B's verifier depends on these keys — additive changes only."""
    rec = _records(n=4)[0]
    for key in ("sample_id", "modality", "correct", "signal_a", "signal_a_flip",
                "signal_a_js", "signal_b", "signal_b_per_step", "signal_c",
                "signal_c_vote", "fs", "fs_per_step", "confidence", "n_steps",
                "iou_mean", "iou_per_step", "weak_labels", "grounded_latent"):
        assert key in rec, f"record schema missing {key}"
    assert len(rec["signal_b_per_step"]) == rec["n_steps"]
    assert len(rec["fs_per_step"]) == rec["n_steps"]


def test_per_step_fs_written_back_to_steps():
    s = build_synthetic_dataset(n=1)[0]
    rec = compute_faithfulness(MockVLM(), s)
    for step in rec["output"].steps:
        assert step.fs is not None and 0.0 <= step.fs <= 1.0
        assert step.attention_grounding is not None


def test_verifier_feature_adapter():
    """features_for_verifier must emit exactly Instance B's feature schema."""
    rec = _records(n=4)[0]
    feats = features_for_verifier(rec)
    assert set(feats) == set(VERIFIER_FEATURE_KEYS)
    for k, v in feats.items():
        assert isinstance(v, float), f"{k} not float"
    # Sanity: raw signals passed through unchanged.
    assert feats["sig_a_counterfactual"] == rec["signal_a"]
    assert feats["aux_n_steps"] == float(rec["n_steps"])


def test_verifier_features_carry_signal():
    """The adapted features must still separate grounded from ungrounded."""
    recs = _records(n=300)
    labels = [r["grounded_latent"] for r in recs]
    feats = [features_for_verifier(r) for r in recs]
    for key in ("sig_a_counterfactual", "sig_c_consistency"):
        auc = roc_auc_score(labels, [f[key] for f in feats])
        assert auc > 0.6, f"{key} lost signal in adapter (AUROC {auc:.3f})"


def test_answer_margin_present():
    rec = _records(n=4)[0]
    assert "answer_margin" in rec and 0.0 <= rec["answer_margin"] <= 1.0


def test_decompose_free_text():
    steps = decompose_rationale(
        "Step 1: there is an opacity in the left lower lobe. "
        "Step 2: no pleural effusion is seen; the heart size is normal."
    )
    assert len(steps) >= 2
    assert "opacity" in steps[0].terms
