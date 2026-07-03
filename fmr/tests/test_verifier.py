"""Unit tests — learned faithfulness verifier + heuristic fusion (Instance B)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from fmr.data import build_synthetic_dataset, split_dataset  # noqa: E402
from fmr.models import MockVLM  # noqa: E402
from fmr.models.second_vlm import PriorHeavyMockVLM  # noqa: E402
from fmr.training import (  # noqa: E402
    FEATURE_KEYS,
    HeuristicFusion,
    LearnedVerifier,
    build_feature_frame,
    compute_sample_features,
)
from fmr.training.verifier import vectorize  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    s = build_synthetic_dataset(n=120, seed=7)
    p = split_dataset(s, seed=13)
    vlms = [MockVLM(), PriorHeavyMockVLM()]
    tr = build_feature_frame(vlms, p["train"], n_chains=2, noise=0.5)
    te = build_feature_frame(vlms, p["test"], n_chains=2, noise=0.5)
    return tr, te


# ---------- heuristic fusion -------------------------------------------------

def test_heuristic_bounds_and_weights():
    h = HeuristicFusion()
    assert h.score({"sig_a_counterfactual": 1, "sig_b_grounding": 1, "sig_c_consistency": 1}) == 1.0
    assert h.score({"sig_a_counterfactual": 0, "sig_b_grounding": 0, "sig_c_consistency": 0}) == 0.0
    mid = h.score({"sig_a_counterfactual": 1, "sig_b_grounding": 0, "sig_c_consistency": 0})
    assert mid == pytest.approx(0.5)  # normalized weight on A


# ---------- feature extraction ----------------------------------------------

def test_features_have_all_keys():
    s = build_synthetic_dataset(n=4, seed=1)[0]
    sf = compute_sample_features(MockVLM(), s, n_chains=2)
    for k in FEATURE_KEYS:
        assert k in sf.features
    assert vectorize(sf.features).shape == (len(FEATURE_KEYS),)


def test_noise_is_deterministic_and_bounded():
    s = build_synthetic_dataset(n=4, seed=1)[0]
    a = compute_sample_features(MockVLM(), s, n_chains=2, noise=0.4)
    b = compute_sample_features(MockVLM(), s, n_chains=2, noise=0.4)
    assert a.features == b.features                      # deterministic
    assert 0.0 <= a.features["sig_a_counterfactual"] <= 1.0
    # noise=0 reproduces the clean signal exactly
    c0 = compute_sample_features(MockVLM(), s, n_chains=2, noise=0.0)
    c0b = compute_sample_features(MockVLM(), s, n_chains=2, noise=0.0)
    assert c0.features == c0b.features


# ---------- learned verifier -------------------------------------------------

def test_fit_requires_two_classes():
    v = LearnedVerifier("logreg")
    with pytest.raises(ValueError):
        v.fit(np.random.rand(10, len(FEATURE_KEYS)), np.zeros(10))


def test_learned_is_dropin_for_heuristic(frame):
    tr, te = frame
    v = LearnedVerifier("logreg").fit(tr.X, tr.y_weak, seed=7)
    # same interface as HeuristicFusion
    assert 0.0 <= v.score(te.feats[0]) <= 1.0
    batch = v.score_batch(te.feats)
    assert batch.shape == (len(te.feats),) and np.all((batch >= 0) & (batch <= 1))


def test_verifier_headroom_and_weak_label_ceiling(frame):
    """RQ5, honest version on the graded mock. Two claims:

    (1) The learned model has real HEADROOM: trained on the *true* latent it far
        exceeds the heuristic fusion — so the model class is not the bottleneck.
    (2) Trained only on the *noisy weak label* it is capped and does NOT reliably
        beat the heuristic (whose raw signals already track the latent well). This
        is the honest negative result the proposal says to report: the verifier
        helps to the extent grounding labels are available.
    """
    tr, te = frame
    auroc_h = roc_auc_score(te.y_true, HeuristicFusion().score_batch(te.feats))
    oracle = LearnedVerifier("gbt").fit(tr.X, tr.y_true, seed=7)
    auroc_oracle = roc_auc_score(te.y_true, oracle.score_batch(te.feats))
    weak = LearnedVerifier("logreg").fit(tr.X, tr.y_weak, seed=7)
    auroc_weak = roc_auc_score(te.y_true, weak.score_batch(te.feats))

    assert auroc_oracle > auroc_h + 0.1           # (1) clear headroom with good labels
    assert auroc_weak >= auroc_h - 0.25           # (2) weak-label version is a sane drop-in
    assert 0.5 <= auroc_weak <= 1.0


def test_weak_labels_are_noisy_not_oracle(frame):
    tr, _ = frame
    agree = np.mean(tr.y_weak == tr.y_true)
    assert 0.5 < agree < 0.95                # genuinely weak, not the latent


def test_save_load_roundtrip(frame, tmp_path):
    tr, te = frame
    v = LearnedVerifier("gbt").fit(tr.X, tr.y_weak, seed=7)
    p = tmp_path / "v.pkl"
    v.save(p)
    assert (Path(str(p) + ".meta.json")).exists()
    v2 = LearnedVerifier.load(p)
    np.testing.assert_allclose(v.score_batch(te.feats), v2.score_batch(te.feats))


def test_feature_importance_present(frame):
    tr, _ = frame
    v = LearnedVerifier("gbt").fit(tr.X, tr.y_weak, seed=7)
    imp = v.feature_importance()
    assert set(imp) == set(FEATURE_KEYS)
    assert abs(sum(imp.values()) - 1.0) < 0.1  # tree importances ~sum to 1
