"""Unit tests — Stage 4 correction module (Instance B)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmr.correction import (  # noqa: E402
    CorrectionConfig,
    clue_support,
    correct_sample,
    post_correction_fs,
    post_correction_sensitivity,
    trace_clue_region,
    vcd_answer,
    verify_and_revise,
)
from fmr.correction.vcd import vcd_contrast  # noqa: E402
from fmr.data import build_synthetic_dataset  # noqa: E402
from fmr.faithfulness.counterfactual import counterfactual_signal  # noqa: E402
from fmr.models import MockVLM  # noqa: E402
from fmr.models.second_vlm import PriorHeavyMockVLM, load_second_vlm  # noqa: E402
from fmr.utils import softmax  # noqa: E402


# ---------- vcd math ---------------------------------------------------------

def test_vcd_contrast_cancels_prior_and_amplifies_evidence():
    # vocab: [prior, gt, other]; prior fires in both, evidence only with image.
    pref_orig = np.array([1.8, 1.4, 0.0])
    pref_blank = np.array([1.8, 0.0, 0.0])
    p = vcd_contrast(softmax(pref_orig), softmax(pref_blank), alpha=1.0)
    assert int(np.argmax(p)) == 1  # evidence wins after the contrast


def test_vcd_contrast_noop_when_distributions_match():
    pref = np.array([2.0, 0.5, 0.1])
    p_o = softmax(pref)
    p = vcd_contrast(p_o, p_o, alpha=1.0)
    assert int(np.argmax(p)) == int(np.argmax(p_o))


def test_vcd_plausibility_mask_blocks_implausible_promotion():
    # 'gt' is implausible under the original (p ratio < beta) -> must stay blocked
    # even though the raw contrast would promote it.
    pref_orig = np.array([4.0, -2.0])
    pref_blank = np.array([4.0, -5.0])
    p = vcd_contrast(softmax(pref_orig), softmax(pref_blank), alpha=1.0, beta=0.1)
    assert int(np.argmax(p)) == 0
    assert p[1] == 0.0


# ---------- fixtures ---------------------------------------------------------

@pytest.fixture(scope="module")
def samples():
    return build_synthetic_dataset(n=80, seed=7)


@pytest.fixture(scope="module")
def mock():
    return MockVLM()


@pytest.fixture(scope="module")
def prior_vlm():
    return PriorHeavyMockVLM()


def _first(samples, grounded: bool):
    for s in samples:
        if bool(s.meta["grounded"]) == grounded:
            return s
    raise AssertionError("no such sample")


# ---------- clue tracing -----------------------------------------------------

def test_clue_trace_separates_grounded_from_ungrounded(mock, samples):
    grounded_confs = []
    blind_confs = []
    for s in samples:
        tr = trace_clue_region(mock, s, n_probes=3)
        (grounded_confs if s.meta["grounded"] else blind_confs).append(tr.confidence)
    assert np.mean(grounded_confs) > np.mean(blind_confs) + 0.2


def test_clue_trace_region_near_gt_when_grounded(mock, samples):
    s = _first(samples, grounded=True)
    tr = trace_clue_region(mock, s, n_probes=3)
    assert tr.region is not None and s.gt_region is not None
    assert tr.region.iou(s.gt_region) > 0.4


# ---------- verify & revise --------------------------------------------------

def test_verify_and_revise_drops_unsupported_steps(mock, samples):
    s = _first(samples, grounded=False)
    vcd_res = vcd_answer(mock, s)
    tr = trace_clue_region(mock, s, n_probes=3)
    revised, diag = verify_and_revise(s, vcd_res.outputs["original"], tr, vcd_res)
    assert diag["n_kept"] < diag["n_steps"]  # scattered regions -> steps dropped
    # original output must not be mutated
    assert all(st.supported is None for st in vcd_res.outputs["original"].steps)


def test_verify_and_revise_answer_policy_conservative(mock, samples):
    s = _first(samples, grounded=True)
    vcd_res = vcd_answer(mock, s)
    tr = trace_clue_region(mock, s, n_probes=3)
    revised, diag = verify_and_revise(s, vcd_res.outputs["original"], tr, vcd_res,
                                      vcd_margin=1e9)  # margin gate never met
    assert revised.answer == vcd_res.original_answer


def test_verify_and_revise_accepts_injected_supports(mock, samples):
    """Ablation hook: an injected support vector overrides clue-support, keeping
    exactly the steps at/above threshold."""
    s = samples[0]
    vcd_res = vcd_answer(mock, s)
    tr = trace_clue_region(mock, s, n_probes=3)
    orig = vcd_res.outputs["original"]
    supports = [0.9 if i % 2 == 0 else 0.0 for i in range(len(orig.steps))]
    revised, diag = verify_and_revise(s, orig, tr, vcd_res, support_threshold=0.5,
                                      vcd_margin=float("inf"), supports=supports)
    assert diag["n_kept"] == sum(x >= 0.5 for x in supports)
    assert diag["supports"] == supports


# ---------- rescore ----------------------------------------------------------

def test_rescore_shape_matches_signal_a(mock, samples):
    s = samples[0]
    out = mock.generate(s)
    post = post_correction_sensitivity(mock, s, out)
    ref = counterfactual_signal(mock, s)
    assert set(post) == {"corrected", "counterfactual", "flip_rate", "js_divergence"}
    # identical output => identical sensitivity as Signal A on the raw model
    assert post["counterfactual"] == pytest.approx(ref["counterfactual"])


def test_post_correction_fs_default_fusion(mock, samples):
    s = samples[0]
    out = mock.generate(s)
    res = post_correction_fs(mock, s, out)
    assert set(res) >= {"fs", "signal_a", "signal_b", "signal_c"}
    assert 0.0 <= res["fs"] <= 1.0


def test_post_correction_fs_accepts_injected_A_functions(mock, samples):
    """Simulates the merge: Instance A passes their attention_signal + fuse."""
    s = samples[0]
    out = mock.generate(s)
    calls = {}

    def fake_attention(corrected):
        calls["att"] = True
        return {"attention": 0.9, "per_step": [0.9]}

    def fake_fuse(a, b, c):
        calls["fuse"] = (a, b, c)
        return 0.123

    res = post_correction_fs(mock, s, out, attention_fn=fake_attention,
                             consistency_c=0.5, fuse_fn=fake_fuse)
    assert calls["att"] and res["signal_b"] == pytest.approx(0.9)
    assert res["signal_c"] == pytest.approx(0.5)
    assert res["fs"] == pytest.approx(0.123)


# ---------- pipeline ---------------------------------------------------------

def test_pipeline_selective_application(mock, samples):
    cfg = CorrectionConfig()
    for s in samples[:30]:
        r = correct_sample(mock, s, config=cfg)
        assert r.applied == (r.fs_before < cfg.trigger_threshold)
        if not r.applied:
            assert r.corrected is r.original and r.fs_after == r.fs_before


def test_pipeline_rescues_prior_dominated_answers(prior_vlm, samples):
    """The core Stage-4 claim on the fixable pathology: VCD flips a majority of
    prior-dominated wrong answers to the ground truth."""
    fixed, fixable = 0, 0
    for s in samples:
        if not s.meta["grounded"]:
            continue
        r = correct_sample(prior_vlm, s, config=CorrectionConfig())
        if not r.applied or r.original.answer == s.answer:
            continue
        fixable += 1
        fixed += int(r.corrected.answer == s.answer)
    assert fixable >= 5  # the profile must actually produce the pathology
    assert fixed / fixable > 0.6


def test_pipeline_leaves_image_blind_cases_to_abstention(mock, samples):
    """Image-blind cases: correction must not fabricate a confident answer —
    fs_after stays below the gate so they flow to abstention."""
    blind = [s for s in samples if not s.meta["grounded"]]
    rs = [correct_sample(mock, s, config=CorrectionConfig()) for s in blind]
    applied = [r for r in rs if r.applied]
    assert applied, "expected image-blind samples to be flagged"
    still_low = [r.fs_after < 0.5 for r in applied]
    assert np.mean(still_low) > 0.8


def test_second_vlm_factory():
    vlm = load_second_vlm()
    assert vlm.name == "mock-prior-heavy" and vlm.is_reasoning
    with pytest.raises(ValueError):
        load_second_vlm({"backend": "nope"})
