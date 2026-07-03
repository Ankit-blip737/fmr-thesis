"""Tests for the headline replication-verdict logic (review fix #2)."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_drift_slope_sign():
    import run_blind_test as bt
    decaying = {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1}
    rising = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    assert bt._drift_slope(decaying) < 0
    assert bt._drift_slope(rising) > 0
    assert bt._drift_slope({"0": 0.3}) == 0.0


def test_replication_verdict_positive():
    import run_blind_test as bt
    results = {"models": {
        "r": {"name": "reasoner", "is_reasoning": True,
              "iou_vs_step_index": {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1},
              "grounding_drift_slope": bt._drift_slope({"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1})},
        "p": {"name": "plain", "is_reasoning": False,
              "iou_vs_step_index": {"0": 0.35}, "grounding_drift_slope": 0.0},
    }}
    v = bt._replication_verdict(results)
    assert v["tested"] and v["replicated"] and v["within_model_decay"]
    assert v["primary_evidence"] == "drift"


def test_replication_verdict_negative_not_forced():
    """No drift signal and no comparator => inconclusive, framing not forced."""
    import run_blind_test as bt
    results = {"models": {
        "r": {"name": "reasoner", "is_reasoning": True,
              "iou_vs_step_index": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
              "grounding_drift_slope": bt._drift_slope({"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4})},
    }}
    v = bt._replication_verdict(results)
    assert v["tested"] and not v["replicated"]
    assert v["primary_evidence"] in ("none", "blind_gap")


def test_replication_verdict_blind_gap_lens():
    """When per-step drift is unavailable, a smaller reasoning blind_gap supports
    the hypothesis via the blind-gap lens, with the accuracy confound flagged."""
    import run_blind_test as bt
    results = {"models": {
        "r": {"name": "reasoner", "is_reasoning": True, "iou_vs_step_index": {},
              "grounding_drift_slope": 0.0, "blind_gap": 0.07,
              "accuracy": {"original": 0.40}},
        "p": {"name": "plain", "is_reasoning": False, "iou_vs_step_index": {},
              "grounding_drift_slope": 0.0, "blind_gap": 0.13,
              "accuracy": {"original": 0.57}},
    }}
    v = bt._replication_verdict(results)
    assert v["replicated"] and v["primary_evidence"] == "blind_gap"
    assert v["blind_gap_supports"] is True
    assert v["accuracy_confound"] is True  # reasoning model notably weaker
