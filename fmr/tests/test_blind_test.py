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


def test_replication_verdict_negative_not_forced():
    """If grounding does NOT decay, the verdict must say so (no forced framing)."""
    import run_blind_test as bt
    results = {"models": {
        "r": {"name": "reasoner", "is_reasoning": True,
              "iou_vs_step_index": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
              "grounding_drift_slope": bt._drift_slope({"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4})},
    }}
    v = bt._replication_verdict(results)
    assert v["tested"] and not v["replicated"]
    assert "NOT replicated" in v["note"]
