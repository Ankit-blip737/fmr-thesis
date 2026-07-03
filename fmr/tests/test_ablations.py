"""Fast unit tests for the ablation helpers (no model calls)."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _fake_records(n=400, seed=0):
    """Synthetic records where signals are noisy observations of a latent."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        g = int(rng.random() < 0.5)
        base = 0.65 if g else 0.35
        a = float(np.clip(base + rng.normal(0, 0.15), 0, 1))
        b = float(np.clip(base + rng.normal(0, 0.2), 0, 1))
        c = float(np.clip(base + rng.normal(0, 0.2), 0, 1))
        fs = 0.4 * a + 0.3 * b + 0.3 * c
        recs.append({"signal_a": a, "signal_b": b, "signal_c": c, "fs": fs,
                     "grounded_latent": g, "correct": int(rng.random() < base + 0.2)})
    return recs


def test_weight_sensitivity_bounds():
    import run_ablations
    recs = _fake_records()
    labels = [r["grounded_latent"] for r in recs]
    ws = run_ablations.weight_sensitivity(recs, labels, step=0.2)
    assert ws["auroc_min"] <= ws["auroc_mean"] <= ws["auroc_max"]
    assert 0.5 < ws["auroc_max"] <= 1.0
    assert set(ws["best"]["weights"]) == {"a", "b", "c"}


def test_abstention_power_monotone():
    import run_ablations
    recs = _fake_records(n=1600)
    p = run_ablations.abstention_power(recs, alphas=(0.05, 0.20),
                                       cal_sizes=(100, 400, 800), delta=0.1)
    # A looser alpha must be certifiable with no more data than a tighter one.
    lo = p["min_cal_size_for_alpha"]["0.05"]
    hi = p["min_cal_size_for_alpha"]["0.2"]
    if lo is not None and hi is not None:
        assert hi <= lo
    # Alpha-curve entries carry the expected fields.
    for a, row in p["alpha_curve"].items():
        assert set(row) >= {"feasible", "coverage", "retained_error"}
