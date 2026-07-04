import numpy as np

from fmr.abstention import (
    binomial_ucb,
    calibrate_threshold,
    evaluate_selective,
    risk_coverage_curve,
)


def test_binomial_ucb_bounds():
    assert binomial_ucb(0, 0, 0.95) == 1.0
    assert binomial_ucb(5, 5, 0.95) == 1.0
    ucb = binomial_ucb(0, 100, 0.95)
    assert 0.0 < ucb < 0.05  # 1-(0.05)^(1/100) ~ 0.0295
    assert binomial_ucb(0, 1000, 0.95) < ucb  # tightens with n


def _world(n, rng):
    """Scores predictive of correctness with noise: P(correct) = sigmoid."""
    scores = rng.uniform(0, 1, size=n)
    p_correct = 1.0 / (1.0 + np.exp(-8 * (scores - 0.35)))
    correct = (rng.uniform(0, 1, size=n) < p_correct).astype(int)
    return scores, correct


def test_guarantee_holds_across_many_worlds():
    """Empirical check of the SGR bound: with prob >=1-delta over the calibration
    draw, retained error on the *population* is <= alpha. We approximate the
    population with a large fresh test set and count violations across trials."""
    alpha, delta = 0.10, 0.10
    trials, violations, feasible = 60, 0, 0
    rng = np.random.default_rng(0)
    for _ in range(trials):
        cal_s, cal_c = _world(800, rng)
        test_s, test_c = _world(20000, rng)
        res = calibrate_threshold(cal_s, cal_c, alpha=alpha, delta=delta)
        if not res.feasible:
            continue
        feasible += 1
        ev = evaluate_selective(test_s, test_c, res.threshold)
        if ev["n_retained"] > 0 and ev["retained_error"] > alpha:
            violations += 1
    assert feasible > trials * 0.8, "bound should usually be certifiable here"
    # Allow small slack over delta for test-set noise.
    assert violations / max(feasible, 1) <= delta + 0.05, (
        f"guarantee violated in {violations}/{feasible} feasible trials"
    )


def test_infeasible_returns_abstain_all():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 50)
    correct = rng.integers(0, 2, 50)  # coin-flip correctness, tiny cal set
    res = calibrate_threshold(scores, correct, alpha=0.01, delta=0.05)
    assert not res.feasible
    assert res.threshold == float("inf")
    ev = evaluate_selective(scores, correct, res.threshold)
    assert ev["n_retained"] == 0


def test_risk_coverage_curve_shape():
    rng = np.random.default_rng(2)
    scores, correct = _world(500, rng)
    rc = risk_coverage_curve(scores, correct)
    assert len(rc["coverage"]) == 500
    assert abs(rc["coverage"][-1] - 1.0) < 1e-9
    # Risk at full coverage equals overall error rate.
    assert abs(rc["risk"][-1] - (1 - correct.mean())) < 1e-9
    assert 0.0 <= rc["aurc"] <= 1.0


def test_coverage_at_risk_and_risk_at_coverage():
    from fmr.abstention import coverage_at_risk, risk_at_coverage
    rng = np.random.default_rng(7)
    scores, correct = _world(3000, rng)
    # A stricter risk target permits no more coverage than a looser one.
    c05 = coverage_at_risk(scores, correct, 0.05)
    c20 = coverage_at_risk(scores, correct, 0.20)
    assert 0.0 <= c05 <= c20 <= 1.0
    # Answering fewer (higher-scored) cases yields lower risk than answering all.
    assert risk_at_coverage(scores, correct, 0.3) <= risk_at_coverage(scores, correct, 1.0) + 1e-9
    # risk@coverage=1.0 equals the overall error rate.
    assert abs(risk_at_coverage(scores, correct, 1.0) - (1 - correct.mean())) < 1e-9


def test_better_scores_lower_aurc():
    rng = np.random.default_rng(3)
    scores, correct = _world(2000, rng)
    good = risk_coverage_curve(scores, correct)["aurc"]
    random_scores = rng.uniform(0, 1, 2000)
    bad = risk_coverage_curve(random_scores, correct)["aurc"]
    assert good < bad
