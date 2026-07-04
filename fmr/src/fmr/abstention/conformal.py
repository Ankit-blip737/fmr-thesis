"""Faithfulness-aware abstention with a distribution-free guarantee.

Goal: choose a threshold tau on the Faithfulness Score such that, among the
cases the system chooses to ANSWER (FS >= tau), the error rate is at most
``alpha`` — everything else is deferred to a clinician.

Method: Selection with Guaranteed Risk (Geifman & El-Yaniv, NeurIPS 2017) — a
split-calibration procedure with a finite-sample, distribution-free bound. On a
held-out calibration set we sweep candidate thresholds (the observed scores),
compute the exact binomial upper confidence bound on the retained error at each,
Bonferroni-corrected across candidates, and keep the threshold with maximal
coverage whose bound is <= alpha. With probability >= 1 - delta over the draw of
the calibration set, the true retained error at the chosen tau is <= alpha.

This is stronger than naive split-conformal quantiles for our use case because
the quantity we need to control (error *conditional on answering*) is a
selective risk, not marginal coverage. The calibration set must be disjoint
from anything used to tune correction or verifier hyperparameters — enforced by
``split_dataset`` upstream and logged in DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def binomial_ucb(errors: int, n: int, confidence: float) -> float:
    """Exact (Clopper-Pearson) upper confidence bound on an error probability."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    # Upper bound of the one-sided CP interval at level `confidence`.
    return float(stats.beta.ppf(confidence, errors + 1, n - errors))


@dataclass
class CalibrationResult:
    threshold: float          # answer iff score >= threshold
    alpha: float              # target retained-error rate
    delta: float              # failure probability of the guarantee
    cal_coverage: float       # fraction of calibration set retained
    cal_error: float          # empirical retained error on calibration
    cal_error_ucb: float      # the bound that certified this threshold
    n_cal: int
    feasible: bool            # False -> no threshold met the bound; abstain on all


def calibrate_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.05,
    max_candidates: int = 50,
) -> CalibrationResult:
    """Pick the max-coverage threshold whose certified retained error <= alpha.

    Candidate thresholds are restricted to a quantile grid of at most
    ``max_candidates`` values. The guarantee holds over this finite candidate
    set, and the smaller set makes the Bonferroni correction far less
    conservative — which is what keeps the bound feasible on small real
    calibration sets (e.g. VQA-RAD/SLAKE, a few hundred items). With many
    unique continuous scores, per-threshold Bonferroni over *all* of them would
    demand near-zero retained errors and force abstain-all.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = len(scores)
    assert n == len(correct) and n > 0

    uniq = np.unique(scores)
    if len(uniq) > max_candidates:
        qs = np.linspace(0.0, 1.0, max_candidates)
        candidates = np.unique(np.quantile(uniq, qs))
    else:
        candidates = uniq
    m = len(candidates)
    confidence = 1.0 - delta / m    # Bonferroni across candidate thresholds

    best: CalibrationResult | None = None
    for tau in candidates:
        keep = scores >= tau
        n_keep = int(keep.sum())
        errors = int((1 - correct[keep]).sum())
        ucb = binomial_ucb(errors, n_keep, confidence)
        if ucb <= alpha:
            cov = n_keep / n
            if best is None or cov > best.cal_coverage:
                best = CalibrationResult(
                    threshold=float(tau),
                    alpha=alpha,
                    delta=delta,
                    cal_coverage=cov,
                    cal_error=errors / n_keep if n_keep else 0.0,
                    cal_error_ucb=ucb,
                    n_cal=n,
                    feasible=True,
                )
    if best is None:
        # No threshold certifiable -> the safe output is "always defer".
        best = CalibrationResult(
            threshold=float("inf"), alpha=alpha, delta=delta,
            cal_coverage=0.0, cal_error=0.0, cal_error_ucb=1.0,
            n_cal=n, feasible=False,
        )
    return best


def evaluate_selective(scores: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """Coverage / retained accuracy / retained error at a fixed threshold."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    keep = scores >= threshold
    n_keep = int(keep.sum())
    retained_acc = float(correct[keep].mean()) if n_keep else float("nan")
    return {
        "coverage": n_keep / len(scores) if len(scores) else 0.0,
        "n_retained": n_keep,
        "retained_accuracy": retained_acc,
        "retained_error": 1.0 - retained_acc if n_keep else float("nan"),
    }


def _tie_aware_expected_error(scores: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Per-position EXPECTED error after answering highest-scored first, honest
    about ties.

    A deferral trigger cannot distinguish samples that share the same score, so
    the order *within* a tied group is arbitrary. Breaking ties by array index
    (what ``argsort`` does) fabricates a specific curve — e.g. a CONSTANT signal
    (all scores equal) would otherwise get a meaningful-looking AURC purely from
    input order. We instead replace each sample's correctness with the MEAN
    correctness of its equal-score group, so any prefix that cuts a tied group
    contributes that group's expected error. Consequence: a constant signal
    yields a flat risk = base error at every coverage (AURC = base error),
    correctly reading as "no discrimination". Returns the per-position expected
    error along the score-descending order.
    """
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    c_sorted = correct[order].astype(float)
    eff = c_sorted.copy()
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        eff[i:j] = c_sorted[i:j].mean()   # expected correctness within the tie group
        i = j
    return 1.0 - eff                       # expected error per position


def risk_at_coverage(scores: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Expected retained error when answering the top-``coverage`` fraction
    (tie-aware — see :func:`_tie_aware_expected_error`)."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = len(scores)
    if n == 0:
        return float("nan")
    k = max(1, int(round(coverage * n)))
    err = _tie_aware_expected_error(scores, correct)
    return float(err[:k].mean())


def coverage_at_risk(scores: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Max coverage (answer highest-scored first, tie-aware) whose expected
    retained error <= target. Standard selective-prediction operating metric."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = len(scores)
    if n == 0:
        return 0.0
    err = _tie_aware_expected_error(scores, correct)
    cum = np.cumsum(err)
    best_cov = 0.0
    for i in range(n):
        if cum[i] / (i + 1) <= target_risk:
            best_cov = (i + 1) / n
    return float(best_cov)


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> dict:
    """Risk-coverage trade-off swept over all thresholds (for plotting/AURC),
    tie-aware so constant/degenerate scores read as no-discrimination.

    Also reports ``n_distinct`` and ``degenerate`` (a single distinct score) so
    callers can label a non-discriminative trigger instead of showing a
    misleading tie-order-artifact AURC.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = len(scores)
    err = _tie_aware_expected_error(scores, correct)
    coverages, risks = [], []
    cum = 0.0
    for i in range(n):
        cum += err[i]
        coverages.append((i + 1) / n)
        risks.append(cum / (i + 1))
    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float("nan")
    return {"coverage": coverages, "risk": risks, "aurc": aurc,
            "n_distinct": int(len(np.unique(scores))),
            "degenerate": bool(len(np.unique(scores)) <= 1)}
