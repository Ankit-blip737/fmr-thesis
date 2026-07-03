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
) -> CalibrationResult:
    """Pick the max-coverage threshold whose certified retained error <= alpha."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = len(scores)
    assert n == len(correct) and n > 0

    candidates = np.unique(scores)  # thresholds worth trying
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


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> dict:
    """Risk-coverage trade-off swept over all thresholds (for plotting/AURC)."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    order = np.argsort(-scores)  # answer highest-scored cases first
    sorted_correct = correct[order]
    n = len(scores)
    coverages, risks = [], []
    errors = 0
    for i in range(n):
        errors += 1 - sorted_correct[i]
        coverages.append((i + 1) / n)
        risks.append(errors / (i + 1))
    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float("nan")
    return {"coverage": coverages, "risk": risks, "aurc": aurc}
