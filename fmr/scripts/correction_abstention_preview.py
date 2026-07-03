"""Illustrative selective-prediction preview: does correction shrink the abstain set?

This quantifies external-review framing #1 — correction exists to make the
abstention story stronger. It is NOT the conformal gate (Instance A owns
`abstention/`); it is a simple threshold sweep over faithfulness, used only to
compare two deferral *signals* on the SAME data:

  * NAIVE gate  — threshold the PRE-correction faithfulness score; answer with
    the raw model's answer.
  * FMR gate    — threshold the POST-correction faithfulness score; answer with
    the corrected answer.

If correction works, the FMR curve dominates: at matched coverage the retained
set is more accurate, and fewer cases must be deferred to hit a target accuracy.
Reported: risk (error on retained) vs coverage, the risk-coverage AUC (lower is
better), and retained accuracy at fixed coverages. Uses the prior-dominated
backend (where correction actually changes answers) as the headline.

Usage: python fmr/scripts/correction_abstention_preview.py [--n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmr.correction import CorrectionConfig, correct_sample  # noqa: E402
from fmr.data import build_synthetic_dataset  # noqa: E402
from fmr.models import MockVLM  # noqa: E402
from fmr.models.second_vlm import PriorHeavyMockVLM  # noqa: E402


def risk_coverage(scores: np.ndarray, correct: np.ndarray, grid: int = 101) -> list[dict]:
    """Sweep the retain-threshold; return (coverage, risk, retained_acc) points.

    A case is *retained* (answered) when score >= tau, else deferred. Risk is the
    error rate on the retained set; coverage is the retained fraction.
    """
    pts = []
    for tau in np.linspace(0.0, 1.0, grid):
        retained = scores >= tau
        cov = float(np.mean(retained))
        if cov == 0:
            pts.append({"tau": float(tau), "coverage": 0.0, "risk": 0.0, "retained_acc": 1.0})
            continue
        acc = float(np.mean(correct[retained]))
        pts.append({"tau": float(tau), "coverage": cov, "risk": 1.0 - acc, "retained_acc": acc})
    return pts


def rc_auc(pts: list[dict]) -> float:
    """Area under the risk-coverage curve (risk integrated over coverage)."""
    xs = np.array([p["coverage"] for p in pts])
    ys = np.array([p["risk"] for p in pts])
    order = np.argsort(xs)
    return float(np.trapz(ys[order], xs[order]))


def acc_at_coverage(pts: list[dict], target_cov: float) -> float:
    best = min(pts, key=lambda p: abs(p["coverage"] - target_cov))
    return best["retained_acc"]


def coverage_at_acc(pts: list[dict], target_acc: float) -> float:
    """Max coverage achievable while keeping retained accuracy >= target."""
    ok = [p for p in pts if p["retained_acc"] >= target_acc and p["coverage"] > 0]
    return max((p["coverage"] for p in ok), default=0.0)


def analyze(vlm, samples) -> dict:
    gt = {s.sample_id: s.answer for s in samples}
    fs_before, fs_after, ok_before, ok_after = [], [], [], []
    for s in samples:
        r = correct_sample(vlm, s, config=CorrectionConfig())
        fs_before.append(r.fs_before)
        fs_after.append(r.fs_after)
        ok_before.append(r.original.answer == gt[s.sample_id])
        ok_after.append(r.corrected.answer == gt[s.sample_id])
    fs_before = np.array(fs_before); fs_after = np.array(fs_after)
    ok_before = np.array(ok_before); ok_after = np.array(ok_after)

    naive = risk_coverage(fs_before, ok_before)
    fmr = risk_coverage(fs_after, ok_after)

    return {
        "model": vlm.name,
        "n": len(samples),
        "overall_acc_before": float(np.mean(ok_before)),
        "overall_acc_after": float(np.mean(ok_after)),
        "rc_auc_naive(lower=better)": rc_auc(naive),
        "rc_auc_fmr(lower=better)": rc_auc(fmr),
        "retained_acc@coverage": {
            f"{c:.1f}": {"naive": acc_at_coverage(naive, c), "fmr": acc_at_coverage(fmr, c)}
            for c in (0.3, 0.5, 0.7, 0.9)
        },
        "coverage@retained_acc>=0.9": {
            "naive": coverage_at_acc(naive, 0.9), "fmr": coverage_at_acc(fmr, 0.9),
        },
        "coverage@retained_acc>=0.95": {
            "naive": coverage_at_acc(naive, 0.95), "fmr": coverage_at_acc(fmr, 0.95),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    samples = build_synthetic_dataset(n=args.n, seed=args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for tag, vlm in (("prior", PriorHeavyMockVLM()), ("mock", MockVLM())):
        rep = analyze(vlm, samples)
        reports[tag] = rep
        print(f"\n===== {vlm.name} =====")
        print(json.dumps(rep, indent=2))

    (out_dir / "correction_abstention_preview.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
