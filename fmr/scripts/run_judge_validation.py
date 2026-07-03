"""Validate the LLM-as-judge before any pipeline metric trusts it (fix #3).

Runs the heuristic judge against the hand-authored gold set and reports
agreement (accuracy, Cohen's κ, per-class precision/recall, confusion). If a
``complete`` callable is wired (open-LLM on Colab / API judge), the same harness
scores the LLM judge too — that path is exercised by
`notebooks/colab_judge_llm.ipynb`. Here (CPU, no key) we validate the heuristic
fallback, which is what runs when no LLM is available.

If κ is weak, the script does NOT silently pass: it prints a revised-rubric
suggestion and exits non-zero so the failure is visible.

Usage: python fmr/scripts/run_judge_validation.py [--kappa-floor 0.6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmr.eval.judge import HeuristicJudge, evaluate_judge_agreement  # noqa: E402
from fmr.eval.gold_data import JUDGE_GOLD, label_distribution  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa-floor", type=float, default=0.6,
                    help="minimum acceptable Cohen's kappa vs human labels")
    ap.add_argument("--partial-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "judge_validation_heuristic.json"))
    args = ap.parse_args()

    judge = HeuristicJudge(partial_threshold=args.partial_threshold)
    report = evaluate_judge_agreement(judge, JUDGE_GOLD)
    report["gold_label_distribution"] = label_distribution()
    report["judge"] = "heuristic"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    kappa = report["cohens_kappa"]
    print(f"\nGold set: N={report['n']} dist={report['gold_label_distribution']}")
    print(f"3-way accuracy   : {report['accuracy']:.3f}")
    print(f"binary accuracy  : {report['binary_accuracy(correct_vs_not)']:.3f}")
    print(f"Cohen's kappa    : {kappa:.3f}  (floor {args.kappa_floor})")

    # Show the disagreements explicitly so they can be audited.
    print("\n--- disagreements (judge != gold) ---")
    for g in JUDGE_GOLD:
        v = judge(g["question"], g["prediction"], g["reference"])
        if v.label != g["label"]:
            print(f"  q={g['question']!r} pred={g['prediction']!r} ref={g['reference']!r} "
                  f"gold={g['label']} judge={v.label} ({v.rationale})")

    if kappa < args.kappa_floor:
        print(
            f"\n[FAIL] kappa {kappa:.3f} < floor {args.kappa_floor}. "
            "Revise: extend _SYNONYMS clusters for the missed clinical terms, or "
            "adjust partial_threshold, then re-run before trusting judge scores."
        )
        return 1
    print(f"\n[PASS] judge agreement acceptable (kappa {kappa:.3f} >= {args.kappa_floor}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
