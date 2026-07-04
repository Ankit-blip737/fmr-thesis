"""Recompute the abstention block of each real fmr_results.json from its stored
records, using the (tie-aware) conformal metrics.

The real fmr_results were produced on Colab with the pre-fix risk_coverage_curve,
so their per-signal AURCs contained tie-order artifacts for constant signals
(Signal B / Signal C on the small real sets). The conformal GATE itself
(calibrate_threshold / evaluate_selective) is unaffected — it sweeps thresholds
directly — but we rebuild the whole block for consistency and to attach the
`degenerate`/`n_distinct` flags the dashboard uses. Reads only the stored records
(no model), so it is CPU-safe and deterministic.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from _common import CONFIG_DIR  # noqa: F401 (puts src on path)

from fmr.abstention import calibrate_threshold, evaluate_selective, risk_coverage_curve

ALPHA_DEFAULT, DELTA_DEFAULT = 0.10, 0.10  # real small-set defaults (see power finding)


def _selective(cal, test, key, alpha, delta):
    cal_s = np.array([r[key] for r in cal], float); cal_c = np.array([r["correct"] for r in cal], int)
    te_s = np.array([r[key] for r in test], float); te_c = np.array([r["correct"] for r in test], int)
    calres = calibrate_threshold(cal_s, cal_c, alpha=alpha, delta=delta)
    ev = evaluate_selective(te_s, te_c, calres.threshold)
    rc = risk_coverage_curve(te_s, te_c)
    return {
        "threshold": calres.threshold, "feasible": calres.feasible,
        "cal_coverage": calres.cal_coverage, "cal_error": calres.cal_error,
        "test": ev,
        "guarantee_holds": (not calres.feasible) or ev["n_retained"] == 0 or ev["retained_error"] <= alpha,
        "aurc": rc["aurc"], "risk_coverage": {"coverage": rc["coverage"], "risk": rc["risk"]},
        "n_distinct": rc["n_distinct"], "degenerate": rc["degenerate"],
    }


def main():
    updated = []
    for recf in sorted(glob.glob("fmr/outputs/real/*/fmr_records.json")):
        ds = os.path.basename(os.path.dirname(recf))
        resf = os.path.join(os.path.dirname(recf), "fmr_results.json")
        if not os.path.exists(resf):
            continue
        recs = json.load(open(recf, encoding="utf-8"))
        cal, test = recs.get("cal", []), recs.get("test", [])
        if not cal or not test:
            continue
        res = json.load(open(resf, encoding="utf-8"))
        alpha = res.get("abstention", {}).get("alpha", ALPHA_DEFAULT)
        delta = res.get("abstention", {}).get("delta", DELTA_DEFAULT)
        res["abstention"] = {
            "alpha": alpha, "delta": delta,
            "provisional_pre_correction": res.get("abstention", {}).get("provisional_pre_correction", True),
            "recomputed_tie_aware": True,
            "fs": _selective(cal, test, "fs", alpha, delta),
            "confidence": _selective(cal, test, "confidence", alpha, delta),
            "signal_a_only": _selective(cal, test, "signal_a", alpha, delta),
            "signal_b_only": _selective(cal, test, "signal_b", alpha, delta),
            "signal_c_only": _selective(cal, test, "signal_c", alpha, delta),
        }
        json.dump(res, open(resf, "w", encoding="utf-8"), indent=2)
        fs = res["abstention"]["fs"]
        print(f"[recompute] {ds}: fs AURC={fs['aurc']:.3f} feasible={fs['feasible']} "
              f"| signal_b degenerate={res['abstention']['signal_b_only']['degenerate']}")
        updated.append(ds)
    print(f"[recompute] updated: {updated}")


if __name__ == "__main__":
    main()
