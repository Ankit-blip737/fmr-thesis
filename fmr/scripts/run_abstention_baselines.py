"""Head-to-head abstention-trigger comparison (proposal Section 8).

Compares the Faithfulness Score (ours) against the standard deferral triggers as
the thing that decides ANSWER vs ABSTAIN:
  * confidence-thresholding   — the model's own answer confidence (max logit)
  * self-consistency / RadFlag — agreement across sampled chains (Signal C /
    raw vote share). RadFlag (Zhang et al., ML4H 2025) is a black-box
    consistency flag; on our closed-vocab setup it is the same underlying
    signal as self-consistency thresholding, so we report both scalings.
  * each single FMM signal    — Signal A / B / C alone
  * fused FS (ours)           — the multi-signal Faithfulness Score

Metrics per trigger (empirical selective-prediction, no calibration needed):
  AURC (area under risk-coverage; lower=better), coverage@risk<=α for α in
  {0.05,0.10,0.20}, and risk@coverage for coverage in {0.5,0.8}.

Works on BOTH the mock (scored on the fly) and any existing real-model records
(fmr/outputs/real/<ds>/fmr_records.json) — no new GPU output needed.

Usage:
    python fmr/scripts/run_abstention_baselines.py            # mock + all real
    python fmr/scripts/run_abstention_baselines.py --source real:vqa_rad
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from _common import load_all_configs, resolve_dataset_and_splits, resolve_vlm

from fmr.abstention import coverage_at_risk, risk_at_coverage, risk_coverage_curve
from fmr.faithfulness import score_dataset
from fmr.utils import save_json

# trigger name -> (record key, higher-is-more-confident?)
TRIGGERS = {
    "fs_ours": ("fs", True),
    "confidence": ("confidence", True),
    "self_consistency": ("signal_c", True),      # Signal C (rescaled vote agreement)
    "radflag": ("signal_c_vote", True),          # raw vote share (RadFlag-style)
    "signal_a": ("signal_a", True),
    "signal_b": ("signal_b", True),
}
RISK_TARGETS = [0.05, 0.10, 0.20]
COV_TARGETS = [0.5, 0.8]
# Common coverage grid for the matched-coverage head-to-head table (proposal §9).
COV_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def _downsample_curve(rc: dict, k: int = 40) -> dict:
    """Thin the risk-coverage curve to <=k points for a compact JSON the
    dashboard can plot without smoothing over the real (possibly jagged) shape."""
    cov, risk = rc["coverage"], rc["risk"]
    n = len(cov)
    if n <= k:
        idx = list(range(n))
    else:
        idx = sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))
    return {"coverage": [round(cov[i], 4) for i in idx], "risk": [round(risk[i], 4) for i in idx]}


def _metrics(records: list[dict]) -> dict:
    correct = np.array([r["correct"] for r in records], dtype=int)
    out = {"n": len(records), "base_accuracy": float(correct.mean()), "triggers": {}}
    for name, (key, _) in TRIGGERS.items():
        if not records or key not in records[0] or records[0][key] is None:
            continue
        scores = np.array([r.get(key, 0.0) or 0.0 for r in records], dtype=float)
        rc = risk_coverage_curve(scores, correct)
        out["triggers"][name] = {
            "aurc": rc["aurc"],
            "coverage_at_risk": {f"{t:.2f}": coverage_at_risk(scores, correct, t) for t in RISK_TARGETS},
            "risk_at_coverage": {f"{c:.2f}": risk_at_coverage(scores, correct, c) for c in COV_GRID},
            "curve": _downsample_curve(rc),   # points for the dashboard overlay
        }
    # Matched-coverage error table: {coverage -> {trigger -> retained error}}.
    out["matched_coverage_error"] = {
        f"{c:.2f}": {t: out["triggers"][t]["risk_at_coverage"][f"{c:.2f}"] for t in out["triggers"]}
        for c in COV_GRID
    }
    # Rank triggers by AURC (lower better); note whether FS wins.
    ranked = sorted(out["triggers"].items(), key=lambda kv: kv[1]["aurc"])
    out["ranking_by_aurc"] = [k for k, _ in ranked]
    if "fs_ours" in out["triggers"]:
        out["fs_is_best_aurc"] = ranked[0][0] == "fs_ours"
        out["fs_vs_confidence_aurc_delta"] = (
            out["triggers"]["fs_ours"]["aurc"] - out["triggers"].get("confidence", {}).get("aurc", float("nan")))
    return out


def _mock_records(n_consistency: int = 5) -> list[dict]:
    cfg = load_all_configs()
    _, splits = resolve_dataset_and_splits(cfg["data"])
    vlm = resolve_vlm(cfg["models"], samples=splits["test"])
    return score_dataset(vlm, splits["test"], n_consistency_samples=n_consistency)


def _real_sources() -> dict[str, list[dict]]:
    srcs = {}
    for f in sorted(glob.glob("fmr/outputs/real/*/fmr_records.json")):
        ds = os.path.basename(os.path.dirname(f))
        recs = json.load(open(f, encoding="utf-8")).get("test", [])
        if recs:
            srcs[f"real:{ds}"] = recs
    return srcs


def run(source: str | None, out_dir: str) -> dict:
    results = {"risk_targets": RISK_TARGETS, "coverage_targets": COV_TARGETS, "sources": {}}
    sources: dict[str, list[dict]] = {}
    if source is None or source == "mock":
        sources["mock"] = _mock_records()
    if source is None or source.startswith("real"):
        real = _real_sources()
        sources.update({k: v for k, v in real.items() if source in (None, k)})

    for name, recs in sources.items():
        m = _metrics(recs)
        results["sources"][name] = m
        if m.get("triggers"):
            fs = m["triggers"].get("fs_ours", {})
            conf = m["triggers"].get("confidence", {})
            print(f"[abst-base] {name} (n={m['n']}, acc={m['base_accuracy']:.3f}): "
                  f"best-by-AURC={m['ranking_by_aurc'][0]} | "
                  f"FS AURC={fs.get('aurc', float('nan')):.3f} vs confidence AURC={conf.get('aurc', float('nan')):.3f} | "
                  f"FS cov@risk0.10={fs.get('coverage_at_risk', {}).get('0.10', float('nan')):.2f}")

    save_json(results, f"{out_dir}/abstention_baselines.json")
    print(f"[abst-base] wrote {out_dir}/abstention_baselines.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="mock | real:<dataset> | (default: all)")
    ap.add_argument("--out", default="fmr/outputs")
    args = ap.parse_args()
    run(args.source, args.out)
