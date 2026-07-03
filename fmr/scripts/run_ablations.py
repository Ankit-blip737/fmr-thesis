"""Stage 3/5 robustness & sensitivity ablations (the headline components).

Per the external review, the FS (Stage 3) and abstention (Stage 5) get the
deepest ablation treatment. This script quantifies how robust the results are to
the choices a skeptic will question:

  1. **FS fusion-weight sensitivity.** Sweep the (a, b, c) weights over the
     simplex; report the AUROC spread. A robust FS should not hinge on a lucky
     weighting — and the best learned fusion (Instance B) should beat the whole
     hand-weighted envelope.
  2. **Signal-B grid sensitivity.** Vary the grid resolution used for regions;
     report Signal-B AUROC vs grid. Guards against "the grounding signal is a
     grid-size artifact."
  3. **Seed stability.** Re-draw the synthetic world across seeds; report the
     mean/std of every signal's AUROC and the fused FS.
  4. **Abstention alpha sweep + calibration-size power curve.** Coverage vs the
     target error, and the smallest calibration set that certifies each alpha —
     making the "≳1k cal points for alpha=0.05" claim quantitative.

All on the mock (machinery). The Colab run reuses the same functions on real
scores by pointing --config-dir at a real config.

Usage:
    python fmr/scripts/run_ablations.py [--out fmr/outputs/ablations]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from _common import load_all_configs
from sklearn.metrics import roc_auc_score

from fmr.abstention import calibrate_threshold, evaluate_selective
from fmr.data.synthetic import build_synthetic_dataset
from fmr.faithfulness import fuse, score_dataset
from fmr.models.mock_vlm import MockVLM
from fmr.utils import save_json


def _auroc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")


def weight_sensitivity(records, labels, step=0.1):
    """AUROC of the fused FS across the (a,b,c) weight simplex."""
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    aurocs = []
    best = {"auroc": -1.0}
    for wa, wb in itertools.product(grid, grid):
        wc = 1.0 - wa - wb
        if wc < -1e-9:
            continue
        w = {"a": float(wa), "b": float(wb), "c": float(max(0.0, wc))}
        fs = [fuse(r["signal_a"], r["signal_b"], r["signal_c"], w) for r in records]
        au = _auroc(labels, fs)
        aurocs.append(au)
        if au > best["auroc"]:
            best = {"auroc": au, "weights": w}
    arr = np.array(aurocs)
    return {"n_weightings": len(aurocs), "auroc_mean": float(arr.mean()),
            "auroc_std": float(arr.std()), "auroc_min": float(arr.min()),
            "auroc_max": float(arr.max()), "best": best,
            "default_weights_auroc": _auroc(labels, [r["fs"] for r in records])}


def grid_sensitivity(grids=(2, 3, 4, 6, 8), n=800, n_cons=3):
    """Signal-B / FS AUROC as the region grid resolution changes."""
    out = {}
    for g in grids:
        samples = build_synthetic_dataset(n=n, grid=g, seed=7)
        recs = score_dataset(MockVLM(), samples, n_consistency_samples=n_cons)
        labels = [r["grounded_latent"] for r in recs]
        out[str(g)] = {"auroc_signal_b": _auroc(labels, [r["signal_b"] for r in recs]),
                       "auroc_fs": _auroc(labels, [r["fs"] for r in recs])}
    return out


def seed_stability(seeds=(1, 2, 3, 4, 5), n=800, n_cons=3):
    """Signal AUROC stability across independent synthetic draws."""
    keys = ("signal_a", "signal_b", "signal_c", "fs")
    acc = {k: [] for k in keys}
    for s in seeds:
        samples = build_synthetic_dataset(n=n, seed=s)
        recs = score_dataset(MockVLM(), samples, n_consistency_samples=n_cons)
        labels = [r["grounded_latent"] for r in recs]
        for k in keys:
            acc[k].append(_auroc(labels, [r[k] for r in recs]))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in acc.items()}


def abstention_power(records, alphas=(0.02, 0.05, 0.10, 0.15, 0.20),
                     cal_sizes=(100, 200, 500, 1000, 1500), delta=0.05):
    """Alpha sweep + smallest feasible calibration size per alpha."""
    scores = np.array([r["fs"] for r in records])
    correct = np.array([r["correct"] for r in records])
    n = len(scores)
    rng = np.random.default_rng(0)

    # Alpha sweep on a fixed cal/test split.
    idx = rng.permutation(n)
    half = n // 2
    cal_i, test_i = idx[:half], idx[half:]
    alpha_curve = {}
    for a in alphas:
        res = calibrate_threshold(scores[cal_i], correct[cal_i], alpha=a, delta=delta)
        ev = evaluate_selective(scores[test_i], correct[test_i], res.threshold)
        alpha_curve[str(a)] = {"feasible": res.feasible, "coverage": ev["coverage"],
                               "retained_error": ev["retained_error"]}

    # Power: smallest cal size that certifies each alpha (feasible).
    power = {}
    for a in alphas:
        smallest = None
        for m in cal_sizes:
            if m > n:
                break
            res = calibrate_threshold(scores[:m], correct[:m], alpha=a, delta=delta)
            if res.feasible and res.cal_coverage > 0.05:
                smallest = m
                break
        power[str(a)] = smallest
    return {"alpha_curve": alpha_curve, "min_cal_size_for_alpha": power}


def run(out_dir: str, config_dir=None):
    cfg = load_all_configs(config_dir)
    n_cons = int(cfg["experiment"]["signals"].get("consistency", {}).get("n_samples", 5))
    samples = build_synthetic_dataset(n=1500, seed=7)
    records = score_dataset(MockVLM(), samples, n_consistency_samples=n_cons)
    labels = [r["grounded_latent"] for r in records]

    results = {
        "weight_sensitivity": weight_sensitivity(records, labels),
        "grid_sensitivity": grid_sensitivity(n_cons=n_cons),
        "seed_stability": seed_stability(n_cons=n_cons),
        "abstention_power": abstention_power(records),
    }
    ws = results["weight_sensitivity"]
    print(f"[abl] FS weight sensitivity: AUROC {ws['auroc_min']:.3f}-{ws['auroc_max']:.3f} "
          f"(mean {ws['auroc_mean']:.3f} +/- {ws['auroc_std']:.3f}); best {ws['best']['weights']}")
    print(f"[abl] grid sensitivity (Signal B AUROC): "
          + "  ".join(f"g{g}={v['auroc_signal_b']:.3f}" for g, v in results["grid_sensitivity"].items()))
    print("[abl] seed stability:", {k: f"{v['mean']:.3f}+/-{v['std']:.3f}"
                                     for k, v in results["seed_stability"].items()})
    print("[abl] min cal size for alpha:", results["abstention_power"]["min_cal_size_for_alpha"])
    save_json(results, f"{out_dir}/ablations.json")
    print(f"[abl] wrote {out_dir}/ablations.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fmr/outputs/ablations")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()
    run(args.out, args.config_dir)
