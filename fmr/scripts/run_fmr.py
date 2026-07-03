"""Stages 3+5 — full FMR pipeline: signals -> FS -> conformal gate -> selective eval.

Produces:
  * per-sample records (cal + test) with RAW per-signal scores — this file is
    the training input for Instance B's learned verifier (see DECISIONS.md
    interface contract),
  * signal validation (AUROC of each signal + fused FS against the grounding
    labels; IoU validation of Signal B),
  * calibrated abstention (SGR bound) with the empirical guarantee check,
  * risk-coverage comparison: fused FS vs confidence vs each single signal.

Usage:
    python fmr/scripts/run_fmr.py [--out fmr/outputs] [--alpha 0.05]
    # --post-correction: set once Instance B's correction module is applied
    #   upstream, so results stop being labeled provisional.
"""
from __future__ import annotations

import argparse

import numpy as np
from _common import load_all_configs, resolve_dataset_and_splits, resolve_vlm
from sklearn.metrics import roc_auc_score

from fmr.abstention import calibrate_threshold, evaluate_selective, risk_coverage_curve
from fmr.faithfulness import score_dataset
from fmr.utils import save_json


def _strip(records: list[dict]) -> list[dict]:
    """Drop the non-serializable VLMOutput before writing records to JSON."""
    return [{k: v for k, v in r.items() if k != "output"} for r in records]


def _auroc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run(out_dir: str, alpha: float, delta: float, post_correction: bool,
        config_dir: str | None = None) -> dict:
    cfg = load_all_configs(config_dir)
    _, splits = resolve_dataset_and_splits(cfg["data"])
    sig_cfg = cfg["experiment"]["signals"]
    weights = sig_cfg.get("weights")
    cons = sig_cfg.get("consistency", {})

    vlm = resolve_vlm(cfg["models"], samples=splits["train"] + splits["cal"] + splits["test"])
    print(f"[fmr] scoring cal ({len(splits['cal'])}) + test ({len(splits['test'])}) "
          f"with {vlm.name} (n_consistency={cons.get('n_samples', 5)})")

    kw = dict(weights=weights,
              n_consistency_samples=int(cons.get("n_samples", 5)),
              consistency_temperature=float(cons.get("temperature", 0.7)))
    cal_records = score_dataset(vlm, splits["cal"], **kw)
    test_records = score_dataset(vlm, splits["test"], **kw)
    train_records = score_dataset(vlm, splits["train"], **kw)  # for Instance B's verifier

    # ---- Signal validation: does each signal separate grounded from not? ----
    # Sample-level label: the hidden latent on synthetic data; weak IoU labels
    # (majority over steps) on real box datasets.
    def _labels(records: list[dict]) -> list[int] | None:
        if records and records[0]["grounded_latent"] is not None:
            return [int(r["grounded_latent"]) for r in records]
        if records and records[0]["weak_labels"] is not None:
            return [int(np.mean(r["weak_labels"]) >= 0.5) for r in records]
        return None

    validation: dict = {"label_source": "grounded_latent" if test_records[0]["grounded_latent"] is not None
                        else "iou_weak_labels"}
    labels = _labels(test_records)
    if labels is not None:
        for sig in ("signal_a", "signal_b", "signal_c", "fs", "confidence"):
            validation[f"auroc_{sig}"] = _auroc(labels, [r[sig] for r in test_records])
        # Signal B IoU validation: mean IoU for latent-grounded vs ungrounded.
        with_iou = [r for r in test_records if r["iou_mean"] is not None]
        if with_iou and with_iou[0]["grounded_latent"] is not None:
            g = [r["iou_mean"] for r in with_iou if r["grounded_latent"] == 1]
            u = [r["iou_mean"] for r in with_iou if r["grounded_latent"] == 0]
            validation["mean_iou_grounded"] = float(np.mean(g)) if g else None
            validation["mean_iou_ungrounded"] = float(np.mean(u)) if u else None
        print("[fmr] AUROC vs grounding labels: "
              + "  ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in validation.items()
                          if k.startswith("auroc_")))

    # ---- Conformal abstention: calibrate on cal, evaluate on test ------------
    def _selective(score_key: str) -> dict:
        cal_s = np.array([r[score_key] for r in cal_records])
        cal_c = np.array([r["correct"] for r in cal_records])
        test_s = np.array([r[score_key] for r in test_records])
        test_c = np.array([r["correct"] for r in test_records])
        calres = calibrate_threshold(cal_s, cal_c, alpha=alpha, delta=delta)
        ev = evaluate_selective(test_s, test_c, calres.threshold)
        rc = risk_coverage_curve(test_s, test_c)
        return {
            "threshold": calres.threshold, "feasible": calres.feasible,
            "cal_coverage": calres.cal_coverage, "cal_error": calres.cal_error,
            "cal_error_ucb": calres.cal_error_ucb,
            "test": ev, "guarantee_holds": (not calres.feasible) or ev["n_retained"] == 0
                                          or ev["retained_error"] <= alpha,
            "aurc": rc["aurc"], "risk_coverage": rc,
        }

    abstention = {
        "alpha": alpha, "delta": delta,
        "provisional_pre_correction": not post_correction,
        "fs": _selective("fs"),
        "confidence": _selective("confidence"),
        "signal_a_only": _selective("signal_a"),
        "signal_b_only": _selective("signal_b"),
        "signal_c_only": _selective("signal_c"),
    }
    for k in ("fs", "confidence"):
        a = abstention[k]
        print(f"[fmr] abstain[{k:>10s}] tau={a['threshold']:.3f} feasible={a['feasible']} "
              f"cov(test)={a['test']['coverage']:.3f} err(test)={a['test']['retained_error']:.4f} "
              f"(target<={alpha})  AURC={a['aurc']:.4f}")

    # ---- Per-modality accuracy + FS -----------------------------------------
    per_mod: dict = {}
    for r in test_records:
        d = per_mod.setdefault(r["modality"], {"n": 0, "correct": 0, "fs": []})
        d["n"] += 1
        d["correct"] += r["correct"]
        d["fs"].append(r["fs"])
    per_mod = {m: {"n": d["n"], "accuracy": d["correct"] / d["n"], "mean_fs": float(np.mean(d["fs"]))}
               for m, d in sorted(per_mod.items())}

    results = {
        "model": vlm.name, "dataset": cfg["data"].get("dataset"),
        "n_train": len(train_records), "n_cal": len(cal_records), "n_test": len(test_records),
        "validation": validation, "abstention": abstention, "per_modality": per_mod,
    }
    save_json(results, f"{out_dir}/fmr_results.json")
    save_json({"train": _strip(train_records), "cal": _strip(cal_records),
               "test": _strip(test_records)}, f"{out_dir}/fmr_records.json")
    print(f"[fmr] wrote {out_dir}/fmr_results.json and {out_dir}/fmr_records.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fmr/outputs")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--post-correction", action="store_true")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()
    exp = load_all_configs(args.config_dir)["experiment"]["abstention"]
    run(args.out,
        alpha=args.alpha if args.alpha is not None else float(exp.get("alpha", 0.05)),
        delta=args.delta if args.delta is not None else float(exp.get("delta", 0.05)),
        post_correction=args.post_correction,
        config_dir=args.config_dir)
