"""Stage 6 — full benchmark: correction integration, ablations, model-agnosticism.

This is the Stage-5-onward pipeline with Instance B's correction module wired in
and the complete ablation suite. Key design points:

  * **Calibration ordering (review fix #4).** The conformal gate is calibrated on
    the *post-correction* faithfulness score — the value the deployed system
    actually emits — not the raw pre-correction FS. We report both to show the
    effect of getting the ordering right.
  * **Guarded correction import.** If ``fmr.correction`` is present (Instance B's
    module, matched to its real ``correct_sample``/``CorrectionResult``
    interface) it is used; otherwise an identity fallback runs so the whole
    benchmark still executes on this branch, clearly labeled
    ``correction_present: false``.
  * **Ablations:** signals A→AB→ABC incrementally; FS vs confidence vs each
    single signal as the abstention trigger; correction on/off.
  * **Model-agnosticism:** the benchmark runs across ≥2 base-model configs.

Usage:
    python fmr/scripts/run_fmr_full.py [--models mock_reasoner mock_reasoner_b]
                                       [--alpha 0.05] [--out fmr/outputs/full]
"""
from __future__ import annotations

import argparse

import numpy as np
from _common import load_all_configs, resolve_dataset_and_splits, resolve_vlm
from sklearn.metrics import roc_auc_score

from fmr.abstention import calibrate_threshold, evaluate_selective, risk_coverage_curve
from fmr.faithfulness import attention_signal, fuse, score_dataset
from fmr.utils import save_json


# ---- correction hook (real module if present, else identity) ----------------
def _load_correction():
    try:
        from fmr.correction import CorrectionConfig, correct_sample  # Instance B
        return correct_sample, CorrectionConfig, True
    except Exception:
        return None, None, False


def _apply_correction(correct_sample, cfg, vlm, sample, record, weights):
    """Return (fs_after, corrected_correct, applied) for one scored sample.

    The DEPLOYED post-correction FS is *my fused score recomputed on the
    corrected output* — not Instance B's ``fs_after`` (which is Signal A only,
    its rescore trigger). Review fix #4 requires calibrating on the score the
    system actually emits, so we fuse:
      A = post-correction counterfactual sensitivity (Instance B's fs_after),
      B = attention coherence of the corrected reasoning chain,
      C = pre-correction self-consistency (a model/sample property the
          deterministic revision doesn't change; reuse noted in the thesis).
    """
    if correct_sample is None:
        return record["fs"], record["correct"], False
    res = correct_sample(vlm, sample, fs=record["fs"], original=record["output"], config=cfg)
    if not res.applied:
        return record["fs"], record["correct"], False
    a_after = float(res.fs_after)
    b_after = attention_signal(res.corrected)["attention"]
    c_after = record["signal_c"]
    fs_after = fuse(a_after, b_after, c_after, weights)
    corrected_correct = int(res.corrected.answer == sample.answer)
    return fs_after, corrected_correct, True


def _auroc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")


def _selective(cal_s, cal_c, test_s, test_c, alpha, delta):
    calres = calibrate_threshold(np.asarray(cal_s), np.asarray(cal_c), alpha=alpha, delta=delta)
    ev = evaluate_selective(np.asarray(test_s), np.asarray(test_c), calres.threshold)
    rc = risk_coverage_curve(np.asarray(test_s), np.asarray(test_c))
    guarantee = (not calres.feasible) or ev["n_retained"] == 0 or ev["retained_error"] <= alpha
    return {"threshold": calres.threshold, "feasible": calres.feasible,
            "cal_coverage": calres.cal_coverage, "test": ev,
            "aurc": rc["aurc"], "guarantee_holds": guarantee}


def _benchmark_one_model(cfg, model_key, alpha, delta, correction, corr_cfg):
    correct_sample, _, corr_present = correction
    _, splits = resolve_dataset_and_splits(cfg["data"])
    sig = cfg["experiment"]["signals"]
    kw = dict(weights=sig.get("weights"),
              n_consistency_samples=int(sig.get("consistency", {}).get("n_samples", 5)),
              consistency_temperature=float(sig.get("consistency", {}).get("temperature", 0.7)))

    all_samples = {s.sample_id: s for part in splits.values() for s in part}
    vlm = resolve_vlm(cfg["models"], model_key, samples=list(all_samples.values()))

    def score_split(name):
        recs = score_dataset(vlm, splits[name], **kw)
        # Attach post-correction FS + corrected correctness for each record.
        for r in recs:
            fs_after, corr_c, applied = _apply_correction(
                correct_sample, corr_cfg, vlm, all_samples[r["sample_id"]], r,
                sig.get("weights"))
            r["fs_after"] = fs_after
            r["correct_after"] = corr_c
            r["correction_applied"] = applied
        return recs

    cal, test = score_split("cal"), score_split("test")

    # ---- signal validation (incremental fusion ablation) -------------------
    def _labels(recs):
        if recs[0]["grounded_latent"] is not None:
            return [int(r["grounded_latent"]) for r in recs]
        if recs[0]["weak_labels"] is not None:
            return [int(np.mean(r["weak_labels"]) >= 0.5) for r in recs]
        return None

    labels = _labels(test)
    validation = {}
    if labels is not None:
        w = sig.get("weights", {"a": 0.4, "b": 0.3, "c": 0.3})
        fs_a = [r["signal_a"] for r in test]
        fs_ab = [fuse(r["signal_a"], r["signal_b"], r["signal_c"],
                      {"a": w["a"], "b": w["b"], "c": 0.0}) for r in test]
        fs_abc = [r["fs"] for r in test]
        validation = {
            "auroc_signal_a": _auroc(labels, [r["signal_a"] for r in test]),
            "auroc_signal_b": _auroc(labels, [r["signal_b"] for r in test]),
            "auroc_signal_c": _auroc(labels, [r["signal_c"] for r in test]),
            "auroc_fs_A": _auroc(labels, fs_a),
            "auroc_fs_AB": _auroc(labels, fs_ab),
            "auroc_fs_ABC": _auroc(labels, fs_abc),
            "auroc_confidence": _auroc(labels, [r["confidence"] for r in test]),
        }

    # ---- abstention: post-correction FS (deployed) vs pre-correction vs others
    def col(recs, k):
        return [r[k] for r in recs]

    abst = {
        "alpha": alpha, "delta": delta, "correction_present": corr_present,
        # THE deployed gate: calibrate on post-correction FS, score corrected answers.
        "fs_post_correction": _selective(col(cal, "fs_after"), col(cal, "correct_after"),
                                         col(test, "fs_after"), col(test, "correct_after"), alpha, delta),
        # Ablation: pre-correction FS (wrong ordering) for contrast.
        "fs_pre_correction": _selective(col(cal, "fs"), col(cal, "correct"),
                                        col(test, "fs"), col(test, "correct"), alpha, delta),
        "confidence": _selective(col(cal, "confidence"), col(cal, "correct"),
                                 col(test, "confidence"), col(test, "correct"), alpha, delta),
        "signal_a_only": _selective(col(cal, "signal_a"), col(cal, "correct"),
                                    col(test, "signal_a"), col(test, "correct"), alpha, delta),
        "signal_b_only": _selective(col(cal, "signal_b"), col(cal, "correct"),
                                    col(test, "signal_b"), col(test, "correct"), alpha, delta),
        "signal_c_only": _selective(col(cal, "signal_c"), col(cal, "correct"),
                                    col(test, "signal_c"), col(test, "correct"), alpha, delta),
    }

    # ---- correction effect --------------------------------------------------
    n_applied = sum(r["correction_applied"] for r in test)
    correction_effect = {
        "n_applied": n_applied,
        "acc_before": float(np.mean(col(test, "correct"))),
        "acc_after": float(np.mean(col(test, "correct_after"))),
        "mean_fs_before": float(np.mean(col(test, "fs"))),
        "mean_fs_after": float(np.mean(col(test, "fs_after"))),
    }

    return {"model": vlm.name, "is_reasoning": vlm.is_reasoning,
            "n_cal": len(cal), "n_test": len(test),
            "validation": validation, "abstention": abst,
            "correction_effect": correction_effect}


def run(models, alpha, delta, out_dir, config_dir=None):
    cfg = load_all_configs(config_dir)
    correction = _load_correction()
    corr_cfg = None
    if correction[2]:
        _, CorrectionConfig, _ = correction
        corr_cfg = CorrectionConfig.from_dict(cfg.get("correction"))
    print(f"[full] correction module present: {correction[2]}")

    results = {"dataset": cfg["data"].get("dataset"), "alpha": alpha,
               "correction_present": correction[2], "models": {}}
    for key in models:
        print(f"[full] === model: {key} ===")
        res = _benchmark_one_model(cfg, key, alpha, delta, correction, corr_cfg)
        results["models"][key] = res
        v, ab = res["validation"], res["abstention"]
        if v:
            print(f"[full]   incremental fusion AUROC: A={v['auroc_fs_A']:.3f} "
                  f"AB={v['auroc_fs_AB']:.3f} ABC={v['auroc_fs_ABC']:.3f}")
        dep = ab["fs_post_correction"]
        print(f"[full]   deployed gate (post-corr FS): feasible={dep['feasible']} "
              f"cov={dep['test']['coverage']:.3f} err={dep['test']['retained_error']} "
              f"guarantee={dep['guarantee_holds']}")
        ce = res["correction_effect"]
        print(f"[full]   correction: applied={ce['n_applied']} acc {ce['acc_before']:.3f}->"
              f"{ce['acc_after']:.3f} meanFS {ce['mean_fs_before']:.3f}->{ce['mean_fs_after']:.3f}")

    save_json(results, f"{out_dir}/full_benchmark.json")
    print(f"[full] wrote {out_dir}/full_benchmark.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mock_reasoner", "mock_reasoner_b"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", default="fmr/outputs/full")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()
    run(args.models, args.alpha, args.delta, args.out, args.config_dir)
