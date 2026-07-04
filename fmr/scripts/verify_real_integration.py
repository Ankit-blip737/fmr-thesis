"""STEP 1 — Adapter + verifier + calibration end-to-end on REAL VQA-RAD data.

Runs on CPU using the real VQA-RAD records already pushed to this branch
(fmr/outputs/real/vqa_rad/fmr_records.json — produced by MedVLM-R1 on Colab), NOT
mock fixtures. Verifies three integration points and reports actual numbers:

  1. adapter.frame_from_records maps real Signal A/B/C records -> the exact 13-key
     verifier feature schema (FEATURE_KEYS), end-to-end.
  2. The learned verifier TRAINS on the real weak labels and is evaluated on a
     held-out real test split — reported as (a) held-out weak-label AUROC
     (learnability) and (b) FS-vs-answer-correctness AUROC (downstream utility,
     since real data has no ground-truth grounding latent). Heuristic fusion is
     the baseline.
  3. correction.post_correction_fs wiring: (a) the real conformal gate is
     calibrated on the real records at alpha in {0.05, 0.10} and feasibility is
     reported; (b) post_correction_fs is exercised via dependency injection with
     Instance A's attention_signal + fuse to confirm it emits a fused FS the gate
     can consume (the real correction *pass itself* is GPU-bound — Colab).

Honest caveats printed inline: tiny real n (train 40 / cal 20 / test 20); Signal
B is a constant 0.5 on real data (attention->region extraction is still the
stub); no true grounding labels on real VQA-RAD.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _common import CONFIG_DIR  # noqa: F401 (ensures src on path)
from sklearn.metrics import roc_auc_score

from fmr.abstention import calibrate_threshold, evaluate_selective
from fmr.training import FEATURE_KEYS, HeuristicFusion, LearnedVerifier
from fmr.training.adapter import frame_from_records, features_from_record

REAL = Path("fmr/outputs/real/vqa_rad/fmr_records.json")


def _auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _schema_check(records):
    feats = [features_from_record(r) for r in records[:5]]
    ok = all(set(f.keys()) == set(FEATURE_KEYS) for f in feats)
    frame = frame_from_records(records)
    return {
        "n": len(records),
        "schema_ok": ok,
        "n_feature_keys": len(FEATURE_KEYS),
        "X_shape": list(frame["X"].shape),
        "y_true_all_missing": bool((frame["y_true"] == -1).all()),
        "y_weak_pos_rate": float(np.mean(frame["y_weak"])),
        "signal_b_constant": bool(np.allclose([r["signal_b"] for r in records],
                                              records[0]["signal_b"])),
    }


def _train_and_eval(train, test):
    tr = frame_from_records(train)
    te = frame_from_records(test)
    y_correct_te = np.array([r["correct"] for r in test])

    heur = HeuristicFusion()
    s_heur = heur.score_batch(te["feats"])
    out = {
        "heuristic": {
            "weak_auroc": _auc(te["y_weak"], s_heur),
            "correctness_auroc": _auc(y_correct_te, s_heur),
        },
        "learned": {},
    }
    for kind in ("logreg", "gbt"):
        v = LearnedVerifier(model_kind=kind).fit(tr["X"], tr["y_weak"], seed=0)
        s = v.score_batch(te["feats"])
        out["learned"][kind] = {
            "weak_auroc": _auc(te["y_weak"], s),
            "correctness_auroc": _auc(y_correct_te, s),
        }
    # Fused-FS-vs-correctness directly from the record (Instance A's shipped FS).
    out["record_fs_correctness_auroc"] = _auc(y_correct_te, [r["fs"] for r in test])
    return out


def _calibration(cal, test):
    res = {}
    cal_s = np.array([r["fs"] for r in cal]); cal_c = np.array([r["correct"] for r in cal])
    te_s = np.array([r["fs"] for r in test]); te_c = np.array([r["correct"] for r in test])
    for alpha in (0.05, 0.10):
        c = calibrate_threshold(cal_s, cal_c, alpha=alpha, delta=0.10)
        ev = evaluate_selective(te_s, te_c, c.threshold)
        res[f"alpha_{alpha:.2f}"] = {
            "feasible": c.feasible, "threshold": None if c.threshold == float("inf") else round(c.threshold, 4),
            "cal_coverage": round(c.cal_coverage, 3),
            "test_coverage": round(ev["coverage"], 3),
            "test_retained_error": None if np.isnan(ev["retained_error"]) else round(ev["retained_error"], 4),
        }
    return res


def _post_correction_wiring_check():
    """Confirm post_correction_fs accepts Instance A's fuse + attention_signal and
    emits a fused FS the gate can consume. Uses MockVLM on a synthetic sample (the
    real correction pass needs the GPU model); this validates the injection path
    (review fix #4), not real correction numbers."""
    from fmr.correction import post_correction_fs
    from fmr.data.synthetic import build_synthetic_dataset
    from fmr.faithfulness import fuse
    from fmr.faithfulness.attention import attention_signal
    from fmr.models.mock_vlm import MockVLM

    s = build_synthetic_dataset(n=1)[0]
    vlm = MockVLM()
    corrected = vlm.generate(s, variant="original")
    out = post_correction_fs(vlm, s, corrected, attention_fn=attention_signal,
                             consistency_c=0.7, fuse_fn=lambda a, b, c: fuse(a, b, c))
    return {"wiring_ok": ("fs" in out and 0.0 <= out["fs"] <= 1.0),
            "fused_fs_example": round(float(out["fs"]), 4),
            "note": "real correction pass on Qwen/MedVLM is GPU-bound (Colab)"}


def main():
    records = json.load(open(REAL, encoding="utf-8"))
    train, cal, test = records["train"], records["cal"], records["test"]

    schema = _schema_check(test)
    verif = _train_and_eval(train, test)
    calib = _calibration(cal, test)
    wiring = _post_correction_wiring_check()

    print("=== STEP 1: real-data (VQA-RAD) adapter integration ===")
    print("schema:", json.dumps(schema))
    print("verifier:", json.dumps(verif))
    print("calibration:", json.dumps(calib))
    print("post_correction wiring:", json.dumps(wiring))

    result = {"dataset": "vqa_rad (real, MedVLM-R1)",
              "splits": {"train": len(train), "cal": len(cal), "test": len(test)},
              "schema": schema, "verifier": verif, "calibration": calib,
              "post_correction_wiring": wiring}
    Path("fmr/outputs/real/vqa_rad").mkdir(parents=True, exist_ok=True)
    json.dump(result, open("fmr/outputs/real/vqa_rad/adapter_integration.json", "w"), indent=2)
    print("wrote fmr/outputs/real/vqa_rad/adapter_integration.json")
    return result


if __name__ == "__main__":
    main()
