"""Stage 2b — the blind test + the headline drift experiment.

Two questions, per the proposal:
  1. How much do models actually use the image? Compare accuracy under
     original vs blank vs mismatched images. `blind_gap` = acc(original) -
     acc(blank); a small gap means the model answers from language priors.
  2. Does grounding decay along the reasoning chain ("more reasoning -> less
     grounded")? Track step-region IoU vs GT and the coherence Signal B as a
     function of step index, reasoning vs non-reasoning model.

Usage:
    python fmr/scripts/run_blind_test.py [--split test] [--out fmr/outputs]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from _common import load_all_configs, resolve_dataset_and_splits, resolve_vlm

from fmr.faithfulness import attention_signal
from fmr.utils import save_json

VARIANTS = ("original", "blank", "mismatch")


def _drift_slope(curve: dict) -> float:
    """Least-squares slope of grounding vs step index (negative = decays)."""
    if len(curve) < 2:
        return 0.0
    xs = np.array(sorted(int(k) for k in curve), dtype=float)
    ys = np.array([curve[str(int(x))] for x in xs], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


def _replication_verdict(results: dict) -> dict:
    """Test the headline hypothesis explicitly (review fix #2).

    Replicated iff the reasoning model's grounding *decays* along the chain
    (negative drift slope) AND it is less grounded overall at the final step
    than the non-reasoning model's single step. If not, DON'T force the framing —
    this verdict is what the thesis reports, honestly, whatever it says.
    """
    reasoning = [m for m in results["models"].values() if m["is_reasoning"]]
    plain = [m for m in results["models"].values() if not m["is_reasoning"]]
    if not reasoning:
        return {"tested": False, "reason": "no reasoning model in comparison"}
    r = reasoning[0]
    slope = r["grounding_drift_slope"]
    within_model_decay = slope < -1e-3
    verdict = {
        "tested": True,
        "reasoning_model": r["name"],
        "drift_slope": slope,
        "within_model_decay": within_model_decay,
    }
    if plain:
        p = plain[0]
        r_last = list(r["iou_vs_step_index"].values())[-1] if r["iou_vs_step_index"] else None
        p_g = list(p["iou_vs_step_index"].values())[0] if p["iou_vs_step_index"] else None
        if r_last is not None and p_g is not None:
            verdict["reasoning_final_vs_plain"] = r_last - p_g
            verdict["reasoning_less_grounded_than_plain"] = r_last < p_g
    verdict["replicated"] = within_model_decay
    verdict["note"] = ("headline 'more reasoning -> less grounded' REPLICATED"
                       if within_model_decay else
                       "NOT replicated — report the actual effect, do not force the framing")
    return verdict


def run(split: str, out_dir: str, config_dir: str | None = None) -> dict:
    cfg = load_all_configs(config_dir)
    _, splits = resolve_dataset_and_splits(cfg["data"])
    eval_set = splits[split]
    comparison = cfg["models"].get("comparison", {})
    model_keys = [comparison.get("reasoning_model", "mock_reasoner"),
                  comparison.get("non_reasoning_model", "mock_plain")]

    results: dict = {"dataset": cfg["data"].get("dataset"), "split": split, "n": len(eval_set), "models": {}}

    for key in model_keys:
        vlm = resolve_vlm(cfg["models"], key, samples=eval_set)
        acc = {}
        flips = {"blank": [], "mismatch": []}
        outputs_orig = []
        for v in VARIANTS:
            hits = []
            for s in eval_set:
                out = vlm.generate(s, variant=v, temperature=0.0)
                hits.append(int(out.answer == s.answer))
                if v == "original":
                    outputs_orig.append((s, out))
            acc[v] = float(np.mean(hits))
        # Answer-flip rates vs the original answer.
        for s, orig in outputs_orig:
            for v in ("blank", "mismatch"):
                flips[v].append(int(vlm.generate(s, variant=v, temperature=0.0).answer != orig.answer))

        # Drift: per-step-index grounding, both vs-GT IoU and the runtime Signal B.
        iou_by_step: dict[int, list[float]] = defaultdict(list)
        sigb_by_step: dict[int, list[float]] = defaultdict(list)
        for s, orig in outputs_orig:
            att = attention_signal(orig)
            for k, step in enumerate(orig.steps):
                if s.gt_region is not None and step.pred_region is not None:
                    iou_by_step[k].append(step.pred_region.iou(s.gt_region))
                sigb_by_step[k].append(att["per_step"][k])

        iou_curve = {str(k): float(np.mean(v)) for k, v in sorted(iou_by_step.items())}
        sigb_curve = {str(k): float(np.mean(v)) for k, v in sorted(sigb_by_step.items())}
        results["models"][key] = {
            "name": vlm.name,
            "is_reasoning": vlm.is_reasoning,
            "accuracy": acc,
            "blind_gap": acc["original"] - acc["blank"],
            "mismatch_gap": acc["original"] - acc["mismatch"],
            "flip_rate": {v: float(np.mean(f)) for v, f in flips.items()},
            "iou_vs_step_index": iou_curve,
            "signal_b_vs_step_index": sigb_curve,
            "grounding_drift_slope": _drift_slope(iou_curve),
        }
        r = results["models"][key]
        print(f"[blind] {key:>16s}  acc(orig)={acc['original']:.3f}  acc(blank)={acc['blank']:.3f}  "
              f"acc(mismatch)={acc['mismatch']:.3f}  blind_gap={r['blind_gap']:.3f}")
        print(f"[blind] {'':>16s}  IoU by step: "
              + "  ".join(f"s{k}={v:.3f}" for k, v in sorted((int(a), b) for a, b in r['iou_vs_step_index'].items())))

        # Free memory before the next model
        del vlm
        import gc; import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["replication"] = _replication_verdict(results)
    print(f"[blind] HEADLINE: {results['replication']['note']} "
          f"(drift slope={results['replication'].get('drift_slope', float('nan')):.4f})")

    path = save_json(results, f"{out_dir}/blind_test.json")
    print(f"[blind] wrote {path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="fmr/outputs")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()
    run(args.split, args.out, args.config_dir)
