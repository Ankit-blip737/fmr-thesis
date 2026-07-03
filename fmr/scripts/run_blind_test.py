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


def run(split: str, out_dir: str, config_dir: str | None = None) -> dict:
    cfg = load_all_configs(config_dir)
    _, splits = resolve_dataset_and_splits(cfg["data"])
    eval_set = splits[split]
    comparison = cfg["models"].get("comparison", {})
    model_keys = [comparison.get("reasoning_model", "mock_reasoner"),
                  comparison.get("non_reasoning_model", "mock_plain")]

    results: dict = {"dataset": cfg["data"].get("dataset"), "split": split, "n": len(eval_set), "models": {}}

    for key in model_keys:
        vlm = resolve_vlm(cfg["models"], key)
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

        results["models"][key] = {
            "name": vlm.name,
            "is_reasoning": vlm.is_reasoning,
            "accuracy": acc,
            "blind_gap": acc["original"] - acc["blank"],
            "mismatch_gap": acc["original"] - acc["mismatch"],
            "flip_rate": {v: float(np.mean(f)) for v, f in flips.items()},
            "iou_vs_step_index": {str(k): float(np.mean(v)) for k, v in sorted(iou_by_step.items())},
            "signal_b_vs_step_index": {str(k): float(np.mean(v)) for k, v in sorted(sigb_by_step.items())},
        }
        r = results["models"][key]
        print(f"[blind] {key:>16s}  acc(orig)={acc['original']:.3f}  acc(blank)={acc['blank']:.3f}  "
              f"acc(mismatch)={acc['mismatch']:.3f}  blind_gap={r['blind_gap']:.3f}")
        print(f"[blind] {'':>16s}  IoU by step: "
              + "  ".join(f"s{k}={v:.3f}" for k, v in sorted((int(a), b) for a, b in r['iou_vs_step_index'].items())))

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
