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
    """Test the headline hypothesis explicitly and honestly (review fix #2).

    Two independent lenses on "more reasoning -> less grounded":
      (1) BLIND-GAP lens (always available): compare image-reliance
          `blind_gap = acc(orig) - acc(blank)` for the reasoning vs the
          non-reasoning model. If the reasoning model's answers depend on the
          image LESS (smaller blind_gap), that supports the hypothesis. We flag
          the accuracy confound (a weaker model can have a small gap for the
          wrong reason).
      (2) DRIFT lens (needs per-step attended regions): the reasoning model's
          grounding decays along the chain (negative IoU-vs-step slope). This is
          the sharper test but requires attention->region extraction, which the
          real HF backend only provides when attention grounding is enabled.

    `replicated` is set from whichever lens has evidence; if neither does, say so
    plainly and do not force the framing.
    """
    reasoning = [m for m in results["models"].values() if m["is_reasoning"]]
    plain = [m for m in results["models"].values() if not m["is_reasoning"]]
    if not reasoning:
        return {"tested": False, "reason": "no reasoning model in comparison"}
    r = reasoning[0]
    verdict = {"tested": True, "reasoning_model": r["name"]}

    # --- Lens 1: blind-gap comparison (reasoning vs non-reasoning) -----------
    blind_supports = None
    if plain:
        p = plain[0]
        rg, pg = r.get("blind_gap"), p.get("blind_gap")
        if rg is not None and pg is not None:
            verdict["blind_gap_reasoning"] = rg
            verdict["blind_gap_nonreasoning"] = pg
            verdict["blind_gap_delta"] = rg - pg           # <0 => reasoning less grounded
            # Guard: if BOTH gaps are ~0 or negative, neither model relies on the
            # image — a smaller reasoning gap is then NOT evidence of "reasoning
            # less grounded", it's just noise on two image-independent models.
            _EPS = 0.03
            both_image_independent = (rg <= _EPS and pg <= _EPS)
            verdict["both_image_independent"] = both_image_independent
            blind_supports = (rg < pg - 1e-6) and not both_image_independent
            verdict["blind_gap_supports"] = blind_supports
            # Accuracy confound flag.
            ra = (r.get("accuracy") or {}).get("original")
            pa = (p.get("accuracy") or {}).get("original")
            if ra is not None and pa is not None:
                verdict["accuracy_reasoning"] = ra
                verdict["accuracy_nonreasoning"] = pa
                verdict["accuracy_confound"] = ra < pa - 0.05  # reasoning notably weaker

    # --- Lens 2: within-model grounding drift (needs per-step regions) -------
    slope = r.get("grounding_drift_slope", 0.0)
    has_drift = bool(r.get("iou_vs_step_index"))
    within_model_decay = has_drift and slope < -1e-3
    verdict["drift_slope"] = slope
    verdict["drift_available"] = has_drift
    verdict["within_model_decay"] = within_model_decay

    # --- Combine -------------------------------------------------------------
    if within_model_decay:
        verdict["replicated"] = True
        verdict["primary_evidence"] = "drift"
        verdict["note"] = "REPLICATED (per-step grounding decays along the chain)"
    elif blind_supports:
        verdict["replicated"] = True
        verdict["primary_evidence"] = "blind_gap"
        conf = " — but confounded by lower reasoning-model accuracy" if verdict.get("accuracy_confound") else ""
        verdict["note"] = (f"SUPPORTED via blind-gap: reasoning model relies on the image less "
                           f"(gap {verdict.get('blind_gap_reasoning'):.3f} < {verdict.get('blind_gap_nonreasoning'):.3f}){conf}"
                           + ("" if has_drift else "; per-step drift pending attention instrumentation"))
    elif verdict.get("both_image_independent"):
        verdict["replicated"] = False
        verdict["primary_evidence"] = "blind_gap"
        verdict["note"] = (f"NOT a grounding effect: BOTH models are image-independent on this "
                           f"dataset (blind_gap reasoning {verdict.get('blind_gap_reasoning'):.3f}, "
                           f"non-reasoning {verdict.get('blind_gap_nonreasoning'):.3f} — both ≤ 0.03). "
                           f"The dataset is answerable from language priors; the smaller reasoning gap is noise, not evidence.")
    elif blind_supports is False:
        verdict["replicated"] = False
        verdict["primary_evidence"] = "blind_gap"
        verdict["note"] = "NOT supported: reasoning model relies on the image at least as much as non-reasoning — report the actual effect"
    else:
        verdict["replicated"] = False
        verdict["primary_evidence"] = "none"
        verdict["note"] = "inconclusive — no drift signal and no non-reasoning comparator"
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

        # Free GPU memory before the next model loads (explicit unload > del).
        if hasattr(vlm, "unload"):
            vlm.unload()
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
