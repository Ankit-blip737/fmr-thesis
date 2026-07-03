"""Per-component correction ablation (proposal Section 9).

The selective-pipeline number alone doesn't attribute the gain to a component, so
this script decomposes correction into its three primitives and reports each one's
marginal contribution on the SAME flagged (low-FS) samples:

  baseline        raw model, no correction
  +VCD            answer-level contrast only (rationale untouched)
  +Verify(noclue) drop unsupported steps using a no-trace self-consistency support
  +Verify+Clue    drop unsupported steps using the clue-traced evidence region
  Full            VCD + clue-tracing + verify-revise

Row deltas give the ablation:
  (+VCD − baseline)          = VCD's contribution (answer accuracy)
  (+Verify(noclue) − base)   = step verification's contribution (rationale support)
  (+Verify+Clue − +Verify)   = clue-tracing's contribution (better support source)
  (Full − +Verify+Clue)      = VCD on top of a cleaned rationale

Metrics (hidden grounding latent used for EVAL only):
  acc                 answer accuracy on flagged samples
  step_support_rate   fraction of RETAINED steps that are genuinely image-grounded
                      (IoU(step region, GT region) >= 0.5) — rationale faithfulness
  keep_frac           fraction of steps retained (rationale compression)
  fs_after            post-correction faithfulness (Signal A)
  regr_broken         correct answers broken on the well-grounded holdout (safety)

Usage: python fmr/scripts/ablate_correction.py [--model both] [--n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmr.correction import CorrectionConfig  # noqa: E402
from fmr.correction.clue_tracing import ClueTrace, clue_support, trace_clue_region  # noqa: E402
from fmr.correction.rescore import post_correction_sensitivity  # noqa: E402
from fmr.correction.vcd import VCDResult, vcd_answer  # noqa: E402
from fmr.correction.verify_revise import verify_and_revise  # noqa: E402
from fmr.faithfulness.counterfactual import counterfactual_signal  # noqa: E402
from fmr.data import build_synthetic_dataset  # noqa: E402
from fmr.models import load_vlm  # noqa: E402
from fmr.models.second_vlm import load_second_vlm  # noqa: E402

CONFIGS = ("baseline", "+VCD", "+Verify(noclue)", "+Verify+Clue", "Full")


def _noop_vcd(vcd: VCDResult) -> VCDResult:
    """A VCD result that changes nothing (answer stays original) — for VCD-off rows."""
    orig = vcd.outputs["original"]
    return VCDResult(
        answer=vcd.original_answer, original_answer=vcd.original_answer, changed=False,
        p_vcd=orig.answer_logits, margin=0.0, n_plausible=vcd.n_plausible,
        contrast_variants=vcd.contrast_variants, outputs=vcd.outputs,
    )


def _self_support(output) -> list[float]:
    """No-clue-tracing support baseline: IoU of each step's region with the medoid
    of the greedy chain's OWN step regions (single-chain self-consistency, no
    external multi-chain corroboration)."""
    regions = [s.pred_region for s in output.steps if s.pred_region is not None]
    if len(regions) < 2:
        return [0.0 for _ in output.steps]
    best_i, best = 0, -1.0
    for i, r in enumerate(regions):
        agree = np.mean([r.iou(o) for j, o in enumerate(regions) if j != i])
        if agree > best:
            best_i, best = i, agree
    medoid = regions[best_i]
    conf = float(max(0.0, best))
    return [float(s.pred_region.iou(medoid) * conf) if s.pred_region is not None else 0.0
            for s in output.steps]


def _step_grounded(step, sample) -> bool:
    if step.pred_region is None or sample.gt_region is None:
        return False
    return step.pred_region.iou(sample.gt_region) >= 0.5


def apply_config(vlm, sample, cfg: CorrectionConfig, name: str, cache: dict):
    """Return (answer, retained_steps, fs_after) for one ablation config."""
    cf = cache["cf"]
    orig = cf["orig"]
    if name == "baseline":
        return orig.answer, list(orig.steps), float(cf["counterfactual"])

    vcd = cache["vcd"]
    trace = cache["trace"]

    if name == "+VCD":
        # keep all steps, adopt the VCD answer per margin
        out, _ = verify_and_revise(sample, orig, trace, vcd, support_threshold=-1.0,
                                   vcd_margin=cfg.vcd_margin,
                                   supports=[1.0] * len(orig.steps))
    elif name == "+Verify(noclue)":
        out, _ = verify_and_revise(sample, orig, ClueTrace(None, 0.0), _noop_vcd(vcd),
                                   support_threshold=cfg.support_threshold,
                                   vcd_margin=float("inf"), supports=_self_support(orig))
    elif name == "+Verify+Clue":
        out, _ = verify_and_revise(sample, orig, trace, _noop_vcd(vcd),
                                   support_threshold=cfg.support_threshold,
                                   vcd_margin=float("inf"),
                                   supports=[clue_support(s, trace) for s in orig.steps])
    elif name == "Full":
        out, _ = verify_and_revise(sample, orig, trace, vcd,
                                   support_threshold=cfg.support_threshold,
                                   vcd_margin=cfg.vcd_margin,
                                   supports=[clue_support(s, trace) for s in orig.steps])
    else:
        raise ValueError(name)

    fs_after = float(post_correction_sensitivity(vlm, sample, out, reuse=vcd.outputs)["counterfactual"])
    # retained steps = those with a real region (collapsed-note step has None)
    retained = [s for s in out.steps if s.pred_region is not None]
    return out.answer, retained, fs_after


def run_model(vlm, samples, cfg: CorrectionConfig) -> dict:
    gt = {s.sample_id: s.answer for s in samples}
    # trigger on Signal A, exactly like the pipeline
    flagged, unflagged_grounded = [], []
    caches = {}
    for s in samples:
        cf = counterfactual_signal(vlm, s)
        caches[s.sample_id] = {"cf": cf}
        if cf["counterfactual"] < cfg.trigger_threshold:
            flagged.append(s)
        elif s.meta.get("grounded"):
            unflagged_grounded.append(s)

    # precompute vcd + trace once per flagged sample (shared across configs)
    for s in flagged:
        caches[s.sample_id]["vcd"] = vcd_answer(vlm, s, alpha=cfg.alpha, beta=cfg.beta,
                                                contrast_variants=cfg.contrast_variants)
        caches[s.sample_id]["trace"] = trace_clue_region(vlm, s, n_probes=cfg.n_probes,
                                                          temperature=cfg.probe_temperature,
                                                          early_weight=cfg.early_weight)

    table = {}
    for name in CONFIGS:
        accs, supp_rates, keeps, fss = [], [], [], []
        for s in flagged:
            ans, retained, fs = apply_config(vlm, s, cfg, name, caches[s.sample_id])
            accs.append(ans == gt[s.sample_id])
            n_all = len(caches[s.sample_id]["cf"]["orig"].steps)
            if retained:
                supp_rates.append(np.mean([_step_grounded(st, s) for st in retained]))
            keeps.append(len(retained) / n_all if n_all else 0.0)
            fss.append(fs)
        table[name] = {
            "acc": float(np.mean(accs)) if accs else float("nan"),
            "step_support_rate": float(np.mean(supp_rates)) if supp_rates else float("nan"),
            "keep_frac": float(np.mean(keeps)) if keeps else float("nan"),
            "fs_after": float(np.mean(fss)) if fss else float("nan"),
        }

    # Safety regression: force each config on the well-grounded holdout, count breaks.
    for s in unflagged_grounded:
        c = caches[s.sample_id]
        c["vcd"] = vcd_answer(vlm, s, alpha=cfg.alpha, beta=cfg.beta,
                              contrast_variants=cfg.contrast_variants)
        c["trace"] = trace_clue_region(vlm, s, n_probes=cfg.n_probes,
                                       temperature=cfg.probe_temperature, early_weight=cfg.early_weight)
    for name in CONFIGS:
        broken = 0
        for s in unflagged_grounded:
            ans, _, _ = apply_config(vlm, s, cfg, name, caches[s.sample_id])
            was_right = caches[s.sample_id]["cf"]["orig"].answer == gt[s.sample_id]
            broken += int(was_right and ans != gt[s.sample_id])
        table[name]["regr_broken"] = broken

    return {
        "model": vlm.name,
        "n_flagged": len(flagged),
        "n_wellgrounded_holdout": len(unflagged_grounded),
        "configs": table,
    }


def _print_table(rep: dict) -> None:
    print(f"\n===== {rep['model']}  (flagged={rep['n_flagged']}, "
          f"holdout={rep['n_wellgrounded_holdout']}) =====")
    hdr = f"{'config':<16}{'acc':>7}{'step_supp':>11}{'keep':>7}{'fs_after':>10}{'regr_broken':>13}"
    print(hdr); print("-" * len(hdr))
    for name, m in rep["configs"].items():
        print(f"{name:<16}{m['acc']:>7.3f}{m['step_support_rate']:>11.3f}"
              f"{m['keep_frac']:>7.3f}{m['fs_after']:>10.3f}{m['regr_broken']:>13d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mock", "prior", "both"], default="both")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=str(ROOT / "configs" / "correction.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    from fmr.utils import load_config
    cfg = CorrectionConfig.from_dict(load_config(args.config).get("correction"))
    samples = build_synthetic_dataset(n=args.n, seed=args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    if args.model in ("mock", "both"):
        models.append(load_vlm({"backend": "mock"}))
    if args.model in ("prior", "both"):
        models.append(load_second_vlm({"backend": "mock_prior"}))

    reports = {}
    for vlm in models:
        rep = run_model(vlm, samples, cfg)
        reports[vlm.name] = rep
        _print_table(rep)

    (out_dir / "correction_ablation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_dir / 'correction_ablation.json'}")


if __name__ == "__main__":
    main()
