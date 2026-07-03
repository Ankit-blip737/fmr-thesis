"""Stage 4 verification — run + audit the training-free correction module.

Runs the selective correction pipeline over the synthetic dataset for one or
both mock backends and reports, with the hidden grounding latent used strictly
for *evaluation*:

  * accuracy before/after correction on flagged (low-FS) samples,
  * the fixable-vs-must-abstain split (prior-dominated rescued; image-blind
    unchanged and still low-FS -> flows to the abstention gate),
  * step-level revision stats (kept/dropped, support),
  * a REGRESSION CHECK: correction force-applied to well-grounded (unflagged)
    samples must not corrupt them,
  * a manual-audit dump of flagged examples (JSONL + stdout).

Usage:  python fmr/scripts/run_correction.py [--model mock|prior|both] [--n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmr.correction import CorrectionConfig, correct_sample  # noqa: E402
from fmr.data import build_synthetic_dataset  # noqa: E402
from fmr.models import load_vlm  # noqa: E402
from fmr.models.second_vlm import load_second_vlm  # noqa: E402
from fmr.utils import load_config  # noqa: E402


def run_one_model(vlm, samples, cfg: CorrectionConfig, audit_k: int = 20) -> tuple[dict, list[dict]]:
    results = [correct_sample(vlm, s, config=cfg) for s in samples]
    by_id = {s.sample_id: s for s in samples}

    applied = [r for r in results if r.applied]
    unflagged = [r for r in results if not r.applied]

    def acc(rs, which: str) -> float:
        if not rs:
            return float("nan")
        outs = [getattr(r, which).answer == by_id[r.sample_id].answer for r in rs]
        return sum(outs) / len(rs)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    # Eval-only split by the hidden latent: which flagged cases are "fixable"
    # (image evidence exists) vs "image-blind" (abstention's job).
    app_grounded = [r for r in applied if by_id[r.sample_id].meta.get("grounded")]
    app_blind = [r for r in applied if not by_id[r.sample_id].meta.get("grounded")]

    # Regression check: force correction on the unflagged (high-FS) samples.
    force_cfg = replace(cfg, trigger_threshold=1.1)
    regression = []
    for r in unflagged:
        fr = correct_sample(vlm, by_id[r.sample_id], config=force_cfg)
        regression.append(fr)
    reg_changed = [r for r in regression if r.answer_changed]
    reg_broke = [
        r for r in reg_changed
        if r.original.answer == by_id[r.sample_id].answer
        and r.corrected.answer != by_id[r.sample_id].answer
    ]

    summary = {
        "model": vlm.name,
        "n": len(samples),
        "n_flagged": len(applied),
        "trigger_threshold": cfg.trigger_threshold,
        "flagged": {
            "acc_before": acc(applied, "original"),
            "acc_after": acc(applied, "corrected"),
            "fs_before_mean": mean(r.fs_before for r in applied),
            "fs_after_mean": mean(r.fs_after for r in applied),
            "answer_changed": sum(r.answer_changed for r in applied),
            "keep_frac_mean": mean(r.diagnostics.get("keep_frac", 0.0) for r in applied),
        },
        "flagged_fixable(grounded latent)": {
            "n": len(app_grounded),
            "acc_before": acc(app_grounded, "original"),
            "acc_after": acc(app_grounded, "corrected"),
            "fs_after_mean": mean(r.fs_after for r in app_grounded),
            "recovers_gate": mean(r.fs_after >= cfg.trigger_threshold for r in app_grounded),
        },
        "flagged_image_blind(latent)": {
            "n": len(app_blind),
            "acc_before": acc(app_blind, "original"),
            "acc_after": acc(app_blind, "corrected"),
            "fs_after_mean": mean(r.fs_after for r in app_blind),
            "still_low_fs(->abstain)": mean(r.fs_after < cfg.trigger_threshold for r in app_blind),
        },
        "regression_check(forced on unflagged)": {
            "n": len(regression),
            "acc_before": acc(regression, "original"),
            "acc_after": acc(regression, "corrected"),
            "answers_changed": len(reg_changed),
            "correct_answers_broken": len(reg_broke),
        },
    }

    audit = []
    for r in applied[:audit_k]:
        s = by_id[r.sample_id]
        audit.append({
            "sample_id": r.sample_id,
            "question": s.question,
            "gt_answer": s.answer,
            "grounded_latent(eval_only)": int(s.meta.get("grounded", -1)),
            "answer_before": r.original.answer,
            "answer_after": r.corrected.answer,
            "fs_before": round(r.fs_before, 4),
            "fs_after": round(r.fs_after, 4),
            "vcd_changed": bool(r.diagnostics.get("vcd_changed")),
            "vcd_margin": round(float(r.diagnostics.get("vcd_margin", 0.0)), 3),
            "steps_kept": f"{r.diagnostics.get('n_kept')}/{r.diagnostics.get('n_steps')}",
            "clue_confidence": round(float(r.diagnostics.get("clue_confidence", 0.0)), 3),
        })
    return summary, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mock", "prior", "both"], default="both")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=str(ROOT / "configs" / "correction.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--audit-k", type=int, default=20)
    args = ap.parse_args()

    cfg = CorrectionConfig.from_dict(load_config(args.config).get("correction"))
    samples = build_synthetic_dataset(n=args.n, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    if args.model in ("mock", "both"):
        models.append(("mock", load_vlm({"backend": "mock"})))
    if args.model in ("prior", "both"):
        models.append(("prior", load_second_vlm({"backend": "mock_prior"})))

    for tag, vlm in models:
        summary, audit = run_one_model(vlm, samples, cfg, audit_k=args.audit_k)
        (out_dir / f"correction_summary_{tag}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with open(out_dir / f"correction_audit_{tag}.jsonl", "w", encoding="utf-8") as fh:
            for row in audit:
                fh.write(json.dumps(row) + "\n")

        print(f"\n===== {vlm.name} =====")
        print(json.dumps(summary, indent=2))
        print(f"\n--- manual audit (first {min(15, len(audit))} flagged) ---")
        for row in audit[:15]:
            print(
                f"{row['sample_id']} latent={row['grounded_latent(eval_only)']} "
                f"gt={row['gt_answer']!r:10s} before={row['answer_before']!r:10s} "
                f"after={row['answer_after']!r:10s} fs {row['fs_before']:.2f}->{row['fs_after']:.2f} "
                f"steps {row['steps_kept']} clue_conf={row['clue_confidence']}"
            )


if __name__ == "__main__":
    main()
