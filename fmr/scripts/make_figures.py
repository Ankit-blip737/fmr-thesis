"""Regenerate all figures from the JSON artifacts in fmr/outputs.

Every figure is reproducible from the saved run outputs alone — no model calls.

Usage:
    python fmr/scripts/make_figures.py [--out fmr/outputs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import load_all_configs  # noqa: E402,F401  (keeps sys.path shim)
from fmr.utils import ensure_dir  # noqa: E402


def _load(out_dir: Path, name: str) -> dict | None:
    p = out_dir / name
    if not p.exists():
        print(f"[figures] skip: {name} not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fig_drift(blind: dict, fig_dir: Path) -> None:
    """Headline: grounding (IoU vs GT) as a function of reasoning-step index."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, m in blind["models"].items():
        curve = m.get("iou_vs_step_index", {})
        if not curve:
            continue
        xs = sorted(int(k) for k in curve)
        ys = [curve[str(k)] for k in xs]
        style = "-o" if m["is_reasoning"] else "--s"
        ax.plot([x + 1 for x in xs], ys, style, label=f"{m['name']}"
                + (" (reasoning)" if m["is_reasoning"] else " (non-reasoning)"))
    ax.set_xlabel("Reasoning step index")
    ax.set_ylabel("Mean IoU with ground-truth evidence region")
    ax.set_title("Grounding decays along the reasoning chain")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_grounding_drift.png", dpi=150)
    plt.close(fig)


def fig_blind(blind: dict, fig_dir: Path) -> None:
    """Accuracy under original / blank / mismatch per model."""
    models = list(blind["models"].items())
    variants = ["original", "blank", "mismatch"]
    x = np.arange(len(variants))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (key, m) in enumerate(models):
        ys = [m["accuracy"][v] for v in variants]
        ax.bar(x + i * width, ys, width, label=m["name"])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Accuracy")
    ax.set_title("Blind test: how much does the answer depend on the image?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_blind_test.png", dpi=150)
    plt.close(fig)


def fig_signal_separation(records: dict, fig_dir: Path) -> None:
    """Distribution of each signal for grounded vs ungrounded cases (test split)."""
    test = records["test"]
    if test and test[0].get("grounded_latent") is None:
        print("[figures] skip fig3: no grounding labels in records")
        return
    sigs = ["signal_a", "signal_b", "signal_c", "fs"]
    fig, axes = plt.subplots(1, len(sigs), figsize=(14, 3.2), sharey=True)
    for ax, sig in zip(axes, sigs):
        g = [r[sig] for r in test if r["grounded_latent"] == 1]
        u = [r[sig] for r in test if r["grounded_latent"] == 0]
        ax.hist(u, bins=20, alpha=0.6, label="ungrounded", density=True)
        ax.hist(g, bins=20, alpha=0.6, label="grounded", density=True)
        ax.set_title(sig)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("Faithfulness signals separate grounded from ungrounded reasoning")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_signal_separation.png", dpi=150)
    plt.close(fig)


def fig_risk_coverage(results: dict, fig_dir: Path) -> None:
    """Risk-coverage: fused FS vs confidence vs single signals + the target line."""
    ab = results["abstention"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [("fs", "Faithfulness Score (fused)"), ("confidence", "Answer confidence"),
                       ("signal_a_only", "Signal A only"), ("signal_b_only", "Signal B only"),
                       ("signal_c_only", "Signal C only")]:
        rc = ab[key]["risk_coverage"]
        lw = 2.5 if key == "fs" else 1.2
        ax.plot(rc["coverage"], rc["risk"], label=f"{label} (AURC={ab[key]['aurc']:.3f})", linewidth=lw)
    ax.axhline(ab["alpha"], color="k", linestyle=":", label=f"target error α={ab['alpha']}")
    # Mark the calibrated FS operating point.
    t = ab["fs"]["test"]
    if t["n_retained"] > 0:
        ax.plot(t["coverage"], t["retained_error"], "r*", markersize=15,
                label=f"calibrated gate (cov={t['coverage']:.2f}, err={t['retained_error']:.3f})")
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Risk (error on answered cases)")
    title = "Selective prediction: faithfulness beats confidence as the deferral trigger"
    if ab.get("provisional_pre_correction"):
        title += "\n(provisional: pre-correction FS)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_risk_coverage.png", dpi=150)
    plt.close(fig)


def fig_incremental_fusion(full: dict, fig_dir: Path) -> None:
    """Ablation: AUROC as signals are added A -> A+B -> A+B+C, per model."""
    models = full["models"]
    if not any(m.get("validation") for m in models.values()):
        print("[figures] skip fig5: no validation labels")
        return
    stages = ["auroc_fs_A", "auroc_fs_AB", "auroc_fs_ABC"]
    labels = ["A", "A+B", "A+B+C"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, m in models.items():
        v = m.get("validation", {})
        if not v:
            continue
        ax.plot(labels, [v[s] for s in stages], "-o", label=m["model"])
    ax.set_ylabel("AUROC vs grounding label")
    ax.set_xlabel("Signals fused")
    ax.set_title("Incremental fusion ablation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_incremental_fusion.png", dpi=150)
    plt.close(fig)


def fig_model_agnosticism(full: dict, fig_dir: Path) -> None:
    """Deployed-gate coverage + retained error across base models (α line)."""
    models = full["models"]
    keys = list(models.keys())
    cov = [models[k]["abstention"]["fs_post_correction"]["test"]["coverage"] for k in keys]
    err = [models[k]["abstention"]["fs_post_correction"]["test"]["retained_error"] or 0.0 for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, cov, 0.4, label="coverage")
    ax.bar(x + 0.2, err, 0.4, label="retained error")
    ax.axhline(full["alpha"], color="k", ls=":", label=f"α={full['alpha']}")
    ax.set_xticks(x)
    ax.set_xticklabels([models[k]["model"] for k in keys], rotation=15)
    ax.set_title("Model-agnosticism: deployed gate across base models")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_model_agnosticism.png", dpi=150)
    plt.close(fig)


def fig_weight_sensitivity(abl: dict, fig_dir: Path) -> None:
    """FS AUROC envelope across fusion weights vs the default + best-hand."""
    ws = abl["weight_sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["min", "mean", "max"], [ws["auroc_min"], ws["auroc_mean"], ws["auroc_max"]],
           color="#8ca0c8", width=0.5)
    ax.axhline(ws["default_weights_auroc"], color="green", ls="--",
               label=f"default weights ({ws['default_weights_auroc']:.3f})")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Fused-FS AUROC")
    ax.set_title("FS is robust to fusion weights\n(learned verifier target = beat the max)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_weight_sensitivity.png", dpi=150)
    plt.close(fig)


def fig_abstention_power(abl: dict, fig_dir: Path) -> None:
    """Min calibration size needed to certify each target error rate."""
    power = abl["abstention_power"]["min_cal_size_for_alpha"]
    alphas = sorted(power, key=float)
    ys = [power[a] if power[a] is not None else np.nan for a in alphas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(a) for a in alphas], ys, "-o")
    ax.set_xlabel("Target error rate α")
    ax.set_ylabel("Min calibration size to certify")
    ax.set_title("Conformal power: tighter guarantees need more calibration data")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_abstention_power.png", dpi=150)
    plt.close(fig)


def fig_abstention_baselines(ab: dict, fig_dir: Path) -> None:
    """Head-to-head AURC per abstention trigger, grouped by data source."""
    srcs = ab.get("sources", {})
    src_names = [s for s in srcs if srcs[s].get("triggers")]
    if not src_names:
        print("[figures] skip fig9: no abstention-baseline data")
        return
    trig_order = ["fs_ours", "confidence", "self_consistency", "radflag", "signal_a", "signal_b"]
    trigs = [t for t in trig_order if any(t in srcs[s]["triggers"] for s in src_names)]
    colors = {"fs_ours": "#16a34a", "confidence": "#f59e0b", "self_consistency": "#db2777",
              "radflag": "#a855f7", "signal_a": "#6366f1", "signal_b": "#0e7490"}
    x = np.arange(len(src_names)); w = 0.8 / max(1, len(trigs))
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, t in enumerate(trigs):
        vals = [srcs[s]["triggers"].get(t, {}).get("aurc", np.nan) for s in src_names]
        ax.bar(x + j * w, vals, w, label=t, color=colors.get(t, "#888"))
    ax.set_xticks(x + w * (len(trigs) - 1) / 2)
    ax.set_xticklabels(src_names, rotation=10)
    ax.set_ylabel("AURC (lower = better)")
    ax.set_title("Abstention triggers head-to-head — FS (ours) vs confidence / self-consistency / RadFlag")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_abstention_baselines.png", dpi=150)
    plt.close(fig)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    fig_dir = ensure_dir(out / "figures")
    blind = _load(out, "blind_test.json")
    results = _load(out, "fmr_results.json")
    records = _load(out, "fmr_records.json")
    full = _load(out, "full_benchmark.json")
    abl = _load(out, "ablations.json")
    abst_base = _load(out, "abstention_baselines.json")
    if blind:
        fig_drift(blind, fig_dir)
        fig_blind(blind, fig_dir)
    if records:
        fig_signal_separation(records, fig_dir)
    if results:
        fig_risk_coverage(results, fig_dir)
    if full:
        fig_incremental_fusion(full, fig_dir)
        fig_model_agnosticism(full, fig_dir)
    if abl:
        fig_weight_sensitivity(abl, fig_dir)
        fig_abstention_power(abl, fig_dir)
    if abst_base:
        fig_abstention_baselines(abst_base, fig_dir)
    print(f"[figures] wrote figures to {fig_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fmr/outputs")
    args = ap.parse_args()
    main(args.out)
