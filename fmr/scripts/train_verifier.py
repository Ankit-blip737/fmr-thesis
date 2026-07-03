"""Train + benchmark the learned faithfulness verifier vs the heuristic fusion.

Design (honest experiment, per the proposal):
  * Features come from TWO mock backends (image-blind + prior-dominated) so the
    signals genuinely conflict on some samples.
  * The verifier trains on the NOISY counterfactual weak label (no GT boxes).
  * Everyone is evaluated against the TRUE hidden latent on a held-out split that
    is disjoint from train AND from the calibration split reserved for Instance
    A's conformal gate.
  * Because real faithfulness signals are noisy (not the clean latent a
    deterministic mock exposes), we sweep a measurement-noise level and report the
    heuristic-vs-learned AUROC curve. This tests the thesis's actual claim: a
    *learned* fusion of noisy signals should degrade more gracefully than fixed
    hand-weighting. The winner at the headline noise level ships as default; if
    the heuristic wins, it stays — reported as an honest negative result (fix #4).

Usage: python fmr/scripts/train_verifier.py [--n 400] [--n-chains 4]
                 [--noise-sweep 0,0.15,0.3,0.45] [--headline-noise 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from fmr.data import build_synthetic_dataset, split_dataset  # noqa: E402
from fmr.models import MockVLM  # noqa: E402
from fmr.models.second_vlm import PriorHeavyMockVLM  # noqa: E402
from fmr.training import (  # noqa: E402
    HeuristicFusion,
    LearnedVerifier,
    SIGNAL_KEYS,
    build_feature_frame,
)


def _auc(y, s) -> tuple[float, float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def run_at_noise(tr_samples, te_samples, vlms, noise, n_chains, seed, save_model_to=None):
    tr = build_feature_frame(vlms, tr_samples, n_chains=n_chains, noise=noise)
    te = build_feature_frame(vlms, te_samples, n_chains=n_chains, noise=noise)

    heur = HeuristicFusion()
    s_heur = heur.score_batch(te.feats)
    heur_auroc, heur_auprc = _auc(te.y_true, s_heur)

    per_signal = {}
    for k in SIGNAL_KEYS:
        s = np.array([f[k] for f in te.feats])
        per_signal[k] = _auc(te.y_true, s)[0]

    learned = {}
    for kind in ("logreg", "gbt"):
        v = LearnedVerifier(model_kind=kind).fit(tr.X, tr.y_weak, seed=seed)
        s = v.score_batch(te.feats)
        auroc, auprc = _auc(te.y_true, s)
        learned[kind] = {
            "auroc": auroc, "auprc": auprc,
            "pearson_vs_heuristic": float(np.corrcoef(s, s_heur)[0, 1]),
            "label_agreement_vs_heuristic@0.5": float(np.mean((s >= 0.5) == (s_heur >= 0.5))),
            "importance": v.feature_importance(),
        }
        if save_model_to and kind == "gbt":
            v.save(save_model_to)

    oracle = LearnedVerifier(model_kind="gbt").fit(tr.X, tr.y_true, seed=seed)
    o_auroc, _ = _auc(te.y_true, oracle.score_batch(te.feats))

    best_kind = max(learned, key=lambda k: learned[k]["auroc"])
    best_auroc = learned[best_kind]["auroc"]

    return {
        "noise": noise,
        "weak_vs_true_agreement(train)": float(np.mean(tr.y_weak == tr.y_true)),
        "per_signal_auroc": per_signal,
        "heuristic_auroc": heur_auroc, "heuristic_auprc": heur_auprc,
        "learned": learned,
        "oracle_auroc": o_auroc,
        "best_learned_kind": best_kind, "best_learned_auroc": best_auroc,
        "learned_minus_heuristic": best_auroc - heur_auroc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "verifier.yaml"),
                    help="YAML defaults (CLI flags override); missing file is fine")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-chains", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--noise-sweep", default=None)
    ap.add_argument("--headline-noise", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    # Resolve config defaults <- YAML, then apply any CLI overrides.
    cfg = {}
    if Path(args.config).exists():
        from fmr.utils import load_config
        cfg = load_config(args.config).get("verifier", {}) or {}
    args.n = args.n if args.n is not None else int(cfg.get("n_samples", 400))
    args.n_chains = args.n_chains if args.n_chains is not None else int(cfg.get("n_chains", 4))
    args.seed = args.seed if args.seed is not None else int(cfg.get("seed", 7))
    args.headline_noise = (args.headline_noise if args.headline_noise is not None
                           else float(cfg.get("headline_noise", 0.3)))
    args.margin = args.margin if args.margin is not None else float(cfg.get("margin", 0.01))
    if args.noise_sweep is not None:
        pass
    elif cfg.get("noise_sweep"):
        args.noise_sweep = ",".join(str(x) for x in cfg["noise_sweep"])
    else:
        args.noise_sweep = "0,0.15,0.3,0.45"

    samples = build_synthetic_dataset(n=args.n, seed=args.seed)
    parts = split_dataset(samples, fractions=(0.5, 0.25, 0.25), seed=13)
    vlms = [MockVLM(), PriorHeavyMockVLM()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    noises = [float(x) for x in args.noise_sweep.split(",")]
    if args.headline_noise not in noises:
        noises.append(args.headline_noise)
    noises = sorted(set(noises))

    print(f"Backends={[v.name for v in vlms]}  train={len(parts['train'])} "
          f"cal(reserved for A)={len(parts['cal'])} test={len(parts['test'])}")
    print(f"Noise sweep: {noises}")

    sweep = []
    for nz in noises:
        save_to = (out_dir / "verifier_gbt.pkl") if nz == args.headline_noise else None
        r = run_at_noise(parts["train"], parts["test"], vlms, nz, args.n_chains, args.seed, save_to)
        sweep.append(r)
        print(f"  noise={nz:>4}: heuristic={r['heuristic_auroc']:.3f}  "
              f"learned({r['best_learned_kind']})={r['best_learned_auroc']:.3f}  "
              f"d={r['learned_minus_heuristic']:+.3f}  oracle={r['oracle_auroc']:.3f}  "
              f"per-signal A/B/C={r['per_signal_auroc']['sig_a_counterfactual']:.2f}/"
              f"{r['per_signal_auroc']['sig_b_grounding']:.2f}/"
              f"{r['per_signal_auroc']['sig_c_consistency']:.2f}")

    headline = next(r for r in sweep if r["noise"] == args.headline_noise)
    ships_learned = headline["learned_minus_heuristic"] >= args.margin

    report = {
        "config": {"n_samples": args.n, "n_chains": args.n_chains,
                   "backends": [v.name for v in vlms],
                   "headline_noise": args.headline_noise, "margin": args.margin},
        "splits": {"train_rows": 2 * len(parts["train"]), "test_rows": 2 * len(parts["test"]),
                   "cal_reserved_for_A": len(parts["cal"]),
                   "test_true_pos_rate": float(np.mean(
                       [1 if s.meta.get("grounded") else 0 for s in parts["test"]]))},
        "weak_label_source": "counterfactual flip_rate>=0.5 (noisy; no GT boxes)",
        "noise_sweep": sweep,
        "headline": {
            "noise": args.headline_noise,
            "heuristic_auroc": headline["heuristic_auroc"],
            "best_learned_kind": headline["best_learned_kind"],
            "best_learned_auroc": headline["best_learned_auroc"],
            "delta": headline["learned_minus_heuristic"],
            "oracle_auroc": headline["oracle_auroc"],
            "agreement_vs_heuristic@0.5":
                headline["learned"][headline["best_learned_kind"]]["label_agreement_vs_heuristic@0.5"],
            "ships_learned_as_default": bool(ships_learned),
            "verdict": ("LEARNED VERIFIER WINS at headline noise -> ships as default"
                        if ships_learned else
                        "heuristic retained at headline noise (learned did not clear margin) "
                        "— honest negative result; fallback stands"),
        },
    }
    (out_dir / "verifier_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
