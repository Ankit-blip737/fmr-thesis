"""Stage 2a — base-model baselines: VQA accuracy per model, per modality.

Usage:
    python fmr/scripts/run_baselines.py [--models mock_reasoner mock_plain]
                                        [--split test] [--out fmr/outputs]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from _common import load_all_configs, resolve_dataset_and_splits, resolve_vlm

from fmr.utils import save_json


def run(models: list[str], split: str, out_dir: str, config_dir: str | None = None) -> dict:
    cfg = load_all_configs(config_dir)
    _, splits = resolve_dataset_and_splits(cfg["data"])
    eval_set = splits[split]

    results: dict = {"dataset": cfg["data"].get("dataset"), "split": split, "n": len(eval_set), "models": {}}
    for key in models:
        vlm = resolve_vlm(cfg["models"], key)
        per_mod: dict[str, list[int]] = defaultdict(list)
        correct = []
        for s in eval_set:
            out = vlm.generate(s, variant="original", temperature=0.0)
            hit = int(out.answer == s.answer)
            correct.append(hit)
            per_mod[s.modality].append(hit)
        results["models"][key] = {
            "name": vlm.name,
            "is_reasoning": vlm.is_reasoning,
            "accuracy": sum(correct) / len(correct),
            "per_modality": {m: sum(v) / len(v) for m, v in sorted(per_mod.items())},
        }
        print(f"[baselines] {key:>16s}  acc={results['models'][key]['accuracy']:.3f}  "
              f"per-modality={results['models'][key]['per_modality']}")

    path = save_json(results, f"{out_dir}/baselines.json")
    print(f"[baselines] wrote {path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mock_reasoner", "mock_plain"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="fmr/outputs")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()
    run(args.models, args.split, args.out, args.config_dir)
