"""GPU/Colab orchestrator — run the full FMR pipeline on REAL models + data.

This never runs on the CPU dev box (real-model inference). It is the single
entry point the Colab notebooks call. It:
  1. loads the base YAML configs and applies real overrides (dataset, models,
     sample caps) into a temp config dir,
  2. runs baselines -> blind test -> full FMR pipeline against that config,
  3. writes everything under fmr/outputs/real/<dataset>/,
  4. regenerates the figures.

Git commit+push back to the branch is handled by the notebook (or --push here),
so results land on the branch automatically per the GPU handoff protocol.

Example (inside Colab):
    python fmr/scripts/run_real.py \
        --dataset slake --reasoning medvlm_r1 --non-reasoning qwen25_vl_3b \
        --max-samples 400 --n-consistency 5 --push
"""
from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import yaml
from _common import CONFIG_DIR

import run_baselines
import run_blind_test
import run_fmr
import make_figures


def _write_configs(dataset: str, model_key: str, reasoning: str, non_reasoning: str,
                    max_samples: int | None, n: int | None, split: str,
                    image_root: str | None = None) -> Path:
    """Materialize an override config dir; returns its path."""
    base = {name: yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
            for name in ("data", "models", "experiment")}
    data, models = copy.deepcopy(base["data"]), copy.deepcopy(base["models"])

    data["dataset"] = dataset
    if dataset == "synthetic":
        if n is not None:
            data["synthetic"]["n"] = n
    elif dataset in data:  # real datasets accept max_samples + split
        if max_samples is not None:
            data[dataset]["max_samples"] = max_samples
        data[dataset]["split"] = split
        if image_root:
            data[dataset]["image_root"] = image_root

    models["model"] = model_key or reasoning
    models["comparison"] = {"reasoning_model": reasoning, "non_reasoning_model": non_reasoning}

    tmp = Path(tempfile.mkdtemp(prefix="fmr_real_cfg_"))
    (tmp / "data.yaml").write_text(yaml.safe_dump(data))
    (tmp / "models.yaml").write_text(yaml.safe_dump(models))
    (tmp / "experiment.yaml").write_text(yaml.safe_dump(base["experiment"]))
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="slake")
    ap.add_argument("--reasoning", default="medvlm_r1", help="models.yaml key for the reasoning VLM")
    ap.add_argument("--non-reasoning", default="qwen25_vl_3b", help="models.yaml key for the non-reasoning VLM")
    ap.add_argument("--model", default=None, help="single model for run_fmr (defaults to --reasoning)")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--n", type=int, default=None, help="synthetic size override")
    ap.add_argument("--n-consistency", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-root", default=None,
                    help="dir of extracted images (SLAKE needs this - mirror has no inline images)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--push", action="store_true", help="git commit+push outputs to origin")
    args = ap.parse_args()

    cfg_dir = _write_configs(args.dataset, args.model, args.reasoning, args.non_reasoning,
                             args.max_samples, args.n, args.split, args.image_root)
    out = f"fmr/outputs/real/{args.dataset}"
    Path(out).mkdir(parents=True, exist_ok=True)

    if args.n_consistency is not None:
        exp = yaml.safe_load((cfg_dir / "experiment.yaml").read_text())
        exp["signals"]["consistency"]["n_samples"] = args.n_consistency
        (cfg_dir / "experiment.yaml").write_text(yaml.safe_dump(exp))

    print(f"[run_real] dataset={args.dataset} reasoning={args.reasoning} "
          f"non_reasoning={args.non_reasoning} out={out}")
    run_baselines.run([args.reasoning, args.non_reasoning], args.split, out, config_dir=str(cfg_dir))
    run_blind_test.run(args.split, out, config_dir=str(cfg_dir))
    run_fmr.run(out, alpha=args.alpha, delta=args.delta, post_correction=False, config_dir=str(cfg_dir))
    make_figures.main(out)

    if args.push:
        from push_results import push
        push(out, message=f"[A] real results: {args.dataset} ({args.reasoning} vs {args.non_reasoning})")


if __name__ == "__main__":
    main()
