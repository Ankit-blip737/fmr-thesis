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

Resume support (for GPU-limit recovery across Colab sessions):
  --skip-if-done   skip the entire dataset if run_status.json shows all stages ok.
                   Use this to safely re-run all three datasets; completed ones
                   finish in <1 second.
  --resume         skip individual stages already marked "ok" in run_status.json.
                   Use this to continue a partially-completed dataset after a
                   GPU-limit interruption.

Example (new Colab session after hitting GPU limit mid-PathVQA):
    # vqa_rad already done — skip it instantly
    python fmr/scripts/run_real.py --dataset vqa_rad --skip-if-done --push
    # pathvqa was interrupted — resume from where it stopped
    python fmr/scripts/run_real.py --dataset pathvqa --resume --push
    # slake not started yet — run fully
    python fmr/scripts/run_real.py --dataset slake --push
"""
from __future__ import annotations

import argparse
import copy
import json
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


def _load_existing_status(out: str) -> dict:
    """Load run_status.json if it exists; otherwise infer from output files."""
    p = Path(out) / "run_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("status", {})
        except Exception:
            pass
    # Infer from files if status JSON is missing (e.g. crashed mid-run)
    status = {}
    if (Path(out) / "baselines.json").exists():
        status["baselines"] = "ok"
    if (Path(out) / "blind_test.json").exists():
        status["blind_test"] = "ok"
    if (Path(out) / "fmr_results.json").exists():
        status["fmr"] = "ok"
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="slake")
    ap.add_argument("--reasoning", default="medvlm_r1",
                    help="models.yaml key for the reasoning VLM")
    ap.add_argument("--non-reasoning", default="qwen25_vl_3b",
                    help="models.yaml key for the non-reasoning VLM")
    ap.add_argument("--model", default=None,
                    help="single model for run_fmr (defaults to --reasoning)")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--n", type=int, default=None, help="synthetic size override")
    ap.add_argument("--n-consistency", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-root", default=None,
                    help="dir of extracted images (SLAKE needs this)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--push", action="store_true",
                    help="git commit+push outputs to origin")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="skip entire dataset if all stages already ok in run_status.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip individual stages already marked ok in run_status.json")
    args = ap.parse_args()

    out = f"fmr/outputs/real/{args.dataset}"
    Path(out).mkdir(parents=True, exist_ok=True)

    # --- Resume / skip logic ------------------------------------------------
    existing_status = _load_existing_status(out)
    all_stages = ["baselines", "blind_test", "fmr"]

    if args.skip_if_done:
        if all(existing_status.get(s) == "ok" for s in all_stages):
            print(f"[run_real] {args.dataset}: all stages already OK — "
                  f"skipping (--skip-if-done).")
            return

    cfg_dir = _write_configs(args.dataset, args.model, args.reasoning, args.non_reasoning,
                             args.max_samples, args.n, args.split, args.image_root)

    if args.n_consistency is not None:
        exp = yaml.safe_load((cfg_dir / "experiment.yaml").read_text())
        exp["signals"]["consistency"]["n_samples"] = args.n_consistency
        (cfg_dir / "experiment.yaml").write_text(yaml.safe_dump(exp))

    print(f"[run_real] dataset={args.dataset} reasoning={args.reasoning} "
          f"non_reasoning={args.non_reasoning} out={out}")

    if args.resume and existing_status:
        done = [s for s in all_stages if existing_status.get(s) == "ok"]
        if done:
            print(f"[run_real] --resume: carrying forward already-ok stages: {done}")

    def _push(stage: str) -> None:
        if not args.push:
            return
        try:
            import make_dashboard          # rebuild the dashboard bundle from all
            make_dashboard.build()          # current outputs (mock + real) before push
        except Exception as exc:
            print(f"[run_real] dashboard rebuild before {stage} failed (non-fatal): {exc}")
        try:
            from push_results import push
            push(out, message=f"[A] real results ({stage}): {args.dataset} "
                              f"({args.reasoning} vs {args.non_reasoning})")
        except Exception as exc:  # never let a push failure abort the run
            print(f"[run_real] push after {stage} failed (non-fatal): {exc}")

    # Each stage runs independently and pushes its own outputs the moment it
    # succeeds. A CUDA OOM / timeout in a later (heavier) stage therefore cannot
    # discard the results of an earlier one — critically, the blind-test HEADLINE
    # (grounding-drift replication verdict) survives even if the full FMR stage
    # dies. Stages are ordered cheapest -> heaviest for exactly this reason.
    status: dict[str, str] = dict(existing_status)  # carry forward existing ok stages
    stages = [
        ("baselines", lambda: run_baselines.run(
            [args.reasoning, args.non_reasoning], args.split, out,
            config_dir=str(cfg_dir))),
        ("blind_test", lambda: run_blind_test.run(
            args.split, out, config_dir=str(cfg_dir))),
        ("fmr", lambda: run_fmr.run(
            out, alpha=args.alpha, delta=args.delta, post_correction=False,
            config_dir=str(cfg_dir))),
    ]
    for name, fn in stages:
        # --resume: skip stages already marked ok in the previous run
        if args.resume and status.get(name) == "ok":
            print(f"[run_real] --resume: stage {name!r} already ok, skipping.")
            continue
        try:
            fn()
            status[name] = "ok"
        except Exception as exc:
            import traceback
            status[name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[run_real] stage {name!r} FAILED (continuing): {exc}")
            traceback.print_exc()
        # Regenerate figures from whatever JSON exists so far, then push.
        try:
            make_figures.main(out)
        except Exception as exc:
            print(f"[run_real] make_figures after {name} failed (non-fatal): {exc}")
        _push(name)

    # Persist a machine-readable run status alongside the outputs.
    from fmr.utils import save_json
    save_json({"dataset": args.dataset, "reasoning": args.reasoning,
               "non_reasoning": args.non_reasoning, "max_samples": args.max_samples,
               "n_consistency": args.n_consistency, "status": status},
              f"{out}/run_status.json")
    _push("status")
    print(f"[run_real] done. stage status: {status}")


if __name__ == "__main__":
    main()
