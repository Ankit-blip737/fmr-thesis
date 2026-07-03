"""Shared plumbing for the run scripts: config loading, model/data resolution."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow running the scripts without `pip install -e` (e.g. straight from a clone).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmr.data.loaders import load_dataset, split_dataset  # noqa: E402
from fmr.models.base_vlm import load_vlm  # noqa: E402
from fmr.utils import load_config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_all_configs(config_dir: str | Path | None = None) -> dict[str, dict]:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    return {
        "data": load_config(d / "data.yaml"),
        "models": load_config(d / "models.yaml"),
        "experiment": load_config(d / "experiment.yaml"),
    }


def resolve_dataset_and_splits(data_cfg: dict) -> tuple[list, dict[str, list]]:
    name = data_cfg.get("dataset", "synthetic")
    samples = load_dataset(dict(data_cfg[name]))
    split_cfg = data_cfg.get("split", {})
    splits = split_dataset(
        samples,
        fractions=tuple(split_cfg.get("fractions", (0.5, 0.25, 0.25))),
        holdout_modality=split_cfg.get("holdout_modality"),
        seed=int(split_cfg.get("seed", 13)),
    )
    return samples, splits


def resolve_vlm(models_cfg: dict, key: str | None = None, samples: list | None = None) -> Any:
    key = key or models_cfg.get("model", "mock_reasoner")
    vlm = load_vlm(dict(models_cfg[key]))
    # Real HF backends need a pool of images to build the 'mismatch' variant.
    if samples and hasattr(vlm, "set_mismatch_pool"):
        pool = [s.image for s in samples if s.image is not None][:64]
        if pool:
            vlm.set_mismatch_pool(pool)
    return vlm


def accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    return sum(p == g for p, g in pairs) / len(pairs)
