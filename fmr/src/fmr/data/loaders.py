"""Unified dataset loading + splitting.

``load_dataset`` returns a list of :class:`Sample` regardless of source, so the
harness code never branches on dataset identity. The synthetic source always
works offline; the real loaders (VQA-RAD, SLAKE, PathVQA, OmniMedVQA) are
gated behind file presence and raise a clear message when the data is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..types import Sample
from .synthetic import build_synthetic_dataset


def load_dataset(config: dict[str, Any] | None = None) -> list[Sample]:
    config = dict(config or {})
    name = config.pop("name", "synthetic")

    if name == "synthetic":
        return build_synthetic_dataset(**config)
    if name in ("vqa_rad", "slake", "pathvqa", "omnimedvqa"):
        return _load_real(name, config)
    raise ValueError(f"Unknown dataset {name!r}.")


def _load_real(name: str, config: dict[str, Any]) -> list[Sample]:
    root = config.get("root")
    if not root or not Path(root).exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at root={root!r}. Download it and set "
            f"`data.yaml:{name}.root`, or use name='synthetic' for the offline pipeline. "
            "Loader parsing for the real annotation format goes here (question/answer/"
            "image path and, for SLAKE/VQA-RAD, the bounding-box -> Region mapping)."
        )
    # Real parsing intentionally left as a typed stub — the schema differs per
    # dataset and requires the downloaded files to implement against.
    raise NotImplementedError(f"Parsing for '{name}' not yet implemented; data present at {root}.")


def split_dataset(
    samples: list[Sample],
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    holdout_modality: str | None = None,
    seed: int = 13,
) -> dict[str, list[Sample]]:
    """Split into train / calibration / test.

    ``train`` feeds the learned verifier, ``cal`` feeds the split-conformal gate,
    ``test`` is held out for reporting. If ``holdout_modality`` is set, all of
    that modality is moved into a separate ``holdout`` split to test
    generalization to an unseen modality (per the proposal's held-out experiment).
    """
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)

    pool = list(samples)
    holdout: list[Sample] = []
    if holdout_modality is not None:
        holdout = [s for s in pool if s.modality == holdout_modality]
        pool = [s for s in pool if s.modality != holdout_modality]

    idx = rng.permutation(len(pool))
    n_train = int(fractions[0] * len(pool))
    n_cal = int(fractions[1] * len(pool))
    train = [pool[i] for i in idx[:n_train]]
    cal = [pool[i] for i in idx[n_train:n_train + n_cal]]
    test = [pool[i] for i in idx[n_train + n_cal:]]

    out = {"train": train, "cal": cal, "test": test}
    if holdout_modality is not None:
        out["holdout"] = holdout
    return out
