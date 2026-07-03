"""Assemble the verifier's feature frame from one or more VLM backends.

Mixing backends is intentional: the ``mock-reasoner`` (image-blind pathology) and
``mock-prior-heavy`` (prior-dominated, where Signal A is *misleadingly low* on
genuinely image-grounded steps) create signal-conflict regions where a fixed
linear weighting of A/B/C is provably suboptimal — the setting in which a learned
fusion can actually win. Using a single well-behaved backend would make the
heuristic look artificially unbeatable and hide the effect the thesis claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..models.base_vlm import BaseVLM
from ..types import Sample
from .labels import true_latent, weak_label_counterfactual
from .signals import compute_sample_features
from .verifier import FEATURE_KEYS, vectorize


@dataclass
class FeatureFrame:
    X: np.ndarray                 # (n, n_features)
    feats: list[dict]             # raw feature dicts (for heuristic scoring)
    y_true: np.ndarray            # true latent (eval only)
    y_weak: np.ndarray            # counterfactual weak label (training)
    groups: list[str]             # backend name per row
    sample_ids: list[str]
    keys: tuple = FEATURE_KEYS


def build_feature_frame(
    vlms: Sequence[BaseVLM],
    samples: Sequence[Sample],
    n_chains: int = 4,
    noise: float = 0.0,
) -> FeatureFrame:
    rows, feats, yt, yw, groups, ids = [], [], [], [], [], []
    for vlm in vlms:
        for s in samples:
            sf = compute_sample_features(vlm, s, n_chains=n_chains, noise=noise)
            rows.append(vectorize(sf.features))
            feats.append(sf.features)
            yt.append(true_latent(s))
            yw.append(weak_label_counterfactual(sf))
            groups.append(vlm.name)
            ids.append(f"{vlm.name}:{s.sample_id}")
    return FeatureFrame(
        X=np.vstack(rows),
        feats=feats,
        y_true=np.array(yt),
        y_weak=np.array(yw),
        groups=groups,
        sample_ids=ids,
    )
