"""Optional trained-enhancement track (Instance B).

The **learned faithfulness verifier**: a small trained head that fuses the
per-signal faithfulness measurements (Signals A/B/C + cheap auxiliaries) into a
grounding probability, *replacing* the hand-weighted heuristic fusion — but only
as a drop-in, reversible upgrade. The heuristic fusion (``HeuristicFusion``) is
always available and is the guaranteed fallback (external-review fix #4): the
rest of the pipeline never hard-depends on the trained head.

Dependency note (DECISIONS.md [B]): the *real* per-signal scores are owned by
Instance A (`faithfulness/score.py`, Signals B/C). Until that lands, features are
computed by ``training.signals`` from committed code (counterfactual signal +
region geometry + self-consistency sampling), which mirrors the intended
per-signal interface. Rewire ``build_feature_frame`` to Instance A's API when
present — the verifier, labels, and eval code are agnostic to the feature source.
"""
from .verifier import (
    HeuristicFusion,
    LearnedVerifier,
    FEATURE_KEYS,
    SIGNAL_KEYS,
)
from .signals import compute_sample_features, SampleFeatures
from .labels import weak_label_counterfactual, true_latent
from .dataset import build_feature_frame

__all__ = [
    "HeuristicFusion",
    "LearnedVerifier",
    "FEATURE_KEYS",
    "SIGNAL_KEYS",
    "compute_sample_features",
    "SampleFeatures",
    "weak_label_counterfactual",
    "true_latent",
    "build_feature_frame",
]
