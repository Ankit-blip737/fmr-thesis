"""Grounding labels for the learned verifier — weak (training) and true (eval).

Per the proposal, grounding labels come from data the pipeline already produces,
with *no manual-annotation bottleneck*:

* ``weak_label_counterfactual`` — the training label. A step/sample is weakly
  "grounded" if its answer is counterfactually sensitive (flips when the image is
  removed/swapped). This needs no ground-truth boxes, so it transfers to datasets
  without region annotations. It is deliberately NOISY — that is the point: we
  train on a cheap noisy label and test whether a *learned fusion* still recovers
  the true latent better than a hand-weighted score.

* ``true_latent`` — the EVAL target only. On synthetic data the hidden ``grounded``
  latent is the ground truth for whether the reasoning genuinely used the image.
  On SLAKE/VQA-RAD this role is played by attention-region↔GT-box IoU. The
  verifier is scored (AUROC/AUPRC) against this, never trained on it (except in a
  clearly-labelled oracle-upper-bound ablation).
"""
from __future__ import annotations

from .signals import SampleFeatures
from ..types import Sample


def weak_label_counterfactual(feat: SampleFeatures, flip_threshold: float = 0.5) -> int:
    """Noisy training label from Signal-A counterfactual behaviour (no GT needed)."""
    return int(feat.features.get("a_flip_rate", 0.0) >= flip_threshold)


def weak_label_iou(feat: SampleFeatures, iou_threshold: float = 0.5) -> int:
    """Alternative weak label from Signal-B grounding (needs GT boxes / regions)."""
    return int(feat.features.get("b_iou_max", 0.0) >= iou_threshold)


def true_latent(sample: Sample) -> int:
    """Eval-only ground truth: is this sample's reasoning genuinely image-grounded?

    Synthetic: the hidden ``grounded`` latent. Real datasets should supply the
    IoU-derived label here instead (kept out of training).
    """
    return int(sample.meta.get("grounded", 0))
