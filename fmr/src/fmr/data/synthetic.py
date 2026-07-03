"""Synthetic medical-VQA dataset for offline runs.

Each sample carries a hidden ``grounded`` latent and a ground-truth evidence
region. The MockVLM's behaviour is driven by that latent, so the faithfulness
signals become predictive of it — which is exactly what lets us validate the
measurement module, the learned verifier, and the conformal gate without any
real model or data download.

The four modalities mirror the real study (X-ray / CT / MRI / pathology) so the
per-modality reporting code has something to group by.
"""
from __future__ import annotations

import numpy as np

from ..types import Sample
from .regions import Region

MODALITIES = ("xray", "ct", "mri", "pathology")
FINDINGS = {
    "xray": ["opacity", "effusion", "cardiomegaly", "nodule"],
    "ct": ["lesion", "hemorrhage", "mass", "infarct"],
    "mri": ["lesion", "edema", "atrophy", "enhancement"],
    "pathology": ["mitosis", "necrosis", "tumor cells", "inflammation"],
}
# Closed-form answer vocabulary (keeps metrics clean, per the proposal).
ANSWER_VOCAB = ["yes", "no", "mild", "moderate", "severe", "absent"]


def build_synthetic_dataset(
    n: int = 400,
    grounded_fraction: float = 0.5,
    grid: int = 4,
    seed: int = 7,
) -> list[Sample]:
    """Return ``n`` synthetic samples with hidden grounding latents + GT regions."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for i in range(n):
        modality = MODALITIES[i % len(MODALITIES)]
        finding = str(rng.choice(FINDINGS[modality]))
        gt_idx = int(rng.integers(0, len(ANSWER_VOCAB)))
        answer = ANSWER_VOCAB[gt_idx]

        grounded = int(rng.random() < grounded_fraction)
        # Grounded faithfulness varies (modulates the image peak in MockVLM);
        # ungrounded cases ignore the image entirely (binary latent). Signal
        # imperfection on this fixture comes from logit noise + reasoning drift,
        # keeping "image-blind" cleanly defined for the correction / distillation
        # / verifier logic while still not perfectly separable.
        strength = float(rng.uniform(0.5, 1.0))
        # Ungrounded items lean on a language prior that is sometimes right by luck.
        prior_idx = gt_idx if rng.random() < 0.4 else int(rng.integers(0, len(ANSWER_VOCAB)))

        r, c = int(rng.integers(0, grid)), int(rng.integers(0, grid))
        gt_region = Region.from_grid_cell(r, c, grid, grid)

        samples.append(
            Sample(
                sample_id=f"syn-{i:05d}",
                question=f"Is there {finding} present, and how severe?",
                answer=answer,
                modality=modality,
                image=None,
                gt_region=gt_region,
                answer_choices=list(ANSWER_VOCAB),
                meta={
                    "grounded": grounded,          # hidden latent (eval only)
                    "ground_strength": strength,
                    "prior_idx": prior_idx,
                    "finding": finding,
                    "grid": grid,
                    "synthetic": True,
                },
            )
        )
    return samples
