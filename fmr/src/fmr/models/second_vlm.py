"""Second base VLM (Instance B) — for the model-agnosticism / cross-model axis.

Two backends behind ``load_second_vlm``:

* ``mock_prior`` (default, offline) — :class:`PriorHeavyMockVLM`, a synthetic
  model with a *different pathology* than Instance A's ``MockVLM``. MockVLM's
  ungrounded mode is fully image-blind (no decode-time method can help — those
  cases belong to abstention). PriorHeavyMockVLM instead *sees* the evidence but
  lets a language prior out-vote it: the prior boost is present under every
  image variant, while a weaker ground-truth boost appears only with the real
  image. That is exactly the failure VCD provably repairs (the prior term
  cancels in the contrast; the evidence term is amplified), so the cross-model
  comparison exercises both "fixable" and "must-abstain" pathologies.

* ``hf`` — :class:`SecondHFVLM`, a thin scaffold for a second *real* model
  family (default MedGemma; license-gated — see BLOCKERS.md). Same ``generate``
  contract as Instance A's ``HFVLM``; filled in on GPU via the handoff notebook.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Sample
from ..utils import rng_from
from .base_vlm import BaseVLM
from .mock_vlm import MockVLM


class PriorHeavyMockVLM(MockVLM):
    """Synthetic reasoning VLM dominated by its language prior.

    Subclasses Instance A's ``MockVLM`` (read-only reuse — see DECISIONS.md [B])
    and overrides only the answer-preference mechanism plus defaults: a longer,
    faster-drifting chain, and preferences of the form

        pref = prior_peak * onehot(prior) [all variants]
             + evidence_peak * (0.5 + 0.5*strength) * onehot(gt) [original & grounded]

    with ``evidence_peak * (0.5 + 0.5*strength) < prior_peak``, so the greedy
    answer follows the prior even though genuine image evidence is present.
    """

    def __init__(
        self,
        name: str = "mock-prior-heavy",
        is_reasoning: bool = True,
        n_steps: int | None = 5,
        drift: float = 0.5,
        prior_peak: float = 1.8,
        evidence_peak: float = 1.4,
        seed: int = 1,
    ) -> None:
        super().__init__(
            name=name,
            is_reasoning=is_reasoning,
            n_steps=n_steps,
            drift=drift,
            prior_peak=prior_peak,
            seed=seed,
        )
        self.evidence_peak = evidence_peak

    def _pref(self, sample: Sample, variant: str) -> np.ndarray:
        vocab = sample.answer_choices or [sample.answer]
        v = len(vocab)
        gt_idx = vocab.index(sample.answer) if sample.answer in vocab else 0
        prior_idx = int(sample.meta.get("prior_idx", gt_idx))
        grounded = int(sample.meta.get("grounded", 1))
        strength = float(sample.meta.get("ground_strength", 1.0))

        rng = rng_from(self.seed, sample.sample_id, variant, "pref-prior-heavy")
        pref = rng.normal(0.0, 0.25, size=v)

        # The language prior fires regardless of what the image shows.
        pref[prior_idx] += self.prior_peak
        # Latent image evidence: real, but too weak to win the greedy argmax.
        if variant == "original" and grounded:
            pref[gt_idx] += self.evidence_peak * (0.5 + 0.5 * strength)
        return pref


class SecondHFVLM:
    """Scaffold for the second real model family (see ``HFVLM`` for the notes).

    Kept deliberately parallel to Instance A's ``HFVLM``: same constructor
    shape, same ``generate`` contract, different default checkpoint so the
    Stage 6 cross-model comparison spans two model families.
    """

    is_reasoning = True

    def __init__(self, model_id: str = "google/medgemma-4b-it", device: str = "cuda", **kw: Any) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: F401
        except Exception as exc:  # pragma: no cover - only hit without the deps
            raise ImportError(
                "The 'hf' backend for the second VLM needs `transformers` + `torch` "
                "+ accepted model license (MedGemma is gated; see BLOCKERS.md). "
                "Use backend='mock_prior' for the offline pipeline."
            ) from exc
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self._kw = kw

    def generate(self, sample: Sample, variant: str = "original",
                 temperature: float = 0.0, draw: int = 0):  # pragma: no cover - requires weights
        raise NotImplementedError(
            "SecondHFVLM.generate is a hardware scaffold; the working implementation "
            "for GPU runs lives in fmr/notebooks/colab_stage4_correction_real.ipynb "
            "as an adapter over the same generate contract."
        )


def load_second_vlm(config: dict[str, Any] | None = None) -> BaseVLM:
    """Factory mirroring ``models.base_vlm.load_vlm`` for the second model.

    Kept separate so Instance A's factory file stays untouched; ``load_vlm``
    can dispatch here (backend='second') whenever we merge.
    """
    config = dict(config or {})
    backend = config.pop("backend", "mock_prior")

    if backend == "mock_prior":
        return PriorHeavyMockVLM(**config)
    if backend == "hf":
        return SecondHFVLM(**config)
    raise ValueError(f"Unknown second-VLM backend: {backend!r} (expected 'mock_prior' or 'hf').")
