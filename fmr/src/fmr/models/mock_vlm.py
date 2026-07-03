"""MockVLM — a fully offline, deterministic stand-in for a medical reasoning VLM.

Why this exists: the whole FMR pipeline is model-agnostic, but we still need to
exercise it end-to-end without a GPU or multi-GB checkpoints (for tests, CI, and
demos). MockVLM emits answers, per-step attended regions, and answer-logit
distributions whose behaviour is driven by a hidden per-sample *grounding*
latent (``sample.meta['grounded']``). That makes the three faithfulness signals
genuinely informative — each depends on the latent through an independent
mechanism — so the learned verifier really can learn to fuse them and beat any
single signal, and the reasoning-vs-non-reasoning gap is reproducible.

It is clearly labelled synthetic. No claim is made that these numbers reflect a
real clinical model; they exist to validate the *machinery*.
"""
from __future__ import annotations

import numpy as np

from ..types import Sample, Step, VLMOutput
from ..utils import rng_from, softmax
from ..data.regions import Region

_VARIANTS = ("original", "blank", "mismatch")


class MockVLM:
    """Deterministic synthetic reasoning VLM."""

    def __init__(
        self,
        name: str = "mock-reasoner",
        is_reasoning: bool = True,
        n_steps: int | None = None,
        drift: float = 0.35,
        grounded_peak: float = 3.5,
        prior_peak: float = 1.3,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.is_reasoning = is_reasoning
        # Reasoning models emit a longer chain; non-reasoning models answer in one step.
        self.n_steps = n_steps if n_steps is not None else (4 if is_reasoning else 1)
        # `drift` models the "more reasoning -> less grounded" effect: later steps
        # of a reasoning chain attend less to the image. Non-reasoning => no drift.
        self.drift = drift if is_reasoning else 0.0
        self.grounded_peak = grounded_peak
        self.prior_peak = prior_peak
        self.seed = seed

    # ---- internal: preference logits over the answer vocabulary -------------
    def _pref(self, sample: Sample, variant: str) -> np.ndarray:
        """Answer-preference logits driven by a binary ``grounded`` latent.

        Grounded samples get an image-dependent peak on the true answer that
        vanishes when the image is removed/swapped (so the answer flips —
        high Signal A); ungrounded samples lean on a fixed language prior that
        is unchanged by the image (no flip — low Signal A). ``ground_strength``
        modulates the grounded peak so grounded faithfulness still varies. The
        logit noise keeps signals imperfect but the classes are cleanly
        separable enough that "image-blind" cases are well-defined — which the
        correction / distillation / verifier logic downstream depends on.
        Realistic sub-1.0 AUROCs are demonstrated on the real-model data, not on
        this deliberately-clean machinery fixture.
        """
        vocab = sample.answer_choices or [sample.answer]
        v = len(vocab)
        gt_idx = vocab.index(sample.answer) if sample.answer in vocab else 0
        prior_idx = int(sample.meta.get("prior_idx", gt_idx))
        grounded = int(sample.meta.get("grounded", 1))
        strength = float(sample.meta.get("ground_strength", 1.0))

        rng = rng_from(self.seed, sample.sample_id, variant, "pref")
        pref = rng.normal(0.0, 0.25, size=v)

        image_present = variant == "original"
        if image_present:
            if grounded:
                pref[gt_idx] += self.grounded_peak * (0.5 + 0.5 * strength)
            else:
                pref[prior_idx] += self.prior_peak
        else:
            # Image removed / swapped.
            if grounded:
                pass  # diffuse: relied on the (now-gone) image -> near-uniform
            else:
                pref[prior_idx] += self.prior_peak  # ignores image -> unchanged
        return pref

    def _answer(self, sample: Sample, pref: np.ndarray, temperature: float, draw: int) -> tuple[str, int]:
        vocab = sample.answer_choices or [sample.answer]
        if temperature <= 0:
            idx = int(np.argmax(pref))
        else:
            rng = rng_from(self.seed, sample.sample_id, draw, "ans")
            idx = int(rng.choice(len(vocab), p=softmax(pref, temperature)))
        return vocab[idx], idx

    def _steps(self, sample: Sample, variant: str, answer: str) -> list[Step]:
        grounded = int(sample.meta.get("grounded", 1))
        gt = sample.gt_region
        rng = rng_from(self.seed, sample.sample_id, variant, "steps")
        n_rows = n_cols = int(sample.meta.get("grid", 4))
        steps: list[Step] = []
        for k in range(self.n_steps):
            # Later steps of a reasoning chain drift away from the evidence region.
            drift_k = self.drift * (k / max(1, self.n_steps - 1))
            grounded_here = grounded and variant == "original" and rng.random() > drift_k
            if grounded_here and gt is not None:
                # Jitter tightly around the ground-truth cell -> high IoU.
                jx = rng.normal(0, 0.05)
                jy = rng.normal(0, 0.05)
                region = Region(gt.x0 + jx, gt.y0 + jy, gt.x1 + jx, gt.y1 + jy)
            else:
                r, c = rng.integers(0, n_rows), rng.integers(0, n_cols)
                region = Region.from_grid_cell(int(r), int(c), n_rows, n_cols)
            term = sample.meta.get("finding", "finding")
            text = f"Step {k + 1}: the {term} in this region supports '{answer}'."
            steps.append(Step(text=text, terms=[term], pred_region=region))
        return steps

    def generate(
        self,
        sample: Sample,
        variant: str = "original",
        temperature: float = 0.0,
        draw: int = 0,
    ) -> VLMOutput:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
        pref = self._pref(sample, variant)
        answer, _ = self._answer(sample, pref, temperature, draw)
        logits = softmax(pref)  # reported answer distribution
        steps = self._steps(sample, variant, answer)
        return VLMOutput(
            sample_id=sample.sample_id,
            answer=answer,
            steps=steps,
            answer_logits=logits,
            variant=variant,
            meta={"model": self.name, "is_reasoning": self.is_reasoning},
        )
