"""Signal C - Self-consistency (RadFlag-style).

Sample N reasoning chains at non-zero temperature and measure agreement. A model
whose answer is pinned down by image evidence produces the same answer across
samples; a model guessing from a weak language prior scatters. Score = vote
share of the modal answer, rescaled so that chance-level agreement maps to ~0.

This signal is model-agnostic and needs nothing but repeated ``generate`` calls,
which is also why it is the most expensive signal (N extra forward passes) —
the config exposes ``n_samples`` so the cost/quality trade-off is explicit.
"""
from __future__ import annotations

from collections import Counter

from ..models.base_vlm import BaseVLM
from ..types import Sample
from ..utils import clip01


def consistency_signal(
    vlm: BaseVLM,
    sample: Sample,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> dict:
    """Agreement among ``n_samples`` sampled answers, in [0, 1]."""
    answers = [
        vlm.generate(sample, variant="original", temperature=temperature, draw=i).answer
        for i in range(n_samples)
    ]
    counts = Counter(answers)
    modal_answer, modal_count = counts.most_common(1)[0]
    vote_share = modal_count / n_samples

    # Rescale: chance agreement for k choices is 1/k; map [1/k, 1] -> [0, 1].
    k = len(sample.answer_choices) if sample.answer_choices else max(len(counts), 2)
    chance = 1.0 / k
    score = clip01((vote_share - chance) / (1.0 - chance)) if k > 1 else vote_share

    return {
        "consistency": float(score),
        "vote_share": float(vote_share),
        "modal_answer": modal_answer,
        "n_samples": n_samples,
        "answers": answers,
    }
