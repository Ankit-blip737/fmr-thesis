"""Unified base-VLM interface + a factory.

Everything downstream (faithfulness signals, correction, abstention) depends only
on this interface, never on a concrete model. That is what makes FMR
model-agnostic: swap the backend, keep the whole pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Sample, VLMOutput


class BaseVLM(ABC):
    """Abstract reasoning VLM.

    A backend must be able to answer a ``Sample`` under three image *variants*
    (``original`` / ``blank`` / ``mismatch``) — that trio is what Signal A
    (counterfactual sensitivity) needs — and to produce a step-decomposed
    rationale with per-step attended regions (for Signal B) at a chosen sampling
    temperature (for Signal C).
    """

    name: str = "base"
    is_reasoning: bool = True

    @abstractmethod
    def generate(
        self,
        sample: Sample,
        variant: str = "original",
        temperature: float = 0.0,
        draw: int = 0,
    ) -> VLMOutput:
        """Return an answer + rationale for ``sample`` under ``variant``."""
        raise NotImplementedError


def load_vlm(config: dict[str, Any] | None = None) -> BaseVLM:
    """Instantiate a base VLM from a (models.yaml-style) config dict.

    ``backend: mock`` (default) returns the offline MockVLM. ``backend: hf``
    lazily imports the Hugging Face wrapper, which requires ``transformers``.
    """
    config = dict(config or {})
    backend = config.pop("backend", "mock")

    if backend == "mock":
        from .mock_vlm import MockVLM

        return MockVLM(**config)
    if backend == "hf":
        from .hf_vlm import HFVLM

        return HFVLM(**config)
    raise ValueError(f"Unknown VLM backend: {backend!r} (expected 'mock' or 'hf').")
