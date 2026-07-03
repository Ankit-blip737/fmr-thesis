"""Question→vision clue tracing (ClueTracer-style, arXiv:2602.02004).

Goal: locate the image region that actually carries the evidence for the
question ("the clue"), independently of any single reasoning step, so that each
step can then be checked against it.

Model-agnostic realization through the ``BaseVLM`` contract: probe the model for
one or more reasoning chains, collect every step's attended region, and take the
*weighted medoid* — the region with the highest weighted IoU-agreement with all
other attended regions. Early steps get more weight because grounding drifts as
the chain grows (the very effect this thesis measures). The medoid's mean
agreement doubles as a *confidence*: grounded chains cluster tightly around the
evidence (high confidence), ungrounded chains scatter across the image (low).

For real HF models the same interface is fed from attention-rollout/relevancy
regions (Signal B's machinery); nothing here assumes the mock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.regions import Region
from ..faithfulness.decompose import extract_terms
from ..models.base_vlm import BaseVLM
from ..types import Sample, Step


@dataclass
class ClueTrace:
    """Traced evidence region + how much to trust it."""

    region: Optional[Region]
    confidence: float                 # weighted mean IoU-agreement at the medoid, [0, 1]
    terms: list[str] = field(default_factory=list)   # clue terms from the question
    n_chains: int = 0
    n_regions: int = 0


def trace_clue_region(
    vlm: BaseVLM,
    sample: Sample,
    n_probes: int = 3,
    temperature: float = 0.7,
    early_weight: float = 0.5,
) -> ClueTrace:
    """Probe ``n_probes`` chains and return the weighted-medoid evidence region.

    ``early_weight`` in [0, 1) controls how strongly late (drift-prone) steps are
    down-weighted: step k of n gets weight ``1 - early_weight * k/(n-1)``.
    Deterministic backends can return identical chains for every draw; duplicates
    are collapsed so they do not inflate the confidence.
    """
    chains: list[list[Step]] = []
    seen: set[str] = set()
    for draw in range(max(1, n_probes)):
        temp = 0.0 if draw == 0 else temperature   # first probe = the greedy chain
        out = vlm.generate(sample, variant="original", temperature=temp, draw=draw)
        key = out.rationale
        if key in seen:
            continue
        seen.add(key)
        chains.append(out.steps)

    regions: list[Region] = []
    weights: list[float] = []
    for steps in chains:
        n = len(steps)
        for k, step in enumerate(steps):
            if step.pred_region is None:
                continue
            regions.append(step.pred_region)
            weights.append(1.0 - early_weight * (k / max(1, n - 1)))

    terms = extract_terms(sample.question)
    if not regions:
        return ClueTrace(region=None, confidence=0.0, terms=terms, n_chains=len(chains))
    if len(regions) == 1:
        # A single attended region cannot corroborate itself.
        return ClueTrace(region=regions[0], confidence=0.0, terms=terms,
                         n_chains=len(chains), n_regions=1)

    best_idx, best_score = 0, -1.0
    for i, r_i in enumerate(regions):
        agree = sum(w_j * r_i.iou(r_j) for j, (r_j, w_j) in enumerate(zip(regions, weights)) if j != i)
        norm = sum(w_j for j, w_j in enumerate(weights) if j != i)
        score = agree / norm if norm > 0 else 0.0
        if score > best_score:
            best_idx, best_score = i, score

    return ClueTrace(
        region=regions[best_idx],
        confidence=float(max(0.0, best_score)),
        terms=terms,
        n_chains=len(chains),
        n_regions=len(regions),
    )


def clue_support(step: Step, trace: ClueTrace) -> float:
    """Visual support of one reasoning step given the traced clue, in [0, 1].

    IoU with the traced evidence region, scaled by the trace's confidence — a
    step can only be as supported as the clue itself is trustworthy.
    """
    if step.pred_region is None or trace.region is None:
        return 0.0
    return float(step.pred_region.iou(trace.region) * trace.confidence)
