"""CoT -> atomic claims.

For MockVLM the rationale already arrives as :class:`Step` objects. For real
models the rationale is free text, so this module provides a rule-based splitter
+ light clinical-term extraction. It is deliberately simple and pluggable (an
LLM-based decomposer can be dropped in behind ``decompose_rationale``).
"""
from __future__ import annotations

import re

from ..types import Step

_SPLIT = re.compile(r"(?<=[.;])\s+|\n+|(?=Step\s*\d+\s*:)", re.IGNORECASE)
# A tiny illustrative clinical lexicon; replace/extend with a real ontology (UMLS).
_TERMS = re.compile(
    r"\b(opacity|effusion|cardiomegaly|nodule|lesion|hemorrhage|mass|infarct|"
    r"edema|atrophy|enhancement|mitosis|necrosis|tumor|inflammation|fracture|"
    r"consolidation|pneumothorax)\b",
    re.IGNORECASE,
)


def extract_terms(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TERMS.finditer(text)]


def decompose_rationale(text: str) -> list[Step]:
    """Split a free-text rationale into atomic reasoning steps."""
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    return [Step(text=p, terms=extract_terms(p)) for p in parts]
