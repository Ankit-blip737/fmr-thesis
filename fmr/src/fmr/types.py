"""Core data types shared across the FMR pipeline.

Keeping these as small dataclasses (rather than passing raw dicts around) makes
the data flow between the base model, the faithfulness signals, the correction
module and the abstention gate explicit and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .data.regions import Region


@dataclass
class Sample:
    """One medical VQA item."""

    sample_id: str
    question: str
    answer: str                      # ground-truth answer (used only for eval)
    modality: str = "unknown"        # xray | ct | mri | pathology | ...
    image: Any = None                # array / path / synthetic descriptor
    gt_region: Optional[Region] = None
    answer_choices: Optional[list[str]] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Step:
    """One atomic reasoning step / claim extracted from a chain-of-thought."""

    text: str
    terms: list[str] = field(default_factory=list)     # key clinical terms
    pred_region: Optional[Region] = None               # attended region for this step

    # Faithfulness signals (filled by the measurement module), each in [0, 1].
    counterfactual: Optional[float] = None             # Signal A
    attention_grounding: Optional[float] = None        # Signal B
    consistency: Optional[float] = None                # Signal C
    fs: Optional[float] = None                          # aggregate step faithfulness

    grounded_label: Optional[int] = None               # derived label (verifier training)
    supported: Optional[bool] = None                   # kept after verify/revise?


@dataclass
class VLMOutput:
    """What a base VLM returns for one (image, question) under one image variant."""

    sample_id: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    answer_logits: Optional[np.ndarray] = None         # distribution over answer vocab
    variant: str = "original"                          # original | blank | mismatch
    meta: dict = field(default_factory=dict)

    @property
    def rationale(self) -> str:
        return " ".join(s.text for s in self.steps)
