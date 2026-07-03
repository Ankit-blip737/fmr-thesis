"""Pillar 2 — training-free correction (Stage 4).

Role in the system (deliberately scoped): correction is *supporting
infrastructure for abstention*, not a headline contribution. It fixes the
fixable — answers where image evidence exists but is dominated by a language
prior — so that fewer cases need deferring. Cases where the model is genuinely
image-blind are left low-faithfulness on purpose: they belong to the
conformally-calibrated abstention gate (Pillar 3), not to a forced rewrite.

Components (all decode-time, no training):
  * ``vcd``           — Visual Contrastive Decoding on answer distributions
  * ``clue_tracing``  — question→vision clue tracing (evidence-region consensus)
  * ``verify_revise`` — CoRGI-style step verification + rationale revision
  * ``rescore``       — post-correction faithfulness (what calibration must use)
  * ``pipeline``      — selective application: only when faithfulness is low
"""
from .vcd import VCDResult, vcd_answer
from .clue_tracing import ClueTrace, trace_clue_region, clue_support
from .verify_revise import verify_and_revise
from .rescore import post_correction_sensitivity
from .pipeline import CorrectionConfig, CorrectionResult, correct_sample

__all__ = [
    "VCDResult",
    "vcd_answer",
    "ClueTrace",
    "trace_clue_region",
    "clue_support",
    "verify_and_revise",
    "post_correction_sensitivity",
    "CorrectionConfig",
    "CorrectionResult",
    "correct_sample",
]
