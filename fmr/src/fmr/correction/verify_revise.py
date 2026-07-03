"""Step verify-and-revise (CoRGI-style, arXiv:2508.00378).

Verify: score every reasoning step's visual support against the traced clue
region. Revise: rebuild the rationale from the *supported* steps only, and adopt
the image-anchored (VCD) answer when the contrast flipped it decisively.

The revision is conservative by design (see risk table: "correction hurts
accuracy"): the original answer is kept unless VCD both changed it *and* did so
with a clear margin; a low-margin flip is treated as contrast noise.
"""
from __future__ import annotations

from dataclasses import replace

from ..types import Sample, VLMOutput
from .clue_tracing import ClueTrace, clue_support
from .vcd import VCDResult

_COLLAPSED_NOTE = (
    "All {n} reasoning steps lacked traceable visual support; rationale withheld "
    "and the answer was re-derived with image-anchored contrastive decoding."
)


def verify_and_revise(
    sample: Sample,
    output: VLMOutput,
    trace: ClueTrace,
    vcd_res: VCDResult,
    support_threshold: float = 0.25,
    vcd_margin: float = 0.25,
) -> tuple[VLMOutput, dict]:
    """Return (revised output, diagnostics). ``output`` is not mutated.

    * Steps with clue-support >= ``support_threshold`` are kept (``supported=True``).
    * Unsupported steps are dropped from the revised rationale (counted in meta).
    * Answer: VCD's answer iff it changed with margin >= ``vcd_margin``; else the
      original. ``answer_logits`` always become the VCD-corrected distribution so
      downstream (re)scoring sees the corrected system's distribution.
    """
    supports = [clue_support(s, trace) for s in output.steps]
    kept = [
        replace(s, supported=True)
        for s, sup in zip(output.steps, supports)
        if sup >= support_threshold
    ]
    n_dropped = len(output.steps) - len(kept)

    adopt_vcd = vcd_res.changed and vcd_res.margin >= vcd_margin
    answer = vcd_res.answer if adopt_vcd else vcd_res.original_answer

    steps = kept
    if not steps:
        steps = [replace(output.steps[0], text=_COLLAPSED_NOTE.format(n=len(output.steps)),
                         pred_region=None, supported=False)] if output.steps else []

    diagnostics = {
        "supports": [float(s) for s in supports],
        "n_steps": len(output.steps),
        "n_kept": len(kept),
        "n_dropped": n_dropped,
        "keep_frac": len(kept) / len(output.steps) if output.steps else 0.0,
        "adopted_vcd_answer": bool(adopt_vcd),
        "clue_confidence": trace.confidence,
    }
    revised = VLMOutput(
        sample_id=sample.sample_id,
        answer=answer,
        steps=steps,
        answer_logits=vcd_res.p_vcd,
        variant="corrected",
        meta={**output.meta, "correction": diagnostics},
    )
    return revised, diagnostics
