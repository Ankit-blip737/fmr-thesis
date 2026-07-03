"""Selective correction pipeline: trigger → VCD → clue trace → verify/revise → rescore.

Correction is applied *only* when faithfulness is low (per the proposal's risk
table — indiscriminate correction risks hurting accuracy on already-grounded
cases). The trigger score is the fused Faithfulness Score when the caller has
one; until Instance A's ``faithfulness/score.py`` lands, the fallback trigger is
Signal A alone (see DECISIONS.md [B]).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..faithfulness.counterfactual import counterfactual_signal
from ..models.base_vlm import BaseVLM
from ..types import Sample, VLMOutput
from .clue_tracing import ClueTrace, trace_clue_region
from .rescore import post_correction_sensitivity
from .vcd import VCDResult, vcd_answer
from .verify_revise import verify_and_revise


@dataclass
class CorrectionConfig:
    trigger_threshold: float = 0.5      # correct only when FS < this
    alpha: float = 1.0                  # VCD contrast strength
    beta: float = 0.1                   # VCD adaptive-plausibility cutoff
    contrast_variants: tuple[str, ...] = ("blank", "mismatch")
    support_threshold: float = 0.25     # min clue-support to keep a step
    vcd_margin: float = 0.25            # min log-margin to adopt a VCD answer flip
    n_probes: int = 3                   # chains probed for clue tracing
    probe_temperature: float = 0.7
    early_weight: float = 0.5           # down-weighting of late (drift-prone) steps

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CorrectionConfig":
        d = dict(d or {})
        if "contrast_variants" in d:
            d["contrast_variants"] = tuple(d["contrast_variants"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CorrectionResult:
    """Everything downstream needs about one sample's (possible) correction.

    ``fs_after`` is the faithfulness score of the *deployed* output — the value
    the conformal calibration must consume (equals ``fs_before`` when correction
    was not triggered).
    """

    sample_id: str
    applied: bool
    fs_before: float
    fs_after: float
    original: VLMOutput
    corrected: VLMOutput                 # == original when not applied
    vcd: Optional[VCDResult] = None
    trace: Optional[ClueTrace] = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def answer_changed(self) -> bool:
        return self.corrected.answer != self.original.answer


def correct_sample(
    vlm: BaseVLM,
    sample: Sample,
    fs: Optional[float] = None,
    original: Optional[VLMOutput] = None,
    config: Optional[CorrectionConfig] = None,
) -> CorrectionResult:
    """Apply selective, training-free correction to one sample.

    ``fs``: pre-computed (fused) faithfulness score in [0, 1]. When ``None``,
    Signal A is computed here as the trigger. ``original`` may carry the already
    generated greedy output to avoid regeneration.
    """
    cfg = config or CorrectionConfig()

    if fs is None:
        sig_a = counterfactual_signal(vlm, sample)
        fs = float(sig_a["counterfactual"])
        original = original or sig_a["orig"]
    elif original is None:
        original = vlm.generate(sample, variant="original", temperature=0.0)

    if fs >= cfg.trigger_threshold:
        return CorrectionResult(
            sample_id=sample.sample_id,
            applied=False,
            fs_before=fs,
            fs_after=fs,
            original=original,
            corrected=original,
            diagnostics={"reason": "fs above trigger threshold"},
        )

    vcd_res = vcd_answer(
        vlm, sample, alpha=cfg.alpha, beta=cfg.beta, contrast_variants=cfg.contrast_variants
    )
    trace = trace_clue_region(
        vlm,
        sample,
        n_probes=cfg.n_probes,
        temperature=cfg.probe_temperature,
        early_weight=cfg.early_weight,
    )
    corrected, ver_diag = verify_and_revise(
        sample,
        vcd_res.outputs["original"],
        trace,
        vcd_res,
        support_threshold=cfg.support_threshold,
        vcd_margin=cfg.vcd_margin,
    )
    rescored = post_correction_sensitivity(vlm, sample, corrected, reuse=vcd_res.outputs)

    return CorrectionResult(
        sample_id=sample.sample_id,
        applied=True,
        fs_before=fs,
        fs_after=float(rescored["counterfactual"]),
        original=original,
        corrected=corrected,
        vcd=vcd_res,
        trace=trace,
        diagnostics={
            **ver_diag,
            "vcd_changed": vcd_res.changed,
            "vcd_margin": vcd_res.margin,
            "post_flip_rate": rescored["flip_rate"],
            "post_js": rescored["js_divergence"],
        },
    )
