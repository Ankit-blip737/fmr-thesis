"""Faithfulness-LoRA (optional/stretch) — move grounding into the weights.

Two halves, split by where they can run:

* **Data construction (CPU, tested here).** The training targets are the
  *verified grounded rationales* the training-free correction module already
  produces: for samples whose POST-correction faithfulness clears a bar, we take
  the corrected answer + the revised (supported-steps-only) rationale as a
  self-distillation target. Preference pairs (grounded ≻ ungrounded) for DPO are
  built the same way. This logic is deterministic and unit-tested against the
  mock backends — no GPU needed to validate it.

* **The QLoRA fit itself (GPU).** ``train_faithfulness_lora`` is guarded behind
  PEFT/TRL/bitsandbytes; on this CPU box it raises a clear message pointing to
  the Colab handoff notebook. The frozen base model is always the default;
  faithfulness-LoRA is reported strictly as an ablation (fix #4 — never a
  dependency).

Design note (fix #1): this is a stretch ablation answering RQ3 "can grounding be
learned, not just decoded?", not a headline. It reuses correction outputs already
computed, so it costs only a few GPU-hours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..correction import CorrectionConfig, correct_sample
from ..models.base_vlm import BaseVLM
from ..types import Sample


@dataclass
class FaithfulnessLoRAConfig:
    keep_threshold: float = 0.5          # min POST-correction FS to accept as a grounded target
    max_reject_fs: float = 0.2           # max FS for the "rejected" chain in a preference pair
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    # LoRA / QLoRA hyperparams (consumed on GPU by the notebook)
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 1
    load_in_4bit: bool = True


@dataclass
class DistillExample:
    sample_id: str
    question: str
    target_answer: str
    target_rationale: str
    fs: float
    source_correction_applied: bool


@dataclass
class PreferencePair:
    sample_id: str
    question: str
    chosen: str                          # grounded rationale+answer text
    rejected: str                        # ungrounded rationale+answer text
    chosen_fs: float
    rejected_fs: float


def _render(answer: str, rationale: str) -> str:
    return f"{rationale}\nAnswer: {answer}".strip()


def build_self_distillation_set(
    vlm: BaseVLM,
    samples: Sequence[Sample],
    config: Optional[FaithfulnessLoRAConfig] = None,
) -> list[DistillExample]:
    """Verified-grounded (answer, rationale) targets from correction outputs.

    Keeps a sample iff its post-correction faithfulness >= ``keep_threshold`` —
    i.e., the corrected output is one we can *verify* is grounded. Image-blind
    samples (which correction cannot rescue) are intentionally excluded: we never
    distill an ungrounded rationale into the weights.
    """
    cfg = config or FaithfulnessLoRAConfig()
    out: list[DistillExample] = []
    for s in samples:
        r = correct_sample(vlm, s, config=cfg.correction)
        if r.fs_after < cfg.keep_threshold:
            continue
        out.append(DistillExample(
            sample_id=s.sample_id,
            question=s.question,
            target_answer=r.corrected.answer,
            target_rationale=r.corrected.rationale,
            fs=r.fs_after,
            source_correction_applied=r.applied,
        ))
    return out


def build_preference_pairs(
    vlm: BaseVLM,
    samples: Sequence[Sample],
    config: Optional[FaithfulnessLoRAConfig] = None,
) -> list[PreferencePair]:
    """Grounded-≻-ungrounded pairs for DPO-style faithfulness tuning.

    ``chosen`` = the corrected (grounded) output; ``rejected`` = the raw output
    when it was both ungrounded (low pre-correction FS) and correction actually
    changed it — so the pair genuinely contrasts a grounded vs an ungrounded chain
    for the same question.
    """
    cfg = config or FaithfulnessLoRAConfig()
    pairs: list[PreferencePair] = []
    for s in samples:
        r = correct_sample(vlm, s, config=cfg.correction)
        if not r.applied or r.fs_after < cfg.keep_threshold or r.fs_before > cfg.max_reject_fs:
            continue
        if r.corrected.answer == r.original.answer and r.corrected.rationale == r.original.rationale:
            continue  # no contrast
        pairs.append(PreferencePair(
            sample_id=s.sample_id,
            question=s.question,
            chosen=_render(r.corrected.answer, r.corrected.rationale),
            rejected=_render(r.original.answer, r.original.rationale),
            chosen_fs=r.fs_after,
            rejected_fs=r.fs_before,
        ))
    return pairs


def train_faithfulness_lora(*args: Any, **kwargs: Any):  # pragma: no cover - GPU only
    """QLoRA fine-tune the base VLM on verified grounded rationales.

    Intentionally not runnable on this CPU box. The working implementation lives
    in ``fmr/notebooks/colab_faithfulness_lora.ipynb`` (self-distillation SFT with
    PEFT/QLoRA; optional TRL DPO on the preference pairs). This stub exists so the
    data-construction API above is importable and testable offline.
    """
    raise NotImplementedError(
        "Faithfulness-LoRA training is GPU-only. Build the self-distillation set / "
        "preference pairs here (CPU, tested), then run the fit in "
        "fmr/notebooks/colab_faithfulness_lora.ipynb on a 24GB GPU (QLoRA)."
    )
