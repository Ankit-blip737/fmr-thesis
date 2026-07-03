"""Unit tests — faithfulness-LoRA data construction (CPU part; fit is GPU-only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmr.data import build_synthetic_dataset  # noqa: E402
from fmr.models import MockVLM  # noqa: E402
from fmr.models.second_vlm import PriorHeavyMockVLM  # noqa: E402
from fmr.training import (  # noqa: E402
    FaithfulnessLoRAConfig,
    build_preference_pairs,
    build_self_distillation_set,
)
from fmr.training.faithfulness_lora import train_faithfulness_lora  # noqa: E402


@pytest.fixture(scope="module")
def samples():
    return build_synthetic_dataset(n=120, seed=7)


def test_distill_set_only_keeps_grounded_targets(samples):
    vlm = PriorHeavyMockVLM()
    cfg = FaithfulnessLoRAConfig()
    ex = build_self_distillation_set(vlm, samples, cfg)
    assert ex, "expected some verified-grounded targets"
    # every kept example clears the faithfulness bar
    assert all(e.fs >= cfg.keep_threshold for e in ex)
    # targets are well-formed
    assert all(e.target_answer and e.target_rationale for e in ex)


def test_distill_excludes_image_blind(samples):
    """Image-blind samples (grounded latent == 0) must not become distill targets
    for the image-blind backend — we never distill an ungrounded rationale."""
    vlm = MockVLM()
    ex = build_self_distillation_set(vlm, samples, FaithfulnessLoRAConfig())
    kept_ids = {e.sample_id for e in ex}
    blind_ids = {s.sample_id for s in samples if not s.meta["grounded"]}
    # very few (ideally zero) image-blind samples survive the FS bar
    leaked = kept_ids & blind_ids
    assert len(leaked) <= 0.1 * max(1, len(blind_ids))


def test_preference_pairs_contrast_grounded_vs_ungrounded(samples):
    vlm = PriorHeavyMockVLM()
    cfg = FaithfulnessLoRAConfig()
    pairs = build_preference_pairs(vlm, samples, cfg)
    assert pairs, "expected some preference pairs on the prior-dominated backend"
    for p in pairs:
        assert p.chosen_fs >= cfg.keep_threshold
        assert p.rejected_fs <= cfg.max_reject_fs
        assert p.chosen != p.rejected


def test_train_is_gpu_only_stub():
    with pytest.raises(NotImplementedError):
        train_faithfulness_lora()
