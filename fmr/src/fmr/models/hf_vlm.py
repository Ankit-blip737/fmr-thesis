"""Hugging Face reasoning-VLM wrapper (real backend).

This is the plug point for actual open medical VLMs (MedVLM-R1, Qwen2.5-VL,
LLaVA-Med, MedGemma, ...). It is intentionally a thin, dependency-guarded
scaffold: the offline pipeline uses :class:`MockVLM`, and this class is filled in
when a GPU + checkpoint are available. It implements the same ``generate``
contract so nothing downstream changes.

Implementation notes (for whoever runs this on hardware):
  * ``blank``   -> replace the image with a mid-grey / Gaussian-noised tensor.
  * ``mismatch``-> feed a different image from the batch (breaks image-answer link).
  * per-step regions come from attention-rollout / relevancy over image patches
    (see ``fmr.faithfulness.attention``); map the top patches to a Region.
  * ``answer_logits`` -> softmax over the constrained answer-choice token ids.
"""
from __future__ import annotations

from typing import Any

from ..types import Sample, VLMOutput


class HFVLM:
    is_reasoning = True

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda", **kw: Any) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: F401
        except Exception as exc:  # pragma: no cover - only hit without the deps
            raise ImportError(
                "The 'hf' backend needs `transformers`, `torch` and a checkpoint. "
                "Install them and re-run, or use backend='mock' for the offline pipeline."
            ) from exc
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self._kw = kw
        # Lazy load left to `._ensure_loaded()` so importing the module is cheap.
        self._model = None
        self._processor = None

    @staticmethod
    def _patch_config_for_strict_hub() -> None:  # pragma: no cover
        """Patch PretrainedConfig.from_dict to coerce None→True for bool fields.
        Fixes huggingface_hub>=0.34 StrictDataclassFieldValidationError on use_cache=None.
        Patching from_dict (not __init__) is necessary because hf_hub's init_with_validate
        stores a per-class __init__ in each class __dict__, making base-class __init__
        patches ineffective for subclasses like Qwen2TextConfig.
        """
        try:
            from transformers import configuration_utils
            _BOOL_FIELDS = frozenset({
                "use_cache", "output_attentions", "output_hidden_states",
                "return_dict", "tie_word_embeddings", "is_decoder",
                "add_cross_attention", "chunk_size_feed_forward",
            })
            def _sanitize(d: dict) -> None:
                for k, v in list(d.items()):
                    if k in _BOOL_FIELDS and v is None:
                        d[k] = True
                    elif isinstance(v, dict):
                        _sanitize(v)
            _orig = configuration_utils.PretrainedConfig.from_dict.__func__
            def _patched(cls, config_dict: dict, **kwargs):
                _sanitize(config_dict)
                return _orig(cls, config_dict, **kwargs)
            if not getattr(_orig, "_fmr_patched", False):
                _orig._fmr_patched = True
                configuration_utils.PretrainedConfig.from_dict = classmethod(_patched)
        except Exception:
            pass

    def _ensure_loaded(self) -> None:  # pragma: no cover - requires weights
        if self._model is not None:
            return
        from transformers import AutoModelForVision2Seq, AutoProcessor

        import torch
        self._patch_config_for_strict_hub()
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True
        )

    def generate(
        self,
        sample: Sample,
        variant: str = "original",
        temperature: float = 0.0,
        draw: int = 0,
    ) -> VLMOutput:  # pragma: no cover - requires weights
        raise NotImplementedError(
            "HFVLM.generate is a hardware scaffold. Fill in image-variant construction, "
            "constrained decoding for answer_logits, and attention->Region extraction. "
            "The MockVLM backend implements the full contract for offline runs."
        )
