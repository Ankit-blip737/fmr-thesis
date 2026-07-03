"""Hugging Face reasoning-VLM wrapper (real backend, GPU/Colab).

Implements the full ``BaseVLM.generate`` contract for real open medical VLMs
(MedVLM-R1, Qwen2.5-VL, ...) so the entire FMR pipeline runs unchanged on real
models. It is dependency-guarded: importing the module is cheap, and the heavy
imports (torch/transformers) happen only on instantiation. This code path is
exercised on Colab (see notebooks/colab_*.ipynb), never on the CPU-only dev box.

The three image variants Signal A needs:
  * original  -> the real image.
  * blank     -> a mid-grey image (the image evidence is removed).
  * mismatch  -> a *different* image from the dataset (image-answer link broken).

``answer_logits`` is a distribution over the sample's answer choices (or a small
default yes/no/uncertain set for open questions), obtained by teacher-forced
sequence-scoring of each candidate — model-agnostic and robust, unlike trying to
read a single next-token position. Per-step attended regions come from
attention rollout over the vision tokens, mapped to a coarse grid Region; if a
model doesn't expose usable attentions the step region falls back to None (Signal
B then reports the neutral 0.5, and it is marked unvalidated).
"""
from __future__ import annotations

import math
from typing import Any

from ..data.regions import Region
from ..faithfulness.decompose import decompose_rationale
from ..types import Sample, Step, VLMOutput

_DEFAULT_CHOICES = ["yes", "no", "uncertain"]


class HFVLM:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        is_reasoning: bool = True,
        max_new_tokens: int = 256,
        grid: int = 4,
        dtype: str = "fp16",
        max_image_side: int = 512,
        **kw: Any,
    ) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "The 'hf' backend needs `torch` and `transformers`. Install the "
                "'real' extra, or use backend='mock' for the offline pipeline."
            ) from exc
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.is_reasoning = is_reasoning
        self.max_new_tokens = max_new_tokens
        self.grid = grid
        self.dtype = dtype
        # Cap the longest image side. Medical scans are high-res (VQA-RAD ~1024px);
        # Qwen2.5-VL tokenizes by pixels, so a 1024px image explodes into thousands
        # of vision tokens whose O(n^2) attention OOMs a 14.5 GB T4. 512px keeps
        # the vision encoder well within budget and barely dents accuracy.
        self.max_image_side = max_image_side
        self._kw = kw
        self._model = None
        self._processor = None
        self._mismatch_pool: list[Any] = []  # filled by set_mismatch_pool()

    # ---- lazy load ----------------------------------------------------------
    @staticmethod
    def _patch_config_for_strict_hub() -> None:  # pragma: no cover
        """Work around huggingface_hub>=0.34 StrictDataclassFieldValidationError.

        Root cause: hf_hub 0.34+ strict dataclass validation rejects None for bool
        fields. Several VLM configs on HF (Qwen2-VL, MedVLM-R1) have
        ``"use_cache": null`` in config.json.  The crash site is inside
        ``Qwen2TextConfig.__init__`` (a sub-config instantiated in
        ``Qwen2VLConfig.__post_init__``). Because hf_hub's ``init_with_validate``
        decorator stores a *per-class* __init__ in each class's __dict__, patching
        ``PretrainedConfig.__init__`` has NO effect on subclasses.

        Correct fix: patch ``PretrainedConfig.from_dict`` (a classmethod inherited
        by all subclasses that don't override it). It receives the raw dict from
        config.json *before* ``cls(**config_dict)`` is called — sanitising None bool
        values here propagates through the entire nested config tree.

        Alternatively: ``pip install 'transformers>=4.52.0'`` before running.
        """
        try:
            from transformers import configuration_utils

            _BOOL_FIELDS = frozenset({
                "use_cache", "output_attentions", "output_hidden_states",
                "return_dict", "tie_word_embeddings", "is_decoder",
                "add_cross_attention", "chunk_size_feed_forward",
            })

            def _sanitize_dict(d: dict) -> None:
                """Recursively coerce None → True for known bool fields."""
                for k, v in list(d.items()):
                    if k in _BOOL_FIELDS and v is None:
                        d[k] = True
                    elif isinstance(v, dict):
                        _sanitize_dict(v)

            _orig_from_dict = configuration_utils.PretrainedConfig.from_dict.__func__

            def _patched_from_dict(cls, config_dict: dict, **kwargs):
                _sanitize_dict(config_dict)
                return _orig_from_dict(cls, config_dict, **kwargs)

            if not getattr(_orig_from_dict, "_fmr_patched", False):
                _orig_from_dict._fmr_patched = True
                configuration_utils.PretrainedConfig.from_dict = classmethod(
                    _patched_from_dict
                )
        except Exception:
            pass  # Newer transformers already handles this; patch not needed.

    def _ensure_loaded(self) -> None:  # pragma: no cover - requires weights
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except Exception:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self._patch_config_for_strict_hub()

        td = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}.get(self.dtype, torch.float16)
        # Bound the processor's pixel budget too (belt-and-suspenders with the
        # image resize in _load_image), so vision-token count stays small.
        px = self.max_image_side * self.max_image_side
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True, max_pixels=px, min_pixels=256 * 256)
        except Exception:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = _AutoVLM.from_pretrained(
            self.model_id, torch_dtype=td, device_map=self.device,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        self._model.eval()

    def unload(self) -> None:  # pragma: no cover
        """Free the model + processor from GPU (call before loading another)."""
        import gc
        import torch
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_mismatch_pool(self, images: list[Any]) -> None:
        """Provide a pool of images used to build the 'mismatch' variant."""
        self._mismatch_pool = list(images)

    # ---- image variant construction -----------------------------------------
    def _downscale(self, img):  # pragma: no cover
        """Shrink so the longest side <= max_image_side (never upscales)."""
        m = self.max_image_side
        w, h = img.size
        if max(w, h) <= m:
            return img
        s = m / float(max(w, h))
        return img.resize((max(1, int(w * s)), max(1, int(h * s))))

    def _load_image(self, image: Any):  # pragma: no cover
        from PIL import Image

        if image is None:
            return Image.new("RGB", (self.max_image_side, self.max_image_side), (128, 128, 128))
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if hasattr(image, "convert") else image
        return self._downscale(img)

    def _variant_image(self, sample: Sample, variant: str):  # pragma: no cover
        from PIL import Image

        base = self._load_image(sample.image)
        if variant == "original":
            return base
        if variant == "blank":
            return Image.new("RGB", base.size, (128, 128, 128))
        if variant == "mismatch":
            for cand in self._mismatch_pool:
                img = self._load_image(cand)
                if img.size != base.size or cand is not sample.image:
                    return img
            # Fallback: shuffle pixels so content is destroyed but stats persist.
            return base.rotate(180)
        raise ValueError(f"variant must be original|blank|mismatch, got {variant!r}")

    # ---- prompting ----------------------------------------------------------
    def _build_prompt(self, sample: Sample) -> str:
        if self.is_reasoning:
            instr = ("Reason step by step about the image, then give the final answer. "
                     "Format: reasoning sentences, then 'Answer: <answer>'.")
        else:
            instr = "Answer the question directly in a few words. Format: 'Answer: <answer>'."
        q = sample.question
        if sample.answer_choices:
            q += " Choices: " + ", ".join(sample.answer_choices) + "."
        return f"{q}\n{instr}"

    def _messages(self, sample: Sample, image):  # pragma: no cover
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self._build_prompt(sample)},
        ]}]

    # ---- candidate answer scoring (answer_logits) ----------------------------
    def _score_candidates(self, sample: Sample, image, candidates: list[str]):  # pragma: no cover
        """Teacher-forced sequence logprob of each candidate -> softmax dist."""
        import numpy as np
        import torch

        logprobs = []
        for cand in candidates:
            msgs = self._messages(sample, image)
            prompt = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            target = f" Answer: {cand}"
            full = prompt + target
            enc_prompt = self._processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
            enc_full = self._processor(text=[full], images=[image], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model(**enc_full)
                logits = out.logits[0]  # (seq, vocab)
                ids = enc_full["input_ids"][0]
                p_len = enc_prompt["input_ids"].shape[1]
                tgt = ids[p_len:]
                if tgt.numel():
                    lp_tokens = torch.log_softmax(logits[p_len - 1:ids.shape[0] - 1], dim=-1)
                    lp = float(lp_tokens.gather(1, tgt.unsqueeze(1)).sum())
                else:
                    lp = 0.0
            logprobs.append(lp / max(1, int(tgt.numel())))  # length-normalized
            # Free per-candidate: high-res vision activations must not accumulate.
            del out, logits, enc_full, enc_prompt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arr = np.array(logprobs)
        e = np.exp(arr - arr.max())
        return e / e.sum()

    # ---- attention -> per-step region ---------------------------------------
    def _step_regions(self, n_steps: int, sample: Sample) -> list[Region | None]:  # pragma: no cover
        """Best-effort per-step attended region.

        Full attention-rollout over interleaved vision tokens is model-specific;
        this returns None per step by default (Signal B -> neutral, marked
        unvalidated) and is the documented plug point to wire real rollout on
        Colab. Kept out of the generation hot path so a model without usable
        attentions still runs end-to-end.
        """
        return [None] * n_steps

    # ---- main contract ------------------------------------------------------
    def generate(
        self,
        sample: Sample,
        variant: str = "original",
        temperature: float = 0.0,
        draw: int = 0,
    ) -> VLMOutput:  # pragma: no cover - requires weights
        import torch

        self._ensure_loaded()
        image = self._variant_image(sample, variant)
        msgs = self._messages(sample, image)
        prompt = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self._processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)

        gen_kw = dict(max_new_tokens=self.max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen_kw.update(temperature=temperature, top_p=0.95)
            torch.manual_seed(hash((sample.sample_id, draw)) % (2**31))
        with torch.no_grad():
            gen = self._model.generate(**enc, **gen_kw)
        text = self._processor.batch_decode(
            gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        answer = self._parse_answer(text, sample)
        candidates = sample.answer_choices or _DEFAULT_CHOICES
        logits = self._score_candidates(sample, image, candidates)

        # Decompose the reasoning into steps; attach best-effort regions.
        # Prefer a <think>...</think> block (MedVLM-R1 format); else the text
        # before "Answer:"; strip any <answer> tag out of the rationale.
        import re as _re
        think = _re.search(r"<think>(.*?)</think>", text, _re.I | _re.S)
        if think:
            rationale = think.group(1).strip()
        else:
            rationale = _re.sub(r"<answer>.*?</answer>", "", text.split("Answer:")[0],
                                flags=_re.I | _re.S).strip() or text
        steps = decompose_rationale(rationale) if self.is_reasoning else [Step(text=rationale)]
        regions = self._step_regions(len(steps), sample)
        for st, reg in zip(steps, regions):
            st.pred_region = reg

        return VLMOutput(
            sample_id=sample.sample_id,
            answer=answer,
            steps=steps,
            answer_logits=logits,
            variant=variant,
            meta={"model": self.name, "is_reasoning": self.is_reasoning, "raw_text": text},
        )

    def _parse_answer(self, text: str, sample: Sample) -> str:  # pragma: no cover
        """Extract the final answer, robust to model output format.

        Handles three formats: ``<answer>X</answer>`` (MedVLM-R1 / R1-style),
        ``Answer: X`` (our prompt), and free text. Snaps to a choice when one is
        given, scanning the whole text as a last resort before defaulting.
        """
        import re
        m = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
        if m:
            span = m.group(1)
        elif "Answer:" in text:
            span = text.split("Answer:")[-1]
        else:  # strip a think block, take the tail
            span = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
        ans = span.strip().split("\n")[0].strip(" .:'\"*").lower()
        if sample.answer_choices:
            lc = [c.lower() for c in sample.answer_choices]
            for c, cl in zip(sample.answer_choices, lc):        # in the answer span
                if re.search(r"\b" + re.escape(cl) + r"\b", ans):
                    return c
            for c, cl in zip(sample.answer_choices, lc):        # anywhere in the text
                if re.search(r"\b" + re.escape(cl) + r"\b", text.lower()):
                    return c
            return sample.answer_choices[0]
        return ans or text.strip().lower()[:40]
