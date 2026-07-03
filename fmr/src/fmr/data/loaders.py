"""Unified dataset loading + splitting.

``load_dataset`` returns a list of :class:`Sample` regardless of source, so the
harness never branches on dataset identity. The synthetic source always works
offline. The real loaders pull public Hugging Face mirrors (no token needed for
VQA-RAD / SLAKE / PathVQA) and normalize every record to the ``Sample`` schema,
including SLAKE/VQA-RAD bounding boxes → normalized :class:`Region` for Signal B.

Design choices (see DECISIONS.md):
  * HF hub mirrors: vqa_rad -> flaviagiammarino/vqa-rad,
    slake -> BoKelvin/SLAKE, pathvqa -> flaviagiammarino/path-vqa.
  * OmniMedVQA needs a manual download (gated distribution) -> file-based loader.
  * Images are kept as PIL objects in ``Sample.image`` (the HF backend converts
    them to model tensors + builds the blank/mismatch variants). On the offline
    mock path ``image`` stays None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..types import Sample
from .regions import Region
from .synthetic import build_synthetic_dataset

_HF_REPOS = {
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "slake": "BoKelvin/SLAKE",
    "pathvqa": "flaviagiammarino/path-vqa",
}
_MODALITY_KEYWORDS = {  # crude modality tagging from question/location text
    "xray": ("x-ray", "xray", "chest", "radiograph"),
    "ct": ("ct", "computed tomography"),
    "mri": ("mri", "magnetic resonance", "t1", "t2", "flair"),
    "pathology": ("pathology", "histolog", "microscop", "stain", "h&e"),
}


def load_dataset(config: dict[str, Any] | None = None) -> list[Sample]:
    config = dict(config or {})
    name = config.pop("name", "synthetic")

    if name == "synthetic":
        return build_synthetic_dataset(**config)
    if name in _HF_REPOS:
        return _load_hf(name, config)
    if name == "omnimedvqa":
        return _load_omnimedvqa(config)
    raise ValueError(f"Unknown dataset {name!r}.")


def _guess_modality(text: str, default: str = "unknown") -> str:
    t = (text or "").lower()
    for mod, kws in _MODALITY_KEYWORDS.items():
        if any(k in t for k in kws):
            return mod
    return default


def _norm_answer(ans: Any) -> str:
    return str(ans).strip().lower()


_YESNO = {"yes", "no"}


def _load_hf(name: str, config: dict[str, Any]) -> list[Sample]:
    """Load VQA-RAD / SLAKE / PathVQA from a public HF mirror.

    Per-dataset schema notes (verified 2026-07-03 against the live mirrors):
      * vqa_rad (flaviagiammarino/vqa-rad): columns image/question/answer only.
        No modality field (tagged from text) and no per-QA boxes in this mirror.
      * slake (BoKelvin/SLAKE): rich JSON — modality, answer_type (OPEN/CLOSED),
        q_lang (en/zh), location, img_name. Images are referenced by path, NOT
        inline; set ``image_root`` to the extracted imgs/ dir to attach PIL
        images. No per-QA bbox in this mirror either (SLAKE ships masks
        separately) — Signal B IoU on real SLAKE stays unvalidated until masks
        are wired (see BLOCKERS.md); we still get true modality labels here.
      * pathvqa (flaviagiammarino/path-vqa): image/question/answer; all pathology.
    """
    try:
        from datasets import load_dataset as hf_load
    except Exception as exc:  # pragma: no cover
        raise ImportError("`datasets` is required for real loaders (`pip install datasets`).") from exc

    repo = _HF_REPOS[name]
    split = config.get("split", "test")
    max_samples = config.get("max_samples")
    cache_dir = config.get("cache_dir")
    english_only = config.get("english_only", True)
    image_root = config.get("image_root")

    ds = hf_load(repo, split=split, cache_dir=cache_dir)

    samples: list[Sample] = []
    for i, row in enumerate(ds):
        if name == "slake" and english_only and row.get("q_lang", "en") != "en":
            continue
        q = row.get("question") or row.get("question_text") or ""
        a = _norm_answer(row.get("answer") or row.get("answer_text") or "")

        # Modality: real label on SLAKE; text-guess elsewhere.
        if name == "slake" and row.get("modality"):
            modality = str(row["modality"]).lower().replace("-", "").replace(" ", "")
            modality = {"xray": "xray", "cxr": "xray"}.get(modality, modality)
        else:
            modality = _guess_modality(f"{q} {row.get('location', '')}",
                                       default="pathology" if name == "pathvqa" else "unknown")

        # Image: inline PIL where present; else resolve img_name against image_root.
        img = row.get("image")
        if img is None and name == "slake" and image_root and row.get("img_name"):
            img = str(Path(image_root) / row["img_name"])

        # Closed (yes/no or explicitly CLOSED) -> give binary choices for clean metrics.
        answer_type = str(row.get("answer_type", "")).upper()
        choices = ["yes", "no"] if (a in _YESNO and answer_type != "OPEN") else None

        samples.append(
            Sample(
                sample_id=f"{name}-{split}-{i:06d}",
                question=str(q),
                answer=a,
                modality=modality,
                image=img,
                gt_region=None,   # no per-QA boxes in these mirrors; see docstring
                answer_choices=choices,
                meta={
                    "source": name, "raw_answer": row.get("answer"),
                    "answer_type": answer_type or None,
                    "closed": choices is not None,
                    "img_name": row.get("img_name"),
                    "location": row.get("location"),
                },
            )
        )
        if max_samples and len(samples) >= int(max_samples):
            break
    return samples


def bbox_to_region(bbox: list[float], image_width: float, image_height: float) -> Region | None:
    """Convert a pixel [x, y, w, h] box to a normalized Region.

    Kept as a reusable helper for the Colab SLAKE-mask path: when the SLAKE
    segmentation masks are downloaded, derive a bounding box per (image,
    location) and call this to attach ``gt_region`` for Signal B IoU validation.
    """
    if not bbox or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return None
    x, y, w, h = (float(v) for v in bbox)
    return Region(x / image_width, y / image_height,
                  (x + w) / image_width, (y + h) / image_height)


def _load_omnimedvqa(config: dict[str, Any]) -> list[Sample]:
    """OmniMedVQA from a manually-downloaded local root (see BLOCKERS.md)."""
    root = config.get("root")
    if not root or not Path(root).exists():
        raise FileNotFoundError(
            f"OmniMedVQA not found at root={root!r}. Download the open-access subset "
            "and set data.yaml:omnimedvqa.root. (Gated distribution — see BLOCKERS.md.)"
        )
    raise NotImplementedError(
        "OmniMedVQA local parser stub: iterate the JSON QA files under root, map "
        "image_path/question/gt_answer/option_* to Sample, tag modality from the "
        "dataset's 'Modality_Type' field. Fill in once the download is present."
    )


def split_dataset(
    samples: list[Sample],
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    holdout_modality: str | None = None,
    seed: int = 13,
) -> dict[str, list[Sample]]:
    """Split into train / calibration / test.

    ``train`` feeds the learned verifier, ``cal`` feeds the split-conformal gate,
    ``test`` is held out for reporting. If ``holdout_modality`` is set, all of
    that modality is moved into a separate ``holdout`` split to test
    generalization to an unseen modality (per the proposal's held-out experiment).
    """
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)

    pool = list(samples)
    holdout: list[Sample] = []
    if holdout_modality is not None:
        holdout = [s for s in pool if s.modality == holdout_modality]
        pool = [s for s in pool if s.modality != holdout_modality]

    idx = rng.permutation(len(pool))
    n_train = int(fractions[0] * len(pool))
    n_cal = int(fractions[1] * len(pool))
    train = [pool[i] for i in idx[:n_train]]
    cal = [pool[i] for i in idx[n_train:n_train + n_cal]]
    test = [pool[i] for i in idx[n_train + n_cal:]]

    out = {"train": train, "cal": cal, "test": test}
    if holdout_modality is not None:
        out["holdout"] = holdout
    return out
