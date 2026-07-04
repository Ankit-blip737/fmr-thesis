"""OmniMedVQA parser + held-out-modality generalization split.

The real OmniMedVQA download is gated (see BLOCKERS.md), so we validate the
parser against a synthetic fixture that mirrors the open-access QA-JSON schema,
and confirm the held-out-modality machinery works on the modalities we already
have. No network / no download needed.
"""
import json

from fmr.data.loaders import load_dataset, split_dataset
from fmr.data.synthetic import build_synthetic_dataset


def _write_omni_fixture(tmp_path):
    root = tmp_path / "omnimedvqa"
    (root / "Images").mkdir(parents=True)
    (root / "Images" / "a.png").write_bytes(b"x")
    (root / "Images" / "b.png").write_bytes(b"x")
    qa = [
        {"question_id": "q1", "image_path": "Images/a.png", "question": "What modality?",
         "gt_answer": "CT", "option_A": "CT", "option_B": "MRI", "option_C": "X-Ray",
         "option_D": "US", "modality_type": "CT(Computed Tomography)", "dataset": "src1"},
        {"question_id": "q2", "image_path": "Images/b.png", "question": "Lesion present?",
         "gt_answer": "Yes", "option_A": "Yes", "option_B": "No",
         "modality_type": "Microscopy Images", "dataset": "src2"},
    ]
    (root / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    return root


def test_omnimedvqa_parser(tmp_path):
    root = _write_omni_fixture(tmp_path)
    samples = load_dataset({"name": "omnimedvqa", "root": str(root)})
    assert len(samples) == 2
    s0 = samples[0]
    assert s0.modality == "ct" and s0.answer == "ct"
    assert s0.answer_choices == ["ct", "mri", "x-ray", "us"]
    assert s0.meta["closed"] is True and s0.meta["answer_type"] == "CLOSED"
    assert str(root) in str(s0.image)  # image path resolved against root
    # Microscopy -> pathology; 2-option still closed.
    assert samples[1].modality == "pathology"


def test_omnimedvqa_modality_filter(tmp_path):
    root = _write_omni_fixture(tmp_path)
    only_ct = load_dataset({"name": "omnimedvqa", "root": str(root), "modalities": ["ct"]})
    assert len(only_ct) == 1 and only_ct[0].modality == "ct"


def test_omnimedvqa_missing_root_message(tmp_path):
    try:
        load_dataset({"name": "omnimedvqa", "root": str(tmp_path / "nope")})
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "OmniMedVQA" in str(e)


def test_holdout_modality_works_on_existing_data():
    """Held-out-modality generalization does not require OmniMedVQA — it works on
    the modalities we already have (synthetic mirrors xray/ct/mri/pathology)."""
    samples = build_synthetic_dataset(n=400)
    splits = split_dataset(samples, holdout_modality="pathology")
    assert splits["holdout"] and all(s.modality == "pathology" for s in splits["holdout"])
    for part in ("train", "cal", "test"):
        assert all(s.modality != "pathology" for s in splits[part])
