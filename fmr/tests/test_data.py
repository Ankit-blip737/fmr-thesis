from fmr.data.loaders import load_dataset, split_dataset
from fmr.data.synthetic import build_synthetic_dataset


def test_synthetic_size_and_fields():
    samples = build_synthetic_dataset(n=100)
    assert len(samples) == 100
    s = samples[0]
    assert s.question and s.answer in s.answer_choices
    assert s.gt_region is not None
    assert s.meta["grounded"] in (0, 1)


def test_split_disjoint_and_complete():
    samples = build_synthetic_dataset(n=200)
    splits = split_dataset(samples, fractions=(0.5, 0.25, 0.25), seed=13)
    ids = [s.sample_id for part in splits.values() for s in part]
    assert len(ids) == 200
    assert len(set(ids)) == 200  # disjoint
    assert len(splits["train"]) == 100 and len(splits["cal"]) == 50


def test_split_holdout_modality():
    samples = build_synthetic_dataset(n=200)
    splits = split_dataset(samples, holdout_modality="pathology")
    assert all(s.modality == "pathology" for s in splits["holdout"])
    for part in ("train", "cal", "test"):
        assert all(s.modality != "pathology" for s in splits[part])


def test_split_deterministic():
    samples = build_synthetic_dataset(n=100)
    a = split_dataset(samples, seed=13)
    b = split_dataset(samples, seed=13)
    assert [s.sample_id for s in a["test"]] == [s.sample_id for s in b["test"]]


def test_loader_dispatch():
    samples = load_dataset({"name": "synthetic", "n": 10})
    assert len(samples) == 10
