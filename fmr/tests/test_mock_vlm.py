from fmr.data.synthetic import build_synthetic_dataset
from fmr.models.mock_vlm import MockVLM


def test_deterministic_at_zero_temperature():
    samples = build_synthetic_dataset(n=5)
    vlm = MockVLM()
    for s in samples:
        a = vlm.generate(s, variant="original", temperature=0.0)
        b = vlm.generate(s, variant="original", temperature=0.0)
        assert a.answer == b.answer
        assert [st.pred_region for st in a.steps] == [st.pred_region for st in b.steps]


def test_draws_vary_at_temperature():
    samples = build_synthetic_dataset(n=40)
    vlm = MockVLM()
    varied = 0
    for s in samples:
        answers = {vlm.generate(s, temperature=1.5, draw=i).answer for i in range(5)}
        varied += len(answers) > 1
    assert varied > 5  # at high temperature a decent share of samples scatter


def test_reasoning_vs_plain_chain_length():
    s = build_synthetic_dataset(n=1)[0]
    assert len(MockVLM(is_reasoning=True).generate(s).steps) == 4
    assert len(MockVLM(is_reasoning=False).generate(s).steps) == 1


def test_variant_validation():
    s = build_synthetic_dataset(n=1)[0]
    try:
        MockVLM().generate(s, variant="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
