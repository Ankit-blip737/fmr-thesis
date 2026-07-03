"""Unit tests — LLM-as-judge + validation harness (Instance B).

Includes HELD-OUT probes (cases NOT in the gold set) so the tests guard against
the heuristic being overfit to the gold set it was tuned on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmr.eval.judge import (  # noqa: E402
    HeuristicJudge,
    LLMJudge,
    build_judge,
    evaluate_judge_agreement,
    _cohens_kappa,
)
from fmr.eval.gold_data import JUDGE_GOLD  # noqa: E402


@pytest.fixture(scope="module")
def judge():
    return HeuristicJudge()


# ---------- core behaviors ---------------------------------------------------

@pytest.mark.parametrize("pred,ref,expected", [
    ("yes", "yes", "correct"),
    ("enlarged heart", "cardiomegaly", "correct"),
    ("no effusion", "effusion", "incorrect"),
    ("pneumothorax present", "no pneumothorax", "incorrect"),
    ("", "yes", "incorrect"),
])
def test_core(judge, pred, ref, expected):
    assert judge("q", pred, ref).label == expected


# ---------- HELD-OUT probes (not in gold; general principles) ---------------

@pytest.mark.parametrize("pred,ref,expected", [
    # synonyms the gold didn't include verbatim
    ("bleeding", "hemorrhage", "correct"),
    ("broken bone", "fracture", "correct"),
    ("three lesions", "3 lesions", "correct"),
    # polarity flips
    ("no mass", "mass", "incorrect"),
    ("tumor absent", "tumor present", "incorrect"),
    # wrong finding
    ("fracture", "effusion", "incorrect"),
    # multi-finding partial (held-out combo)
    ("nodule", "nodule and effusion", "partial"),
    # underspecified single finding
    ("effusion", "large loculated pleural effusion left base", "partial"),
    # normality synonyms
    ("unremarkable", "no abnormality", "correct"),
])
def test_heldout_probes(judge, pred, ref, expected):
    assert judge("q", pred, ref).label == expected, (pred, ref)


# ---------- score mapping ----------------------------------------------------

def test_score_mapping(judge):
    assert judge("q", "yes", "yes").score == 1.0
    assert judge("q", "fracture", "effusion").score == 0.0
    v = judge("q", "nodule", "nodule and effusion")
    assert v.label == "partial" and v.score == 0.5


# ---------- kappa correctness ------------------------------------------------

def test_cohens_kappa_perfect_and_chance():
    yt = ["correct", "incorrect", "partial", "correct"]
    assert _cohens_kappa(yt, yt) == pytest.approx(1.0)
    # total disagreement flips correct<->incorrect
    assert _cohens_kappa(["correct", "incorrect"], ["incorrect", "correct"]) < 0.0


# ---------- agreement harness on the gold set --------------------------------

def test_gold_agreement_strong(judge):
    rep = evaluate_judge_agreement(judge, JUDGE_GOLD)
    # tuned heuristic: expect near-perfect on the (hand-authored) gold set.
    assert rep["cohens_kappa"] >= 0.85
    assert rep["binary_accuracy(correct_vs_not)"] >= 0.9
    assert rep["sources"]["heuristic"] == len(JUDGE_GOLD)


# ---------- LLM judge parsing + robust fallback ------------------------------

def test_llm_judge_parses_verdict():
    j = LLMJudge(complete=lambda prompt: "CORRECT\nClinically equivalent.")
    v = j("q", "enlarged heart", "cardiomegaly")
    assert v.label == "correct" and v.source == "llm"


def test_llm_judge_falls_back_on_error():
    def boom(prompt):
        raise RuntimeError("provider down")
    j = LLMJudge(complete=boom)
    v = j("q", "no effusion", "effusion")
    assert v.source == "llm-fallback" and v.label == "incorrect"  # heuristic got it


def test_llm_judge_falls_back_on_garbage():
    j = LLMJudge(complete=lambda prompt: "the weather is nice today")
    v = j("q", "yes", "yes")
    assert v.source == "llm-fallback" and v.label == "correct"


def test_build_judge_dispatch():
    assert isinstance(build_judge("heuristic"), HeuristicJudge)
    assert isinstance(build_judge("llm", complete=lambda p: "PARTIAL\nx"), LLMJudge)
    with pytest.raises(ValueError):
        build_judge("llm")  # missing complete
