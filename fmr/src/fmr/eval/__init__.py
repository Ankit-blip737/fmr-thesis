"""Evaluation utilities owned by Instance B.

``judge`` — LLM-as-judge for open-ended answer correctness, with a validated
heuristic fallback and an agreement harness (required-fix #3: the judge must be
validated against human labels before any metric trusts it).
"""
from .judge import (
    HeuristicJudge,
    LLMJudge,
    JudgeVerdict,
    build_judge,
    evaluate_judge_agreement,
)

__all__ = [
    "HeuristicJudge",
    "LLMJudge",
    "JudgeVerdict",
    "build_judge",
    "evaluate_judge_agreement",
]
