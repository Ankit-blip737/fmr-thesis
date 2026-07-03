"""LLM-as-judge for open-ended medical-VQA answer correctness — with validation.

Why this file is careful. A generic LLM scoring clinical answer correctness is an
*underspecified* risk: if the judge is wrong, every open-ended metric downstream
(including Instance A's Stage-6 benchmark) inherits that error silently. So this
module ships three things, not one:

1. ``HeuristicJudge`` — a deterministic, dependency-free judge (normalization +
   clinical synonym/negation-aware matching). It is the always-works fallback and
   the thing we can validate offline on this CPU box.
2. ``LLMJudge`` — the real LLM judge behind a provider-agnostic ``complete``
   callable, with a rubric-constrained prompt. No network is required to import
   it; a callable is injected (open-LLM on Colab, or an API judge if a key is
   supplied — see BLOCKERS.md).
3. ``evaluate_judge_agreement`` — the required validation: judge-vs-gold
   agreement (accuracy, Cohen's κ, per-class precision/recall). Nothing should
   trust a judge whose κ against human labels is weak.

Verdicts are ``correct`` / ``incorrect`` / ``partial`` (mapped to a score in
[0, 1]); the calibration harness treats ``partial`` as configurable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

# --------------------------------------------------------------------------- #
# Verdict type
# --------------------------------------------------------------------------- #

_LABELS = ("correct", "partial", "incorrect")
_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class JudgeVerdict:
    label: str                          # one of _LABELS
    score: float                        # _SCORE[label]
    rationale: str = ""
    source: str = "heuristic"           # heuristic | llm | llm-fallback
    meta: dict = field(default_factory=dict)

    @classmethod
    def of(cls, label: str, rationale: str = "", source: str = "heuristic", **meta: Any) -> "JudgeVerdict":
        label = label if label in _LABELS else "incorrect"
        return cls(label=label, score=_SCORE[label], rationale=rationale, source=source, meta=meta)


# --------------------------------------------------------------------------- #
# Clinical text normalization (shared by both judges)
# --------------------------------------------------------------------------- #

# Small, illustrative clinical synonym clusters. Extend with UMLS for production.
_SYNONYMS: list[set[str]] = [
    {"absent", "none", "no", "negative", "not present", "no evidence"},
    {"present", "yes", "positive", "seen", "evident"},
    {"cardiomegaly", "enlarged heart", "enlarged cardiac silhouette"},
    {"effusion", "pleural effusion", "fluid"},
    {"opacity", "opacities", "opacification", "consolidation"},
    {"mass", "lesion", "nodule", "tumor", "tumour", "neoplasm"},
    {"hemorrhage", "haemorrhage", "bleed", "bleeding"},
    {"fracture", "break", "broken"},
    {"mild", "slight", "minimal"},
    {"moderate", "intermediate"},
    {"severe", "marked", "extensive"},
    {"normal", "unremarkable", "no abnormality"},
]
# Flat set of single-token synonyms, for conservative plural stripping.
_SYNONYM_TOKENS = {t for cluster in _SYNONYMS for t in cluster if " " not in t}
_NEG = re.compile(r"\b(no|not|without|absent|negative|denies|free of|rule[sd]? out|none)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
# Clusters that denote "nothing abnormal" — used for polarity, never as a finding.
_ABSENT_CLUSTERS = (
    frozenset({"absent", "none", "no", "negative", "not present", "no evidence"}),
    frozenset({"normal", "unremarkable", "no abnormality"}),
)
# Modifier clusters (polarity + severity) — never counted as *findings* (they
# describe a finding rather than being one), so they must not trigger the
# multi-finding coverage logic or stand in for a pathology.
_AFFIRM_CLUSTER = frozenset({"present", "yes", "positive", "seen", "evident"})
_SEVERITY_CLUSTERS = (
    frozenset({"mild", "slight", "minimal"}),
    frozenset({"moderate", "intermediate"}),
    frozenset({"severe", "marked", "extensive"}),
)
_MODIFIER_CLUSTERS = set(_ABSENT_CLUSTERS) | {_AFFIRM_CLUSTER} | set(_SEVERITY_CLUSTERS)
_MULTI_SPLIT = re.compile(r"\s*(?:,|;|/|\band\b|\bplus\b|&|\bwith\b)\s*")
_STOPWORDS = {"in", "the", "of", "on", "is", "there", "a", "an", "at", "to", "and",
              "with", "no", "seen", "noted", "present", "evident", "this", "region"}


def _singularize(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es") and tok[:-2] in _SYNONYM_TOKENS:
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = _PUNCT.sub(" ", text)
    toks = [_NUM_WORDS.get(t, t) for t in text.split()]
    toks = [_singularize(t) for t in toks]
    return _WS.sub(" ", " ".join(toks)).strip()


def _synset(token_str: str) -> Optional[frozenset]:
    """Whole-string cluster lookup (kept for back-compat)."""
    for cluster in _SYNONYMS:
        if token_str in cluster:
            return frozenset(cluster)
    return None


def _token_synsets(text: str) -> set[frozenset]:
    """Clusters hit by ANY token/bigram of ``text`` (token-level synonym match)."""
    toks = text.split()
    grams = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
    hits = set()
    for cluster in _SYNONYMS:
        if grams & cluster:
            hits.add(frozenset(cluster))
    return hits


def _finding_synsets(text: str) -> set[frozenset]:
    """Token-level clusters that denote actual findings (exclude polarity/severity)."""
    return {c for c in _token_synsets(text) if c not in _MODIFIER_CLUSTERS}


def _polarity(text: str) -> str:
    """'neg' if the text asserts absence/normality, else 'pos'."""
    if _NEG.search(" " + text + " "):
        return "neg"
    if _token_synsets(text) & set(_ABSENT_CLUSTERS):
        return "neg"
    return "pos"


def _has_negation(text: str) -> bool:
    return _polarity(text) == "neg"


# --------------------------------------------------------------------------- #
# Heuristic judge (offline fallback + validation target)
# --------------------------------------------------------------------------- #


class HeuristicJudge:
    """Deterministic answer-correctness judge.

    Decision order:
      1. exact normalized match             -> correct
      2. same synonym cluster               -> correct
      3. containment either direction, with matching polarity -> correct
      4. containment but *mismatched* negation polarity        -> incorrect
      5. token-overlap Jaccard >= partial_threshold            -> partial
      6. otherwise                                             -> incorrect
    """

    source = "heuristic"

    def __init__(self, partial_threshold: float = 0.5) -> None:
        self.partial_threshold = partial_threshold

    def __call__(self, question: str, prediction: str, reference: str) -> JudgeVerdict:
        p, r = normalize(prediction), normalize(reference)
        if not p:
            return JudgeVerdict.of("incorrect", "empty prediction")
        if p == r:
            return JudgeVerdict.of("correct", "exact match")

        pol_p, pol_r = _polarity(p), _polarity(r)
        fs_p, fs_r = _finding_synsets(p), _finding_synsets(r)

        # (1) Multi-finding reference: score coverage of each listed finding.
        # Detect via conjunctions OR >=2 distinct finding clusters (commas were
        # stripped by normalization, so cluster-count is the robust signal).
        ref_parts = [x for x in _MULTI_SPLIT.split(r) if x.strip()]
        if len(ref_parts) < 2 and len(fs_r) >= 2:
            ref_parts = sorted(next(iter(c & set(r.split()))) for c in fs_r if c & set(r.split()))
        if len(ref_parts) >= 2:
            covered = sum(self._covers(p, fs_p, pol_p, part) for part in ref_parts)
            if covered == len(ref_parts):
                return JudgeVerdict.of("correct", "all listed findings covered")
            if covered >= 1:
                return JudgeVerdict.of("partial", f"{covered}/{len(ref_parts)} findings covered")
            return JudgeVerdict.of("incorrect", "no listed finding covered")

        # (2) Both sides assert absence/normality (no positive finding either) -> agree.
        if pol_p == "neg" and pol_r == "neg" and not fs_p and not fs_r:
            return JudgeVerdict.of("correct", "both assert absence/normality")

        # (3) Shared finding cluster with matching polarity -> correct, unless the
        #     reference specifies materially more (severity/laterality/etc.) than
        #     the prediction, in which case the answer is incomplete -> partial.
        if fs_p & fs_r:
            if pol_p != pol_r:
                return JudgeVerdict.of("incorrect", "shared finding, opposite polarity")
            if self._underspecified(p, r):
                return JudgeVerdict.of("partial", "correct finding but reference more specific")
            return JudgeVerdict.of("correct", "shared finding cluster, matching polarity")

        # (4) Same severity/other cluster (whole-string) -> correct.
        sp, sr = _synset(p), _synset(r)
        if sp is not None and sp == sr and pol_p == pol_r:
            return JudgeVerdict.of("correct", "synonym-cluster match")

        # (5) Opposite polarity on the same subject -> incorrect.
        if pol_p != pol_r and (fs_p & fs_r or (p in r) or (r in p) or fs_p == fs_r):
            return JudgeVerdict.of("incorrect", "opposite polarity")

        # (6) Containment with matching polarity -> correct, unless the prediction
        #     is the contained (shorter) side and gave only a modifier/location
        #     while the reference names a finding -> incomplete -> partial.
        if (p in r) or (r in p):
            if pol_p != pol_r:
                return JudgeVerdict.of("incorrect", "containment but opposite polarity")
            if (p in r) and fs_r and not fs_p:
                return JudgeVerdict.of("partial", "prediction omits the named finding")
            return JudgeVerdict.of("correct", "containment, matching polarity")

        # (7) Distinct finding clusters named on each side -> wrong finding.
        if fs_p and fs_r and not (fs_p & fs_r):
            return JudgeVerdict.of("incorrect", "different finding clusters")

        # (8) Fall back to lexical overlap.
        tp, tr = set(p.split()), set(r.split())
        jac = len(tp & tr) / len(tp | tr) if (tp | tr) else 0.0
        if jac >= self.partial_threshold:
            return JudgeVerdict.of("partial", f"token overlap {jac:.2f}", jaccard=jac)
        return JudgeVerdict.of("incorrect", f"token overlap {jac:.2f}", jaccard=jac)

    @staticmethod
    def _covers(pred: str, pred_findings: set, pred_pol: str, ref_part: str) -> bool:
        """Does the prediction account for one finding from a multi-finding ref?"""
        rp = ref_part.strip()
        if pred_pol != _polarity(rp):        # a polarity flip never "covers" a finding
            return False
        part_findings = _finding_synsets(rp)
        if part_findings and (pred_findings & part_findings):
            return True
        # non-clustered finding word: require a shared content token
        toks = (set(rp.split()) & set(pred.split())) - _STOPWORDS
        return bool(toks)

    @staticmethod
    def _underspecified(pred: str, ref: str) -> bool:
        """True when the reference carries >=2 content tokens the prediction lacks
        (severity/laterality/co-descriptors) — an incomplete but on-topic answer."""
        extra = set(ref.split()) - set(pred.split()) - _STOPWORDS
        return len(extra) >= 2


# --------------------------------------------------------------------------- #
# LLM judge (rubric-constrained; provider-agnostic)
# --------------------------------------------------------------------------- #


class Completer(Protocol):
    def __call__(self, prompt: str) -> str: ...


_RUBRIC = """You are a careful medical examiner grading a candidate's answer to a
visual medical question. You are given the QUESTION, the REFERENCE (gold) answer,
and the CANDIDATE answer. Grade ONLY whether the candidate is clinically
equivalent to the reference — ignore verbosity, phrasing, and reasoning.

Reply with exactly one word on the first line: CORRECT, PARTIAL, or INCORRECT.
- CORRECT: clinically equivalent to the reference (synonyms/negations count).
- PARTIAL: captures some but not all of the reference's clinical content, or is
  underspecified.
- INCORRECT: contradicts the reference, wrong finding, or wrong polarity.
Then, on a second line, give a one-sentence justification.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {prediction}
"""

_VERDICT_RE = re.compile(r"\b(correct|partial|incorrect)\b", re.IGNORECASE)


class LLMJudge:
    """Rubric-constrained LLM judge. Robust to malformed completions.

    ``complete`` is any callable ``str -> str`` (open-LLM generate on Colab, or an
    API call). If it errors or returns something unparseable, the judge falls
    back to ``HeuristicJudge`` and marks the source, so a flaky judge can never
    silently emit garbage verdicts.
    """

    source = "llm"

    def __init__(self, complete: Completer, fallback: Optional[HeuristicJudge] = None,
                 rubric: str = _RUBRIC) -> None:
        self.complete = complete
        self.fallback = fallback or HeuristicJudge()
        self.rubric = rubric

    def __call__(self, question: str, prediction: str, reference: str) -> JudgeVerdict:
        prompt = self.rubric.format(question=question, reference=reference, prediction=prediction)
        try:
            raw = self.complete(prompt)
        except Exception as exc:  # provider hiccup -> heuristic, never crash the eval
            v = self.fallback(question, prediction, reference)
            return JudgeVerdict.of(v.label, f"llm error -> heuristic: {exc}", source="llm-fallback")
        m = _VERDICT_RE.search(raw or "")
        if not m:
            v = self.fallback(question, prediction, reference)
            return JudgeVerdict.of(v.label, "unparseable llm output -> heuristic", source="llm-fallback")
        label = m.group(1).lower()
        rationale = (raw.strip().splitlines() or [""])[-1][:300]
        return JudgeVerdict.of(label, rationale, source="llm", raw=raw[:500])


def build_judge(kind: str = "heuristic", complete: Optional[Completer] = None,
                **kw: Any) -> Callable[[str, str, str], JudgeVerdict]:
    if kind == "heuristic":
        return HeuristicJudge(**kw)
    if kind == "llm":
        if complete is None:
            raise ValueError("LLM judge needs a `complete` callable (str->str).")
        return LLMJudge(complete=complete, **kw)
    raise ValueError(f"Unknown judge kind {kind!r}.")


# --------------------------------------------------------------------------- #
# Validation harness (required-fix #3)
# --------------------------------------------------------------------------- #


def _cohens_kappa(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = _LABELS) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        conf[idx[t]][idx[p]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) / n for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 1.0


def evaluate_judge_agreement(
    judge: Callable[[str, str, str], JudgeVerdict],
    gold: list[dict],
) -> dict:
    """Agreement of ``judge`` vs human ``gold`` labels.

    ``gold`` items: ``{question, prediction, reference, label}`` with ``label``
    in _LABELS. Returns accuracy, Cohen's κ, per-class precision/recall, a binary
    (correct-vs-not) accuracy, and the confusion matrix — everything needed to
    decide whether the judge is trustworthy.
    """
    y_true = [g["label"] for g in gold]
    verdicts = [judge(g["question"], g["prediction"], g["reference"]) for g in gold]
    y_pred = [v.label for v in verdicts]
    n = len(gold)

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else float("nan")
    # Binary collapse: correct vs {partial, incorrect}. This is what most
    # downstream accuracy metrics actually consume.
    bt = [t == "correct" for t in y_true]
    bp = [p == "correct" for p in y_pred]
    bin_acc = sum(t == p for t, p in zip(bt, bp)) / n if n else float("nan")

    per_class = {}
    for lab in _LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        per_class[lab] = {"precision": prec, "recall": rec, "support": tp + fn}

    conf = {t: {p: 0 for p in _LABELS} for t in _LABELS}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    return {
        "n": n,
        "accuracy": acc,
        "binary_accuracy(correct_vs_not)": bin_acc,
        "cohens_kappa": _cohens_kappa(y_true, y_pred),
        "per_class": per_class,
        "confusion": conf,
        "sources": {s: sum(v.source == s for v in verdicts) for s in {"heuristic", "llm", "llm-fallback"}},
    }
