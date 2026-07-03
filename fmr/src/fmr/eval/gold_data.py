"""Hand-authored judge gold set (required-fix #3).

Each row is a (question, prediction, reference, human-label) tuple that a
clinically-literate rater would assign. The set is deliberately adversarial for
a naive string-match judge: it stresses synonyms, negation/polarity flips,
severity gradations, containment, and partial-credit cases — so the reported
agreement genuinely certifies the judge rather than rewarding exact-match luck.

`label` in {"correct","partial","incorrect"}. These are *my* labels authored in
lieu of an external annotator on this CPU box (flagged in BLOCKERS.md); the LLM
judge is additionally cross-checked against them on Colab. Provenance is marked
so the thesis can report "synthetic gold, N=..." honestly.
"""
from __future__ import annotations

JUDGE_GOLD: list[dict] = [
    # --- exact / trivial correct ---
    {"question": "Is there cardiomegaly?", "prediction": "yes", "reference": "yes", "label": "correct"},
    {"question": "Is the study normal?", "prediction": "normal", "reference": "normal", "label": "correct"},
    # --- synonym clusters -> correct ---
    {"question": "Is there cardiomegaly?", "prediction": "enlarged heart", "reference": "cardiomegaly", "label": "correct"},
    {"question": "Any fluid in the pleural space?", "prediction": "pleural effusion", "reference": "effusion", "label": "correct"},
    {"question": "Is the study normal?", "prediction": "unremarkable", "reference": "normal", "label": "correct"},
    {"question": "Is a mass seen?", "prediction": "there is a nodule", "reference": "mass", "label": "correct"},
    {"question": "Bleeding present?", "prediction": "haemorrhage", "reference": "hemorrhage", "label": "correct"},
    {"question": "Is there consolidation?", "prediction": "opacification", "reference": "consolidation", "label": "correct"},
    {"question": "How severe?", "prediction": "marked", "reference": "severe", "label": "correct"},
    {"question": "Finding present?", "prediction": "positive", "reference": "present", "label": "correct"},
    # --- negation / polarity flips -> incorrect ---
    {"question": "Is there effusion?", "prediction": "no effusion", "reference": "effusion", "label": "incorrect"},
    {"question": "Is there pneumothorax?", "prediction": "pneumothorax present", "reference": "no pneumothorax", "label": "incorrect"},
    {"question": "Is there a fracture?", "prediction": "yes", "reference": "no", "label": "incorrect"},
    {"question": "Cardiomegaly?", "prediction": "absent", "reference": "present", "label": "incorrect"},
    {"question": "Is the lung normal?", "prediction": "no abnormality", "reference": "consolidation", "label": "incorrect"},
    {"question": "Is there edema?", "prediction": "free of edema", "reference": "edema", "label": "incorrect"},
    # --- wrong finding -> incorrect ---
    {"question": "What abnormality?", "prediction": "effusion", "reference": "cardiomegaly", "label": "incorrect"},
    {"question": "What abnormality?", "prediction": "fracture", "reference": "mass", "label": "incorrect"},
    {"question": "Which lobe?", "prediction": "left lower lobe", "reference": "right upper lobe", "label": "incorrect"},
    {"question": "Severity?", "prediction": "mild", "reference": "severe", "label": "incorrect"},
    {"question": "Severity?", "prediction": "severe", "reference": "mild", "label": "incorrect"},
    # --- containment, matching polarity -> correct ---
    {"question": "Describe the finding.", "prediction": "there is moderate cardiomegaly", "reference": "cardiomegaly", "label": "correct"},
    {"question": "Any pneumothorax?", "prediction": "no", "reference": "no pneumothorax", "label": "correct"},
    {"question": "Finding?", "prediction": "large pleural effusion on the right", "reference": "pleural effusion", "label": "correct"},
    # --- partial credit (underspecified / incomplete) ---
    {"question": "What are the findings?", "prediction": "cardiomegaly", "reference": "cardiomegaly and pleural effusion", "label": "partial"},
    {"question": "Location and severity?", "prediction": "left side", "reference": "severe left-sided effusion", "label": "partial"},
    {"question": "Describe fully.", "prediction": "mass in the lung", "reference": "spiculated mass in the right upper lobe", "label": "partial"},
    {"question": "What abnormalities?", "prediction": "effusion", "reference": "effusion, atelectasis, cardiomegaly", "label": "partial"},
    # --- empty / evasive -> incorrect ---
    {"question": "Is there a nodule?", "prediction": "", "reference": "yes", "label": "incorrect"},
    {"question": "Diagnosis?", "prediction": "cannot determine", "reference": "pneumonia", "label": "incorrect"},
    # --- more synonym/paraphrase correct ---
    {"question": "Is the heart enlarged?", "prediction": "enlarged cardiac silhouette", "reference": "cardiomegaly", "label": "correct"},
    {"question": "Any lesion?", "prediction": "neoplasm noted", "reference": "tumor", "label": "correct"},
    {"question": "Is it present?", "prediction": "evident", "reference": "seen", "label": "correct"},
    {"question": "Fracture?", "prediction": "broken", "reference": "fracture", "label": "correct"},
    {"question": "How much?", "prediction": "minimal", "reference": "mild", "label": "correct"},
    # --- tricky: same words, opposite meaning -> incorrect ---
    {"question": "Is there effusion?", "prediction": "no evidence of effusion", "reference": "effusion present", "label": "incorrect"},
    {"question": "Pneumothorax?", "prediction": "rule out pneumothorax", "reference": "pneumothorax", "label": "incorrect"},
    # --- numeric / laterality correct ---
    {"question": "Which side?", "prediction": "right", "reference": "right side", "label": "correct"},
    {"question": "How many nodules?", "prediction": "two nodules", "reference": "2 nodules", "label": "correct"},
    # --- borderline partial vs incorrect ---
    {"question": "Full findings?", "prediction": "normal heart", "reference": "cardiomegaly with effusion", "label": "incorrect"},
    {"question": "Findings?", "prediction": "opacity in lung", "reference": "left lower lobe opacity", "label": "partial"},
    # --- more clear correct/incorrect to balance classes ---
    {"question": "Is there atelectasis?", "prediction": "yes", "reference": "positive", "label": "correct"},
    {"question": "Normal exam?", "prediction": "abnormal", "reference": "unremarkable", "label": "incorrect"},
    {"question": "Any acute finding?", "prediction": "none", "reference": "no acute finding", "label": "correct"},
]


def label_distribution() -> dict:
    d: dict[str, int] = {}
    for row in JUDGE_GOLD:
        d[row["label"]] = d.get(row["label"], 0) + 1
    return d
