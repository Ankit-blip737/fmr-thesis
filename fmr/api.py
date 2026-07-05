"""FastAPI backend for LIVE FMR inference (Google Colab GPU + ngrok).

This is the "dynamic" half of the thesis demo: the static dashboard shows the
pre-computed benchmark; this endpoint runs the *actual* FMR pipeline on a single
image + question uploaded from the browser, on a free Colab T4.

Contract (what the Vercel frontend calls):
    POST /analyze   multipart form:
        file      : the medical image (X-ray / CT / ... )   [required]
        question  : the clinical question                   [required]
        choices   : optional comma-separated answer choices (e.g. "yes,no")
        alpha,delta: optional conformal targets (default from env / experiment.yaml)
    -> JSON: { fmr_score, decision ("answer"|"abstain"), model_answer,
               signals{A,B,C}, reasoning_steps, gate{...}, ... }

It reuses the real pipeline unchanged:
  * ``HFVLM``                     — the frozen MedVLM-R1 reasoning VLM wrapper.
  * ``compute_faithfulness``      — Signals A/B/C -> fused Faithfulness Score (FS),
                                    with ``n_consistency_samples = 5`` (the "5
                                    consistency passes" the UI advertises).
  * ``calibrate_threshold``       — the conformal gate (module
                                    ``fmr.abstention.conformal``; your prompt
                                    called it ``fmr.conformal``). Answer iff
                                    ``FS >= tau``, exactly like the dashboard's
                                    ANSWER/ABSTAIN logic (``decisionFor``).

The gate threshold ``tau`` is calibrated *once* at startup from a persisted
calibration split (``outputs/real/slake/fmr_records.json`` by default — the
largest real split, n=75, and the only one whose calibration set admits a
non-trivial operating point at the default alpha; vqa_rad/pathvqa are abstain-all
at that n). A single live sample cannot self-calibrate a distribution-free bound.
If the split-conformal bound is infeasible at the (small) real calibration n, we
fall back to the same honest, uncertified "operating point" the dashboard's
alpha-slider uses and label it as such (``certified: false``) rather than
fabricating a guarantee. Override the split with ``FMR_CAL_RECORDS``.

Run locally / on Colab:  ``uvicorn api:app --host 0.0.0.0 --port 8000``
(see the Colab cell in ``colab_serve_api.py`` for the pyngrok tunnel).
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# --- Make the `fmr` package importable without an editable install ----------
# Mirrors scripts/_common.py so this file runs straight from a git clone on
# Colab (`fmr/src` holds the package: fmr.models.hf_vlm, fmr.abstention, ...).
_HERE = Path(__file__).resolve().parent          # .../<repo>/fmr
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fmr.abstention import calibrate_threshold          # the conformal gate logic
from fmr.faithfulness import DEFAULT_WEIGHTS, compute_faithfulness
from fmr.models.hf_vlm import HFVLM
from fmr.types import Sample

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("FMR_MODEL_ID", "JZPeterPan/MedVLM-R1")
N_CONSISTENCY = int(os.environ.get("FMR_N_CONSISTENCY", 5))    # the "5 passes"
CONSISTENCY_TEMPERATURE = float(os.environ.get("FMR_CONS_TEMP", 0.7))
ALPHA = float(os.environ.get("FMR_ALPHA", 0.15))               # retained-error target
DELTA = float(os.environ.get("FMR_DELTA", 0.05))               # guarantee failure prob
# Persisted calibration split the gate is calibrated on (produced by run_fmr.py).
# Default: slake — largest real cal set (n=75) and the only one that yields a
# usable ANSWER/ABSTAIN threshold at the default alpha. Override via env.
CAL_RECORDS_PATH = Path(
    os.environ.get("FMR_CAL_RECORDS",
                   str(_HERE / "outputs" / "real" / "slake" / "fmr_records.json"))
)

# One frozen model on one GPU: load lazily, and serialize inference so two
# in-flight requests can't corrupt each other's CUDA state / OOM the T4.
_LOCK = threading.Lock()
_VLM: Optional[HFVLM] = None
_GATE_CACHE: dict[tuple[float, float], dict[str, Any]] = {}


def get_vlm() -> HFVLM:
    """Lazily construct the MedVLM-R1 wrapper (weights load on first ``generate``)."""
    global _VLM
    if _VLM is None:
        _VLM = HFVLM(model_id=MODEL_ID, device="cuda", is_reasoning=True)
    return _VLM


# --------------------------------------------------------------------------- #
# Conformal gate (calibrated once from the persisted calibration split)        #
# --------------------------------------------------------------------------- #
def _load_calibration() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (fs, correct) arrays from the calibration split, or (None, None)."""
    if not CAL_RECORDS_PATH.exists():
        return None, None
    try:
        data = json.loads(CAL_RECORDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    cal = data.get("cal") or []
    if not cal:
        return None, None
    fs = np.asarray([r["fs"] for r in cal], dtype=float)
    correct = np.asarray([r["correct"] for r in cal], dtype=int)
    return fs, correct


def _empirical_threshold(fs: np.ndarray, correct: np.ndarray, alpha: float) -> Optional[float]:
    """Largest-coverage FS cut whose *empirical* retained error <= alpha on the
    calibration set. This mirrors the dashboard's alpha-slider ("operating point
    computed client-side from the risk-coverage data") — an honest fallback, NOT
    a certified bound — used only when the conformal SGR bound is infeasible at
    the small real calibration n."""
    order = np.argsort(-fs, kind="stable")   # answer highest-FS first
    err = 1 - correct[order]
    fss = fs[order]
    best_tau: Optional[float] = None
    for k in range(1, len(fss) + 1):
        if err[:k].mean() <= alpha:
            best_tau = float(fss[k - 1])     # min FS among the retained top-k
    return best_tau


def build_gate(alpha: float = ALPHA, delta: float = DELTA) -> dict[str, Any]:
    """Calibrate the ANSWER/ABSTAIN threshold. Prefers the certified split-conformal
    bound; falls back to an honest uncertified operating point when infeasible."""
    fs, correct = _load_calibration()
    if fs is None or len(fs) == 0:
        return {"threshold": 0.5, "alpha": alpha, "delta": delta,
                "feasible": False, "certified": False, "n_cal": 0,
                "mode": "default tau=0.5 (no calibration split found)"}

    res = calibrate_threshold(fs, correct, alpha=alpha, delta=delta)
    if res.feasible:
        return {"threshold": float(res.threshold), "alpha": alpha, "delta": delta,
                "feasible": True, "certified": True, "n_cal": int(res.n_cal),
                "cal_coverage": float(res.cal_coverage),
                "cal_error_ucb": float(res.cal_error_ucb),
                "mode": "split-conformal SGR bound (distribution-free, certified)"}

    tau = _empirical_threshold(fs, correct, alpha)
    if tau is None:                          # nothing meets alpha -> safe default
        return {"threshold": float("inf"), "alpha": alpha, "delta": delta,
                "feasible": False, "certified": False, "n_cal": int(len(fs)),
                "mode": "abstain-all (no FS cut meets alpha on the calibration set)"}
    return {"threshold": float(tau), "alpha": alpha, "delta": delta,
            "feasible": False, "certified": False, "n_cal": int(len(fs)),
            "mode": "empirical operating point (uncertified — small calibration n)"}


def gate_for(alpha: float, delta: float) -> dict[str, Any]:
    key = (round(float(alpha), 4), round(float(delta), 4))
    if key not in _GATE_CACHE:
        _GATE_CACHE[key] = build_gate(alpha, delta)
    return _GATE_CACHE[key]


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(title="FMR Live API", version="1.0",
              description="Live Faithful-Medical-Reasoning inference (MedVLM-R1 + conformal gate).")

# CORS: allow the Vercel frontend (any origin) to POST here. No cookies are used,
# so credentials stay off — required when allow_origins is the wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "FMR Live API",
        "model": MODEL_ID,
        "n_consistency": N_CONSISTENCY,
        "endpoints": {"analyze": "POST /analyze (multipart: file, question)"},
        "gate": gate_for(ALPHA, DELTA),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_loaded": _VLM is not None, "model": MODEL_ID}


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    question: str = Form(...),
    choices: str = Form(""),
    alpha: float = Form(ALPHA),
    delta: float = Form(DELTA),
) -> dict[str, Any]:
    """Run the full FMR pipeline on one uploaded (image, question) and gate it.

    Signal A samples the original + blank + mismatched image (3 passes); Signal C
    resamples the reasoning chain N_CONSISTENCY (=5) times — this is the GPU work
    the UI spinner narrates. Returns the fused FMR score and the conformal
    ANSWER/ABSTAIN decision.
    """
    from PIL import Image

    if not question or not question.strip():
        raise HTTPException(status_code=422, detail="`question` is required.")
    try:
        raw = file.file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface a clean 400 to the browser
        raise HTTPException(status_code=400, detail=f"Could not read image file: {exc}")

    choice_list = [c.strip() for c in choices.split(",") if c.strip()] or None
    # answer="" — ground truth is unknown for a live case (used only for eval,
    # never for the FMR score or the gate). modality is cosmetic here.
    sample = Sample(
        sample_id="live",
        question=question.strip(),
        answer="",
        modality="xray",
        image=image,
        answer_choices=choice_list,
    )

    t0 = time.time()
    with _LOCK:  # one T4, one model — serialize the ~8 forward passes per request
        vlm = get_vlm()
        # For a single live image there is no dataset pool for the 'mismatch'
        # counterfactual variant, so HFVLM falls back to its documented
        # content-destroying transform (rotate 180°). That is sufficient for the
        # image-reliance probe on one case.
        record = compute_faithfulness(
            vlm,
            sample,
            weights=DEFAULT_WEIGHTS,
            n_consistency_samples=N_CONSISTENCY,
            consistency_temperature=CONSISTENCY_TEMPERATURE,
        )
    elapsed = time.time() - t0

    gate = gate_for(float(alpha), float(delta))
    fs = float(record["fs"])
    decision = "answer" if fs >= gate["threshold"] else "abstain"

    return {
        "fmr_score": fs,                                   # the fused Faithfulness Score
        "decision": decision,                              # "answer" | "abstain"
        "abstain": decision == "abstain",
        "model_answer": record["answer"],
        "question": sample.question,
        "signals": {
            "counterfactual_A": float(record["signal_a"]),
            "grounding_B": float(record["signal_b"]),
            "consistency_C": float(record["signal_c"]),
        },
        "confidence": float(record["confidence"]),
        "reasoning_steps": list(record["steps_text"]),
        "fs_per_step": [float(x) for x in record["fs_per_step"]],
        "n_consistency": N_CONSISTENCY,
        "gate": gate,                                      # threshold, alpha, feasible, certified, mode
        "elapsed_seconds": round(elapsed, 2),
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
