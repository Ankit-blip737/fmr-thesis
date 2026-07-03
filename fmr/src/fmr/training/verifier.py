"""The faithfulness fusion head — heuristic fallback + learned drop-in.

``HeuristicFusion`` is the always-works, training-free score: a fixed weighted
sum of Signals A/B/C in [0, 1]. It is the guaranteed fallback (fix #4).

``LearnedVerifier`` is the optional trained upgrade: a small sklearn classifier
(logistic regression = learned-linear; gradient boosting = learned-nonlinear)
that predicts P(grounded) from the full feature vector. It exposes the *same*
``score(features) -> float`` interface as ``HeuristicFusion`` so it is a genuine
drop-in — the pipeline calls ``score`` and never needs to know which is behind it.

Persistence uses joblib when available, else pickle; both are optional so the
module imports with only numpy present (sklearn is needed only to *fit*).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .signals import SIGNAL_KEYS

# Full ordered feature vector the learned head consumes. The first three are the
# raw signals; the rest are aggregates/auxiliaries the heuristic cannot use.
FEATURE_KEYS = (
    "sig_a_counterfactual",
    "sig_b_grounding",
    "sig_c_consistency",
    "a_flip_rate",
    "a_js",
    "b_iou_max",
    "b_iou_first",
    "b_iou_last",
    "b_iou_slope",
    "c_answer_consistency",
    "c_region_consistency",
    "aux_answer_margin",
    "aux_n_steps",
)


def vectorize(features: dict, keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in keys], dtype=float)


@dataclass
class HeuristicFusion:
    """Fixed-weight fusion of Signals A/B/C -> FS in [0, 1] (the fallback)."""

    w_a: float = 0.5
    w_b: float = 0.3
    w_c: float = 0.2

    def score(self, features: dict) -> float:
        w = np.array([self.w_a, self.w_b, self.w_c])
        w = w / w.sum()
        vals = np.array([features.get(k, 0.0) for k in SIGNAL_KEYS], dtype=float)
        return float(np.clip(np.dot(w, vals), 0.0, 1.0))

    def score_batch(self, feats: Sequence[dict]) -> np.ndarray:
        return np.array([self.score(f) for f in feats])


class LearnedVerifier:
    """Learned fusion head (sklearn). Drop-in replacement for HeuristicFusion.

    ``model_kind``: 'gbt' (GradientBoosting, non-linear) or 'logreg' (linear).
    """

    def __init__(self, model_kind: str = "gbt", keys: Sequence[str] = FEATURE_KEYS,
                 model: Any = None, scaler: Any = None) -> None:
        self.model_kind = model_kind
        self.keys = tuple(keys)
        self.model = model
        self.scaler = scaler
        self._fitted = model is not None

    # --- training ---------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "LearnedVerifier":
        from sklearn.preprocessing import StandardScaler
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("LearnedVerifier.fit needs both classes present in y.")
        if self.model_kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            self.scaler = StandardScaler().fit(X)
            self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            self.model.fit(self.scaler.transform(X), y)
        elif self.model_kind == "gbt":
            from sklearn.ensemble import GradientBoostingClassifier
            self.scaler = None  # trees don't need scaling
            # Regularized for the small, weak-labelled training set (shallow trees,
            # subsampling, leaf floor) so it fuses signals instead of memorizing
            # weak-label noise.
            self.model = GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=seed
            )
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown model_kind {self.model_kind!r}.")
        self._fitted = True
        return self

    # --- inference (drop-in for HeuristicFusion.score) --------------------
    def _proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LearnedVerifier is not fitted.")
        Xt = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict_proba(Xt)[:, 1]

    def score(self, features: dict) -> float:
        x = vectorize(features, self.keys).reshape(1, -1)
        return float(self._proba(x)[0])

    def score_batch(self, feats: Sequence[dict]) -> np.ndarray:
        X = np.vstack([vectorize(f, self.keys) for f in feats])
        return self._proba(X)

    def feature_importance(self) -> Optional[dict]:
        if not self._fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
        else:
            return None
        return {k: float(v) for k, v in zip(self.keys, imp)}

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model_kind": self.model_kind, "keys": self.keys,
                         "model": self.model, "scaler": self.scaler}, fh)
        meta = {"model_kind": self.model_kind, "keys": list(self.keys),
                "importance": self.feature_importance()}
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LearnedVerifier":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(model_kind=d["model_kind"], keys=d["keys"], model=d["model"], scaler=d["scaler"])
