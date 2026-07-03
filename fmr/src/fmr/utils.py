"""Small shared utilities: seeding, config loading, and probability helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a declared dependency
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs (`pip install pyyaml`).")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def seed_from(*parts: Any) -> int:
    """Deterministic 32-bit seed derived from arbitrary (hashable) parts.

    Used so the MockVLM is reproducible per (sample, variant, draw) without any
    global RNG state leaking between calls.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=4).hexdigest(), 16)


def rng_from(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(seed_from(*parts))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temperature, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions, in [0, 1] (log base 2)."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log2(a) - np.log2(b))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    """JSON-dump with numpy-type coercion; returns the written path."""
    import json

    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return p
