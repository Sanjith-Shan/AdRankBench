"""Small shared helpers for the inference package.

These are the pieces every inference module needs and none of them owns. Keeping
them here means the TensorRT modules, the backend registry, and the benchmark
script all agree on the same sigmoid, the same not available marker, and the same
way of turning a raw byte count into something a reader can scan.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# The single string used everywhere a measurement could not be taken. Reports
# print this instead of a number so an absent value never reads like a zero.
NOT_AVAILABLE: str = "not available"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid over a numpy array.

    The positive and negative halves use different but algebraically identical
    forms so neither branch ever evaluates exp of a large positive number.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_neg = np.exp(arr[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


def human_bytes(n: Optional[float]) -> str:
    """Format a byte count as a short human readable string."""
    if n is None:
        return NOT_AVAILABLE
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def fmt(value: Any, spec: str = ".3f") -> str:
    """Format a number for a report cell, or return the not available marker.

    A None value, a non numeric value, and a NaN all collapse to the same marker
    so a report never shows an empty cell or a misleading zero.
    """
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if np.isnan(number) or np.isinf(number):
        return NOT_AVAILABLE
    return format(number, spec)


def jsonable(obj: Any) -> Any:
    """Convert numpy scalars and arrays into plain python for json.dump.

    The benchmark writes raw measurements into a json artifact and numpy types
    are not serializable by default, so every record passes through here first.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def percentile(values: np.ndarray, q: float) -> float:
    """Return a percentile of a latency array, or nan when it is empty."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def merge_notes(*notes: Optional[str]) -> str:
    """Join the non empty notes from several probes into one sentence."""
    parts = [n.strip() for n in notes if n]
    return " ".join(parts)


def empty_record() -> Dict[str, Any]:
    """Return a fresh dict, used as the default for optional record fields."""
    return {}
