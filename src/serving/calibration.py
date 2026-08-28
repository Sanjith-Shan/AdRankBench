"""The served probability calibrator.

The score endpoint promises a calibrated click probability rather than a
ranking score, and those are different things. A ranker that orders ads
correctly can still be systematically confident or systematically shy, and in an
ad system that bias does not stay cosmetic. A bid is a click probability
multiplied by a value, and pacing spends against expected cost, so a probability
that is off by a constant factor turns straight into mispriced auctions. This is
the same argument docs/METHODOLOGY.md makes for reporting expected calibration
error next to AUC, moved onto the request path.

The calibrator here is a Platt scaler, which is a one dimensional logistic
regression fitted on the model logit. It is fitted on the validation split for
the same reason the INT8 calibration set in docs/INFERENCE.md comes from
validation. Fitting a correction on the test split would tune a parameter on the
evaluation data and make the reported calibration look better than it is.

Fitting it is not the same as shipping it. The build measures expected
calibration error on validation before and after, and keeps the scaler only when
it actually improves. When it does not, the bundle records an identity
calibrator and says so, and the service serves the model's own sigmoid output
untouched. A calibration layer that is applied without being checked is a second
place for the served numbers to drift away from the offline ones.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

# The clamp applied before a probability is turned back into a logit. A backend
# returns probabilities rather than logits, so the scaler has to invert the
# sigmoid, and an exact zero or one has no finite logit. The bound corresponds
# to a logit of about plus or minus 34, which is far outside anything this model
# produces, so the clamp is a guard rather than an active transform.
_EPS: float = 1e-15


def probabilities_to_logits(probs: np.ndarray) -> np.ndarray:
    """Invert the sigmoid on a probability array with a numerically safe clamp."""
    arr = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(arr) - np.log1p(-arr)


class Calibrator:
    """Applies a stored calibration to a batch of model probabilities.

    Two methods exist. The identity method returns the input unchanged and is
    what ships when the model is already well calibrated. The platt method maps
    the model logit through a fitted affine transform and back through a
    sigmoid.
    """

    def __init__(self, method: str = "identity", a: float = 1.0, b: float = 0.0) -> None:
        self.method = str(method)
        self.a = float(a)
        self.b = float(b)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for a batch of raw model outputs."""
        arr = np.asarray(probs, dtype=np.float64)
        if self.method != "platt":
            return arr
        from src.inference.common import sigmoid

        return sigmoid(self.a * probabilities_to_logits(arr) + self.b)

    def as_dict(self) -> Dict[str, Any]:
        """Return the json friendly record that goes into the bundle manifest."""
        return {"method": self.method, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, record: Optional[Mapping[str, Any]]) -> "Calibrator":
        """Rebuild a calibrator from a manifest record, defaulting to identity."""
        if not record:
            return cls()
        return cls(
            method=str(record.get("method", "identity")),
            a=float(record.get("a", 1.0)),
            b=float(record.get("b", 0.0)),
        )

    def describe(self) -> str:
        """Return the one line description the health endpoint reports."""
        if self.method != "platt":
            return "identity, the model sigmoid output is served unchanged"
        return f"platt scaling on the model logit with a {self.a:.4f} and b {self.b:.4f}"


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Fit a Platt scaler on held out probabilities and their labels.

    The fit is a logistic regression of the label on the single logit feature,
    which is the standard Platt formulation. Regularization is set very weak
    because there is only one coefficient and one intercept to estimate and
    shrinking them toward zero would flatten the very correction being fitted.
    """
    from sklearn.linear_model import LogisticRegression

    logits = probabilities_to_logits(probs).reshape(-1, 1)
    targets = np.asarray(labels, dtype=np.int64).ravel()
    if targets.size == 0 or len(np.unique(targets)) < 2:
        return Calibrator()
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, targets)
    return Calibrator(
        method="platt",
        a=float(model.coef_.ravel()[0]),
        b=float(model.intercept_.ravel()[0]),
    )


def select_calibrator(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Fit a Platt scaler on validation and keep it only when it helps.

    Returns the chosen calibrator alongside the expected calibration error
    before and after, so the decision is auditable from the manifest rather than
    being a claim in a docstring.
    """
    from src.evaluation.calibration import expected_calibration_error

    raw = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64).ravel()
    ece_before = float(expected_calibration_error(targets, raw, n_bins=n_bins))

    fitted = fit_platt(raw, targets)
    if fitted.method != "platt":
        return {
            "calibrator": Calibrator(),
            "ece_before": ece_before,
            "ece_after": ece_before,
            "applied": False,
            "reason": (
                "the validation split carried a single label class, so no "
                "calibration could be fitted and the model output is served "
                "unchanged"
            ),
        }

    adjusted = fitted.apply(raw)
    ece_after = float(expected_calibration_error(targets, adjusted, n_bins=n_bins))
    if ece_after < ece_before:
        return {
            "calibrator": fitted,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "applied": True,
            "reason": (
                f"platt scaling lowered validation expected calibration error "
                f"from {ece_before:.4f} to {ece_after:.4f}, so it is applied on "
                "the request path"
            ),
        }
    return {
        "calibrator": Calibrator(),
        "ece_before": ece_before,
        "ece_after": ece_before,
        "applied": False,
        "reason": (
            f"platt scaling did not lower validation expected calibration error, "
            f"which was {ece_before:.4f} raw against {ece_after:.4f} scaled, so "
            "the model output is served unchanged"
        ),
    }
