"""Logistic regression baseline.

This is the linear reference point for the benchmark. It uses scikit-learn and
only sees the dense numerical features plus the frequency encodings of the
sparse fields. It has no access to feature interactions, which is exactly why
the factorization and deep models are expected to beat it on data with real
pairwise signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel
from src.schema import Dataset, FeatureMeta


def _build_features(data: Dataset) -> np.ndarray:
    """Stack the dense numerical block with the frequency encoded sparse block.

    The logistic baseline does not use the hashed category indices or the
    crosses. Those carry no linear meaning. It uses the normalized frequency of
    each category instead, which is already a scaled value in [0, 1].
    """
    return np.hstack([data.numerical, data.cat_freq]).astype(np.float64)


class LogisticModel(BaseModel):
    """L2 regularized logistic regression over numerical and frequency features."""

    def __init__(self) -> None:
        self.clf = None
        self._constant_proba = None

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        meta: FeatureMeta,
        config: dict,
    ) -> "LogisticModel":
        """Fit on the training split. The validation split is ignored here.

        If the training labels happen to contain a single class we cannot fit a
        classifier, so we fall back to predicting that constant rate.
        """
        x = _build_features(train)
        y = train.label.astype(np.int64)

        unique = np.unique(y)
        if unique.shape[0] < 2:
            # Degenerate single class case. Predict the observed constant rate.
            self._constant_proba = float(unique[0])
            self.clf = None
            return self

        self._constant_proba = None
        self.clf = LogisticRegression(
            C=config["C"],
            max_iter=config["max_iter"],
            solver="lbfgs",
        )
        self.clf.fit(x, y)
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        """Return click probabilities for each row as a (N,) float array."""
        n = len(data)
        if self.clf is None:
            value = 0.0 if self._constant_proba is None else self._constant_proba
            return np.full(n, value, dtype=np.float64)
        x = _build_features(data)
        return self.clf.predict_proba(x)[:, 1]

    def get_name(self) -> str:
        return "LogisticRegression"
