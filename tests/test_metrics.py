"""Tests for the ranking and calibration metrics.

These tests use tiny synthetic arrays and never train a model. They check that
perfect separation gives an AUC of 1.0, that random predictions give an AUC near
0.5, that near perfect predictions give a low logloss, that perfectly calibrated
synthetic data gives an ECE near 0, that group AUC behaves sanely on a tiny
example, and that predicting the base rate constant gives a normalized entropy
near 1.0.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_auc,
    compute_logloss,
    group_auc,
    normalized_entropy,
)
from src.evaluation.calibration import expected_calibration_error


def test_auc_perfect_separation_is_one():
    """Predictions that perfectly order the classes give an AUC of 1.0."""
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(y_true, y_pred) == 1.0


def test_auc_random_is_near_half():
    """Random predictions give an AUC close to 0.5."""
    rng = np.random.default_rng(42)
    n = 4000
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    auc = compute_auc(y_true, y_pred)
    assert abs(auc - 0.5) < 0.05


def test_logloss_of_near_perfect_predictions_is_low():
    """Confident and correct predictions give a small logloss."""
    y_true = [0, 0, 1, 1]
    y_pred = [0.01, 0.02, 0.98, 0.99]
    ll = compute_logloss(y_true, y_pred)
    assert ll < 0.1


def test_ece_of_perfectly_calibrated_data_is_near_zero():
    """When the predicted probability equals the observed fraction ECE is near 0.

    We build several blocks. Inside each block every prediction equals a fixed
    probability p and exactly that fraction of the labels are positive. Each
    block lands in its own equal width bin, so the mean prediction matches the
    observed positive fraction in every bin and the calibration error vanishes.
    """
    probs = [0.05, 0.25, 0.55, 0.85]
    block = 1000
    y_true = []
    y_pred = []
    for p in probs:
        n_pos = int(round(p * block))
        labels = [1] * n_pos + [0] * (block - n_pos)
        y_true.extend(labels)
        y_pred.extend([p] * block)

    ece = expected_calibration_error(np.array(y_true), np.array(y_pred), n_bins=10)
    assert ece < 1e-6


def test_group_auc_sane_on_tiny_example():
    """Group AUC weights per group AUC by group size and stays in [0, 1].

    Group g0 is perfectly ranked and group g1 is perfectly ranked, so the
    impression weighted GAUC is 1.0. A single class group is skipped.
    """
    y_true = [0, 1, 0, 1, 1, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.6]
    groups = ["g0", "g0", "g1", "g1", "g2", "g2"]

    gauc = group_auc(y_true, y_pred, groups)
    # g0 and g1 are each perfectly ranked. g2 has only positives so it is skipped.
    assert 0.0 <= gauc <= 1.0
    assert abs(gauc - 1.0) < 1e-9


def test_group_auc_no_scorable_group_returns_half():
    """When every group has a single class GAUC falls back to 0.5."""
    y_true = [1, 1, 0, 0]
    y_pred = [0.3, 0.7, 0.4, 0.6]
    groups = ["a", "a", "b", "b"]
    assert group_auc(y_true, y_pred, groups) == 0.5


def test_normalized_entropy_of_base_rate_is_near_one():
    """Predicting the constant base rate gives a normalized entropy near 1.0.

    The logloss of the constant base rate predictor equals the entropy of the
    base rate, so the ratio that defines normalized entropy is 1.0.
    """
    rng = np.random.default_rng(42)
    n = 5000
    y_true = (rng.random(n) < 0.3).astype(int)
    base_rate = float(y_true.mean())
    y_pred = np.full(n, base_rate, dtype=np.float64)

    ne = normalized_entropy(y_true, y_pred)
    assert abs(ne - 1.0) < 1e-6
