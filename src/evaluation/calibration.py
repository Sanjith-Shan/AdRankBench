"""Calibration analysis and plotting for click probability models.

A ranking model can order ads well and still produce miscalibrated
probabilities. Calibration matters in ads because bids and pacing decisions
multiply the predicted click probability by a value, so a systematic bias in the
probability turns directly into mispriced auctions. This module provides the
reliability curve data, the expected calibration error, and a plot that overlays
several models against the perfect calibration diagonal.

The matplotlib Agg backend is selected before pyplot is imported so the plotting
works on a headless machine with no display.
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib

# Select a non interactive backend before importing pyplot. This must happen at
# import time so the module is safe to import in a headless environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  import after backend selection
import numpy as np  # noqa: E402


def calibration_curve_data(
    y_true,
    y_pred,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve points using equal width probability bins.

    The prediction range [0, 1] is split into n_bins equal width bins. For each
    non empty bin we report the mean predicted probability, the observed
    fraction of positives, and the bin count. Empty bins are dropped so the
    returned arrays only describe bins that actually carry data.

    Returns a tuple of three arrays (mean_pred_per_bin, mean_true_per_bin,
    count_per_bin), each of length equal to the number of non empty bins.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each prediction to a bin index in [0, n_bins - 1]. The rightmost
    # edge is inclusive so a prediction of exactly 1.0 lands in the last bin.
    bin_ids = np.clip(np.digitize(y_pred, edges[1:-1], right=False), 0, n_bins - 1)

    mean_pred = []
    mean_true = []
    counts = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred.append(float(y_pred[mask].mean()))
        mean_true.append(float(y_true[mask].mean()))
        counts.append(count)

    return (
        np.asarray(mean_pred, dtype=np.float64),
        np.asarray(mean_true, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def expected_calibration_error(y_true, y_pred, n_bins: int = 10) -> float:
    """Expected calibration error over equal width probability bins.

    ECE is the sum over bins of (bin_count / N) times the absolute gap between
    the mean predicted probability and the observed positive fraction in that
    bin. A perfectly calibrated model has ECE close to 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n = len(y_true)
    if n == 0:
        return 0.0

    mean_pred, mean_true, counts = calibration_curve_data(y_true, y_pred, n_bins)
    if len(counts) == 0:
        return 0.0

    weights = counts.astype(np.float64) / float(n)
    gaps = np.abs(mean_pred - mean_true)
    return float(np.sum(weights * gaps))


def plot_calibration(curves: Dict[str, Tuple], out_path: str) -> str:
    """Plot reliability curves for several models on one axes.

    The curves argument maps a model name to a tuple (mean_pred, mean_true), the
    first two arrays returned by calibration_curve_data. Every model is drawn on
    a shared axes together with the y equals x diagonal that marks perfect
    calibration. The figure is saved as a PNG to out_path. Returns out_path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference line.
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

    for name, curve in curves.items():
        mean_pred, mean_true = curve[0], curve[1]
        ax.plot(mean_pred, mean_true, marker="o", label=name)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed click fraction")
    ax.set_title("Calibration reliability curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
