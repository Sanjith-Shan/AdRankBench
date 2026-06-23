"""Numerical feature transformation for the dense Criteo columns.

The dense integer features I1 to I13 are heavy tailed and contain missing
values. This transformer clips negatives to zero, applies log1p to compress the
tail, standardizes with statistics learned on the training split, and appends a
binary is_missing indicator per column. The output width is 26, which is 13
standardized value columns followed by 13 indicator columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import NUM_COLS


class NumericalTransformer:
    """Log transform, standardize, and add missing indicators for dense cols.

    Statistics are fit on the training split only. The transform is fully
    deterministic and uses only numpy and pandas.
    """

    def __init__(self) -> None:
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "NumericalTransformer":
        """Compute per column mean and std after the log1p of clipped values.

        Missing values are ignored when computing the statistics so that the
        fill value used at transform time does not bias the mean. Columns with a
        zero or undefined standard deviation fall back to a std of 1.0.
        """
        values = df[NUM_COLS].to_numpy(dtype=np.float64)
        # Clip negatives to zero then log1p. NaN stays NaN so it can be masked.
        clipped = np.where(np.isnan(values), np.nan, np.clip(values, 0.0, None))
        logged = np.log1p(clipped)

        means = np.nanmean(logged, axis=0)
        stds = np.nanstd(logged, axis=0)
        # Replace NaN means (an all missing column) with 0 and guard zero std.
        means = np.where(np.isnan(means), 0.0, means)
        stds = np.where(np.isnan(stds) | (stds == 0.0), 1.0, stds)

        self.means_ = means.astype(np.float64)
        self.stds_ = stds.astype(np.float64)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (N, 26) float32 array of values then missing indicators.

        For each of the 13 numeric columns we fill NaN with 0, clip negatives to
        0, apply log1p, then standardize with the stored mean and std. We then
        append 13 binary columns that are 1.0 where the original value was NaN.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("NumericalTransformer must be fit before transform.")

        values = df[NUM_COLS].to_numpy(dtype=np.float64)
        missing = np.isnan(values).astype(np.float32)

        filled = np.where(np.isnan(values), 0.0, np.clip(values, 0.0, None))
        logged = np.log1p(filled)
        standardized = (logged - self.means_) / self.stds_

        out = np.concatenate([standardized.astype(np.float32), missing], axis=1)
        return out.astype(np.float32)

    @property
    def out_dim(self) -> int:
        """Output width, which is 13 value columns plus 13 indicator columns."""
        return 2 * len(NUM_COLS)
