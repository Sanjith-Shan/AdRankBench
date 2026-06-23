"""Temporal splitting of the dataset.

Ad ranking data is time ordered. A random split leaks future information into
the training set and inflates offline metrics. This module performs a positional
split that keeps the time order intact. The earliest rows train, the next slice
validates, and the latest rows test.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame positionally into train, val, and test.

    No shuffling happens. The first train_frac of rows become train, the next
    val_frac become val, and the remainder become test. Each returned frame has
    its index reset so downstream code can rely on a clean range index.

    With the defaults the split is 80 percent train, 10 percent val, and 10
    percent test.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    return train_df, val_df, test_df
