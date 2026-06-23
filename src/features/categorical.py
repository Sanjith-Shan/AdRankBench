"""Categorical feature encoding for the sparse Criteo columns.

The 26 sparse fields are high cardinality hex strings. We use the hashing trick
to map each value into a fixed bucket range, which bounds memory and handles
unseen categories at serving time. We also produce a frequency encoding learned
on the training split for the linear baseline. Rare categories below a minimum
training count are folded into a shared rare token before hashing so that noisy
tail values do not each claim a private bucket.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.schema import CAT_COLS

RARE_TOKEN = "__rare__"


def stable_hash(value: str, buckets: int) -> int:
    """Hash a string into [0, buckets) deterministically across processes.

    Python's builtin hash is randomized per process via PYTHONHASHSEED, so it
    cannot be used for reproducible bucketing. We use md5 of the utf-8 bytes and
    reduce the full hex digest modulo the bucket count.
    """
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


class CategoricalEncoder:
    """Hashing and frequency encoder for the sparse fields.

    Frequency counts are learned on the training split only. Rare categories map
    to a shared token before hashing and to a frequency of zero.
    """

    def __init__(self, hash_bucket_size: int = 10000, min_count: int = 10) -> None:
        self.hash_bucket_size = hash_bucket_size
        self.min_count = min_count
        # Per column dict mapping raw value to its training count.
        self.counts_: dict[str, dict[str, int]] = {}
        self.total_rows_: int = 0

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Build per column training frequency dicts and store total row count."""
        self.total_rows_ = int(len(df))
        self.counts_ = {}
        for col in CAT_COLS:
            value_counts = df[col].astype(str).value_counts()
            self.counts_[col] = value_counts.to_dict()
        return self

    def _mapped_value(self, col: str, value: str) -> str:
        """Return the rare token when a value is below min_count on train."""
        count = self.counts_.get(col, {}).get(value, 0)
        if count < self.min_count:
            return RARE_TOKEN
        return value

    def transform_hash(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (N, 26) int64 array of hashed bucket indices.

        Each value is first folded to the rare token when its training count is
        below min_count. We then prefix the value with its column name so that
        the same string in two different fields hashes to different buckets.
        """
        if not self.counts_:
            raise RuntimeError("CategoricalEncoder must be fit before transform.")

        n = len(df)
        out = np.zeros((n, len(CAT_COLS)), dtype=np.int64)
        for j, col in enumerate(CAT_COLS):
            series = df[col].astype(str)
            for i, raw in enumerate(series.to_numpy()):
                mapped = self._mapped_value(col, raw)
                key = col + "=" + mapped
                out[i, j] = stable_hash(key, self.hash_bucket_size)
        return out

    def transform_freq(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (N, 26) float32 array of normalized training frequencies.

        A value maps to its training count divided by the total number of
        training rows, which lies in [0, 1]. Unseen and rare values map to 0.
        """
        if not self.counts_:
            raise RuntimeError("CategoricalEncoder must be fit before transform.")

        n = len(df)
        denom = float(self.total_rows_) if self.total_rows_ > 0 else 1.0
        out = np.zeros((n, len(CAT_COLS)), dtype=np.float32)
        for j, col in enumerate(CAT_COLS):
            series = df[col].astype(str)
            col_counts = self.counts_.get(col, {})
            for i, raw in enumerate(series.to_numpy()):
                count = col_counts.get(raw, 0)
                if count < self.min_count:
                    out[i, j] = 0.0
                else:
                    out[i, j] = count / denom
        return out

    @property
    def vocab_sizes(self) -> list[int]:
        """Per field hash vocabulary size, identical for all 26 fields."""
        return [self.hash_bucket_size] * len(CAT_COLS)
