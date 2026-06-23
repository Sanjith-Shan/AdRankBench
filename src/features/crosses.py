"""Second order feature crosses for the sparse Criteo columns.

Explicit pairwise crosses give linear models access to interaction signal that
they otherwise cannot represent. We pick the most informative categorical
columns by the variance of their training value frequency distribution, then
hash every unordered pair of their values into a fixed bucket range. The deep
factorization models learn interactions implicitly, so these crosses mainly
strengthen the logistic regression baseline.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.schema import CAT_COLS
from src.features.categorical import stable_hash


class CrossGenerator:
    """Generate hashed second order crosses over the top categorical fields.

    The columns to cross are chosen on the training split by ranking each field
    by the variance of its normalized value frequency distribution. The top_k
    fields then produce C(top_k, 2) pairwise crosses.
    """

    def __init__(self, cross_bucket_size: int = 100000, top_k: int = 5) -> None:
        self.cross_bucket_size = cross_bucket_size
        self.top_k = top_k
        # Sorted list of (col_a, col_b) tuples chosen at fit time.
        self.pairs_: list[tuple[str, str]] = []

    def fit(self, df: pd.DataFrame) -> "CrossGenerator":
        """Pick the top_k cat columns by frequency variance and store pairs.

        For each categorical column we compute the normalized value frequency
        distribution on train and take its variance. We rank columns by that
        variance in descending order, take the top_k, sort them by name for
        stable ordering, then store all unordered pairs.
        """
        variances: list[tuple[float, str]] = []
        n = float(len(df)) if len(df) > 0 else 1.0
        for col in CAT_COLS:
            counts = df[col].astype(str).value_counts().to_numpy(dtype=np.float64)
            freqs = counts / n
            var = float(np.var(freqs)) if freqs.size > 0 else 0.0
            variances.append((var, col))

        # Sort by variance descending. Break ties by column name for stability.
        variances.sort(key=lambda pair: (-pair[0], pair[1]))
        top_cols = sorted(col for _, col in variances[: self.top_k])
        self.pairs_ = list(combinations(top_cols, 2))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (N, n_crosses) int64 array of hashed cross indices.

        For each stored pair (a, b) the cross string is f"{a}={va}&{b}={vb}",
        which is hashed into the cross bucket range. Encoding the column names in
        the string keeps crosses from different pairs in separate hash spaces.
        """
        if not self.pairs_:
            raise RuntimeError("CrossGenerator must be fit before transform.")

        n = len(df)
        out = np.zeros((n, len(self.pairs_)), dtype=np.int64)
        for j, (col_a, col_b) in enumerate(self.pairs_):
            series_a = df[col_a].astype(str).to_numpy()
            series_b = df[col_b].astype(str).to_numpy()
            for i in range(n):
                cross = col_a + "=" + series_a[i] + "&" + col_b + "=" + series_b[i]
                out[i, j] = stable_hash(cross, self.cross_bucket_size)
        return out

    @property
    def n_crosses(self) -> int:
        """Number of crosses, which is C(top_k, 2)."""
        return len(self.pairs_)

    @property
    def vocab_sizes(self) -> list[int]:
        """Per cross hash vocabulary size, identical for all crosses."""
        return [self.cross_bucket_size] * len(self.pairs_)
