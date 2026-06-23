"""Tests for the preprocessing transformers and the stable hash.

These tests use tiny synthetic frames so they run fast and never train a model.
They check three properties from the contract. The log transform handles zeros
and negatives without producing nan or inf and clips negatives. The is_missing
indicators are created so a NaN row yields an indicator of 1.0 and a transformed
value of 0. The stable categorical hash is deterministic across two calls and
always lands inside the bucket range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import CAT_COLS, NUM_COLS
from src.features.numerical import NumericalTransformer
from src.features.categorical import stable_hash


def _frame(num_values: dict, n_rows: int) -> pd.DataFrame:
    """Build a small frame with every schema column present.

    The num_values dict overrides specific numerical columns. Any column not
    overridden is filled with a constant so the frame is well formed. The
    categorical columns are filled with a single token because these tests only
    exercise the numerical transformer.
    """
    data = {}
    for col in NUM_COLS:
        if col in num_values:
            data[col] = np.asarray(num_values[col], dtype=np.float64)
        else:
            data[col] = np.full(n_rows, 1.0, dtype=np.float64)
    for col in CAT_COLS:
        data[col] = ["a"] * n_rows
    return pd.DataFrame(data)


def test_log_transform_handles_zeros_and_negatives():
    """Zeros and negatives transform to finite values and negatives clip to 0."""
    # I1 carries a zero, a negative, and a large positive. The fit frame has
    # spread so the std is nonzero and standardization is well defined.
    fit_df = _frame({"I1": [0.0, 5.0, 50.0, 500.0]}, n_rows=4)
    transformer = NumericalTransformer().fit(fit_df)

    test_df = _frame({"I1": [0.0, -10.0, 100.0]}, n_rows=3)
    out = transformer.transform(test_df)

    assert out.shape == (3, transformer.out_dim)
    assert transformer.out_dim == 26
    # No nan or inf anywhere in the output.
    assert np.isfinite(out).all()

    # A zero and a negative both clip to 0 then log1p(0) == 0, so they share the
    # same standardized value. The negative must not blow up to nan.
    zero_val = out[0, 0]
    neg_val = out[1, 0]
    assert np.isclose(zero_val, neg_val)


def test_is_missing_indicator_created():
    """A NaN cell yields indicator 1.0 and a transformed value of 0."""
    fit_df = _frame({"I1": [1.0, 2.0, 3.0, 4.0]}, n_rows=4)
    transformer = NumericalTransformer().fit(fit_df)

    # Row 0 has a NaN in I1, row 1 is present.
    test_df = _frame({"I1": [np.nan, 2.0]}, n_rows=2)
    out = transformer.transform(test_df)

    n_num = len(NUM_COLS)
    # Value block is the first n_num columns, indicator block is the next n_num.
    value_col_i1 = out[:, 0]
    missing_col_i1 = out[:, n_num + 0]

    # The missing row has indicator 1.0 and the present row has indicator 0.0.
    assert missing_col_i1[0] == 1.0
    assert missing_col_i1[1] == 0.0

    # The missing cell is filled with 0 before log1p, so log1p(0) == 0 and the
    # standardized value equals (0 - mean) / std. The raw filled value is 0.
    # We confirm it matches what a literal 0 input produces.
    zero_df = _frame({"I1": [0.0, 2.0]}, n_rows=2)
    zero_out = transformer.transform(zero_df)
    assert np.isclose(value_col_i1[0], zero_out[0, 0])
    # And the value is finite.
    assert np.isfinite(value_col_i1[0])


def test_stable_hash_deterministic_and_in_range():
    """The stable hash repeats across calls and stays inside [0, buckets)."""
    buckets = 1000
    samples = ["C1=abc", "C2=abc", "hello", "", "00ff00ff", "a long value here"]

    for value in samples:
        first = stable_hash(value, buckets)
        second = stable_hash(value, buckets)
        # Deterministic across two calls in the same process.
        assert first == second
        # Inside the bucket range.
        assert 0 <= first < buckets

    # The same string prefixed by different column names should be free to differ.
    # We only require this is possible, not guaranteed, so we check the values are
    # both valid indices rather than asserting inequality.
    h1 = stable_hash("C1=abc", buckets)
    h2 = stable_hash("C2=abc", buckets)
    assert 0 <= h1 < buckets
    assert 0 <= h2 < buckets
