"""Tests for the categorical encoder and the cross generator.

These tests use tiny frames and never train a model. They check that feature
crosses are deterministic so the same frame produces the same array twice, that
the vocabulary is built on train only so a category seen only in test gets a
frequency of 0 with no leakage, and that the cross generator emits exactly
C(5, 2) which is 10 crosses for the default top_k of 5.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from src.schema import CAT_COLS, NUM_COLS
from src.features.categorical import CategoricalEncoder
from src.features.crosses import CrossGenerator


def _cat_frame(overrides: dict, n_rows: int) -> pd.DataFrame:
    """Build a frame with every schema column. Categorical overrides apply.

    Columns not overridden get a constant token. Numerical columns are filled
    with a constant since these tests do not exercise the numerical path.
    """
    data = {}
    for col in NUM_COLS:
        data[col] = np.full(n_rows, 1.0, dtype=np.float64)
    for col in CAT_COLS:
        if col in overrides:
            data[col] = list(overrides[col])
        else:
            data[col] = ["z"] * n_rows
    return pd.DataFrame(data)


def _varied_frame(n_rows: int) -> pd.DataFrame:
    """Build a frame where each cat column has a distinct skew profile.

    The columns are given decreasing skew so the cross generator has a stable
    ranking by frequency variance. A skewed column where one value dominates has
    high variance in its frequency distribution. A near uniform column has low
    variance. This makes the chosen top columns deterministic.
    """
    rng = np.random.default_rng(42)
    data = {}
    for col in NUM_COLS:
        data[col] = np.full(n_rows, 1.0, dtype=np.float64)
    for idx, col in enumerate(CAT_COLS):
        # cardinality grows with the column index so earlier columns are more
        # skewed which gives them higher frequency variance.
        cardinality = 2 + idx
        values = rng.integers(0, cardinality, size=n_rows)
        # bias toward value 0 for the earliest columns to raise their variance.
        if idx < 5:
            bias = rng.random(n_rows) < (0.8 - 0.1 * idx)
            values = np.where(bias, 0, values)
        data[col] = [f"{int(v):04x}" for v in values]
    return pd.DataFrame(data)


def test_feature_crosses_deterministic():
    """The same frame fed twice yields identical cross arrays."""
    df = _varied_frame(n_rows=200)
    gen = CrossGenerator(cross_bucket_size=1000, top_k=5).fit(df)

    first = gen.transform(df)
    second = gen.transform(df)

    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    # Values stay inside the bucket range.
    assert first.min() >= 0
    assert first.max() < 1000


def test_cross_generator_picks_ten_crosses():
    """top_k of 5 yields exactly C(5, 2) which is 10 crosses."""
    df = _varied_frame(n_rows=200)
    gen = CrossGenerator(cross_bucket_size=1000, top_k=5).fit(df)

    expected = comb(5, 2)
    assert expected == 10
    assert gen.n_crosses == 10
    assert len(gen.vocab_sizes) == 10
    assert gen.vocab_sizes == [1000] * 10

    out = gen.transform(df)
    assert out.shape == (200, 10)


def test_vocabulary_built_on_train_only_no_leakage():
    """A category only in test gets frequency 0 with no leakage from test."""
    # Train sees only token "seen" in C1, repeated enough to clear min_count.
    train_df = _cat_frame({"C1": ["seen"] * 20}, n_rows=20)
    encoder = CategoricalEncoder(hash_bucket_size=500, min_count=5).fit(train_df)

    # Test contains a brand new token "unseen" in C1 that train never observed.
    test_df = _cat_frame({"C1": ["seen", "unseen"]}, n_rows=2)
    freq = encoder.transform_freq(test_df)

    col_idx = CAT_COLS.index("C1")
    seen_freq = freq[0, col_idx]
    unseen_freq = freq[1, col_idx]

    # The seen token has a positive train frequency.
    assert seen_freq > 0.0
    # The unseen token has frequency 0 because it was never in train.
    assert unseen_freq == 0.0

    # The hashed output for both rows is a valid bucket index. The unseen value
    # folds to the rare token before hashing, which is still in range.
    hashed = encoder.transform_hash(test_df)
    assert hashed.dtype == np.int64
    assert hashed.min() >= 0
    assert hashed.max() < 500


def test_rare_category_folds_to_zero_frequency():
    """A train category below min_count is treated as rare with frequency 0."""
    # "common" appears 15 times and clears min_count of 5. "rareval" appears
    # twice and falls below it.
    values = ["common"] * 15 + ["rareval"] * 2
    train_df = _cat_frame({"C1": values}, n_rows=len(values))
    encoder = CategoricalEncoder(hash_bucket_size=500, min_count=5).fit(train_df)

    probe = _cat_frame({"C1": ["common", "rareval"]}, n_rows=2)
    freq = encoder.transform_freq(probe)
    col_idx = CAT_COLS.index("C1")

    assert freq[0, col_idx] > 0.0
    # The rare value maps to frequency 0 because its train count is below min_count.
    assert freq[1, col_idx] == 0.0
