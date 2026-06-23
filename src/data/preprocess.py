"""Feature pipeline orchestrator.

This module ties the three feature transformers together into one pipeline.
The pipeline fits all transformers on the training split only to avoid leakage,
then transforms any split into a Dataset container. It also exposes the
FeatureMeta the models need to size their embedding tables.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.features.categorical import CategoricalEncoder
from src.features.crosses import CrossGenerator
from src.features.numerical import NumericalTransformer
from src.schema import Dataset, FeatureMeta, LABEL_COL


class FeaturePipeline:
    """Fit on train, transform any split into a Dataset.

    The pipeline owns one NumericalTransformer, one CategoricalEncoder, and one
    CrossGenerator. All three are fit on the training frame only. Transforming a
    frame produces a Dataset with the numerical block, the hashed categorical
    block, the frequency encoded block, the hashed cross block, and the label.
    """

    def __init__(
        self,
        hash_bucket_size: int = 10000,
        cross_bucket_size: int = 100000,
        min_count: int = 10,
        n_cross_features: int = 5,
    ):
        self.hash_bucket_size = hash_bucket_size
        self.cross_bucket_size = cross_bucket_size
        self.min_count = min_count
        self.n_cross_features = n_cross_features

        self.numerical = NumericalTransformer()
        self.categorical = CategoricalEncoder(
            hash_bucket_size=hash_bucket_size, min_count=min_count
        )
        self.crosses = CrossGenerator(
            cross_bucket_size=cross_bucket_size, top_k=n_cross_features
        )
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Fit all transformers on the training frame only."""
        self.numerical.fit(train_df)
        self.categorical.fit(train_df)
        self.crosses.fit(train_df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dataset:
        """Transform a frame into a Dataset with the correct dtypes."""
        numerical = np.asarray(self.numerical.transform(df), dtype=np.float32)
        categorical = np.asarray(
            self.categorical.transform_hash(df), dtype=np.int64
        )
        cat_freq = np.asarray(
            self.categorical.transform_freq(df), dtype=np.float32
        )
        crosses = np.asarray(self.crosses.transform(df), dtype=np.int64)
        label = np.asarray(df[LABEL_COL].to_numpy(), dtype=np.float32)

        return Dataset(
            numerical=numerical,
            categorical=categorical,
            cat_freq=cat_freq,
            crosses=crosses,
            label=label,
        )

    @property
    def meta(self) -> FeatureMeta:
        """Describe the featurized space for the models.

        n_numerical comes from the numerical transformer output width. The
        categorical vocab sizes are the hash bucket size repeated over the 26
        sparse fields. The cross vocab sizes are the cross bucket size repeated
        over the number of crosses the generator produced.
        """
        n_crosses = self.crosses.n_crosses
        return FeatureMeta(
            n_numerical=self.numerical.out_dim,
            cat_vocab_sizes=[self.hash_bucket_size] * len(self.categorical.vocab_sizes),
            cross_vocab_sizes=[self.cross_bucket_size] * n_crosses,
        )


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    **pipeline_kwargs,
) -> Tuple[Dataset, Dataset, Dataset, FeatureMeta]:
    """Fit the pipeline on train and transform all three splits.

    Returns the three Dataset containers and the FeatureMeta. Any keyword
    arguments are forwarded to the FeaturePipeline constructor.
    """
    pipeline = FeaturePipeline(**pipeline_kwargs)
    pipeline.fit(train_df)

    train_ds = pipeline.transform(train_df)
    val_ds = pipeline.transform(val_df)
    test_ds = pipeline.transform(test_df)
    meta = pipeline.meta

    return train_ds, val_ds, test_ds, meta
