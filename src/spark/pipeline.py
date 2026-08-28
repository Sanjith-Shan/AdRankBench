"""The distributed feature pipeline.

This is a port of `src.data.preprocess.FeaturePipeline` onto Spark SQL. The
contract is that it produces the same numbers, not merely the same shapes, so
every transformation below is written against the reference implementation line
by line rather than reimplemented from the description in the README.

Three things make the port more than a translation exercise.

The md5 bucketing cannot use Spark's `conv`, because a 32 character hex digest
overflows 64 bits. `src.spark.hashing` folds the digest with modular Horner
instead, which is exact.

The fit statistics are aggregates rather than array reductions. The mean and
standard deviation of the log transformed dense columns, the per value training
counts, and the frequency variance that ranks columns for crossing are all
computed as a single grouped aggregation over the training split. Spark skips
nulls in aggregates but propagates NaN, and pandas float columns arrive as NaN,
so ingestion collapses the two before anything is measured.

The transform is a wide expression tree rather than a Python loop. The pandas
encoder hashes one value at a time in interpreted Python, which is what actually
bounds the reference path long before memory does. Here the same work is a
column expression that Spark codegens into JVM bytecode and runs per partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.features.categorical import RARE_TOKEN
from src.schema import CAT_COLS, Dataset, FeatureMeta, LABEL_COL, NUM_COLS
from src.spark.hashing import cross_key, field_key, md5_bucket
from src.spark.io import ROW_INDEX_COL, SPLIT_COL

# Output column prefixes. Flat named columns rather than array columns, because
# a flat Parquet file is what the SQL analytics lane and any downstream reader
# want, and because a column by column parity check is only readable when the
# columns have names.
NUM_VALUE_PREFIX = "num_"
NUM_MISSING_PREFIX = "miss_"
CAT_HASH_PREFIX = "hash_"
CAT_FREQ_PREFIX = "freq_"
CROSS_PREFIX = "cross_"

# Internal join column names, prefixed so they cannot collide with a schema
# column even if the schema grows.
_VOCAB_VALUE = "_adrb_vocab_value"
_VOCAB_COUNT = "_adrb_vocab_count"
_VOCAB_FIELD = "field"
_VOCAB_RAW_VALUE = "value"


def cross_column_name(column_a: str, column_b: str) -> str:
    """Output column name for the cross of two categorical fields."""
    return f"{CROSS_PREFIX}{column_a}_x_{column_b}"


@dataclass
class SparkPipelineConfig:
    """Knobs for the distributed pipeline.

    The first four mirror `FeaturePipeline` exactly and must match it for the
    output to match. The last two are Spark side only and change how the work is
    executed, never what it computes.

    broadcast_vocab_max_rows
        A per column vocabulary below this many distinct kept values is
        broadcast to every executor, so the fact table is never shuffled. Above
        it, Spark falls back to a shuffle join. On synthetic data every field is
        far below the threshold. On real Criteo the largest fields are not, and
        that is where the join strategy starts to matter.
    vocab_shuffle_partitions
        Post shuffle partition count for the vocabulary aggregation, which is
        the most skewed step in the pipeline. None leaves the session default.
    """

    hash_bucket_size: int = 10000
    cross_bucket_size: int = 100000
    min_count: int = 10
    n_cross_features: int = 5
    broadcast_vocab_max_rows: int = 500_000
    vocab_shuffle_partitions: Optional[int] = None


@dataclass
class NumericalStats:
    """Per column mean and standard deviation of the log transformed values."""

    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)


class SparkFeaturePipeline:
    """Fit on the training split, transform any split, all inside Spark.

    Usage mirrors the pandas pipeline. Fit against the training rows only, then
    transform the full frame. Transforming the full frame in one call rather
    than three is the Spark shaped version of the same operation, since the
    split label rides along as a column and becomes the Parquet partition key.
    """

    def __init__(self, config: Optional[SparkPipelineConfig] = None) -> None:
        self.config = config or SparkPipelineConfig()
        self.total_rows_: int = 0
        self.numerical_: NumericalStats = NumericalStats()
        self.vocab_: Any = None
        self.vocab_row_counts_: Dict[str, int] = {}
        self.freq_variances_: Dict[str, float] = {}
        self.pairs_: List[Tuple[str, str]] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: Any) -> "SparkFeaturePipeline":
        """Learn every statistic from the training rows only.

        One pass computes the dense statistics. A second pass builds the long
        form value count table, which serves three purposes at once. It is the
        rare token cutoff, it is the frequency encoding, and its per field
        variance is the ranking that selects columns for crossing. Computing it
        once and reusing it three ways is the main reason this fit is two jobs
        rather than the twenty eight a naive port would run.
        """
        self.total_rows_ = int(train_df.count())
        self.numerical_ = self._fit_numerical(train_df)
        self._fit_vocabulary(train_df)
        self._fit_crosses()
        self._fitted = True
        return self

    def _fit_numerical(self, train_df: Any) -> NumericalStats:
        """Aggregate the mean and population std of log1p of the clipped values.

        The reference uses `np.nanmean` and `np.nanstd`, which ignore missing
        values so the zero fill applied at transform time cannot drag the mean
        down. Spark aggregates ignore null for the same reason, which is why
        ingestion turns NaN into null. `np.nanstd` is a population standard
        deviation, so this is `stddev_pop` and not `stddev`.

        The guards match the reference exactly. An all missing column has an
        undefined mean and becomes zero. An undefined or zero standard deviation
        becomes one, so a constant column standardizes to a constant instead of
        dividing by zero.
        """
        from pyspark.sql import functions as F

        aggregations = []
        for col in NUM_COLS:
            logged = F.log1p(_clip_negative(F.col(col)))
            aggregations.append(F.avg(logged).alias(f"mean__{col}"))
            aggregations.append(F.stddev_pop(logged).alias(f"std__{col}"))

        row = train_df.agg(*aggregations).collect()[0]

        stats = NumericalStats()
        for col in NUM_COLS:
            mean = row[f"mean__{col}"]
            std = row[f"std__{col}"]
            if mean is None or not np.isfinite(mean):
                mean = 0.0
            if std is None or not np.isfinite(std) or std == 0.0:
                std = 1.0
            stats.means[col] = float(mean)
            stats.stds[col] = float(std)
        return stats

    def _fit_vocabulary(self, train_df: Any) -> None:
        """Build the long form (field, value, count) table over the train split.

        The 26 sparse columns are unpivoted into one narrow table with a `stack`
        expression, which turns twenty six separate group by jobs into a single
        shuffle. The result is cached, because the transform reads it once per
        sparse field and the cross ranking reads it again.
        """
        from pyspark.sql import functions as F

        stack_args = ", ".join(f"'{col}', `{col}`" for col in CAT_COLS)
        stack_expr = (
            f"stack({len(CAT_COLS)}, {stack_args}) as ({_VOCAB_FIELD}, {_VOCAB_RAW_VALUE})"
        )
        long_form = train_df.selectExpr(stack_expr)

        if self.config.vocab_shuffle_partitions is not None:
            long_form = long_form.repartition(
                self.config.vocab_shuffle_partitions, _VOCAB_FIELD, _VOCAB_RAW_VALUE
            )

        counts = long_form.groupBy(_VOCAB_FIELD, _VOCAB_RAW_VALUE).count()

        # The reference frequency variance is taken over every distinct value on
        # train, including the ones that later collapse to the rare token, so the
        # variance is computed before the min_count filter is applied.
        denominator = float(self.total_rows_) if self.total_rows_ > 0 else 1.0
        variance_rows = (
            counts.withColumn("freq", F.col("count") / F.lit(denominator))
            .groupBy(_VOCAB_FIELD)
            .agg(F.var_pop("freq").alias("freq_var"))
            .collect()
        )
        self.freq_variances_ = {}
        for row in variance_rows:
            value = row["freq_var"]
            self.freq_variances_[row[_VOCAB_FIELD]] = (
                0.0 if value is None or not np.isfinite(value) else float(value)
            )
        for col in CAT_COLS:
            self.freq_variances_.setdefault(col, 0.0)

        kept = counts.filter(F.col("count") >= F.lit(self.config.min_count))
        kept = kept.select(
            F.col(_VOCAB_FIELD),
            F.col(_VOCAB_RAW_VALUE).alias(_VOCAB_VALUE),
            F.col("count").cast("long").alias(_VOCAB_COUNT),
        ).persist()

        # Materialize once and record the per field size, which is what the
        # broadcast decision at transform time reads.
        size_rows = kept.groupBy(_VOCAB_FIELD).count().collect()
        self.vocab_row_counts_ = {row[_VOCAB_FIELD]: int(row["count"]) for row in size_rows}
        for col in CAT_COLS:
            self.vocab_row_counts_.setdefault(col, 0)
        self.vocab_ = kept

    def _fit_crosses(self) -> None:
        """Rank sparse fields by frequency variance and store the pairs.

        This reproduces `CrossGenerator.fit` exactly, including the tie break.
        The reference sorts by negative variance then by column name, takes the
        top k, sorts those by name again for a stable pair order, and forms every
        unordered pair.
        """
        ranked = sorted(
            ((self.freq_variances_.get(col, 0.0), col) for col in CAT_COLS),
            key=lambda pair: (-pair[0], pair[1]),
        )
        top_columns = sorted(col for _var, col in ranked[: self.config.n_cross_features])
        self.pairs_ = list(combinations(top_columns, 2))

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: Any) -> Any:
        """Featurize a frame, keeping the row index, split label, and label.

        The output is one flat frame with named columns per feature. Nothing is
        collected. The caller either writes it to Parquet or, in the parity
        test, pulls it back to pandas to compare against the reference.
        """
        if not self._fitted:
            raise RuntimeError("SparkFeaturePipeline must be fit before transform.")

        from pyspark.sql import functions as F

        working = self._apply_numerical(df)
        working = self._apply_categorical(working)
        working = self._apply_crosses(working)

        keep = []
        if ROW_INDEX_COL in working.columns:
            keep.append(F.col(ROW_INDEX_COL))
        if SPLIT_COL in working.columns:
            keep.append(F.col(SPLIT_COL))
        keep.append(F.col(LABEL_COL).cast("float").alias(LABEL_COL))
        keep += [F.col(name) for name in self.feature_columns()]
        return working.select(*keep)

    def _apply_numerical(self, df: Any) -> Any:
        """Add the 13 standardized value columns and 13 missing indicators.

        The reference fills missing with zero, clips negatives to zero, applies
        log1p, then standardizes. log1p of zero is zero, so a missing cell and a
        literal zero produce the same standardized value, which is exactly the
        behaviour `tests/test_preprocess.py` pins down. The indicator is what
        carries the distinction between the two.

        All 26 outputs are added in one `select` rather than by 26 chained
        `withColumn` calls. The result is identical and the plan is not. Each
        `withColumn` is a separate Project node, so a loop over a wide schema
        builds a tower of them that Catalyst has to walk on every optimizer pass
        and that the code generator then has to fit inside one JVM method. On a
        26 column schema that is the difference between a plan Spark analyzes
        instantly and one it spends real time on before a single row moves.
        """
        from pyspark.sql import functions as F

        projections = []
        for col in NUM_COLS:
            raw = F.col(col)
            filled = F.coalesce(_clip_negative(raw), F.lit(0.0))
            standardized = (
                F.log1p(filled) - F.lit(self.numerical_.means[col])
            ) / F.lit(self.numerical_.stds[col])
            projections.append(
                standardized.cast("float").alias(f"{NUM_VALUE_PREFIX}{col}")
            )
        for col in NUM_COLS:
            missing = F.when(F.col(col).isNull(), F.lit(1.0)).otherwise(F.lit(0.0))
            projections.append(
                missing.cast("float").alias(f"{NUM_MISSING_PREFIX}{col}")
            )
        return df.select("*", *projections)

    def _apply_categorical(self, df: Any) -> Any:
        """Add the hashed bucket and the frequency encoding per sparse field.

        Each field is left joined against its slice of the training vocabulary.
        A miss means the value was never seen on train or was seen fewer than
        min_count times, which is the same condition in both cases and is why one
        join serves both outputs. A miss hashes as the shared rare token and
        encodes as a frequency of zero.
        """
        from pyspark.sql import functions as F

        denominator = float(self.total_rows_) if self.total_rows_ > 0 else 1.0

        for col in CAT_COLS:
            vocab = self.vocab_.filter(F.col(_VOCAB_FIELD) == F.lit(col)).select(
                _VOCAB_VALUE, _VOCAB_COUNT
            )
            if self.vocab_row_counts_.get(col, 0) <= self.config.broadcast_vocab_max_rows:
                vocab = F.broadcast(vocab)

            carried = list(df.columns)
            joined = df.join(vocab, df[col] == vocab[_VOCAB_VALUE], how="left")

            kept_count = F.col(_VOCAB_COUNT)
            mapped = F.when(kept_count.isNull(), F.lit(RARE_TOKEN)).otherwise(F.col(col))
            # One projection per field rather than two withColumn calls and a
            # drop, for the same plan depth reason as the dense block above.
            df = joined.select(
                *[F.col(name) for name in carried],
                md5_bucket(field_key(col, mapped), self.config.hash_bucket_size).alias(
                    f"{CAT_HASH_PREFIX}{col}"
                ),
                F.coalesce(kept_count / F.lit(denominator), F.lit(0.0))
                .cast("float")
                .alias(f"{CAT_FREQ_PREFIX}{col}"),
            )
        return df

    def _apply_crosses(self, df: Any) -> Any:
        """Add one hashed column per selected pair of sparse fields.

        Crosses hash the raw values, not the rare collapsed ones. That is what
        the reference does, and it is the right choice, because the point of a
        cross is the conjunction of two specific values. Folding both sides into
        a rare token first would erase most of the pairs worth crossing.
        """
        projections = [
            md5_bucket(
                cross_key(column_a, column_b), self.config.cross_bucket_size
            ).alias(cross_column_name(column_a, column_b))
            for column_a, column_b in self.pairs_
        ]
        if not projections:
            return df
        return df.select("*", *projections)

    # ------------------------------------------------------------------
    # Descriptions of the output
    # ------------------------------------------------------------------

    def numerical_columns(self) -> List[str]:
        """The 26 dense output columns, values first and then indicators.

        The order matters. It is the column order of the reference numerical
        block, so a parity check can compare them positionally.
        """
        return [f"{NUM_VALUE_PREFIX}{col}" for col in NUM_COLS] + [
            f"{NUM_MISSING_PREFIX}{col}" for col in NUM_COLS
        ]

    def hash_columns(self) -> List[str]:
        """The 26 hashed sparse columns in schema order."""
        return [f"{CAT_HASH_PREFIX}{col}" for col in CAT_COLS]

    def freq_columns(self) -> List[str]:
        """The 26 frequency encoded sparse columns in schema order."""
        return [f"{CAT_FREQ_PREFIX}{col}" for col in CAT_COLS]

    def cross_columns(self) -> List[str]:
        """The cross columns in the pair order chosen at fit time."""
        return [cross_column_name(a, b) for a, b in self.pairs_]

    def feature_columns(self) -> List[str]:
        """Every feature column, in the order the Dataset blocks expect."""
        return (
            self.numerical_columns()
            + self.hash_columns()
            + self.freq_columns()
            + self.cross_columns()
        )

    @property
    def meta(self) -> FeatureMeta:
        """Describe the featurized space, matching `FeaturePipeline.meta`."""
        return FeatureMeta(
            n_numerical=2 * len(NUM_COLS),
            cat_vocab_sizes=[self.config.hash_bucket_size] * len(CAT_COLS),
            cross_vocab_sizes=[self.config.cross_bucket_size] * len(self.pairs_),
        )

    def unpersist(self) -> None:
        """Release the cached vocabulary table."""
        if self.vocab_ is not None:
            self.vocab_.unpersist()


def _clip_negative(column: Any) -> Any:
    """Clip a column at zero from below while preserving null.

    Spark's `greatest` ignores nulls, so `greatest(null, 0)` returns zero and
    would silently erase missingness before the indicator is computed. An
    explicit conditional propagates null the way `np.clip` propagates NaN.
    """
    from pyspark.sql import functions as F

    return F.when(column < F.lit(0.0), F.lit(0.0)).otherwise(column)


def spark_to_dataset(
    df: Any,
    pipeline: SparkFeaturePipeline,
    split: Optional[str] = None,
) -> Dataset:
    """Collect a featurized Spark frame into the in memory Dataset container.

    This exists for the parity test and for handing a modest split to the
    existing trainer. It collects, so it is bounded by driver memory and is not
    the path a large run should take. A large run writes Parquet and lets the
    trainer stream it back.

    Rows are sorted by the row index before collection, because the Dataset
    arrays are positional and every comparison against the reference depends on
    the two sharing a row order.
    """
    frame = df
    if split is not None:
        frame = frame.filter(frame[SPLIT_COL] == split)
    pandas_frame = frame.orderBy(ROW_INDEX_COL).toPandas()

    numerical = pandas_frame[pipeline.numerical_columns()].to_numpy(dtype=np.float32)
    categorical = pandas_frame[pipeline.hash_columns()].to_numpy(dtype=np.int64)
    cat_freq = pandas_frame[pipeline.freq_columns()].to_numpy(dtype=np.float32)
    cross_columns = pipeline.cross_columns()
    if cross_columns:
        crosses = pandas_frame[cross_columns].to_numpy(dtype=np.int64)
    else:
        crosses = np.zeros((len(pandas_frame), 0), dtype=np.int64)
    label = pandas_frame[LABEL_COL].to_numpy(dtype=np.float32)

    return Dataset(
        numerical=numerical,
        categorical=categorical,
        cat_freq=cat_freq,
        crosses=crosses,
        label=label,
    )
