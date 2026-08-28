"""Reading raw data into Spark and writing featurized data back out.

Ingestion has to reproduce the pandas loader exactly, because parity is decided
before the first transform runs. The loader coerces dense columns to numeric
with bad values becoming NaN, coerces sparse columns to strings, and folds
empty, missing, and the literal strings "nan" and "<NA>" into a single
`__nan__` token so downstream hashing treats missingness as an ordinary
category. This module does the same on the Spark side.

The other job here is row ordering. The pandas path gets its ordering for free
because a DataFrame is a list of rows. Spark has no inherent order, and the
temporal split depends on order absolutely, so an explicit index has to be
attached at read time and carried through every shuffle after it.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import pandas as pd

from src.data.loader import NAN_TOKEN
from src.schema import ALL_COLS, CAT_COLS, LABEL_COL, NUM_COLS

ROW_INDEX_COL = "row_id"
SPLIT_COL = "split"

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


def raw_schema() -> Any:
    """Return the Spark schema for a raw Criteo row.

    Dense columns are doubles rather than integers so that a value the reader
    cannot parse lands as null instead of failing the whole read, which is the
    same forgiving behaviour `pd.to_numeric(errors="coerce")` gives the pandas
    loader.
    """
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    fields = [StructField(LABEL_COL, IntegerType(), True)]
    fields += [StructField(col, DoubleType(), True) for col in NUM_COLS]
    fields += [StructField(col, StringType(), True) for col in CAT_COLS]
    return StructType(fields)


def _sniff_delimiter(data_path: str) -> str:
    """Return the delimiter of a Criteo style file by looking at the first line.

    The reference loader tries tab first and falls back to comma. Reading one
    line is cheaper than reading the file twice and gives the same answer.
    """
    with open(data_path, "r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") >= first.count(",") else ","


def _normalize_columns(df: Any) -> Any:
    """Apply the loader's coercions to a raw Spark frame.

    Sparse columns get the `__nan__` token for null, empty, and the two string
    spellings of missing that survive a naive text read. The label gets a null
    fill of zero. Dense columns are left alone, since a null there is exactly
    the missingness the numerical transformer is built to encode.
    """
    from pyspark.sql import functions as F

    df = df.withColumn(
        LABEL_COL, F.coalesce(F.col(LABEL_COL).cast("int"), F.lit(0))
    )
    for col in CAT_COLS:
        value = F.col(col).cast("string")
        df = df.withColumn(
            col,
            F.when(
                value.isNull() | value.isin("", "nan", "<NA>"), F.lit(NAN_TOKEN)
            ).otherwise(value),
        )
    return df


def read_criteo(
    spark: Any,
    data_path: str,
    sample_size: Optional[int] = None,
) -> Any:
    """Read a Criteo style TSV or CSV file into an indexed Spark frame.

    Passing sample_size keeps only the first N rows by row index, which is a
    positional head and therefore preserves the temporal ordering of the file.
    The result carries the row index column, so the temporal split downstream
    reads an explicit number rather than trusting partition order.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"no data file at {data_path}.")

    delimiter = _sniff_delimiter(data_path)
    df = (
        spark.read.option("sep", delimiter)
        .option("header", "false")
        .option("mode", "PERMISSIVE")
        .schema(raw_schema())
        .csv(data_path)
    )
    from pyspark.sql import functions as F

    df = _normalize_columns(df)
    df = add_row_index(df)
    if sample_size is not None:
        # Filtering on the row index rather than calling limit keeps the head
        # positional and therefore temporal. limit gives Spark permission to
        # return any n rows it finds first, which on a multi partition read is
        # not the first n rows of the file.
        df = df.filter(F.col(ROW_INDEX_COL) < F.lit(int(sample_size)))
    return df


def read_pandas(spark: Any, df: pd.DataFrame) -> Any:
    """Turn an in memory pandas frame into an indexed Spark frame.

    This is the path the synthetic generator and the parity test take. The frame
    already satisfies the schema, so the only work is casting to the Spark types
    and attaching the row index in the pandas row order. Attaching the index in
    pandas rather than in Spark is deliberate here, because the pandas order is
    the ground truth the parity test compares against.
    """
    from pyspark.sql import functions as F

    ordered = df[ALL_COLS].copy()
    ordered[ROW_INDEX_COL] = range(len(ordered))
    for col in NUM_COLS:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce").astype("float64")
    for col in CAT_COLS:
        ordered[col] = ordered[col].astype(str)
    ordered[LABEL_COL] = ordered[LABEL_COL].astype("int32")

    from pyspark.sql.types import LongType, StructField

    schema = raw_schema()
    schema = schema.add(StructField(ROW_INDEX_COL, LongType(), False))

    spark_df = spark.createDataFrame(ordered, schema=schema)
    # A float NaN arriving from pandas is a NaN double in Spark, not a null, and
    # Spark aggregates propagate NaN where they skip null. Collapsing the two
    # here means every expression downstream has one missing value to reason
    # about instead of two.
    for col in NUM_COLS:
        spark_df = spark_df.withColumn(
            col, F.when(F.isnan(F.col(col)), F.lit(None).cast("double")).otherwise(F.col(col))
        )
    return spark_df


def add_row_index(df: Any, name: str = ROW_INDEX_COL) -> Any:
    """Attach a dense zero based row index that respects file read order.

    `monotonically_increasing_id` is monotonic in read order but leaves gaps,
    because it packs the partition id into the high bits. The temporal split
    needs a dense index so it can be compared against an absolute row count, so
    this counts the rows in each partition, turns those counts into a prefix sum
    on the driver, and adds each partition's offset to a local row number. The
    partition count is small, so the offset map is a literal in the plan and
    costs nothing to broadcast.

    The ordering this produces is the file order, which for Criteo is time
    order. That is the property the whole temporal split rests on.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    partition_col = "_adrb_pid"
    local_col = "_adrb_mid"
    indexed = df.withColumn(partition_col, F.spark_partition_id()).withColumn(
        local_col, F.monotonically_increasing_id()
    )
    indexed = indexed.persist()

    counts = indexed.groupBy(partition_col).count().orderBy(partition_col).collect()
    offsets: dict[int, int] = {}
    running = 0
    for row in counts:
        offsets[int(row[partition_col])] = running
        running += int(row["count"])

    if not offsets:
        indexed = indexed.withColumn(name, F.lit(0).cast("long"))
        return indexed.drop(partition_col, local_col)

    offset_pairs = []
    for pid, offset in offsets.items():
        offset_pairs.append(F.lit(pid))
        offset_pairs.append(F.lit(offset).cast("long"))
    offset_map = F.create_map(*offset_pairs)

    window = Window.partitionBy(partition_col).orderBy(local_col)
    indexed = indexed.withColumn(
        name,
        (F.row_number().over(window).cast("long") - F.lit(1))
        + offset_map[F.col(partition_col)],
    )
    result = indexed.drop(partition_col, local_col)
    return result


def add_temporal_split(
    df: Any,
    n_rows: Optional[int] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    index_col: str = ROW_INDEX_COL,
    split_col: str = SPLIT_COL,
) -> Tuple[Any, int, int, int]:
    """Label each row train, val, or test by position, with no shuffle.

    The boundaries use the same integer truncation as `src.data.split`, so the
    Spark split and the pandas split cut at the same row. Assigning a label
    rather than producing three frames means the split survives the joins and
    aggregations that follow, and it becomes the Parquet partition key on write.

    Returns the labelled frame and the three split sizes.
    """
    from pyspark.sql import functions as F

    if n_rows is None:
        n_rows = int(df.count())
    n_train = int(n_rows * train_frac)
    n_val = int(n_rows * val_frac)
    n_test = n_rows - n_train - n_val

    index = F.col(index_col)
    labelled = df.withColumn(
        split_col,
        F.when(index < F.lit(n_train), F.lit(SPLIT_TRAIN))
        .when(index < F.lit(n_train + n_val), F.lit(SPLIT_VAL))
        .otherwise(F.lit(SPLIT_TEST)),
    )
    return labelled, n_train, n_val, n_test


def write_features(
    df: Any,
    output_path: str,
    partition_by: str = SPLIT_COL,
    mode: str = "overwrite",
    n_output_files: Optional[int] = None,
) -> str:
    """Write the featurized frame as Parquet partitioned by split.

    Partitioning on the split column is what a downstream trainer actually
    wants. It reads `split=train` as a directory prune rather than a filter over
    the whole dataset, so loading one split never touches the bytes of the other
    two. Passing n_output_files repartitions before the write, which is the
    difference between a directory of a few well sized files and a directory of
    hundreds of tiny ones that every later reader pays for.
    """
    writer_frame = df
    if n_output_files is not None and n_output_files > 0:
        writer_frame = writer_frame.repartition(n_output_files, partition_by)
    (
        writer_frame.write.mode(mode)
        .partitionBy(partition_by)
        .parquet(output_path)
    )
    return output_path


def directory_size_bytes(path: str) -> int:
    """Total size on disk of a directory tree, in bytes."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
