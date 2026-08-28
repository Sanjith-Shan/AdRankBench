"""Stable md5 bucketing as a Spark SQL expression.

The pandas encoder in `src.features.categorical` hashes a string by taking the
full md5 hex digest, reading it as a single 128 bit integer, and reducing it
modulo the bucket count. Reproducing that in Spark is the one place where the
port is not a mechanical translation, because Spark has no 128 bit integer type
and `conv` saturates a 32 character hex string at the 64 bit boundary. Reducing
a truncated digest gives a different bucket, so a naive port silently produces a
pipeline that is wrong in a way no shape check would catch.

The fix is modular Horner over the digest. Reading the 32 hex characters as
four base 2^32 limbs and folding them left to right with a modulo at every step
computes the same residue as the arbitrary precision path, because modular
arithmetic distributes over the multiply and add. The widest intermediate is
`bucket_count * 2^32 + (2^32 - 1)`, so any bucket count below roughly 2.1e9
stays inside a signed 64 bit long. The pipeline uses 10000 and 100000, which is
nowhere near that ceiling.

This is exact, not approximate. `tests/test_spark_pipeline.py` asserts the Spark
expression and `stable_hash` agree on every value in the test sample rather than
taking the argument above on trust.
"""

from __future__ import annotations

from typing import Any

# Number of hex characters folded per step. Eight hex characters are 32 bits,
# so a limb is at most 4294967295 and the running product stays far inside a
# signed 64 bit long. Eight is also the largest limb that keeps the fold to four
# steps, and the step count matters, because this expression is instantiated
# once per sparse field and once per cross and the whole projection has to fit
# inside the JVM method size limit that Spark's code generator targets.
_HEX_CHARS_PER_LIMB = 8
_LIMB_BASE = 16 ** _HEX_CHARS_PER_LIMB
_DIGEST_HEX_CHARS = 32

# Bucket counts above this would overflow a signed 64 bit long in the running
# product. Far above anything a hashed feature space would ever use, but the
# guard makes the failure loud instead of silent.
MAX_SAFE_BUCKETS = (2 ** 63 - 1 - (_LIMB_BASE - 1)) // _LIMB_BASE


def md5_bucket(column: Any, buckets: int) -> Any:
    """Return a Spark column holding md5(column) reduced modulo buckets.

    The result matches `src.features.categorical.stable_hash` for every input
    string, and is a bigint in the range [0, buckets).

    Parameters
    ----------
    column
        A Spark string column, or anything `pyspark.sql.functions.col` accepts.
    buckets
        Size of the hash space. Must be positive and below MAX_SAFE_BUCKETS.
    """
    from pyspark.sql import functions as F

    if buckets <= 0:
        raise ValueError(f"buckets must be positive, got {buckets}.")
    if buckets > MAX_SAFE_BUCKETS:
        raise ValueError(
            f"buckets={buckets} would overflow the 64 bit Horner fold. "
            f"the ceiling is {MAX_SAFE_BUCKETS}."
        )

    digest = F.md5(column)
    accumulator = F.lit(0).cast("long")
    for start in range(1, _DIGEST_HEX_CHARS + 1, _HEX_CHARS_PER_LIMB):
        limb = F.conv(
            F.substring(digest, start, _HEX_CHARS_PER_LIMB), 16, 10
        ).cast("long")
        accumulator = (accumulator * F.lit(_LIMB_BASE) + limb) % F.lit(buckets)
    return accumulator.cast("long")


def cross_key(column_a: str, column_b: str) -> Any:
    """Build the cross string for a pair of categorical columns.

    The pandas generator forms `f"{a}={va}&{b}={vb}"`. Encoding both column names
    keeps crosses from different pairs in disjoint regions of the hash space, so
    a collision between two different pairs cannot look like a real interaction.
    """
    from pyspark.sql import functions as F

    return F.concat(
        F.lit(column_a + "="),
        F.col(column_a),
        F.lit("&" + column_b + "="),
        F.col(column_b),
    )


def field_key(column: str, value_column: Any) -> Any:
    """Build the single field hash string, which is `column=value`.

    Prefixing with the column name is what keeps the same token appearing in two
    different sparse fields from landing in the same bucket.
    """
    from pyspark.sql import functions as F

    return F.concat(F.lit(column + "="), value_column)
