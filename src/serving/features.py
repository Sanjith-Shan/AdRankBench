"""Persistence and replay of the fitted feature pipeline for the serving path.

This is the module the whole serving layer is built around, because the feature
pipeline is where ranking systems actually break in production. Training fits a
set of parameters on the train split only. The numerical block learns a per
column mean and standard deviation of the log transformed values. The
categorical block learns a per field value count table that decides which values
are rare and what frequency encoding each value gets. The cross generator learns
which categorical fields to cross by ranking them on train frequency variance.
None of that is in the model checkpoint. A checkpoint plus a freshly constructed
pipeline is a model wired to the wrong transform, and the failure is silent. The
service returns plausible probabilities that are simply wrong, and nothing in
the response says so. That is training and serving skew.

The fix is to treat the fitted pipeline as a shipped artifact with the same
status as the weights. This module writes the fitted state to a versioned
gzipped json file and reads it back into a real FeaturePipeline object, so the
served transform is not a reimplementation of the offline transform, it is the
offline transform. Nothing in src/data is modified. The persistence layer sits
around it and reaches into the three fitted transformers by attribute.

One deliberate reduction happens on write. Categorical values whose train count
is below min_count are dropped from the saved count table. That is exactly
behaviour preserving rather than an approximation. CategoricalEncoder folds any
value with a count below min_count into the shared rare token before hashing,
and gives it a frequency of zero. A value that is absent from the table reads
back as count zero, which is also below min_count, so it takes the identical
branch. The saved table is therefore smaller without changing a single output
value, and the parity test in tests/test_serving.py checks that claim against
the unpruned offline pipeline rather than assuming it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.loader import NAN_TOKEN
from src.data.preprocess import FeaturePipeline
from src.schema import ALL_COLS, CAT_COLS, LABEL_COL, NUM_COLS

# Bumped whenever the on disk layout changes in a way an older reader cannot
# understand. The loader refuses a version it does not know rather than reading
# a field it will misinterpret.
FEATURE_STATE_VERSION: int = 1

# The file name the pipeline state always takes inside a serving bundle.
FEATURE_STATE_FILENAME: str = "feature_pipeline.json.gz"

# Every raw field a scoring request is allowed to carry. The label is not one of
# them, because a request that carries a label is either confused or leaking.
REQUEST_FIELDS: frozenset = frozenset(NUM_COLS) | frozenset(CAT_COLS)


def sha256_file(path: str) -> str:
    """Return the hex sha256 of a file, used to fingerprint a bundle member."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_state(pipeline: FeaturePipeline) -> Dict[str, Any]:
    """Extract the fitted state of a pipeline into a plain json friendly dict.

    Everything a transform reads is captured. The numerical means and standard
    deviations, the categorical count tables and the train row count that
    normalizes them, and the cross pairs that were chosen at fit time. The
    constructor arguments come along too, because the bucket sizes and the
    minimum count are as much part of the transform as the learned statistics
    are.
    """
    numerical = pipeline.numerical
    categorical = pipeline.categorical
    crosses = pipeline.crosses

    if numerical.means_ is None or numerical.stds_ is None:
        raise RuntimeError(
            "the numerical transformer is not fitted, so there is no state to "
            "persist. Fit the pipeline on the train split first."
        )
    if not categorical.counts_:
        raise RuntimeError(
            "the categorical encoder is not fitted, so there is no count table "
            "to persist. Fit the pipeline on the train split first."
        )
    if not crosses.pairs_:
        raise RuntimeError(
            "the cross generator is not fitted, so there are no cross pairs to "
            "persist. Fit the pipeline on the train split first."
        )

    min_count = int(categorical.min_count)
    kept: Dict[str, Dict[str, int]] = {}
    dropped = 0
    for col, counts in categorical.counts_.items():
        col_kept: Dict[str, int] = {}
        for value, count in counts.items():
            if int(count) >= min_count:
                col_kept[str(value)] = int(count)
            else:
                dropped += 1
        kept[str(col)] = col_kept

    return {
        "version": FEATURE_STATE_VERSION,
        "config": {
            "hash_bucket_size": int(pipeline.hash_bucket_size),
            "cross_bucket_size": int(pipeline.cross_bucket_size),
            "min_count": int(pipeline.min_count),
            "n_cross_features": int(pipeline.n_cross_features),
        },
        "numerical": {
            "columns": list(NUM_COLS),
            "means": [float(v) for v in np.asarray(numerical.means_).ravel()],
            "stds": [float(v) for v in np.asarray(numerical.stds_).ravel()],
        },
        "categorical": {
            "columns": list(CAT_COLS),
            "min_count": min_count,
            "hash_bucket_size": int(categorical.hash_bucket_size),
            "total_rows": int(categorical.total_rows_),
            "counts": kept,
            "entries_kept": int(sum(len(v) for v in kept.values())),
            "entries_dropped_below_min_count": int(dropped),
        },
        "crosses": {
            "cross_bucket_size": int(crosses.cross_bucket_size),
            "top_k": int(crosses.top_k),
            "pairs": [[str(a), str(b)] for a, b in crosses.pairs_],
        },
    }


def write_feature_state(pipeline: FeaturePipeline, path: str) -> str:
    """Write the fitted pipeline state to a gzipped json file and fingerprint it.

    The write goes to a temporary file first and is renamed into place, so a
    process that dies partway through never leaves a half written pipeline that
    a service would happily load. Returns the sha256 of the finished file.
    """
    state = feature_state(pipeline)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # mtime zero so two writes of identical state produce identical bytes and
    # the fingerprint is a fingerprint of the state rather than of the clock.
    with gzip.GzipFile(filename=tmp, mode="wb", compresslevel=6, mtime=0) as handle:
        handle.write(payload)
    os.replace(tmp, path)
    return sha256_file(path)


def read_feature_state(path: str) -> Dict[str, Any]:
    """Read the raw state dict back off disk, checking the layout version."""
    with gzip.open(path, "rb") as handle:
        state = json.loads(handle.read().decode("utf-8"))
    version = int(state.get("version", -1))
    if version != FEATURE_STATE_VERSION:
        raise ValueError(
            f"the feature pipeline at {path} was written at layout version "
            f"{version} and this build reads version {FEATURE_STATE_VERSION}. "
            "Rebuild the serving bundle rather than serving a state this code "
            "does not understand."
        )
    return state


def pipeline_from_state(state: Mapping[str, Any]) -> FeaturePipeline:
    """Rebuild a real FeaturePipeline from a saved state dict.

    The returned object is the same class the offline path uses, with the same
    three transformers, so the served transform runs the offline code rather
    than a serving side copy of it. That is the only construction that makes a
    bit for bit parity claim meaningful.
    """
    config = state["config"]
    pipeline = FeaturePipeline(
        hash_bucket_size=int(config["hash_bucket_size"]),
        cross_bucket_size=int(config["cross_bucket_size"]),
        min_count=int(config["min_count"]),
        n_cross_features=int(config["n_cross_features"]),
    )

    numerical_state = state["numerical"]
    pipeline.numerical.means_ = np.asarray(numerical_state["means"], dtype=np.float64)
    pipeline.numerical.stds_ = np.asarray(numerical_state["stds"], dtype=np.float64)

    categorical_state = state["categorical"]
    pipeline.categorical.hash_bucket_size = int(categorical_state["hash_bucket_size"])
    pipeline.categorical.min_count = int(categorical_state["min_count"])
    pipeline.categorical.total_rows_ = int(categorical_state["total_rows"])
    pipeline.categorical.counts_ = {
        str(col): {str(k): int(v) for k, v in counts.items()}
        for col, counts in categorical_state["counts"].items()
    }

    cross_state = state["crosses"]
    pipeline.crosses.cross_bucket_size = int(cross_state["cross_bucket_size"])
    pipeline.crosses.top_k = int(cross_state["top_k"])
    pipeline.crosses.pairs_ = [(str(a), str(b)) for a, b in cross_state["pairs"]]

    pipeline._fitted = True
    return pipeline


def load_feature_pipeline(path: str) -> FeaturePipeline:
    """Read a saved state file and return the rebuilt pipeline."""
    return pipeline_from_state(read_feature_state(path))


def _coerce_numeric(value: Any) -> float:
    """Coerce one raw dense field value the way the offline loader coerces it.

    The offline loader runs pandas to_numeric with errors coerced, so anything
    that is not a number becomes NaN and the missing indicator fires. A json
    null arrives here as None and takes the same path, which is what makes a
    request with an absent field behave like a row with an absent field.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _coerce_categorical(value: Any) -> str:
    """Coerce one raw sparse field value the way the offline loader coerces it.

    The offline loader casts to string and rewrites the empty string, the
    literal nan, and the pandas missing marker to the shared missing token. A
    request that omits a field or sends null lands on the same token, so an
    absent categorical at serving time is the same category it was at train
    time rather than a new unseen one.
    """
    if value is None:
        return NAN_TOKEN
    text = str(value)
    if text in ("", "nan", "<NA>", "None"):
        return NAN_TOKEN
    return text


def rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a canonical schema DataFrame from a sequence of raw request rows.

    Each row is a mapping from a Criteo field name to a value. Fields that are
    absent are filled the same way the offline loader fills a missing cell, which
    is NaN for a dense field and the missing token for a sparse field. A label
    column of zero is attached because FeaturePipeline.transform reads it, and it
    is thrown away immediately afterwards. Attaching a dummy label is what lets
    the service call the offline transform unmodified instead of calling the
    three transformers separately and drifting from it.
    """
    n = len(rows)
    data: Dict[str, Any] = {LABEL_COL: np.zeros(n, dtype=np.int64)}
    for col in NUM_COLS:
        data[col] = np.asarray(
            [_coerce_numeric(row.get(col)) for row in rows], dtype=np.float64
        )
    for col in CAT_COLS:
        data[col] = np.asarray(
            [_coerce_categorical(row.get(col)) for row in rows], dtype=object
        )
    return pd.DataFrame(data, columns=ALL_COLS)


def unknown_fields(row: Mapping[str, Any]) -> List[str]:
    """Return the field names in a request row that the schema does not know.

    A typo in a feature name is otherwise invisible. The field is ignored, the
    row is featurized as if that feature were missing, and the score comes back
    looking normal. Rejecting the request is the only way the caller finds out.
    """
    return sorted(str(k) for k in row.keys() if str(k) not in REQUEST_FIELDS)


def frame_to_model_arrays(
    pipeline: FeaturePipeline, frame: pd.DataFrame
) -> "tuple[np.ndarray, np.ndarray]":
    """Transform a canonical frame into the (numerical, cat) model input blocks.

    The transform call is the offline pipeline's own transform, and the array
    layout comes from src.inference.export.dataset_arrays, which is the same
    function the benchmark and the engine builder use. Both halves of the input
    contract are therefore shared with the offline path rather than restated
    here.
    """
    from src.inference.export import dataset_arrays

    dataset = pipeline.transform(frame)
    return dataset_arrays(dataset)


def featurize_rows(
    pipeline: FeaturePipeline, rows: Sequence[Mapping[str, Any]]
) -> "tuple[np.ndarray, np.ndarray]":
    """Turn raw request rows straight into the model input blocks."""
    return frame_to_model_arrays(pipeline, rows_to_frame(rows))


def describe_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the small summary of a pipeline state that /health reports.

    The count table itself is far too large to put in a health response, so what
    goes out is the shape of it. The bucket sizes, the minimum count, the number
    of train rows the frequencies are normalized by, how many count entries
    survived the prune, and which field pairs are crossed.
    """
    categorical = state.get("categorical", {})
    crosses = state.get("crosses", {})
    config = state.get("config", {})
    return {
        "version": int(state.get("version", -1)),
        "hash_bucket_size": int(config.get("hash_bucket_size", 0)),
        "cross_bucket_size": int(config.get("cross_bucket_size", 0)),
        "min_count": int(config.get("min_count", 0)),
        "train_rows": int(categorical.get("total_rows", 0)),
        "count_entries": int(categorical.get("entries_kept", 0)),
        "count_entries_pruned": int(
            categorical.get("entries_dropped_below_min_count", 0)
        ),
        "cross_pairs": ["&".join(pair) for pair in crosses.get("pairs", [])],
    }
