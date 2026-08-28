"""The batch scoring lane.

This is the other half of a decisioning pipeline. Online scoring answers one
auction now. Batch scoring answers every row in a shard as fast as the machine
allows, which is what feeds nightly candidate precomputation, offline audits of
what the live model would have done, backfills after a model swap, and the
retrospective analysis that decides whether the next model ships.

The point of this module is what it does not contain. There is no featurizer
here, no model loader, and no calibration. It constructs the same ScoringEngine
the http service constructs, from the same bundle, and calls the same method.
The only difference between the two lanes is the batch size and where the rows
come from. That is the mechanism that prevents online and offline skew, and it
is checked rather than asserted, because tests/test_serving.py runs the same
rows down both lanes and requires the probabilities to be identical rather than
close.

Chunking exists for memory rather than for speed. A shard can be larger than
memory once it has been featurized into dense blocks, so rows are read, scored,
and written in chunks, and the throughput is reported over the whole job rather
than over the fastest chunk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.schema import CAT_COLS, LABEL_COL, NUM_COLS
from src.serving.runtime import ScoringEngine

# The chunk a shard is scored in when the caller does not say. Large enough that
# the per call overhead is amortized to nothing and small enough that the dense
# featurized block stays comfortably in memory.
DEFAULT_CHUNK_ROWS: int = 4096


@dataclass
class BatchResult:
    """What one batch scoring job did and how fast it did it."""

    input_path: str
    output_path: str
    rows: int
    chunks: int
    wall_seconds: float
    feature_seconds: float
    model_seconds: float
    backend: str
    mean_probability: float
    id_column: str
    hardware_label: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.wall_seconds if self.wall_seconds > 0 else float("nan")

    @property
    def io_seconds(self) -> float:
        """Wall time that was neither featurizing nor running the model.

        Reading the shard, turning frame chunks into row records, and writing
        the scored file. It is reported rather than absorbed, because on a job
        this size it is not a rounding error, and a rows per second figure that
        quietly excluded it would be a model throughput figure wearing a job
        throughput label.
        """
        return max(0.0, self.wall_seconds - self.feature_seconds - self.model_seconds)

    def as_dict(self) -> Dict[str, Any]:
        """Return the json friendly record written next to the scored output."""
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "rows": self.rows,
            "chunks": self.chunks,
            "chunk_rows": self.extra.get("chunk_rows"),
            "wall_seconds": round(self.wall_seconds, 4),
            "feature_seconds": round(self.feature_seconds, 4),
            "model_seconds": round(self.model_seconds, 4),
            "io_seconds": round(self.io_seconds, 4),
            "rows_per_second": round(self.rows_per_second, 1),
            "feature_share": (
                round(self.feature_seconds / self.wall_seconds, 4)
                if self.wall_seconds > 0
                else None
            ),
            "backend": self.backend,
            "mean_probability": round(self.mean_probability, 6),
            "id_column": self.id_column,
            "hardware": self.hardware_label,
            **{k: v for k, v in self.extra.items() if k != "chunk_rows"},
        }


def read_shard(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read a Parquet or CSV shard into a frame with the canonical column names.

    Parquet is read by extension. Anything else goes through the project's own
    Criteo reader, which sniffs tab against comma, applies the canonical schema,
    and performs the same coercions the training path applied. Reusing that
    reader rather than calling read_csv here is what keeps a batch job's idea of
    a missing value identical to the training pipeline's idea of one.
    """
    lowered = path.lower()
    if lowered.endswith(".parquet") or lowered.endswith(".pq"):
        frame = pd.read_parquet(path)
        if limit is not None:
            frame = frame.head(limit)
        return frame.reset_index(drop=True)

    from src.data.loader import load_raw

    return load_raw(path, sample_size=limit)


def resolve_id_column(frame: pd.DataFrame, requested: Optional[str]) -> str:
    """Pick the column that identifies a row in the scored output.

    An explicit request wins. Otherwise a column named ad_id or row_id is used
    when the shard carries one, and when it carries neither the row's position
    in the shard becomes the identifier. A scored file with no way to join back
    to its input is not useful, so there is always an identifier.
    """
    if requested:
        if requested not in frame.columns:
            raise KeyError(
                f"the shard has no column named {requested}. It has "
                f"{sorted(frame.columns)[:12]} and more."
            )
        return requested
    for candidate in ("ad_id", "row_id", "id"):
        if candidate in frame.columns:
            return candidate
    return "row_index"


def _rows_from_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Turn a chunk of a shard into the raw row dicts the engine accepts.

    Only the schema fields are carried across. An identifier column or any other
    passenger column in the shard is not a model feature and must not reach the
    featurizer, where it would be rejected as an unknown field.
    """
    present = [c for c in list(NUM_COLS) + list(CAT_COLS) if c in frame.columns]
    subset = frame[present]
    return subset.to_dict(orient="records")


def score_frame(
    engine: ScoringEngine,
    frame: pd.DataFrame,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> "tuple[np.ndarray, float, float]":
    """Score every row of a frame through the engine and return the probabilities.

    Returns the probability array alongside the total feature time and the total
    model time, so the caller can report which half of the job the wall clock
    went into.
    """
    probabilities: List[np.ndarray] = []
    feature_seconds = 0.0
    model_seconds = 0.0
    for start in range(0, len(frame), chunk_rows):
        chunk = frame.iloc[start : start + chunk_rows]
        probs, timing = engine.score_rows(_rows_from_frame(chunk))
        probabilities.append(np.asarray(probs, dtype=np.float64))
        feature_seconds += timing.feature_seconds
        model_seconds += timing.model_seconds
    if not probabilities:
        return np.zeros(0, dtype=np.float64), 0.0, 0.0
    return np.concatenate(probabilities), feature_seconds, model_seconds


def write_scored(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: str,
    id_column: str,
) -> str:
    """Write the identifier, the probability, and the label when there is one.

    The label rides along when the shard carries one, because the most common
    reason to batch score a labeled shard is to evaluate what the live model
    would have done, and a scored file with no label forces a second join to
    answer that.
    """
    out: Dict[str, Any] = {}
    if id_column == "row_index":
        out[id_column] = np.arange(len(frame), dtype=np.int64)
    else:
        out[id_column] = frame[id_column].to_numpy()
    out["p_click"] = np.asarray(probabilities, dtype=np.float64)
    if LABEL_COL in frame.columns:
        out[LABEL_COL] = frame[LABEL_COL].to_numpy()

    scored = pd.DataFrame(out)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet") or output_path.lower().endswith(".pq"):
        scored.to_parquet(output_path, index=False)
    else:
        scored.to_csv(output_path, index=False)
    return output_path


def run_batch_job(
    engine: ScoringEngine,
    input_path: str,
    output_path: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    limit: Optional[int] = None,
    id_column: Optional[str] = None,
    hardware_label: str = "",
) -> BatchResult:
    """Read a shard, score it through the engine, write it, and time the whole thing.

    The wall clock covers reading, featurizing, scoring, and writing, because
    that is what a scheduled job actually costs. The feature and model shares are
    reported inside it rather than instead of it, so a rows per second figure is
    never quietly a model only figure.
    """
    started = time.perf_counter()
    frame = read_shard(input_path, limit=limit)
    resolved_id = resolve_id_column(frame, id_column)
    probabilities, feature_seconds, model_seconds = score_frame(
        engine, frame, chunk_rows=chunk_rows
    )
    write_scored(frame, probabilities, output_path, resolved_id)
    wall = time.perf_counter() - started

    n_chunks = max(1, (len(frame) + chunk_rows - 1) // chunk_rows) if len(frame) else 0
    return BatchResult(
        input_path=os.path.abspath(input_path),
        output_path=os.path.abspath(output_path),
        rows=int(len(frame)),
        chunks=int(n_chunks),
        wall_seconds=wall,
        feature_seconds=feature_seconds,
        model_seconds=model_seconds,
        backend=engine.backend_key,
        mean_probability=float(np.mean(probabilities)) if len(probabilities) else float("nan"),
        id_column=resolved_id,
        hardware_label=hardware_label,
        extra={"chunk_rows": int(chunk_rows)},
    )
