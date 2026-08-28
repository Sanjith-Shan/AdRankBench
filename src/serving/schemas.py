"""Typed request and response contracts for the scoring service.

The schemas are strict on purpose. A ranking service that accepts anything is a
service where a renamed upstream field becomes a silent quality regression
rather than an error. A request that sends C27, or misspells I3 as l3, is
featurized as though that feature were simply missing. The row still scores, the
probability still looks like a probability, and nothing anywhere says the
request was wrong. Rejecting the unknown field with a 422 is the only version of
this that an upstream team can debug.

The request shape follows the auction rather than the tensor. One request is one
auction, carrying the candidate set for a single opportunity plus the context
fields that are shared across every candidate in it. Sending the context once
rather than duplicating it onto every candidate is how a real bidder does it,
and it means the batch dimension of the model call is the candidate set size,
which is a quantity a capacity plan can actually reason about.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.serving.features import REQUEST_FIELDS

# A raw feature value as it arrives over the wire. Dense fields arrive as
# numbers, sparse fields arrive as strings, and a null is how a caller says a
# field is genuinely absent for this row.
FeatureValue = Optional[Union[float, int, str, bool]]

# The largest candidate set the schema will accept before the engine's own limit
# is consulted. This is a guard against a malformed or hostile request rather
# than a capacity statement.
MAX_CANDIDATES: int = 4096


def _check_fields(features: Dict[str, FeatureValue], where: str) -> Dict[str, FeatureValue]:
    """Reject any field name the Criteo schema does not define."""
    unknown = sorted(k for k in features if k not in REQUEST_FIELDS)
    if unknown:
        raise ValueError(
            f"{where} carries the unknown field names {unknown}. The schema "
            "defines the dense fields I1 to I13 and the sparse fields C1 to C26. "
            "A field this service does not know would be silently treated as "
            "missing, so the request is rejected instead."
        )
    return features


class Candidate(BaseModel):
    """One candidate ad in an auction, with its own feature values."""

    ad_id: str = Field(
        ..., min_length=1, max_length=256, description="Caller's identifier for this ad."
    )
    features: Dict[str, FeatureValue] = Field(
        default_factory=dict,
        description=(
            "Raw Criteo shaped fields for this candidate. Anything absent is "
            "treated as missing, which is a real signal the model was trained on."
        ),
    )

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        return _check_fields(value, "a candidate")


class ScoreRequest(BaseModel):
    """One auction. A candidate set plus the context shared across it."""

    request_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Caller's identifier, echoed back so a slow request can be traced.",
    )
    context: Dict[str, FeatureValue] = Field(
        default_factory=dict,
        description=(
            "Fields shared by every candidate in this auction, such as the user "
            "and page side features. A candidate's own value for a field wins."
        ),
    )
    candidates: List[Candidate] = Field(
        ...,
        min_length=1,
        max_length=MAX_CANDIDATES,
        description="The candidate set to score and rank.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Return only the top k ranked candidates. Defaults to all of them.",
    )

    @field_validator("context")
    @classmethod
    def _validate_context(cls, value: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        return _check_fields(value, "the request context")

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "ScoreRequest":
        seen = set()
        duplicates = set()
        for candidate in self.candidates:
            if candidate.ad_id in seen:
                duplicates.add(candidate.ad_id)
            seen.add(candidate.ad_id)
        if duplicates:
            raise ValueError(
                f"the candidate set repeats the ad ids {sorted(duplicates)}. An "
                "auction ranks distinct candidates, and a duplicate id makes the "
                "returned ranking ambiguous for the caller."
            )
        return self

    def rows(self) -> List[Dict[str, FeatureValue]]:
        """Merge the context into each candidate and return the raw rows.

        The candidate's own value wins on a collision, which is the only sane
        precedence. The context is a default for the auction and the candidate
        is the specific thing being scored.
        """
        return [{**self.context, **candidate.features} for candidate in self.candidates]


class ScoredAd(BaseModel):
    """One scored candidate, in ranked position."""

    ad_id: str
    rank: int = Field(..., ge=1, description="1 is the highest ranked candidate.")
    p_click: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated click probability."
    )


class ScoreTimings(BaseModel):
    """Where the server side wall time of this request went."""

    feature_ms: float
    model_ms: float
    total_ms: float


class ScoreResponse(BaseModel):
    """The ranked candidate set plus the server side timing breakdown."""

    request_id: Optional[str] = None
    backend: str = Field(..., description="The backend key that produced these scores.")
    model_name: str
    n_candidates: int
    ranked: List[ScoredAd]
    timings: ScoreTimings


class HealthResponse(BaseModel):
    """What is loaded, what it is running on, and whether it can serve."""

    status: str
    backend: Dict[str, Any]
    hardware: Dict[str, Any]
    bundle: Dict[str, Any]
    uptime_seconds: float
    requests_total: int


class ErrorResponse(BaseModel):
    """The body every rejected request comes back with."""

    detail: str
