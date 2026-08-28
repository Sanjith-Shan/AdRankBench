"""Tests for the serving layer.

Every test here runs on a cpu only machine with no accelerator, which is the
machine the serving numbers in docs/SERVING.md were measured on. Nothing is
skipped for missing hardware, because nothing here needs any.

The tests are grouped by what they are protecting.

The first group is the persistence layer. A serving bundle that round trips
badly is the failure this whole package exists to prevent, so the state has to
survive a write and a read, the count table prune has to be exactly behaviour
preserving rather than approximately, and a tampered artifact has to be caught
by its fingerprint.

The second group is feature parity, and it is the important one. The served
feature transform has to produce the identical arrays the offline pipeline
produces for the same rows, bit for bit, on both the dense block and the integer
block. Approximately equal is not the bar. A hashed categorical index that is
off by one bucket is not close to correct, it is a different embedding row, and
a standardization statistic that is off in the last decimal is a train and serve
mismatch that will not show up until someone compares an offline metric to an
online one and cannot explain the gap.

The third group is the http surface. The app has to start, /health has to report
a real backend rather than a placeholder, /score has to return the candidate set
ranked in the right order with the right shape, and a malformed request has to
be rejected with a status a caller can act on.

The fourth group is the lane equality. The batch job and the online service have
to return identical probabilities for identical rows, because that is the claim
the whole batch lane rests on.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data.loader import generate_synthetic
from src.data.preprocess import FeaturePipeline
from src.data.split import temporal_split
from src.inference.export import dataset_arrays
from src.schema import CAT_COLS, NUM_COLS
from src.serving.artifact import build_bundle, load_bundle
from src.serving.batch import run_batch_job, score_frame
from src.serving.calibration import Calibrator, probabilities_to_logits
from src.serving.features import (
    featurize_rows,
    feature_state,
    load_feature_pipeline,
    pipeline_from_state,
    rows_to_frame,
    unknown_fields,
    write_feature_state,
)
from src.serving.metrics import ServiceMetrics
from src.serving.runtime import PREFERENCE_ORDER, ScoringEngine, select_backend
from src.serving.service import ServiceConfig, build_service

# Two sample sizes, for two different jobs.
#
# The parity and persistence tests need a fitted pipeline and a handful of rows,
# and nothing about them gets more true with more data, so they use a small
# sample and stay fast. It is still large enough that the categorical count
# table holds both frequent and rare values, which is what makes the prune
# equivalence test a real test rather than a vacuous one.
#
# The bundle tests need a checkpoint and a feature pipeline that were fitted on
# the same rows, since a bundle assembled from mismatched halves is exactly what
# the validation tripwire is there to catch. The bundle fixture therefore builds
# into its own artifact directory and trains its own small checkpoint rather
# than borrowing the one in results/, which was trained against a different
# split and would make the tripwire fire correctly and fail the suite for the
# wrong reason.
SAMPLE_ROWS = 4000
BUNDLE_ROWS = 20000
SEED = 42


@pytest.fixture(scope="module")
def frames():
    """The three temporal splits of a small synthetic sample."""
    frame = generate_synthetic(SAMPLE_ROWS, seed=SEED)
    return temporal_split(frame)


@pytest.fixture(scope="module")
def offline_pipeline(frames):
    """A pipeline fitted the way the offline benchmark fits it, on train only."""
    train_df, _, _ = frames
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    return pipeline


@pytest.fixture(scope="module")
def served_pipeline(offline_pipeline, tmp_path_factory):
    """The same pipeline after a write to disk and a read back."""
    path = str(tmp_path_factory.mktemp("bundle") / "feature_pipeline.json.gz")
    write_feature_state(offline_pipeline, path)
    return load_feature_pipeline(path)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A serving bundle built end to end into a temporary directory.

    Everything is temporary, the checkpoint and the exported graph included, so
    the suite is hermetic and leaves nothing in results/. It also means the
    weights and the feature pipeline in this bundle were fitted on the same
    rows, which is the condition a real bundle has to satisfy and the one the
    validation tripwire checks.
    """
    root = tmp_path_factory.mktemp("serving")
    return build_bundle(
        bundle_dir=str(root / "bundle"),
        artifact_dir=str(root / "artifacts"),
        sample_size=BUNDLE_ROWS,
        synthetic=True,
        seed=SEED,
        verbose=False,
    )


@pytest.fixture(scope="module")
def service(bundle):
    """A service built on that bundle, with warmup, exactly as serve.py builds it."""
    config = ServiceConfig(
        bundle_dir=bundle.bundle_dir,
        sample_size=BUNDLE_ROWS,
        synthetic=True,
        thread_pool_size=4,
        verbose=False,
    )
    return build_service(config)


@pytest.fixture(scope="module")
def client(service):
    """A test client over the built service."""
    with TestClient(service.app) as test_client:
        yield test_client


def _rows_from_frame(frame):
    """Turn a frame into the raw row dicts a caller would send over the wire."""
    columns = list(NUM_COLS) + list(CAT_COLS)
    records = frame[columns].to_dict(orient="records")
    rows = []
    for record in records:
        row = {}
        for key, value in record.items():
            if key in CAT_COLS:
                row[key] = str(value)
            else:
                number = float(value)
                # json cannot carry NaN, so a caller sends null for a missing
                # dense field. Round tripping through that is part of the test.
                row[key] = None if number != number else number
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# The persistence layer
# ---------------------------------------------------------------------------


def test_feature_state_round_trips(offline_pipeline, served_pipeline):
    """The loaded pipeline carries the same fitted parameters it was written with."""
    assert np.array_equal(
        offline_pipeline.numerical.means_, served_pipeline.numerical.means_
    )
    assert np.array_equal(
        offline_pipeline.numerical.stds_, served_pipeline.numerical.stds_
    )
    assert offline_pipeline.crosses.pairs_ == served_pipeline.crosses.pairs_
    assert (
        offline_pipeline.categorical.total_rows_
        == served_pipeline.categorical.total_rows_
    )
    assert offline_pipeline.meta.n_numerical == served_pipeline.meta.n_numerical
    assert (
        offline_pipeline.meta.embed_vocab_sizes()
        == served_pipeline.meta.embed_vocab_sizes()
    )


def test_count_table_prune_is_behaviour_preserving(offline_pipeline, served_pipeline):
    """Dropping counts below min_count must not change a single encoded value.

    The prune is the one place the persistence layer does something other than
    copy. The argument is that a value below min_count and a value missing from
    the table take the same branch, so dropping the first turns it into the
    second with no observable difference. This checks the argument rather than
    trusting it, and it only means anything because the sample really does
    contain values below the threshold.
    """
    state = feature_state(offline_pipeline)
    pruned = int(state["categorical"]["entries_dropped_below_min_count"])
    assert pruned > 0, (
        "the sample produced no rare categorical values, so this test would pass "
        "vacuously. Raise SAMPLE_ROWS or lower min_count."
    )
    assert served_pipeline.categorical.counts_ != offline_pipeline.categorical.counts_


def test_state_version_is_checked(offline_pipeline, tmp_path):
    """A state file written at an unknown layout version is refused, not guessed."""
    import gzip

    from src.serving.features import read_feature_state

    state = feature_state(offline_pipeline)
    state["version"] = 999
    path = str(tmp_path / "bad.json.gz")
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(state).encode("utf-8"))

    with pytest.raises(ValueError, match="layout version"):
        read_feature_state(path)

    # The rebuild path itself does not version check, because it is handed an
    # already validated dict. Feeding it one straight from the reader is the
    # only supported route and that route is checked above.
    state["version"] = 1
    assert pipeline_from_state(state) is not None


def test_bundle_detects_a_changed_artifact(bundle, tmp_path):
    """A manifest whose recorded fingerprint no longer matches must not load.

    An exported graph that changed under a bundle means the fitted feature
    statistics belong to different weights than the ones about to be served.
    That is training and serving skew arriving from the model side rather than
    the feature side, and it is exactly as silent.
    """
    import shutil

    copied = str(tmp_path / "bundle")
    shutil.copytree(bundle.bundle_dir, copied)
    manifest_path = os.path.join(copied, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["model"]["onnx_sha256"] = "0" * 64
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="changed under the bundle"):
        load_bundle(copied, verify=True, verbose=False)


# ---------------------------------------------------------------------------
# Feature parity. The important group.
# ---------------------------------------------------------------------------


def test_served_features_match_offline_exactly(frames, offline_pipeline, served_pipeline):
    """The served transform must reproduce the offline arrays bit for bit.

    The rows go out through the same shape a caller sends, which is a dict per
    candidate with a null wherever a dense value is missing, and come back
    through the serving featurizer. The offline arrays come from the unmodified
    pipeline applied to the original frame. Both integer blocks are compared with
    exact equality because a hash bucket index has no tolerance, and the float
    block is compared with exact equality too, because both paths run the same
    arithmetic on the same inputs and anything other than an identical result
    would mean they did not.
    """
    _, _, test_df = frames
    offline = offline_pipeline.transform(test_df)
    offline_numerical, offline_cat = dataset_arrays(offline)

    rows = _rows_from_frame(test_df)
    served_numerical, served_cat = featurize_rows(served_pipeline, rows)

    assert served_numerical.shape == offline_numerical.shape
    assert served_cat.shape == offline_cat.shape
    assert served_numerical.dtype == offline_numerical.dtype
    assert served_cat.dtype == offline_cat.dtype
    assert np.array_equal(served_cat, offline_cat), (
        "the hashed categorical and cross indices differ between the served and "
        "the offline transform. A different bucket is a different embedding row."
    )
    assert np.array_equal(served_numerical, offline_numerical), (
        "the standardized dense block differs between the served and the offline "
        "transform, which is train and serve skew in the numerical pipeline."
    )


def test_served_features_match_the_unpruned_pipeline(frames, offline_pipeline, served_pipeline):
    """Parity has to hold against the pipeline that still has the rare counts.

    The previous test compares against the offline pipeline object, and the
    served pipeline is the same object's state after the prune. This one checks
    the prune specifically, by transforming with the original unpruned encoder
    and requiring the same answer.
    """
    _, _, test_df = frames
    unpruned_cat = offline_pipeline.categorical.transform_hash(test_df)
    unpruned_freq = offline_pipeline.categorical.transform_freq(test_df)
    pruned_cat = served_pipeline.categorical.transform_hash(test_df)
    pruned_freq = served_pipeline.categorical.transform_freq(test_df)
    assert np.array_equal(unpruned_cat, pruned_cat)
    assert np.array_equal(unpruned_freq, pruned_freq)


def test_absent_fields_become_the_missing_signal(served_pipeline):
    """A field a caller omits has to land on the same value a missing cell lands on.

    Missingness is a trained feature in this pipeline, since the dense block
    carries a binary indicator per column. A request that omits I3 must produce
    the indicator, not a zero that reads as an observed value of zero.
    """
    numerical, _ = featurize_rows(served_pipeline, [{}])
    n_dense = len(NUM_COLS)
    indicators = numerical[0, n_dense:]
    assert np.all(indicators == 1.0)

    explicit_null, _ = featurize_rows(
        served_pipeline, [{col: None for col in NUM_COLS}]
    )
    assert np.array_equal(numerical, explicit_null)


def test_unknown_field_names_are_reported():
    """A typo in a feature name is a detectable error rather than a silent default."""
    assert unknown_fields({"I1": 1.0, "C1": "abc"}) == []
    assert unknown_fields({"l3": 1.0, "C27": "x"}) == ["C27", "l3"]


def test_row_frame_has_the_canonical_schema(served_pipeline):
    """The frame the featurizer builds carries every schema column in order."""
    frame = rows_to_frame([{"I1": 1.0}, {"C1": "abc"}])
    assert list(frame.columns) == ["label"] + list(NUM_COLS) + list(CAT_COLS)
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_identity_calibrator_is_a_no_op():
    """The default calibrator returns exactly what it was given."""
    probs = np.array([0.01, 0.25, 0.5, 0.99])
    assert np.array_equal(Calibrator().apply(probs), probs)


def test_platt_calibrator_inverts_cleanly():
    """A unit Platt scaler must be the identity to floating point."""
    probs = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    identity = Calibrator(method="platt", a=1.0, b=0.0)
    assert np.allclose(identity.apply(probs), probs, atol=1e-12)
    assert np.allclose(
        probabilities_to_logits(probs), np.log(probs / (1.0 - probs)), atol=1e-9
    )


def test_bundle_calibration_is_recorded_and_fitted_on_validation(bundle):
    """The manifest has to say what the calibration did and where it came from."""
    calibration = bundle.manifest["calibration"]
    assert calibration["method"] in ("identity", "platt")
    assert "validation split" in calibration["fitted_on"]
    assert calibration["val_ece_before"] >= 0.0
    if calibration["applied"]:
        assert calibration["val_ece_after"] < calibration["val_ece_before"]


def test_bundle_validation_record_is_a_skew_tripwire(bundle):
    """The bundle must record a validation ranking check that a mismatch would fail."""
    validation = bundle.manifest["validation"]
    assert validation["rows"] > 0
    assert 0.0 <= validation["auc"] <= 1.0
    assert validation["auc"] > 0.6, (
        "the bundle's own validation AUC is near chance, which means the "
        "checkpoint and the feature pipeline in it were not fitted on the same "
        "data."
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selection(bundle):
    """One startup probe, reused. Constructing a backend compiles a graph."""
    return select_backend(bundle, verbose=False)


def test_backend_selection_reports_every_skip(selection):
    """The probe must record a reason for every backend it could not build."""
    assert selection.result.available
    assert selection.key in PREFERENCE_ORDER
    assert selection.order[0] == PREFERENCE_ORDER[0]
    for record in selection.skipped():
        assert record["note"], f"{record['key']} was skipped with no reason given"


def test_reduced_precision_is_opt_in(selection):
    """The default order must not contain a backend that changes the numbers."""
    assert selection.result.spec.precision == "fp32"
    assert not selection.allow_reduced_precision


def test_pinned_backend_is_honoured_or_refused(bundle):
    """Pinning a backend must either get that backend or fail loudly."""
    selection = select_backend(bundle, preferred="pytorch-cpu-fp32", verbose=False)
    assert selection.key == "pytorch-cpu-fp32"
    with pytest.raises(KeyError, match="unknown backend key"):
        select_backend(bundle, preferred="not-a-backend", verbose=False)


# ---------------------------------------------------------------------------
# The http surface
# ---------------------------------------------------------------------------


def test_health_reports_a_real_backend(client, service):
    """The service has to name the backend it actually selected and the hardware."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"]["selected"] == service.engine.backend_key
    assert body["backend"]["selected"] in PREFERENCE_ORDER
    assert body["backend"]["lane"] in ("cpu", "gpu")
    assert body["backend"]["probed"], "the health document has no probe record"
    assert body["hardware"]["host"]["cpu_model"]
    assert body["bundle"]["feature_pipeline_sha256"]
    assert body["bundle"]["checkpoint_sha256"]


def test_score_returns_a_ranked_candidate_set(client):
    """The response must carry every candidate, ranked from one, in score order."""
    candidates = [
        {"ad_id": f"ad-{i}", "features": {"C1": f"{i:08x}", "I1": float(i)}}
        for i in range(12)
    ]
    response = client.post(
        "/score", json={"request_id": "auction-1", "candidates": candidates}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["request_id"] == "auction-1"
    assert body["n_candidates"] == 12
    assert len(body["ranked"]) == 12
    assert [entry["rank"] for entry in body["ranked"]] == list(range(1, 13))
    assert {entry["ad_id"] for entry in body["ranked"]} == {
        c["ad_id"] for c in candidates
    }

    probabilities = [entry["p_click"] for entry in body["ranked"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert body["timings"]["total_ms"] >= 0.0


def test_score_respects_top_k(client):
    """A top_k request returns only the leaders and still ranks them from one."""
    candidates = [{"ad_id": f"ad-{i}", "features": {"C1": f"{i:08x}"}} for i in range(20)]
    full = client.post("/score", json={"candidates": candidates}).json()
    topk = client.post("/score", json={"candidates": candidates, "top_k": 3}).json()
    assert len(topk["ranked"]) == 3
    assert [e["ad_id"] for e in topk["ranked"]] == [
        e["ad_id"] for e in full["ranked"][:3]
    ]


def test_score_is_deterministic(client):
    """The same auction sent twice must come back identical."""
    candidates = [{"ad_id": f"ad-{i}", "features": {"C3": f"{i:08x}"}} for i in range(8)]
    first = client.post("/score", json={"candidates": candidates}).json()["ranked"]
    second = client.post("/score", json={"candidates": candidates}).json()["ranked"]
    assert first == second


def test_context_is_merged_and_overridden_by_the_candidate(client):
    """A context field applies to every candidate unless the candidate sets it."""
    shared = client.post(
        "/score",
        json={
            "context": {"C1": "deadbeef"},
            "candidates": [{"ad_id": "a", "features": {}}],
        },
    ).json()
    explicit = client.post(
        "/score",
        json={"candidates": [{"ad_id": "a", "features": {"C1": "deadbeef"}}]},
    ).json()
    assert shared["ranked"][0]["p_click"] == explicit["ranked"][0]["p_click"]

    overridden = client.post(
        "/score",
        json={
            "context": {"C1": "deadbeef"},
            "candidates": [{"ad_id": "a", "features": {"C1": "0000ffff"}}],
        },
    ).json()
    only = client.post(
        "/score",
        json={"candidates": [{"ad_id": "a", "features": {"C1": "0000ffff"}}]},
    ).json()
    assert overridden["ranked"][0]["p_click"] == only["ranked"][0]["p_click"]


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({}, "no candidates key at all"),
        ({"candidates": []}, "an empty candidate set"),
        ({"candidates": [{"features": {}}]}, "a candidate with no ad id"),
        (
            {"candidates": [{"ad_id": "a", "features": {"C99": "x"}}]},
            "an unknown feature name",
        ),
        (
            {"context": {"nope": 1}, "candidates": [{"ad_id": "a"}]},
            "an unknown context field",
        ),
        (
            {"candidates": [{"ad_id": "a"}, {"ad_id": "a"}]},
            "a duplicated ad id",
        ),
        (
            {"candidates": [{"ad_id": "a"}], "top_k": 0},
            "a top_k below one",
        ),
    ],
)
def test_malformed_requests_are_rejected(client, payload, reason):
    """A bad request has to come back as a 422 with a body, never a 500."""
    response = client.post("/score", json=payload)
    assert response.status_code == 422, f"{reason} was not rejected"
    assert response.json()["detail"]


def test_metrics_expose_counters_and_histograms(client):
    """Both metrics views have to reflect the requests that were served."""
    client.post("/score", json={"candidates": [{"ad_id": "a"}, {"ad_id": "b"}]})

    text = client.get("/metrics")
    assert text.status_code == 200
    body = text.text
    assert "adrank_requests_total" in body
    assert "adrank_request_latency_seconds_bucket" in body
    assert 'le="+Inf"' in body
    assert "adrank_feature_latency_seconds_count" in body
    assert "adrank_model_latency_seconds_count" in body

    as_json = client.get("/metrics", params={"format": "json"}).json()
    assert as_json["requests_total"] > 0
    assert as_json["ads_scored_total"] > 0
    assert as_json["request_latency"]["count"] > 0
    assert as_json["request_latency"]["quantiles_ms"]["p99"] >= 0.0


def test_metrics_start_empty_and_serialize():
    """A fresh histogram reports no observations rather than a misleading zero."""
    metrics = ServiceMetrics()
    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 0
    assert np.isnan(snapshot["request_latency"]["quantiles_ms"]["p99"])
    assert "adrank_requests_total 0" in metrics.render_prometheus()


def test_oversized_request_is_rejected(bundle, selection):
    """A candidate set past the engine limit is refused before it is featurized."""
    engine = ScoringEngine(bundle, selection, max_candidates=4)
    with pytest.raises(ValueError, match="at most 4"):
        engine.score_rows([{} for _ in range(5)])


# ---------------------------------------------------------------------------
# The two lanes
# ---------------------------------------------------------------------------


def test_batch_and_online_scores_are_identical(client, service, frames):
    """The same rows down both lanes must produce the same probabilities.

    This is the claim the batch lane rests on and the reason it shares the
    engine rather than reimplementing it. The comparison is exact rather than
    approximate, because both lanes run the same weights through the same
    runtime on the same featurized arrays, and any difference at all would mean
    one of those three is not actually shared.
    """
    _, _, test_df = frames
    sample = test_df.head(64).reset_index(drop=True)
    rows = _rows_from_frame(sample)

    online = client.post(
        "/score",
        json={
            "candidates": [
                {"ad_id": f"row-{i}", "features": row} for i, row in enumerate(rows)
            ]
        },
    ).json()
    online_by_id = {e["ad_id"]: e["p_click"] for e in online["ranked"]}

    batch_probs, _, _ = score_frame(service.engine, sample, chunk_rows=32)
    assert len(batch_probs) == len(rows)

    for i, probability in enumerate(batch_probs):
        assert online_by_id[f"row-{i}"] == float(probability), (
            f"row {i} scored differently online and in batch, which means the two "
            "lanes are not sharing one artifact and one feature pipeline"
        )


def test_batch_job_writes_scored_rows(service, frames, tmp_path):
    """The batch job has to write a joinable file and report a real throughput."""
    _, _, test_df = frames
    shard = test_df.head(200).reset_index(drop=True)
    shard.insert(0, "ad_id", [f"row-{i}" for i in range(len(shard))])
    input_path = str(tmp_path / "shard.parquet")
    output_path = str(tmp_path / "scored.parquet")
    shard.to_parquet(input_path, index=False)

    result = run_batch_job(
        service.engine, input_path, output_path, chunk_rows=64, hardware_label="test"
    )
    assert result.rows == 200
    assert result.rows_per_second > 0.0
    assert result.backend == service.engine.backend_key
    assert os.path.exists(output_path)

    import pandas as pd

    scored = pd.read_parquet(output_path)
    assert list(scored.columns) == ["ad_id", "p_click", "label"]
    assert len(scored) == 200
    assert scored["p_click"].between(0.0, 1.0).all()
    assert scored["ad_id"].tolist() == shard["ad_id"].tolist()


def test_batch_ignores_passenger_columns(service, frames, tmp_path):
    """A shard column that is not a feature must not reach the featurizer."""
    _, _, test_df = frames
    shard = test_df.head(50).reset_index(drop=True)
    shard.insert(0, "ad_id", [f"row-{i}" for i in range(len(shard))])
    shard["campaign"] = "some passenger value"
    input_path = str(tmp_path / "passengers.parquet")
    shard.to_parquet(input_path, index=False)

    result = run_batch_job(
        service.engine, input_path, str(tmp_path / "out.csv"), chunk_rows=25
    )
    assert result.rows == 50
    assert result.id_column == "ad_id"
