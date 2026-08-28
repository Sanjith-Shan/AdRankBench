"""Tests for the inference serving stack.

Every test here runs and passes on a machine with no NVIDIA gpu, because that
is the machine this project is developed on and a test suite that only proves
something on hardware nobody has is not a test suite. The tests check three
kinds of thing.

The first is that the package imports and behaves on a cpu only host. The
hardware record has to be well formed, the backend registry has to return one
record per requested backend, and a gpu backend has to come back as unavailable
with a stated reason rather than raising.

The second is that the serving path is correct for both models. The ONNX export
has to work for DeepFM and for DCN, the exported graph has to serve a batch size
it was not exported at, and every cpu backend has to reproduce the raw PyTorch
probabilities to within floating point tolerance.

The third is that the analysis code produces sane numbers from a module
definition alone, since that part of the report does not need a device.

Anything that genuinely needs cuda or TensorRT is skipped through
pytest.importorskip so it runs on a gpu host and skips cleanly here.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from src.inference.analysis import bound_verdict, module_cost_model
from src.inference.backends import (
    BackendContext,
    build_backend,
    default_specs,
    make_onnx_runner,
    make_openvino_runner,
    make_torch_runner,
    probe_backends,
    spec_by_key,
)
from src.inference.calibrator import calibration_arrays, calibration_cache_path
from src.inference.common import NOT_AVAILABLE, fmt, jsonable, sigmoid
from src.inference.export import (
    build_module,
    dataset_arrays,
    display_name,
    export_onnx,
    make_batches,
    model_size_bytes,
)
from src.inference.hardware import (
    collect_hardware_record,
    cuda_lane_ready,
    gpu_unavailable_reason,
    hardware_markdown_lines,
)
from src.inference.power import PowerSampler, unavailable_power_record
from src.inference.trt_builder import (
    DOCKER_HINT,
    PRECISIONS,
    build_engine,
    engine_path_for,
    tensorrt_available,
)
from src.inference.trt_runner import LayerProfiler, empty_profile, load_runner
from src.schema import Dataset, FeatureMeta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiny_meta() -> FeatureMeta:
    """A tiny feature space so the exported graphs stay small and fast."""
    return FeatureMeta(n_numerical=4, cat_vocab_sizes=[5, 5, 5], cross_vocab_sizes=[7])


def _tiny_config(model_name: str) -> dict:
    """A tiny config for either supported model."""
    config = {"embed_dim": 4, "hidden": [8, 4], "dropout": 0.0}
    if model_name == "dcn":
        config["cross_layers"] = 2
    return config


def _tiny_module(model_name: str = "deepfm"):
    """Build an untrained tiny module in eval mode."""
    torch.manual_seed(42)
    module = build_module(model_name, _tiny_meta(), _tiny_config(model_name))
    return module.eval()


def _dummy_inputs(meta: FeatureMeta, n_rows: int):
    """Return a valid (numerical, cat) batch for a tiny module."""
    rng = np.random.default_rng(42)
    numerical = rng.standard_normal((n_rows, meta.n_numerical)).astype(np.float32)
    cat = rng.integers(0, 5, size=(n_rows, meta.n_embed_fields)).astype(np.int64)
    return numerical, cat


def _tiny_context(onnx_path: str, module, meta: FeatureMeta, engine_dir: str) -> BackendContext:
    """Build a backend context pointing at a freshly exported tiny graph."""
    return BackendContext(
        model_name="Tiny",
        module=module,
        onnx_path=onnx_path,
        n_numerical=meta.n_numerical,
        n_embed_fields=meta.n_embed_fields,
        max_batch=32,
        engine_paths={
            p: engine_path_for("Tiny", p, 32, engine_dir) for p in PRECISIONS
        },
        calibration_cache=calibration_cache_path("Tiny", engine_dir),
        trt_engine_cache_dir=os.path.join(engine_dir, "ort_trt_cache"),
    )


# ---------------------------------------------------------------------------
# Hardware provenance.
# ---------------------------------------------------------------------------


def test_hardware_record_is_well_formed():
    """The provenance record has every block and never raises on a cpu host."""
    record = collect_hardware_record()

    for block in ("collected_at_utc", "host", "python", "libraries", "gpu"):
        assert block in record

    host = record["host"]
    for field in ("cpu_model", "platform", "system", "machine", "logical_cores"):
        assert field in host
        assert host[field] is not None
    assert isinstance(host["cpu_model"], str) and host["cpu_model"]

    libs = record["libraries"]
    assert libs["torch"]["version"].startswith(torch.__version__[0])
    assert isinstance(libs["torch"]["cuda_available"], bool)
    assert isinstance(libs["onnxruntime"]["available_providers"], list)
    assert "available" in libs["tensorrt"]

    gpu = record["gpu"]
    assert isinstance(gpu["available"], bool)
    if not gpu["available"]:
        # An unavailable gpu must carry a reason, never an empty field.
        assert gpu["reason"]
        assert gpu["name"] == NOT_AVAILABLE


def test_hardware_record_renders_to_markdown():
    """The provenance record renders as a complete markdown table."""
    record = collect_hardware_record()
    lines = hardware_markdown_lines(record)
    assert lines[0].startswith("| Field |")
    assert lines[1].startswith("| ---")
    # Every rendered row is a markdown table row with content in both cells.
    for line in lines[2:]:
        assert line.startswith("| ") and line.endswith(" |")
        assert line.count("|") >= 3


def test_hardware_record_is_json_serializable():
    """The record survives the json conversion the artifacts depend on."""
    import json

    record = collect_hardware_record()
    text = json.dumps(jsonable(record))
    assert "cpu_model" in text


def test_gpu_unavailable_reason_is_stated_when_there_is_no_gpu():
    """A host with no gpu reports a reason rather than an empty string."""
    record = collect_hardware_record()
    if record["gpu"]["available"]:
        pytest.skip("this host has a gpu, so there is no unavailable reason to check")
    reason = gpu_unavailable_reason(record)
    assert isinstance(reason, str) and len(reason) > 20


def test_cuda_lane_ready_matches_torch():
    """The cuda gate agrees with torch and always supplies a reason when closed."""
    ready, reason = cuda_lane_ready()
    assert ready == bool(torch.cuda.is_available())
    if not ready:
        assert reason


# ---------------------------------------------------------------------------
# ONNX export for both models.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["deepfm", "dcn"])
def test_onnx_export_works_for_both_models(tmp_path, model_name):
    """Both DeepFM and DCN export to a single self contained ONNX file."""
    meta = _tiny_meta()
    module = _tiny_module(model_name)
    onnx_path = str(tmp_path / f"{model_name}.onnx")

    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)

    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0
    # A single file with no external data sidecar next to it.
    assert not os.path.exists(onnx_path + ".data")


@pytest.mark.parametrize("model_name", ["deepfm", "dcn"])
def test_cpu_backends_agree_with_pytorch(tmp_path, model_name):
    """Every available cpu backend reproduces the PyTorch probabilities."""
    meta = _tiny_meta()
    module = _tiny_module(model_name)
    onnx_path = str(tmp_path / f"{model_name}.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)

    ctx = _tiny_context(onnx_path, module, meta, str(tmp_path / "trt"))
    numerical, cat = _dummy_inputs(meta, n_rows=8)

    reference = build_backend(spec_by_key("pytorch-cpu-fp32"), ctx)
    assert reference.available
    expected = reference.runner(numerical, cat)
    assert expected.shape == (8,)
    assert np.all((expected >= 0.0) & (expected <= 1.0))

    # The tolerances differ per backend on purpose. ONNX Runtime replays the
    # same fp32 kernels and agrees to within rounding. OpenVINO recompiles the
    # network for the host cpu and reorders and fuses arithmetic, so it lands a
    # little further away. The benchmark measures that drift and reports it
    # rather than hiding it, and this gate is set where the measured drift
    # actually sits rather than where it would be convenient.
    tolerances = {"onnxruntime-cpu-fp32": 1e-5, "openvino-cpu-fp32": 5e-3}
    for key, tolerance in tolerances.items():
        result = build_backend(spec_by_key(key), ctx)
        if not result.available:
            # A missing optional package is a stated reason, not a failure.
            assert result.note
            continue
        got = result.runner(numerical, cat)
        assert got.shape == (8,)
        np.testing.assert_allclose(got, expected, atol=tolerance)


@pytest.mark.parametrize("model_name", ["deepfm", "dcn"])
def test_exported_graph_serves_a_different_batch_size(tmp_path, model_name):
    """The dynamic batch axis lets one graph serve a batch it was not exported at."""
    pytest.importorskip("onnxruntime")
    meta = _tiny_meta()
    module = _tiny_module(model_name)
    onnx_path = str(tmp_path / f"{model_name}.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)

    runner = make_onnx_runner(onnx_path)
    assert runner is not None
    numerical, cat = _dummy_inputs(meta, n_rows=3)
    got = runner(numerical, cat)
    expected = make_torch_runner(module)(numerical, cat)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------


def test_registry_returns_one_record_per_spec(tmp_path):
    """probe_backends returns a complete grid with no holes in it."""
    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    onnx_path = str(tmp_path / "tiny.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)
    ctx = _tiny_context(onnx_path, module, meta, str(tmp_path / "trt"))

    specs = default_specs()
    results = probe_backends(ctx, specs, verbose=False)

    assert len(results) == len(specs)
    assert [r.spec.key for r in results] == [s.key for s in specs]
    for result in results:
        assert isinstance(result.available, bool)
        if not result.available:
            # An unavailable backend must say why, in a full sentence.
            assert result.note and len(result.note) > 10
            assert result.runner is None
        record = result.as_dict()
        for field in ("key", "label", "runtime", "precision", "device", "lane", "available"):
            assert field in record


def test_registry_always_has_the_cpu_reference(tmp_path):
    """The eager PyTorch cpu reference is available on any host."""
    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    onnx_path = str(tmp_path / "tiny.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)
    ctx = _tiny_context(onnx_path, module, meta, str(tmp_path / "trt"))

    results = {r.spec.key: r for r in probe_backends(ctx, verbose=False)}
    assert results["pytorch-cpu-fp32"].available


def test_gpu_backends_are_skipped_not_crashed(tmp_path):
    """On a cpu only host every gpu backend is unavailable with a reason."""
    if torch.cuda.is_available():
        pytest.skip("this host has cuda, so the gpu lane is expected to run")

    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    onnx_path = str(tmp_path / "tiny.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)
    ctx = _tiny_context(onnx_path, module, meta, str(tmp_path / "trt"))

    gpu_specs = [s for s in default_specs() if s.is_gpu]
    assert gpu_specs
    for spec in gpu_specs:
        result = build_backend(spec, ctx)
        assert not result.available
        assert result.runner is None
        assert result.note


def test_precision_filter_keeps_the_cpu_lane():
    """Filtering the grid to fp16 still leaves the fp32 cpu reference in it."""
    specs = default_specs(["fp16"])
    keys = {s.key for s in specs}
    assert "pytorch-cpu-fp32" in keys
    assert "tensorrt-fp16" in keys
    assert "tensorrt-int8" not in keys


def test_legacy_helpers_return_none_when_a_package_is_missing(tmp_path, monkeypatch):
    """The older single backend helpers skip gracefully rather than raising."""
    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    onnx_path = str(tmp_path / "tiny.onnx")
    export_onnx(module, meta, batch_size=4, onnx_path=onnx_path)

    # Mapping a module name to None in sys.modules makes import raise ImportError.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "openvino", None)
    assert make_onnx_runner(onnx_path) is None
    assert make_openvino_runner(onnx_path) is None


def test_backend_constructor_never_raises_on_a_bad_graph(tmp_path):
    """A missing onnx file becomes an unavailable record, not an exception."""
    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    ctx = _tiny_context(str(tmp_path / "does_not_exist.onnx"), module, meta, str(tmp_path))
    result = build_backend(spec_by_key("onnxruntime-cpu-fp32"), ctx)
    assert not result.available
    assert "does not exist" in result.note


# ---------------------------------------------------------------------------
# TensorRT pieces, which degrade cleanly with no gpu.
# ---------------------------------------------------------------------------


def test_tensorrt_availability_reports_a_reason_and_a_container():
    """When TensorRT cannot run the reason names the container to use instead."""
    available, reason = tensorrt_available()
    assert isinstance(available, bool)
    if not available:
        assert reason
        assert "nvcr.io/nvidia/tensorrt" in reason
        assert DOCKER_HINT in reason


def test_engine_path_names_the_model_precision_and_max_batch(tmp_path):
    """The engine file name carries everything that makes an engine unique."""
    path = engine_path_for("DeepFM", "int8", 4096, str(tmp_path))
    assert os.path.basename(path) == "deepfm_int8_bs4096.engine"


def test_build_engine_returns_a_failed_record_rather_than_raising(tmp_path):
    """Building without TensorRT produces a record with ok false and a reason."""
    if tensorrt_available()[0]:
        pytest.skip("this host has TensorRT, so the unavailable path is not exercised")
    record = build_engine(
        onnx_path=str(tmp_path / "missing.onnx"),
        model_name="DeepFM",
        precision="fp16",
        output_dir=str(tmp_path),
    )
    assert record.ok is False
    assert record.message
    assert record.precision == "fp16"


def test_build_engine_rejects_an_unknown_precision(tmp_path):
    """An unknown precision is refused with a message, not a traceback."""
    record = build_engine(
        onnx_path=str(tmp_path / "missing.onnx"),
        model_name="DeepFM",
        precision="fp4",
        output_dir=str(tmp_path),
    )
    assert record.ok is False
    assert "unknown precision" in record.message


def test_trt_runner_load_returns_a_reason_for_a_missing_engine(tmp_path):
    """Loading an engine that is not there returns None and says why."""
    runner, reason = load_runner(str(tmp_path / "nope.engine"), max_batch=8)
    assert runner is None
    assert reason


def test_layer_profiler_buckets_gather_against_matmul():
    """The profiler splits layer time into the two halves of a DLRM network."""
    profiler = LayerProfiler()
    profiler.record("node_Gather_7", 3.0)
    profiler.record("node_MatMul_11", 1.0)
    profiler.iterations = 2

    bucket = profiler.bucket()
    assert bucket["iterations"] == 2
    assert bucket["gather_ms"] == pytest.approx(1.5)
    assert bucket["matmul_ms"] == pytest.approx(0.5)
    assert bucket["gather_share_pct"] == pytest.approx(75.0)
    assert bucket["matmul_share_pct"] == pytest.approx(25.0)


def test_empty_profile_carries_a_reason():
    """A profile that could not be taken says so rather than reporting zeros."""
    profile = empty_profile("no engine ran")
    assert profile["iterations"] == 0
    assert profile["total_ms"] is None
    assert profile["reason"] == "no engine ran"


# ---------------------------------------------------------------------------
# Calibration.
# ---------------------------------------------------------------------------


def test_calibration_rows_come_from_the_split_that_is_handed_in():
    """The calibration pool is a whole number of batches drawn under a seed."""
    rng = np.random.default_rng(0)
    numerical = rng.standard_normal((100, 4)).astype(np.float32)
    cat = rng.integers(0, 5, size=(100, 4)).astype(np.int64)

    a_num, a_cat, a_batches = calibration_arrays(numerical, cat, batch_size=8, max_batches=4, seed=42)
    b_num, b_cat, b_batches = calibration_arrays(numerical, cat, batch_size=8, max_batches=4, seed=42)

    assert a_batches == 4
    assert a_num.shape == (32, 4)
    assert a_cat.shape == (32, 4)
    # The same seed selects the same rows, which is what makes a build reproducible.
    np.testing.assert_array_equal(a_num, b_num)
    np.testing.assert_array_equal(a_cat, b_cat)
    assert b_batches == a_batches


def test_calibration_pool_is_empty_when_the_split_is_too_small():
    """A split smaller than one batch yields zero batches rather than a ragged one."""
    numerical = np.zeros((3, 4), dtype=np.float32)
    cat = np.zeros((3, 4), dtype=np.int64)
    _num, _cat, batches = calibration_arrays(numerical, cat, batch_size=8, max_batches=4)
    assert batches == 0


def test_calibration_cache_path_is_under_the_engine_directory(tmp_path):
    """The calibration cache lands in the committed artifact location."""
    path = calibration_cache_path("DeepFM", str(tmp_path))
    assert path.endswith("deepfm_int8_calibration.cache")
    assert os.path.isdir(os.path.dirname(path))


# ---------------------------------------------------------------------------
# Power.
# ---------------------------------------------------------------------------


def test_power_sampler_is_safe_with_no_gpu():
    """Starting and stopping the sampler on a cpu host reports not available."""
    sampler = PowerSampler(interval_s=0.01)
    sampler.start()
    sampler.stop()
    record = sampler.summarize(n_inferences=1000)

    assert isinstance(record["available"], bool)
    if not record["available"]:
        assert record["reason"]
        assert record["mean_watts"] is None
        assert record["inferences_per_joule"] is None


def test_unavailable_power_record_has_every_field():
    """The placeholder power record has the same shape as a real one."""
    record = unavailable_power_record("no gpu here")
    for field in (
        "available",
        "reason",
        "samples",
        "mean_watts",
        "peak_watts",
        "energy_joules",
        "inferences_per_joule",
    ):
        assert field in record
    assert record["available"] is False
    assert record["reason"] == "no gpu here"


# ---------------------------------------------------------------------------
# Analysis.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["deepfm", "dcn"])
def test_cost_model_produces_sane_counts(model_name):
    """The roofline cost model returns positive counts for both models."""
    meta = _tiny_meta()
    module = _tiny_module(model_name)
    cost = module_cost_model(
        module, meta.n_embed_fields, 4, meta.n_numerical, batch_size=64
    )

    assert cost["flops_per_batch"] > 0
    assert cost["bytes_per_batch"] > 0
    assert cost["embedding_parameters"] > 0
    assert 0.0 < cost["gather_share_of_bytes_pct"] <= 100.0
    assert cost["arithmetic_intensity_flops_per_byte"] > 0


def test_arithmetic_intensity_rises_with_batch_size():
    """Weight traffic amortizes, so a larger batch has a higher intensity."""
    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    small = module_cost_model(module, meta.n_embed_fields, 4, meta.n_numerical, 1)
    large = module_cost_model(module, meta.n_embed_fields, 4, meta.n_numerical, 1024)
    assert (
        large["arithmetic_intensity_flops_per_byte"]
        > small["arithmetic_intensity_flops_per_byte"]
    )


def test_bound_verdict_labels_both_ends():
    """A very low intensity reads memory bound and a very high one compute bound."""
    low = bound_verdict(0.5, None, None)
    high = bound_verdict(500.0, None, None)
    assert low["verdict"] == "memory bound"
    assert high["verdict"] == "compute bound"
    assert low["explanation"] and high["explanation"]


def test_bound_verdict_is_inconclusive_without_an_intensity():
    """With no intensity the verdict refuses to guess."""
    record = bound_verdict(None, None, None)
    assert record["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# Small helpers the report leans on.
# ---------------------------------------------------------------------------


def test_sigmoid_is_stable_at_the_extremes():
    """The sigmoid does not overflow on large magnitude logits."""
    values = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    out = sigmoid(values)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert out[-1] == pytest.approx(1.0)


def test_fmt_collapses_missing_values_to_the_marker():
    """A None and a NaN both render as the not available marker."""
    assert fmt(None) == NOT_AVAILABLE
    assert fmt(float("nan")) == NOT_AVAILABLE
    assert fmt(1.5, ".1f") == "1.5"


def test_batching_preserves_row_order_and_layout():
    """make_batches concatenates categorical then cross fields, in row order."""
    n = 7
    ds = Dataset(
        numerical=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        categorical=np.arange(n * 2, dtype=np.int64).reshape(n, 2),
        cat_freq=np.zeros((n, 2), dtype=np.float32),
        crosses=np.arange(n, dtype=np.int64).reshape(n, 1),
        label=np.zeros(n, dtype=np.float32),
    )
    batches = make_batches(ds, batch_size=3)
    assert [len(b[0]) for b in batches] == [3, 3, 1]

    numerical, cat = dataset_arrays(ds)
    assert cat.shape == (n, 3)
    np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), numerical)
    np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), cat)


def test_model_size_counts_parameters_and_buffers():
    """The reported model size is the parameter and buffer footprint at fp32."""
    module = _tiny_module("deepfm")
    size = model_size_bytes(module)
    expected = sum(p.numel() * p.element_size() for p in module.parameters())
    assert size >= expected
    assert size > 0


def test_display_names_are_canonical():
    """The display names match the checkpoint and engine file names."""
    assert display_name("deepfm") == "DeepFM"
    assert display_name("dcn") == "DCN"


# ---------------------------------------------------------------------------
# The benchmark script itself.
# ---------------------------------------------------------------------------


def _load_benchmark_script():
    """Load the benchmark script by path, since scripts is not a package."""
    path = os.path.join(_ROOT, "scripts", "run_inference_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_inference_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder_script():
    """Load the engine builder script by path."""
    path = os.path.join(_ROOT, "scripts", "build_trt_engines.py")
    spec = importlib.util.spec_from_file_location("build_trt_engines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_keeps_its_legacy_surface():
    """The names the earlier tests and the README rely on are still exported."""
    bench = _load_benchmark_script()
    for name in (
        "export_onnx",
        "make_torch_runner",
        "make_onnx_runner",
        "make_openvino_runner",
        "build_module",
        "load_dataframe",
        "main",
        "parse_args",
        "_sigmoid",
    ):
        assert hasattr(bench, name), name


def test_benchmark_metrics_helpers():
    """The accuracy and drift helpers behave on a small hand made case."""
    bench = _load_benchmark_script()
    y = np.array([0.0, 0.0, 1.0, 1.0])
    preds = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = bench.accuracy_metrics(y, preds)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["logloss"] > 0

    drift = bench.drift_metrics(preds, preds)
    assert drift["mean_abs_diff"] == pytest.approx(0.0)
    assert drift["max_abs_diff"] == pytest.approx(0.0)

    missing = bench.drift_metrics(preds, None)
    assert missing["mean_abs_diff"] is None
    assert missing["note"]


def test_benchmark_timing_helper_returns_one_latency_per_batch():
    """The timing loop records exactly repeats times batches latencies."""
    bench = _load_benchmark_script()
    batches = [(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.int64))] * 4

    calls = {"n": 0}

    def runner(numerical, cat):
        calls["n"] += 1
        return np.zeros(len(numerical))

    latencies = bench.time_backend(runner, batches, warmup=2, repeats=3)
    assert latencies.shape == (12,)
    assert calls["n"] == 14
    assert np.all(latencies >= 0)


def test_builder_script_parses_and_exposes_its_entry_point():
    """The engine builder script imports and exposes main without running it."""
    builder = _load_builder_script()
    assert hasattr(builder, "main")
    assert hasattr(builder, "parse_args")
    assert hasattr(builder, "write_build_report")


# ---------------------------------------------------------------------------
# CUDA only. These skip on a cpu host and run on a gpu one.
# ---------------------------------------------------------------------------


def test_cuda_backends_run_when_there_is_a_gpu(tmp_path):
    """On a cuda host the eager gpu backend matches the cpu reference."""
    if not torch.cuda.is_available():
        pytest.skip("no cuda device on this host")

    meta = _tiny_meta()
    module = _tiny_module("deepfm")
    onnx_path = str(tmp_path / "tiny.onnx")
    export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)
    ctx = _tiny_context(onnx_path, module, meta, str(tmp_path / "trt"))

    numerical, cat = _dummy_inputs(meta, n_rows=8)
    reference = build_backend(spec_by_key("pytorch-cpu-fp32"), ctx).runner(numerical, cat)

    result = build_backend(spec_by_key("pytorch-cuda-fp32"), ctx)
    assert result.available, result.note
    got = result.runner(numerical, cat)
    np.testing.assert_allclose(got, reference, atol=1e-4)


def test_tensorrt_python_api_shape_when_available():
    """When tensorrt imports, the api this project drives is the one that is there."""
    trt = pytest.importorskip("tensorrt")
    # The TensorRT 10 tensor addressing api. The runner is written against these.
    assert hasattr(trt, "IInt8EntropyCalibrator2")
    assert hasattr(trt, "IProfiler")
    assert hasattr(trt, "Builder")
    assert hasattr(trt.IExecutionContext, "execute_async_v3")
    assert hasattr(trt.IExecutionContext, "set_input_shape")
    assert hasattr(trt.IExecutionContext, "set_tensor_address")
    assert hasattr(trt.ICudaEngine, "num_io_tensors")
