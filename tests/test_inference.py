"""Tests for the ONNX export and the inference benchmark backends.

These tests build a tiny DeepFM module, never train it, export it to ONNX, and
check that the ONNX Runtime backend reproduces the raw PyTorch probabilities
within a tight tolerance. They also check that a missing optional backend is
skipped gracefully by returning None rather than raising. The module is kept
small with tiny vocab sizes so the export stays fast and writes a small file.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from src.models.deepfm import DeepFMModule
from src.schema import FeatureMeta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    """Load the run_inference_benchmark script as an importable module.

    The benchmark lives under scripts and is not a package, so it is loaded by
    file path. Importing it does not run the benchmark because the entry point
    is guarded by a main check.
    """
    path = os.path.join(_ROOT, "scripts", "run_inference_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_inference_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_module_and_meta():
    """Build a tiny DeepFM module and its matching feature metadata.

    The vocab sizes are small so the exported ONNX graph stays tiny. Dropout is
    zero and the module is in eval mode so the forward pass is deterministic.
    """
    meta = FeatureMeta(n_numerical=4, cat_vocab_sizes=[5, 5, 5], cross_vocab_sizes=[7])
    torch.manual_seed(42)
    module = DeepFMModule(meta, embed_dim=4, hidden=[8, 4], dropout=0.0).eval()
    return module, meta


def _dummy_inputs(meta, n_rows: int):
    """Return a valid (numerical, cat) batch for the tiny module.

    Every cat index stays below the smallest field vocab so the offset lookup in
    the embedding layer is always in range.
    """
    rng = np.random.default_rng(42)
    numerical = rng.standard_normal((n_rows, meta.n_numerical)).astype(np.float32)
    cat = rng.integers(0, 5, size=(n_rows, meta.n_embed_fields)).astype(np.int64)
    return numerical, cat


def test_onnx_export_matches_pytorch(tmp_path):
    """ONNX Runtime reproduces the PyTorch probabilities after export."""
    bench = _load_script()
    module, meta = _tiny_module_and_meta()

    onnx_path = str(tmp_path / "tiny.onnx")
    bench.export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)
    assert os.path.exists(onnx_path)

    pytest.importorskip("onnxruntime")
    numerical, cat = _dummy_inputs(meta, n_rows=8)

    torch_pred = bench.make_torch_runner(module)(numerical, cat)
    onnx_run = bench.make_onnx_runner(onnx_path)
    assert onnx_run is not None
    onnx_pred = onnx_run(numerical, cat)

    assert torch_pred.shape == (8,)
    np.testing.assert_allclose(onnx_pred, torch_pred, atol=1e-4)


def test_onnx_export_dynamic_batch(tmp_path):
    """The exported graph serves a batch size different from the export size."""
    bench = _load_script()
    module, meta = _tiny_module_and_meta()

    onnx_path = str(tmp_path / "tiny.onnx")
    bench.export_onnx(module, meta, batch_size=8, onnx_path=onnx_path)

    pytest.importorskip("onnxruntime")
    onnx_run = bench.make_onnx_runner(onnx_path)
    # Run at a batch size of 3, which is not the size used at export time.
    numerical, cat = _dummy_inputs(meta, n_rows=3)
    onnx_pred = onnx_run(numerical, cat)
    torch_pred = bench.make_torch_runner(module)(numerical, cat)
    assert onnx_pred.shape == (3,)
    np.testing.assert_allclose(onnx_pred, torch_pred, atol=1e-4)


def test_missing_backend_skips_gracefully(tmp_path, monkeypatch):
    """A missing optional backend returns None instead of raising."""
    bench = _load_script()
    module, meta = _tiny_module_and_meta()

    onnx_path = str(tmp_path / "tiny.onnx")
    bench.export_onnx(module, meta, batch_size=4, onnx_path=onnx_path)

    # Mapping a module name to None in sys.modules makes import raise ImportError.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "openvino", None)
    assert bench.make_onnx_runner(onnx_path) is None
    assert bench.make_openvino_runner(onnx_path) is None
