"""INT8 post training calibration for the TensorRT engine builds.

TensorRT quantizes a network to INT8 by watching real activations flow through
it and choosing a scale per tensor. The entropy calibrator picks the scale that
minimizes the information lost between the float distribution and the quantized
one, which is the right default for a ranking model where the activation
distributions are long tailed rather than bounded.

Two decisions in this module matter for the honesty of the accuracy numbers the
benchmark reports.

The first is which rows feed the calibrator. They come from the validation
split, never from the test split. Calibrating on test rows would let the
quantized engine see the exact data it is later scored on, and the reported INT8
AUC would flatter the engine for a reason that has nothing to do with
quantization. The validation split is already held out of training, so it is the
correct source.

The second is the calibration cache. TensorRT writes the chosen scales to a
cache file, and a later build reads that file instead of running calibration
again. Committing the cache under results/trt makes an INT8 build reproducible
and makes it auditable, because the scales that produced a reported number are
on disk next to the number.

Everything that touches tensorrt is imported inside a function. Importing this
module on a machine with no gpu is safe and does nothing, which is what lets the
whole package import on the Apple Silicon development machine.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.inference.common import NOT_AVAILABLE

# Where calibration caches live. This is a committed artifact location so a
# reported INT8 number can always be traced back to the scales that produced it.
CALIBRATION_DIR = os.path.join("results", "trt")


def calibration_cache_path(model_name: str, output_dir: str = CALIBRATION_DIR) -> str:
    """Return the cache file path for one model, creating the directory."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{model_name.lower()}_int8_calibration.cache")


def calibration_arrays(
    val_numerical: np.ndarray,
    val_cat: np.ndarray,
    batch_size: int,
    max_batches: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Draw a calibration pool from the validation split.

    The rows are sampled without replacement under a fixed seed so two builds on
    the same data calibrate on the same rows. The pool is trimmed to a whole
    number of batches because TensorRT calibrates at one fixed batch size and a
    ragged final batch would be discarded anyway.

    Returns the numerical block, the cat block, and the number of whole batches.
    """
    n_rows = int(val_numerical.shape[0])
    if n_rows == 0:
        return (
            np.zeros((0, val_numerical.shape[1]), dtype=np.float32),
            np.zeros((0, val_cat.shape[1]), dtype=np.int64),
            0,
        )

    wanted = int(batch_size) * int(max_batches)
    rng = np.random.default_rng(seed)
    if wanted >= n_rows:
        index = np.arange(n_rows)
    else:
        index = np.sort(rng.choice(n_rows, size=wanted, replace=False))

    n_batches = int(len(index) // batch_size)
    if n_batches == 0:
        return (
            np.zeros((0, val_numerical.shape[1]), dtype=np.float32),
            np.zeros((0, val_cat.shape[1]), dtype=np.int64),
            0,
        )
    index = index[: n_batches * batch_size]

    numerical = np.ascontiguousarray(val_numerical[index], dtype=np.float32)
    cat = np.ascontiguousarray(val_cat[index], dtype=np.int64)
    return numerical, cat, n_batches


def make_entropy_calibrator(
    feeds: Dict[str, np.ndarray],
    batch_size: int,
    cache_path: str,
    input_dtypes: Optional[Dict[str, np.dtype]] = None,
    device_index: int = 0,
    verbose: bool = True,
) -> Any:
    """Build a trt.IInt8EntropyCalibrator2 over a fixed pool of host arrays.

    The class is defined inside this function because it subclasses a tensorrt
    type, and tensorrt cannot be imported on a machine with no NVIDIA driver. A
    module level class definition would break the import of the whole package on
    the development machine, so the definition is deferred to call time.

    Device memory comes from torch rather than from pycuda or cuda-python. The
    project already depends on torch, torch owns a caching allocator that is
    known good, and a torch cuda tensor exposes a raw device pointer through
    data_ptr, which is exactly what TensorRT asks for. The buffers are held on
    the instance so they outlive every get_batch call, because returning a
    pointer to freed memory is the classic way to corrupt a calibration run.

    Parameters
    ----------
    feeds : dict
        Maps a network input name to the full host array for that input. Every
        array must share the same first dimension.
    batch_size : int
        The fixed batch size TensorRT calibrates at.
    cache_path : str
        Where the calibration cache is read from and written to.
    input_dtypes : dict, optional
        Maps a network input name to the numpy dtype the engine expects. Used to
        cast the cat block down to int32 when the parser lowered it.
    device_index : int
        Which gpu to allocate the calibration buffers on.
    verbose : bool
        Print one line per calibration batch when true.

    Returns
    -------
    An instance ready to be assigned to builder_config.int8_calibrator.
    """
    import tensorrt as trt
    import torch

    if not hasattr(trt, "IInt8EntropyCalibrator2"):
        raise RuntimeError(
            "this TensorRT build does not expose IInt8EntropyCalibrator2. "
            "Implicit quantization through a calibrator was deprecated in "
            "TensorRT 10.1 in favour of explicit quantization, where the "
            "quantize and dequantize nodes are placed in the ONNX graph itself "
            "before the build. Pin a TensorRT 10.0 era build to use this path, "
            "or export a quantized graph instead."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "INT8 calibration needs a cuda device for the calibration buffers "
            "and torch reports none on this host"
        )

    names = list(feeds.keys())
    if not names:
        raise ValueError("the calibrator needs at least one input array")

    lengths = {int(np.asarray(v).shape[0]) for v in feeds.values()}
    if len(lengths) != 1:
        raise ValueError(
            "every calibration array must have the same number of rows, got "
            f"{sorted(lengths)}"
        )
    n_rows = lengths.pop()
    n_batches = int(n_rows // batch_size)

    dtypes = dict(input_dtypes or {})
    device = torch.device(f"cuda:{device_index}")

    torch_dtype_of = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Feed validation batches to TensorRT and cache the resulting scales."""

        def __init__(self) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.cache_path = cache_path
            self.batch_size = int(batch_size)
            self.n_batches = n_batches
            self.current = 0
            self.device_buffers: Dict[str, Any] = {}
            for name in names:
                host = np.asarray(feeds[name])
                want = np.dtype(dtypes.get(name, host.dtype))
                if want not in torch_dtype_of:
                    raise ValueError(
                        f"input {name} has unsupported calibration dtype {want}"
                    )
                width = int(np.prod(host.shape[1:])) if host.ndim > 1 else 1
                self.device_buffers[name] = torch.empty(
                    self.batch_size * width,
                    dtype=torch_dtype_of[want],
                    device=device,
                )

        def get_batch_size(self) -> int:
            """TensorRT calibrates at one fixed batch size and asks for it here."""
            return self.batch_size

        def get_batch(self, names_requested, p_str=None):  # noqa: ARG002 trt passes extra args
            """Return one device pointer per requested input, or None when done.

            Returning None is how a calibrator tells TensorRT the pool is
            exhausted. The pointers must stay valid until the next call, which
            is why the buffers live on the instance.
            """
            if self.current >= self.n_batches:
                return None

            start = self.current * self.batch_size
            end = start + self.batch_size
            pointers: List[int] = []
            for name in names_requested:
                key = name if name in self.device_buffers else None
                if key is None:
                    # A network input the calibration pool does not cover means
                    # the pool and the graph disagree, which is not recoverable.
                    raise KeyError(
                        f"TensorRT asked for calibration input {name} which is "
                        f"not in the calibration pool {sorted(self.device_buffers)}"
                    )
                host = np.asarray(feeds[key])[start:end]
                want = np.dtype(dtypes.get(key, host.dtype))
                if host.dtype != want:
                    host = host.astype(want)
                host = np.ascontiguousarray(host).reshape(-1)
                buffer = self.device_buffers[key]
                buffer.copy_(torch.from_numpy(host))
                pointers.append(int(buffer.data_ptr()))

            self.current += 1
            if verbose:
                print(
                    f"  calibration batch {self.current} of {self.n_batches} "
                    f"at batch size {self.batch_size}."
                )
            # A synchronize here guarantees the copies have landed before
            # TensorRT reads the pointers on its own stream.
            torch.cuda.synchronize(device)
            return pointers

        def read_calibration_cache(self):
            """Return the cached scales when the cache file exists.

            When this returns bytes, TensorRT skips calibration entirely and the
            build becomes reproducible from the committed cache.
            """
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as handle:
                    data = handle.read()
                if verbose:
                    print(
                        f"  reading the int8 calibration cache from "
                        f"{self.cache_path} ({len(data)} bytes), so calibration "
                        "is skipped and the build is reproducible."
                    )
                return data
            return None

        def write_calibration_cache(self, cache) -> None:
            """Write the chosen scales so a later build reproduces this engine."""
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as handle:
                handle.write(cache)
            if verbose:
                print(f"  wrote the int8 calibration cache to {self.cache_path}.")

    return _EntropyCalibrator()


def describe_calibration(cache_path: str, n_batches: int, batch_size: int) -> Dict[str, Any]:
    """Return a record describing a calibration run for the build report."""
    exists = os.path.exists(cache_path)
    return {
        "cache_path": cache_path,
        "cache_exists": exists,
        "cache_bytes": os.path.getsize(cache_path) if exists else None,
        "batches": int(n_batches),
        "batch_size": int(batch_size),
        "source_split": "validation",
        "note": (
            "Calibration rows come from the validation split and never from the "
            "test split, so the reported INT8 accuracy is measured on rows the "
            "quantizer has not seen."
        )
        if n_batches > 0
        else NOT_AVAILABLE,
    }
