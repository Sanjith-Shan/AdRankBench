"""Run a serialized TensorRT engine and profile where its time goes.

This module loads an engine built by trt_builder, allocates the device buffers
once, and exposes a run_batch callable with the same contract every other
backend in this project uses, which is numerical block in, cat block in, click
probabilities out. Keeping the contract identical is what lets the benchmark
compare a TensorRT engine against eager PyTorch without any special casing in
the timing loop.

Three implementation choices are worth stating.

The API used here is the TensorRT 10 tensor addressing API. The old binding
index API and execute_async_v2 are gone in TensorRT 10, so the runner works
through named IO tensors with set_input_shape, set_tensor_address, and
execute_async_v3 on an explicit cuda stream.

Device memory comes from torch rather than from pycuda or cuda-python. This
project already depends on torch, a torch cuda tensor hands out a raw device
pointer through data_ptr, and torch owns a caching allocator that is far better
tested than anything this file would grow. The buffers are allocated once for
the maximum batch and reused, so a per batch measurement is not measuring
cudaMalloc.

Timing is synchronized. Every enqueue on a cuda stream returns immediately, so a
timing loop that does not synchronize measures how fast python can fill a queue
rather than how fast the gpu can empty it. run_batch always synchronizes its
stream before it returns, which means the number the benchmark records is real
work completed and not queue depth.

The layer profiler is the evidence gathering side of the module. A DLRM style
ranker is a large embedding gather followed by a small multilayer perceptron,
and the interesting question is which of those two owns the wall clock. The
profiler reports milliseconds per layer so the report can answer that with
measurement rather than assertion.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.inference.common import NOT_AVAILABLE, sigmoid

# Layer name fragments used to bucket the profiler output into the two halves
# of a DLRM style network. TensorRT names a fused layer after the operations it
# absorbed, so matching on fragments is the practical way to attribute time.
_GATHER_HINTS = ("gather", "embedding", "onehot", "slice", "shuffle", "concat", "reshape")
_MATMUL_HINTS = ("matmul", "gemm", "fully", "conv", "myelin", "mlp", "linear", "add_bias")


class LayerProfiler:
    """Accumulate per layer milliseconds reported by TensorRT.

    This is a thin holder. The object TensorRT actually needs subclasses
    trt.IProfiler and cannot be defined at module scope, so it is created by
    make_layer_profiler below and writes into an instance of this class.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.iterations: int = 0

    def record(self, layer_name: str, ms: float) -> None:
        """Add one reported layer time to the running total."""
        self.totals[layer_name] = self.totals.get(layer_name, 0.0) + float(ms)

    def per_iteration(self) -> Dict[str, float]:
        """Return the mean milliseconds per layer per execution."""
        if self.iterations <= 0:
            return {}
        return {name: total / self.iterations for name, total in self.totals.items()}

    def total_ms(self) -> float:
        """Return the summed layer time per execution."""
        return float(sum(self.per_iteration().values()))

    def bucket(self) -> Dict[str, Any]:
        """Split the profile into gather time, matmul time, and everything else.

        The split is the whole point of profiling this network. If the gather
        bucket dominates then the model is memory bound and a lower precision
        multiply will not help much, and if the matmul bucket dominates then it
        is compute bound and fp16 should pay off. The report states whichever
        the measurement says.
        """
        per_iter = self.per_iteration()
        total = sum(per_iter.values())
        gather_ms = 0.0
        matmul_ms = 0.0
        other_ms = 0.0
        for name, ms in per_iter.items():
            lowered = name.lower()
            if any(hint in lowered for hint in _MATMUL_HINTS):
                matmul_ms += ms
            elif any(hint in lowered for hint in _GATHER_HINTS):
                gather_ms += ms
            else:
                other_ms += ms
        share = lambda part: (part / total * 100.0) if total > 0 else None  # noqa: E731
        return {
            "total_ms": total,
            "gather_ms": gather_ms,
            "matmul_ms": matmul_ms,
            "other_ms": other_ms,
            "gather_share_pct": share(gather_ms),
            "matmul_share_pct": share(matmul_ms),
            "other_share_pct": share(other_ms),
            "layers": per_iter,
            "iterations": self.iterations,
        }


def make_layer_profiler(sink: LayerProfiler):
    """Return a trt.IProfiler that writes into the given LayerProfiler.

    Defined at call time because it subclasses a tensorrt type and tensorrt
    cannot be imported on a machine with no NVIDIA driver.
    """
    import tensorrt as trt

    class _Profiler(trt.IProfiler):
        """Forward every reported layer time into the python side sink."""

        def __init__(self) -> None:
            trt.IProfiler.__init__(self)

        def report_layer_time(self, layer_name, ms) -> None:
            sink.record(str(layer_name), float(ms))

    return _Profiler()


def _torch_dtype_for(trt, dtype):
    """Map a TensorRT dtype to the matching torch dtype."""
    import torch

    name = getattr(dtype, "name", str(dtype))
    mapping = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "BF16": torch.bfloat16,
        "INT8": torch.int8,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "BOOL": torch.bool,
        "UINT8": torch.uint8,
    }
    if name not in mapping:
        raise ValueError(f"unsupported TensorRT tensor dtype {name}")
    return mapping[name]


def _numpy_dtype_for(torch_dtype) -> np.dtype:
    """Map a torch dtype to the numpy dtype the host arrays must be cast to."""
    import torch

    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
        torch.uint8: np.uint8,
    }
    return np.dtype(mapping[torch_dtype])


class TensorRTRunner:
    """Load a serialized engine and score batches through it.

    Parameters
    ----------
    engine_path : str
        A serialized plan written by trt_builder.
    max_batch : int
        The largest batch this runner will be asked for. Buffers are sized for
        it once so no measured call ever allocates. It must not exceed the
        maximum in the engine's optimization profile.
    device_index : int
        Which gpu to run on.
    """

    def __init__(self, engine_path: str, max_batch: int, device_index: int = 0) -> None:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "a TensorRT engine needs a cuda device and torch reports none "
                "on this host"
            )
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"no TensorRT engine at {engine_path}")

        self._trt = trt
        self._torch = torch
        self.engine_path = engine_path
        self.max_batch = int(max_batch)
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT could not deserialize {engine_path}. An engine is "
                "tied to the gpu architecture and the TensorRT version that "
                "built it, so rebuild it on this machine."
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"could not create an execution context for {engine_path}")

        self.stream = torch.cuda.Stream(device=self.device)

        # Walk the IO tensors once and record names, modes, dtypes, and the per
        # row width of each tensor. Everything in this graph is batched on axis
        # zero, so the per row width is the product of the trailing dimensions.
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.torch_dtypes: Dict[str, Any] = {}
        self.numpy_dtypes: Dict[str, np.dtype] = {}
        self.row_widths: Dict[str, int] = {}
        self.static_shapes: Dict[str, Tuple[int, ...]] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _torch_dtype_for(trt, dtype)
            self.torch_dtypes[name] = torch_dtype
            self.numpy_dtypes[name] = _numpy_dtype_for(torch_dtype)
            shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            self.static_shapes[name] = shape
            trailing = shape[1:] if len(shape) > 1 else ()
            width = 1
            for d in trailing:
                width *= max(int(d), 1)
            self.row_widths[name] = int(width)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names or not self.output_names:
            raise RuntimeError(
                f"{engine_path} has {len(self.input_names)} inputs and "
                f"{len(self.output_names)} outputs, which this runner cannot drive"
            )

        # Decide which host array feeds which engine input. Name first, because
        # the export names them numerical and cat, and dtype as the fallback so
        # a renamed export still works. A floating input is the dense block and
        # an integer input is the field index block.
        self.input_source: Dict[str, str] = {}
        for name in self.input_names:
            lowered = name.lower()
            if "num" in lowered or "dense" in lowered:
                self.input_source[name] = "numerical"
            elif "cat" in lowered or "sparse" in lowered or "idx" in lowered:
                self.input_source[name] = "cat"
            elif self.numpy_dtypes[name].kind == "f":
                self.input_source[name] = "numerical"
            else:
                self.input_source[name] = "cat"

        # Resolve the output shapes at the maximum batch so the buffers are
        # allocated once for the largest case this runner will ever see.
        for name in self.input_names:
            trailing = self.static_shapes[name][1:]
            self.context.set_input_shape(name, (self.max_batch,) + tuple(int(d) for d in trailing))
        # all_shape_inputs_specified is deprecated on some TensorRT 10 builds, so
        # it is consulted only when it is there. A graph with a shape tensor
        # input would need a value rather than a dimension and this runner does
        # not supply one, which is worth catching early where it can be.
        specified = getattr(self.context, "all_shape_inputs_specified", True)
        if not specified:
            raise RuntimeError(
                f"{engine_path} has shape tensor inputs that this runner does "
                "not supply, so it cannot be driven from here"
            )

        self.output_row_widths: Dict[str, int] = {}
        for name in self.output_names:
            resolved = tuple(int(d) for d in self.context.get_tensor_shape(name))
            width = 1
            for d in resolved[1:]:
                width *= max(int(d), 1)
            self.output_row_widths[name] = int(width)

        # Allocate the device buffers and the pinned host staging buffers. Pinned
        # host memory is what makes the host to device copy a real asynchronous
        # dma rather than a staged synchronous copy.
        # A note on the torch caching allocator. These buffers are allocated on
        # the default stream and then used on this runner's side stream. That is
        # only hazardous when a tensor is freed while side stream work on it is
        # still pending, and it cannot happen here, because every run
        # synchronizes the stream before it returns and the buffers are held for
        # the life of the runner. Allocating once for the maximum batch also
        # means no measured call ever calls cudaMalloc.
        self.device_in: Dict[str, Any] = {}
        self.host_in: Dict[str, np.ndarray] = {}
        self._host_in_tensors: Dict[str, Any] = {}
        for name in self.input_names:
            elems = self.max_batch * self.row_widths[name]
            self.device_in[name] = torch.empty(
                elems, dtype=self.torch_dtypes[name], device=self.device
            )
            pinned = torch.empty(elems, dtype=self.torch_dtypes[name], pin_memory=True)
            self._host_in_tensors[name] = pinned
            self.host_in[name] = pinned.numpy()

        self.device_out: Dict[str, Any] = {}
        self.host_out: Dict[str, np.ndarray] = {}
        self._host_out_tensors: Dict[str, Any] = {}
        for name in self.output_names:
            elems = self.max_batch * self.output_row_widths[name]
            self.device_out[name] = torch.empty(
                elems, dtype=self.torch_dtypes[name], device=self.device
            )
            pinned = torch.empty(elems, dtype=self.torch_dtypes[name], pin_memory=True)
            self._host_out_tensors[name] = pinned
            self.host_out[name] = pinned.numpy()

        # The first output is the logit tensor the benchmark reads.
        self.primary_output = self.output_names[0]

        self._profiler_sink: Optional[LayerProfiler] = None
        self._profiler_obj = None

    @property
    def device_memory_bytes(self) -> Optional[int]:
        """Scratch device memory the engine reserves, when TensorRT reports it."""
        for attr in ("device_memory_size_v2", "device_memory_size"):
            value = getattr(self.engine, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _bind(self, numerical: np.ndarray, cat: np.ndarray) -> Tuple[int, Tuple[int, ...]]:
        """Stage the inputs, set the shapes and addresses, and return the batch.

        Returns the batch size and the resolved shape of the primary output.
        """
        sources = {"numerical": numerical, "cat": cat}
        batch = int(np.asarray(numerical).shape[0])
        if batch > self.max_batch:
            raise ValueError(
                f"batch of {batch} exceeds the {self.max_batch} this runner "
                "allocated buffers for. Rebuild the engine with a larger max batch."
            )

        for name in self.input_names:
            array = np.asarray(sources[self.input_source[name]])
            want = self.numpy_dtypes[name]
            if array.dtype != want:
                array = array.astype(want, copy=False)
            array = np.ascontiguousarray(array)
            flat = array.reshape(-1)
            count = int(flat.size)
            # Write straight into the pinned staging buffer, then dma it.
            self.host_in[name][:count] = flat
            self.device_in[name][:count].copy_(
                self._host_in_tensors[name][:count], non_blocking=True
            )
            shape = tuple(int(d) for d in array.shape)
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(
                    f"TensorRT rejected the input shape {shape} for {name}. The "
                    "engine optimization profile does not cover this batch size."
                )
            self.context.set_tensor_address(name, int(self.device_in[name].data_ptr()))

        out_shape = tuple(int(d) for d in self.context.get_tensor_shape(self.primary_output))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.device_out[name].data_ptr()))
        return batch, out_shape

    def _execute(self) -> None:
        """Enqueue the engine on this runner's stream."""
        ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned false, the engine did not run")

    def run_logits(self, numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Score one batch and return the raw logits as a host array.

        The stream is synchronized before this returns, so the wall clock a
        caller measures around this call is completed gpu work and not queue
        depth. That synchronization is the difference between a real TensorRT
        latency number and a meaningless one.
        """
        torch = self._torch
        with torch.cuda.stream(self.stream):
            batch, out_shape = self._bind(numerical, cat)
            self._execute()
            count = 1
            for d in out_shape:
                count *= max(int(d), 1)
            self._host_out_tensors[self.primary_output][:count].copy_(
                self.device_out[self.primary_output][:count], non_blocking=True
            )
        self.stream.synchronize()
        flat = np.array(self.host_out[self.primary_output][:count], copy=True)
        return flat.astype(np.float32).reshape(-1)[: int(batch)]

    def run_batch(self, numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Score one batch and return click probabilities in [0, 1]."""
        return sigmoid(self.run_logits(numerical, cat))

    def as_callable(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return the run_batch callable in the shape the benchmark expects."""
        return self.run_batch

    def profile_layers(
        self,
        numerical: np.ndarray,
        cat: np.ndarray,
        iterations: int = 20,
        warmup: int = 5,
    ) -> Dict[str, Any]:
        """Run with a layer profiler attached and return the per layer times.

        A profiler forces TensorRT to time every layer individually, which adds
        overhead and can suppress some kernel fusion, so the absolute
        milliseconds here are not the latency number the benchmark reports. What
        the profile is for is the shape of the distribution, which is where the
        time goes rather than how much of it there is.
        """
        sink = LayerProfiler()
        self._profiler_sink = sink
        self._profiler_obj = make_layer_profiler(sink)
        previous = getattr(self.context, "profiler", None)
        self.context.profiler = self._profiler_obj
        try:
            for _ in range(max(0, warmup)):
                self.run_logits(numerical, cat)
            # Warmup executions also report layer times, so the sink is cleared
            # before the counted iterations begin.
            sink.totals.clear()
            sink.iterations = 0
            for _ in range(max(1, iterations)):
                self.run_logits(numerical, cat)
                sink.iterations += 1
        finally:
            try:
                self.context.profiler = previous
            except Exception:  # noqa: BLE001 restoring is best effort
                pass
            self._profiler_obj = None
        result = sink.bucket()
        result["engine_path"] = self.engine_path
        result["batch_size"] = int(np.asarray(numerical).shape[0])
        return result

    def close(self) -> None:
        """Release the context, the engine, and the device buffers."""
        self.context = None
        self.engine = None
        self.runtime = None
        self.device_in = {}
        self.device_out = {}
        self._host_in_tensors = {}
        self._host_out_tensors = {}
        self.host_in = {}
        self.host_out = {}
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # pragma: no cover - destructor is best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def load_runner(
    engine_path: str, max_batch: int, device_index: int = 0
) -> Tuple[Optional[TensorRTRunner], str]:
    """Load a TensorRT runner, or return None and the reason it could not load.

    This is the shape the backend registry wants. It never raises, so an absent
    engine or an absent gpu becomes a not available row in the report rather
    than a stack trace that ends the sweep.
    """
    try:
        import tensorrt  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return None, f"tensorrt is not importable on this host ({exc})"
    try:
        runner = TensorRTRunner(engine_path, max_batch, device_index)
    except Exception as exc:  # noqa: BLE001
        return None, f"the TensorRT engine could not be loaded ({exc})"
    return runner, ""


def empty_profile(reason: str) -> Dict[str, Any]:
    """Return a layer profile record for a run that could not be profiled."""
    return {
        "total_ms": None,
        "gather_ms": None,
        "matmul_ms": None,
        "other_ms": None,
        "gather_share_pct": None,
        "matmul_share_pct": None,
        "other_share_pct": None,
        "layers": {},
        "iterations": 0,
        "reason": reason or NOT_AVAILABLE,
    }
