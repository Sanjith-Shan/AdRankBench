"""The backend registry for the inference benchmark.

One trained model can be served a lot of different ways. It can run in eager
PyTorch on a cpu or on a gpu, it can run through ONNX Runtime on the cpu
provider or the cuda provider or the TensorRT provider, it can run through a
natively built TensorRT engine at three precisions, and it can run through
OpenVINO on the cpu. Each of those is a real deployment choice with a real cost,
and the only way to compare them is to run all of them on the same weights and
the same rows.

This module is the registry that constructs each one. Every constructor follows
the same contract. It returns a BackendResult that either carries a run_batch
callable or carries None and a sentence saying exactly why the backend could not
be built. Nothing here raises for a missing package or a missing gpu, because a
sweep that dies on the first unavailable runtime is useless on the machine this
project is developed on, which is an Apple Silicon laptop with no cuda at all.

The run_batch contract is the same for every backend. It takes a float32
numerical block and an int64 cat block, both with the batch on axis zero, and it
returns a one dimensional float array of click probabilities. Every gpu backend
synchronizes before it returns, so the wall clock a caller measures around the
call is completed work rather than queue depth.

One thing this registry deliberately does not do is blend lanes. A cpu backend
and a gpu backend are not comparable to each other, they are comparable within
their lane, and every result carries the lane it belongs to so the report can
keep them apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.inference.common import NOT_AVAILABLE, sigmoid
from src.inference.hardware import cuda_lane_ready

RunBatch = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class BackendSpec:
    """One point in the runtime by precision by device grid."""

    key: str
    label: str
    short_label: str
    runtime: str
    precision: str
    device: str
    lane: str

    @property
    def is_gpu(self) -> bool:
        return self.lane == "gpu"


@dataclass
class BackendResult:
    """The outcome of trying to construct one backend."""

    spec: BackendSpec
    runner: Optional[RunBatch] = None
    note: str = ""
    sync: Optional[Callable[[], None]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """True when the backend actually has something to run."""
        return self.runner is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return a json friendly record of the availability decision."""
        return {
            "key": self.spec.key,
            "label": self.spec.label,
            "runtime": self.spec.runtime,
            "precision": self.spec.precision,
            "device": self.spec.device,
            "lane": self.spec.lane,
            "available": self.available,
            "note": self.note or "",
            # Only values that survive a json round trip go into the record.
            # The live runner object lives in extra as well and it belongs in
            # memory, not in an artifact.
            "extra": {
                k: v
                for k, v in self.extra.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            },
        }


@dataclass
class BackendContext:
    """Everything the constructors need to build a backend for one model."""

    model_name: str
    module: Any
    onnx_path: str
    n_numerical: int
    n_embed_fields: int
    max_batch: int = 4096
    engine_paths: Dict[str, str] = field(default_factory=dict)
    calibration_cache: str = ""
    trt_engine_cache_dir: str = os.path.join("results", "trt", "ort_trt_cache")
    device_index: int = 0
    workspace_mb: int = 4096


# The full grid the benchmark sweeps. Order is the order a report shows.
def default_specs(precisions: Optional[List[str]] = None) -> List[BackendSpec]:
    """Return the backend grid, optionally filtered to a set of precisions.

    The cpu lane is always included whatever the precision filter says, because
    it is the reference every accuracy delta is measured against and dropping it
    would leave the report with nothing to compare to.
    """
    grid = [
        BackendSpec(
            key="pytorch-cpu-fp32",
            label="PyTorch eager (CPU, fp32)",
            short_label="PyTorch",
            runtime="pytorch",
            precision="fp32",
            device="cpu",
            lane="cpu",
        ),
        BackendSpec(
            key="onnxruntime-cpu-fp32",
            label="ONNX Runtime (CPU provider, fp32)",
            short_label="ONNX Runtime",
            runtime="onnxruntime",
            precision="fp32",
            device="cpu",
            lane="cpu",
        ),
        BackendSpec(
            key="openvino-cpu-fp32",
            label="OpenVINO (CPU, fp32)",
            short_label="OpenVINO",
            runtime="openvino",
            precision="fp32",
            device="cpu",
            lane="cpu",
        ),
        BackendSpec(
            key="pytorch-cuda-fp32",
            label="PyTorch eager (CUDA, fp32)",
            short_label="PyTorch CUDA fp32",
            runtime="pytorch",
            precision="fp32",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="pytorch-cuda-fp16",
            label="PyTorch eager (CUDA, fp16 autocast)",
            short_label="PyTorch CUDA fp16",
            runtime="pytorch",
            precision="fp16",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="onnxruntime-cuda-fp32",
            label="ONNX Runtime (CUDA provider, fp32)",
            short_label="ORT CUDA",
            runtime="onnxruntime",
            precision="fp32",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="onnxruntime-trt-fp16",
            label="ONNX Runtime (TensorRT provider, fp16)",
            short_label="ORT TRT fp16",
            runtime="onnxruntime-trt",
            precision="fp16",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="onnxruntime-trt-int8",
            label="ONNX Runtime (TensorRT provider, int8)",
            short_label="ORT TRT int8",
            runtime="onnxruntime-trt",
            precision="int8",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="tensorrt-fp32",
            label="TensorRT native engine (fp32)",
            short_label="TensorRT fp32",
            runtime="tensorrt",
            precision="fp32",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="tensorrt-fp16",
            label="TensorRT native engine (fp16)",
            short_label="TensorRT fp16",
            runtime="tensorrt",
            precision="fp16",
            device="cuda",
            lane="gpu",
        ),
        BackendSpec(
            key="tensorrt-int8",
            label="TensorRT native engine (int8)",
            short_label="TensorRT int8",
            runtime="tensorrt",
            precision="int8",
            device="cuda",
            lane="gpu",
        ),
    ]
    if precisions is None:
        return grid
    wanted = {p.lower() for p in precisions}
    return [s for s in grid if s.precision in wanted or s.lane == "cpu"]


def spec_by_key(key: str) -> Optional[BackendSpec]:
    """Look one spec up by its key."""
    for spec in default_specs():
        if spec.key == key:
            return spec
    return None


# ---------------------------------------------------------------------------
# CPU lane. These three are the backends the project has always had, kept with
# exactly the behaviour and the honesty they had before.
# ---------------------------------------------------------------------------


def _torch_cpu(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the eager PyTorch cpu backend, which is the reference for accuracy."""
    import copy

    import torch

    # A deep copy rather than ctx.module.to("cpu"). nn.Module.to moves a module
    # in place and every backend in this registry is constructed up front from
    # the same ctx.module, so moving it here would also move it for the cuda
    # backends and whichever backend was built last would decide the device for
    # all of them. That produced a real failure where cuda inputs met cpu
    # weights. Each torch backend owning its own copy removes the interaction.
    module = copy.deepcopy(ctx.module).to("cpu").eval()

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = module(
                torch.from_numpy(np.ascontiguousarray(numerical, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(cat, dtype=np.int64)),
            )
        return sigmoid(logits.numpy())

    return BackendResult(spec=spec, runner=run_batch, note="")


def _onnxruntime_cpu(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the ONNX Runtime cpu provider backend."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return BackendResult(
            spec=spec,
            note=f"onnxruntime is not installed ({exc}), so this backend was skipped",
        )
    if not os.path.exists(ctx.onnx_path):
        return BackendResult(
            spec=spec, note=f"the onnx graph {ctx.onnx_path} does not exist"
        )

    # Full graph optimization so ONNX Runtime fuses and folds the way it would
    # in a real deployment. Threading stays at the runtime default so it sees
    # the same cores PyTorch and OpenVINO see, which keeps the lane fair.
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            ctx.onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        return BackendResult(spec=spec, note=f"the ONNX Runtime session failed to build ({exc})")

    input_names = [inp.name for inp in session.get_inputs()]

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        feed = {name: (numerical if name == "numerical" else cat) for name in input_names}
        logits = session.run(None, feed)[0]
        return sigmoid(np.asarray(logits).reshape(-1))

    return BackendResult(
        spec=spec,
        runner=run_batch,
        note="",
        extra={"providers": list(session.get_providers())},
    )


def _openvino_cpu(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the OpenVINO cpu backend from the same exported ONNX graph."""
    try:
        import openvino as ov
    except ImportError as exc:
        return BackendResult(
            spec=spec,
            note=f"openvino is not installed ({exc}), so this backend was skipped",
        )
    if not os.path.exists(ctx.onnx_path):
        return BackendResult(
            spec=spec, note=f"the onnx graph {ctx.onnx_path} does not exist"
        )
    try:
        core = ov.Core()
        model = core.read_model(ctx.onnx_path)
        compiled = core.compile_model(model, "CPU")
    except Exception as exc:  # noqa: BLE001
        return BackendResult(spec=spec, note=f"the OpenVINO compile failed ({exc})")

    output_port = compiled.output(0)

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        feed = {}
        for port in compiled.inputs:
            feed[port] = numerical if port.get_any_name() == "numerical" else cat
        result = compiled(feed)
        return sigmoid(np.asarray(result[output_port]).reshape(-1))

    return BackendResult(spec=spec, runner=run_batch, note="", extra={"device": "CPU"})


# ---------------------------------------------------------------------------
# GPU lane.
# ---------------------------------------------------------------------------


def _torch_cuda(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the eager PyTorch cuda backend at fp32 or fp16.

    fp16 goes through autocast rather than a blanket half cast on the module.
    Autocast keeps the reductions and the normalization layers in fp32 and casts
    only the operations that are numerically safe in half, which is what a real
    deployment would do and what keeps the accuracy delta attributable to
    precision rather than to an overflow.
    """
    ready, reason = cuda_lane_ready()
    if not ready:
        return BackendResult(spec=spec, note=reason)

    import copy

    import torch

    # Deep copied for the same reason the cpu backend copies. See the note
    # there. Moving the shared module in place here would drag every other
    # torch backend onto this device too.
    device = torch.device(f"cuda:{ctx.device_index}")
    module = copy.deepcopy(ctx.module).to(device).eval()
    use_fp16 = spec.precision == "fp16"

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        num_t = torch.from_numpy(
            np.ascontiguousarray(numerical, dtype=np.float32)
        ).to(device, non_blocking=False)
        cat_t = torch.from_numpy(np.ascontiguousarray(cat, dtype=np.int64)).to(
            device, non_blocking=False
        )
        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = module(num_t, cat_t)
            else:
                logits = module(num_t, cat_t)
            probs = torch.sigmoid(logits.float())
        # The copy back to the host is what forces the queue to drain, and the
        # explicit synchronize makes that guarantee independent of any future
        # change to how torch implements the copy.
        out = probs.detach().cpu().numpy().reshape(-1)
        torch.cuda.synchronize(device)
        return out.astype(np.float64)

    def sync() -> None:
        torch.cuda.synchronize(device)

    return BackendResult(
        spec=spec,
        runner=run_batch,
        note="",
        sync=sync,
        extra={"device_name": torch.cuda.get_device_name(ctx.device_index)},
    )


def _onnxruntime_cuda(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the ONNX Runtime cuda execution provider backend."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return BackendResult(spec=spec, note=f"onnxruntime is not installed ({exc})")

    available = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        return BackendResult(
            spec=spec,
            note=(
                "this ONNX Runtime build has no CUDAExecutionProvider, it offers "
                f"{available}. Install onnxruntime-gpu on a cuda host."
            ),
        )
    if not os.path.exists(ctx.onnx_path):
        return BackendResult(spec=spec, note=f"the onnx graph {ctx.onnx_path} does not exist")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            ctx.onnx_path,
            sess_options=sess_options,
            providers=[
                ("CUDAExecutionProvider", {"device_id": ctx.device_index}),
                "CPUExecutionProvider",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        return BackendResult(spec=spec, note=f"the cuda ONNX Runtime session failed ({exc})")

    input_names = [inp.name for inp in session.get_inputs()]

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        feed = {name: (numerical if name == "numerical" else cat) for name in input_names}
        logits = session.run(None, feed)[0]
        return sigmoid(np.asarray(logits).reshape(-1))

    return BackendResult(
        spec=spec,
        runner=run_batch,
        note="",
        extra={"providers": list(session.get_providers())},
    )


def _shape_profile_string(ctx: BackendContext, batch: int) -> str:
    """Format a TensorRT execution provider shape profile for one batch size."""
    return f"numerical:{batch}x{ctx.n_numerical},cat:{batch}x{ctx.n_embed_fields}"


def _onnxruntime_trt(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the ONNX Runtime TensorRT execution provider backend.

    The TensorRT execution provider compiles subgraphs into TensorRT engines at
    session build time, which is why the engine cache directory matters. Without
    it every process start pays the full build cost again.

    The int8 path points at the same native calibration cache the standalone
    builder wrote, through trt_int8_use_native_calibration_table. That keeps one
    set of quantization scales behind both the native engine and the execution
    provider engine, so a difference between those two rows is a difference in
    the runtime and not a difference in the calibration.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return BackendResult(spec=spec, note=f"onnxruntime is not installed ({exc})")

    available = list(ort.get_available_providers())
    if "TensorrtExecutionProvider" not in available:
        return BackendResult(
            spec=spec,
            note=(
                "this ONNX Runtime build has no TensorrtExecutionProvider, it "
                f"offers {available}. Install onnxruntime-gpu built against "
                "TensorRT on a cuda host."
            ),
        )
    if not os.path.exists(ctx.onnx_path):
        return BackendResult(spec=spec, note=f"the onnx graph {ctx.onnx_path} does not exist")

    os.makedirs(ctx.trt_engine_cache_dir, exist_ok=True)
    options: Dict[str, Any] = {
        "device_id": ctx.device_index,
        "trt_max_workspace_size": int(ctx.workspace_mb) * 1024 * 1024,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": ctx.trt_engine_cache_dir,
        "trt_timing_cache_enable": True,
        "trt_profile_min_shapes": _shape_profile_string(ctx, 1),
        "trt_profile_opt_shapes": _shape_profile_string(ctx, min(256, ctx.max_batch)),
        "trt_profile_max_shapes": _shape_profile_string(ctx, ctx.max_batch),
    }
    if spec.precision in ("fp16", "int8"):
        options["trt_fp16_enable"] = True
    if spec.precision == "int8":
        options["trt_int8_enable"] = True
        if ctx.calibration_cache and os.path.exists(ctx.calibration_cache):
            options["trt_int8_use_native_calibration_table"] = True
            # The full path rather than the base name, because the execution
            # provider resolves the table relative to the working directory and
            # the cache lives beside the engines rather than beside the graph.
            options["trt_int8_calibration_table_name"] = os.path.abspath(
                ctx.calibration_cache
            )
        else:
            return BackendResult(
                spec=spec,
                note=(
                    "the int8 TensorRT provider needs a calibration table and "
                    f"none exists at {ctx.calibration_cache or NOT_AVAILABLE}. Run "
                    "scripts/build_trt_engines.py first so the scales are written."
                ),
            )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            ctx.onnx_path,
            sess_options=sess_options,
            providers=[
                ("TensorrtExecutionProvider", options),
                ("CUDAExecutionProvider", {"device_id": ctx.device_index}),
                "CPUExecutionProvider",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        return BackendResult(
            spec=spec, note=f"the TensorRT provider session failed to build ({exc})"
        )

    input_names = [inp.name for inp in session.get_inputs()]

    def run_batch(numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        feed = {name: (numerical if name == "numerical" else cat) for name in input_names}
        logits = session.run(None, feed)[0]
        return sigmoid(np.asarray(logits).reshape(-1))

    return BackendResult(
        spec=spec,
        runner=run_batch,
        note="",
        extra={
            "providers": list(session.get_providers()),
            "engine_cache_dir": ctx.trt_engine_cache_dir,
            "options": {k: str(v) for k, v in options.items()},
        },
    )


def _tensorrt_native(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Build the native TensorRT engine backend from a serialized plan."""
    ready, reason = cuda_lane_ready()
    if not ready:
        return BackendResult(spec=spec, note=reason)

    from src.inference.trt_builder import tensorrt_available
    from src.inference.trt_runner import load_runner

    ok, trt_reason = tensorrt_available()
    if not ok:
        return BackendResult(spec=spec, note=trt_reason)

    engine_path = ctx.engine_paths.get(spec.precision, "")
    if not engine_path or not os.path.exists(engine_path):
        return BackendResult(
            spec=spec,
            note=(
                f"no serialized {spec.precision} engine at "
                f"{engine_path or NOT_AVAILABLE}. Run scripts/build_trt_engines.py "
                "on this machine to build it, because an engine is tied to the "
                "gpu architecture and the TensorRT version that built it."
            ),
        )

    runner, load_reason = load_runner(engine_path, ctx.max_batch, ctx.device_index)
    if runner is None:
        return BackendResult(spec=spec, note=load_reason)

    def sync() -> None:
        runner.stream.synchronize()

    return BackendResult(
        spec=spec,
        runner=runner.run_batch,
        note="",
        sync=sync,
        extra={
            "engine_path": engine_path,
            "engine_bytes": os.path.getsize(engine_path),
            "device_memory_bytes": runner.device_memory_bytes,
            "trt_runner": runner,
        },
    )


_CONSTRUCTORS = {
    ("pytorch", "cpu"): _torch_cpu,
    ("pytorch", "cuda"): _torch_cuda,
    ("onnxruntime", "cpu"): _onnxruntime_cpu,
    ("onnxruntime", "cuda"): _onnxruntime_cuda,
    ("onnxruntime-trt", "cuda"): _onnxruntime_trt,
    ("tensorrt", "cuda"): _tensorrt_native,
    ("openvino", "cpu"): _openvino_cpu,
}


def build_backend(spec: BackendSpec, ctx: BackendContext) -> BackendResult:
    """Construct one backend and return its result, never raising.

    A constructor that throws an unexpected exception is caught here and turned
    into an unavailable result with the exception text as the reason, because a
    sweep across eleven backends must not die because one of them is unhappy.
    """
    constructor = _CONSTRUCTORS.get((spec.runtime, spec.device))
    if constructor is None:
        return BackendResult(
            spec=spec,
            note=f"no constructor is registered for {spec.runtime} on {spec.device}",
        )
    try:
        return constructor(spec, ctx)
    except Exception as exc:  # noqa: BLE001 an unavailable backend is a row, not a crash
        return BackendResult(
            spec=spec, note=f"constructing this backend raised {type(exc).__name__} ({exc})"
        )


def probe_backends(
    ctx: BackendContext,
    specs: Optional[List[BackendSpec]] = None,
    verbose: bool = True,
) -> List[BackendResult]:
    """Construct every requested backend and return the availability records.

    This is the function the benchmark and the tests both call. It always
    returns one record per spec, available or not, so a report can print a
    complete grid rather than a grid with holes in it.
    """
    specs = specs if specs is not None else default_specs()
    results: List[BackendResult] = []
    for spec in specs:
        result = build_backend(spec, ctx)
        results.append(result)
        if verbose:
            if result.available:
                print(f"  {spec.label} is available.")
            else:
                print(f"  {spec.label} is {NOT_AVAILABLE}. {result.note}")
    return results


# ---------------------------------------------------------------------------
# Backwards compatible constructors. The first version of the inference
# benchmark exposed these three functions and the existing tests call them, so
# they are kept with the same names, the same signatures, and the same
# behaviour, which is to print a short note and return None when a package is
# missing rather than raising.
# ---------------------------------------------------------------------------


def _bare_context(onnx_path: str, module: Any = None) -> BackendContext:
    """Build a minimal context for the single backend helpers."""
    return BackendContext(
        model_name="model",
        module=module,
        onnx_path=onnx_path,
        n_numerical=0,
        n_embed_fields=0,
    )


def make_torch_runner(module: Any) -> RunBatch:
    """Build a run function for the raw PyTorch eager cpu backend."""
    spec = spec_by_key("pytorch-cpu-fp32")
    result = _torch_cpu(spec, _bare_context("", module))
    return result.runner


def make_onnx_runner(onnx_path: str) -> Optional[RunBatch]:
    """Build a run function for ONNX Runtime on the cpu, or None if unavailable.

    Prints a short note and returns None when onnxruntime cannot be imported, so
    the caller can skip the backend gracefully.
    """
    spec = spec_by_key("onnxruntime-cpu-fp32")
    result = _onnxruntime_cpu(spec, _bare_context(onnx_path))
    if not result.available:
        print(f"ONNX Runtime backend skipped. {result.note}")
        return None
    print(f"ONNX Runtime ready with providers {result.extra.get('providers')}.")
    return result.runner


def make_openvino_runner(onnx_path: str) -> Optional[RunBatch]:
    """Build a run function for OpenVINO, or None if it is not installed.

    OpenVINO reads the ONNX graph directly, compiles it for the host cpu, and
    runs the optimized network. Prints a short note and returns None when
    openvino cannot be imported, so the caller can skip the backend gracefully.
    """
    spec = spec_by_key("openvino-cpu-fp32")
    result = _openvino_cpu(spec, _bare_context(onnx_path))
    if not result.available:
        print(f"OpenVINO backend skipped. {result.note}")
        return None
    print("OpenVINO ready on the CPU device.")
    return result.runner
