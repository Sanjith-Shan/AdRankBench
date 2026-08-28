"""Backend selection and the scoring engine that both lanes run on.

The service does not implement model loading or runtime selection. It asks the
registry in src.inference.backends for each backend in a documented preference
order and takes the first one that actually constructed. That registry is the
same one scripts/run_inference_benchmark.py sweeps, so the runtime that serves a
request is literally the runtime whose latency the benchmark table reports, and
adding a backend there adds it here for free.

The preference order is fastest first, on the measured evidence in
docs/INFERENCE.md rather than on reputation.

1. A natively built TensorRT engine. TensorRT compiles kernels against the exact
   gpu and picks each layer's implementation by benchmarking candidates on the
   real device, so where an engine exists it is the ceiling.
2. The ONNX Runtime cuda execution provider. A gpu path that needs no prebuilt
   engine, so it is the fallback when the engine for this machine was never
   built or was built by a different TensorRT version.
3. OpenVINO on the cpu. It sits above ONNX Runtime cpu because on the Apple M3
   Pro this project is developed on it measured about 1.47 times faster per
   batch than eager PyTorch with a far tighter tail, while the ONNX Runtime cpu
   provider did not beat eager PyTorch at all. On an Intel x86 serving fleet
   that ordering may invert, which is exactly why the selection is a probe at
   startup rather than a constant in a config file.
4. The ONNX Runtime cpu provider. The portable cpu path.
5. Eager PyTorch. The reference implementation and the last resort. It is the
   one backend that cannot be unavailable, because it needs no export and no
   extra package.

Two rules govern the selection and both exist so an operator is never wrong
about what is running.

The first is that the selection is logged with the reason every skipped backend
was skipped, and the whole probe table is served on /health. A service that
quietly fell back to eager PyTorch because an export was missing looks identical
from the outside to one running a compiled engine, right up until the latency
graph is read.

The second is that reduced precision is opt in. fp16 and int8 change the model's
output, and docs/INFERENCE.md argues that int8 in particular may cost real AUC
on a DLRM shaped model whose accuracy lives in the embedding tables. Silently
selecting a faster backend that returns different probabilities than the ones
the offline evaluation measured would be the accuracy equivalent of training and
serving skew. So the default order is fp32 only, and a reduced precision lane is
selectable by flag with the tradeoff stated at startup.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.inference.backends import (
    BackendContext,
    BackendResult,
    build_backend,
    spec_by_key,
)
from src.inference.common import NOT_AVAILABLE
from src.serving.artifact import ServingBundle
from src.serving.features import featurize_rows

# The fp32 preference order, fastest first. Every key is a key from
# src.inference.backends.default_specs.
PREFERENCE_ORDER: Tuple[str, ...] = (
    "tensorrt-fp32",
    "onnxruntime-cuda-fp32",
    "openvino-cpu-fp32",
    "onnxruntime-cpu-fp32",
    "pytorch-cpu-fp32",
)

# The opt in reduced precision order. Tried ahead of the fp32 order only when
# the operator asked for it, because these backends can change the numbers.
REDUCED_PRECISION_ORDER: Tuple[str, ...] = (
    "tensorrt-fp16",
    "tensorrt-int8",
    "onnxruntime-trt-fp16",
    "pytorch-cuda-fp16",
)

# The batch a backend is warmed at when the service starts. The first call into
# any runtime pays for lazy kernel selection, allocator growth, and first touch
# page faults, and docs/INFERENCE.md is explicit that including those in a
# measurement measures startup rather than serving. Paying them before the
# service reports ready keeps them out of every request the load test sees.
WARMUP_BATCHES: Tuple[int, ...] = (1, 8, 64, 256)


@dataclass
class BackendSelection:
    """The outcome of the startup probe."""

    result: BackendResult
    probed: List[Dict[str, Any]] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    allow_reduced_precision: bool = False

    @property
    def key(self) -> str:
        return self.result.spec.key

    @property
    def label(self) -> str:
        return self.result.spec.label

    def skipped(self) -> List[Dict[str, Any]]:
        """Return the records of the backends that were tried and did not build."""
        return [r for r in self.probed if not r.get("available")]

    def as_dict(self) -> Dict[str, Any]:
        """Return the json friendly selection record the health endpoint serves."""
        spec = self.result.spec
        return {
            "selected": spec.key,
            "label": spec.label,
            "runtime": spec.runtime,
            "precision": spec.precision,
            "device": spec.device,
            "lane": spec.lane,
            "preference_order": list(self.order),
            "allow_reduced_precision": bool(self.allow_reduced_precision),
            "probed": list(self.probed),
        }


def _context_for(bundle: ServingBundle, max_batch: int, device_index: int) -> BackendContext:
    """Build the registry context for this bundle.

    The engine and calibration cache paths follow the layout
    scripts/build_trt_engines.py writes, so a gpu host that has already built
    engines is picked up with no extra configuration.
    """
    engine_dir = os.path.join("results", "trt")
    name = bundle.manifest.get("model", {}).get("display_name", "DeepFM")
    engine_paths = {
        precision: os.path.join(
            engine_dir, f"{name}_{precision}_bs{max_batch}.engine"
        )
        for precision in ("fp32", "fp16", "int8")
    }
    return BackendContext(
        model_name=name,
        module=None,
        onnx_path=bundle.onnx_path,
        n_numerical=bundle.meta.n_numerical,
        n_embed_fields=bundle.meta.n_embed_fields,
        max_batch=max_batch,
        engine_paths=engine_paths,
        calibration_cache=os.path.join(engine_dir, f"{name}_int8_calibration.cache"),
        trt_engine_cache_dir=os.path.join(engine_dir, "ort_trt_cache"),
        device_index=device_index,
    )


def select_backend(
    bundle: ServingBundle,
    preferred: Optional[str] = None,
    allow_reduced_precision: bool = False,
    max_batch: int = 4096,
    device_index: int = 0,
    verbose: bool = True,
) -> BackendSelection:
    """Probe the preference order and return the first backend that built.

    When preferred names a key, only that key is tried and a failure to build it
    raises rather than falling back. Pinning a backend and then silently getting
    a different one is worse than not starting.
    """
    ctx = _context_for(bundle, max_batch, device_index)

    if preferred:
        spec = spec_by_key(preferred)
        if spec is None:
            raise KeyError(
                f"unknown backend key {preferred}. The registry offers "
                f"{[s for s in PREFERENCE_ORDER]} plus the reduced precision keys "
                f"{list(REDUCED_PRECISION_ORDER)}."
            )
        order = [preferred]
    else:
        order = list(REDUCED_PRECISION_ORDER if allow_reduced_precision else ())
        order += list(PREFERENCE_ORDER)

    probed: List[Dict[str, Any]] = []
    chosen: Optional[BackendResult] = None
    for key in order:
        spec = spec_by_key(key)
        if spec is None:
            probed.append(
                {"key": key, "available": False, "note": "no such backend key"}
            )
            continue
        if spec.runtime == "pytorch" and ctx.module is None:
            # The eager backends need a live module and the registry does not
            # build one. It is constructed lazily so a service that selected
            # OpenVINO never pays to materialize eighty megabytes of weights.
            ctx.module = bundle.build_module()
        result = build_backend(spec, ctx)
        probed.append(result.as_dict())
        if verbose:
            if result.available:
                print(f"  {spec.label} is available.")
            else:
                print(f"  {spec.label} is {NOT_AVAILABLE}. {result.note}")
        if result.available:
            chosen = result
            break

    if chosen is None:
        reasons = " ".join(
            f"{r.get('label', r.get('key'))} was {NOT_AVAILABLE} because "
            f"{r.get('note', 'no reason was given')}."
            for r in probed
        )
        raise RuntimeError(
            "no backend in the preference order could be constructed, so the "
            f"service cannot serve. {reasons}"
        )

    return BackendSelection(
        result=chosen,
        probed=probed,
        order=order,
        allow_reduced_precision=bool(allow_reduced_precision),
    )


@dataclass
class ScoreTiming:
    """Where the wall time of one scoring call actually went."""

    feature_seconds: float
    model_seconds: float
    total_seconds: float


class ScoringEngine:
    """Featurize, run the model, calibrate. One object, both lanes.

    The online service and the batch job both construct one of these and both
    call score_rows. That is the whole mechanism that prevents online and
    offline skew in this project. There is no second featurizer, no second model
    load, and no second calibration path that could drift, and the test suite
    asserts the two lanes return identical probabilities for identical rows
    rather than merely similar ones.

    The model call is taken under a lock. The registry's run_batch closures make
    no thread safety promise, and at least one of them cannot make it. The
    OpenVINO compiled model shares a single default inference request across
    calls, so two threads entering it concurrently are a data race rather than
    parallelism. Serializing the model call is the correct conservative choice
    and it costs less than it looks like it costs, because the feature transform
    runs outside the lock and is where most of the wall time goes on this model.
    Real horizontal scaling comes from running several worker processes, which is
    what the docker compose file does.
    """

    def __init__(
        self,
        bundle: ServingBundle,
        selection: BackendSelection,
        max_candidates: int = 4096,
    ) -> None:
        self.bundle = bundle
        self.selection = selection
        self.max_candidates = int(max_candidates)
        self._runner = selection.result.runner
        self._lock = threading.Lock()

    @property
    def backend_key(self) -> str:
        return self.selection.key

    def warmup(self, batches: Sequence[int] = WARMUP_BATCHES, verbose: bool = True) -> float:
        """Push a few synthetic batches through featurize and model, discarded.

        The warmup rows are built from the bundle's own schema rather than from
        real data, because the point is to touch every code path and every
        allocator once, not to measure anything.
        """
        started = time.perf_counter()
        for size in batches:
            rows = [{} for _ in range(int(size))]
            self.score_rows(rows)
        elapsed = time.perf_counter() - started
        if verbose:
            print(
                f"warmed the {self.selection.label} backend with batches "
                f"{list(batches)} in {elapsed * 1000:.1f} ms. These calls are "
                "discarded and are not part of any reported number."
            )
        return elapsed

    def featurize(self, rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Run the persisted offline pipeline over raw request rows."""
        return featurize_rows(self.bundle.pipeline, rows)

    def score_arrays(self, numerical: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Run the model on already featurized blocks and calibrate the output."""
        with self._lock:
            raw = self._runner(numerical, cat)
        return self.bundle.calibrator.apply(np.asarray(raw, dtype=np.float64).reshape(-1))

    def score_rows(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> Tuple[np.ndarray, ScoreTiming]:
        """Score raw request rows end to end and report where the time went."""
        if len(rows) > self.max_candidates:
            raise ValueError(
                f"this request carries {len(rows)} candidates and the service is "
                f"configured for at most {self.max_candidates}. Split the "
                "candidate set or raise the limit."
            )
        start = time.perf_counter()
        numerical, cat = self.featurize(rows)
        after_features = time.perf_counter()
        probs = self.score_arrays(numerical, cat)
        end = time.perf_counter()
        return probs, ScoreTiming(
            feature_seconds=after_features - start,
            model_seconds=end - after_features,
            total_seconds=end - start,
        )

    def describe(self) -> Dict[str, Any]:
        """Return the engine description the health endpoint serves."""
        return {
            "backend": self.selection.as_dict(),
            "max_candidates": self.max_candidates,
            "bundle": self.bundle.describe(),
        }
