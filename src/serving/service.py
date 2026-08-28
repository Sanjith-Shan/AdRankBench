"""One function that turns a configuration into a running scoring service.

Both entry points into the online lane go through here. scripts/serve.py builds
a service and hands it to uvicorn, and tests/test_serving.py builds the same
service and drives it with the FastAPI test client. That is on purpose. A test
suite that constructs its own app with its own wiring proves that its wiring
works, not that the served one does.

The order of operations is the order an operator would want it. The bundle loads
and every fingerprint in it is verified, then the backend probe runs and prints
which runtimes were tried and why each one was skipped, then the selected
backend is warmed with a few discarded batches, and only then does the app come
into existence. Anything that goes wrong happens before the port is bound, so a
broken deployment is a process that failed to start rather than a healthy
looking server returning wrong answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.serving.app import create_app
from src.serving.artifact import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BUNDLE_DIR,
    ServingBundle,
    load_or_build_bundle,
)
from src.serving.metrics import ServiceMetrics
from src.serving.runtime import ScoringEngine, select_backend


@dataclass
class ServiceConfig:
    """Everything that varies between one deployment of this service and another."""

    bundle_dir: str = DEFAULT_BUNDLE_DIR
    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    model_name: str = "deepfm"
    data_path: str = "data/criteo.csv"
    sample_size: int = 100000
    synthetic: bool = False
    backend: Optional[str] = None
    allow_reduced_precision: bool = False
    max_candidates: int = 4096
    device_index: int = 0
    thread_pool_size: int = 8
    warmup: bool = True
    verbose: bool = True


@dataclass
class Service:
    """The assembled service and the pieces a caller may want to reach into."""

    app: Any
    engine: ScoringEngine
    metrics: ServiceMetrics
    bundle: ServingBundle
    hardware: Dict[str, Any] = field(default_factory=dict)


def build_service(config: ServiceConfig) -> Service:
    """Load the bundle, select a backend, warm it, and return the app."""
    from src.inference.hardware import collect_hardware_record, print_hardware_record

    hardware = collect_hardware_record(config.device_index)
    if config.verbose:
        print_hardware_record(hardware)

    bundle = load_or_build_bundle(
        bundle_dir=config.bundle_dir,
        verbose=config.verbose,
        artifact_dir=config.artifact_dir,
        model_name=config.model_name,
        data_path=config.data_path,
        sample_size=config.sample_size,
        synthetic=config.synthetic,
    )

    if config.verbose:
        print("probing backends in preference order.")
    selection = select_backend(
        bundle,
        preferred=config.backend,
        allow_reduced_precision=config.allow_reduced_precision,
        max_batch=config.max_candidates,
        device_index=config.device_index,
        verbose=config.verbose,
    )
    if config.verbose:
        host = hardware.get("host", {})
        print(
            f"selected the {selection.label} backend on "
            f"{host.get('cpu_model', 'an unrecorded cpu')}."
        )
        if selection.allow_reduced_precision and selection.result.spec.precision != "fp32":
            print(
                "this is a reduced precision backend, so the probabilities it "
                "returns are not bit identical to the fp32 numbers the offline "
                "evaluation reported."
            )

    engine = ScoringEngine(bundle, selection, max_candidates=config.max_candidates)
    if config.warmup:
        engine.warmup(verbose=config.verbose)

    metrics = ServiceMetrics()
    app = create_app(
        engine,
        metrics=metrics,
        hardware=hardware,
        thread_pool_size=config.thread_pool_size,
    )
    return Service(
        app=app, engine=engine, metrics=metrics, bundle=bundle, hardware=hardware
    )
