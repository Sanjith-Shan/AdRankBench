"""The module level app object uvicorn worker processes import.

Uvicorn cannot hand an already constructed application object to worker
processes it forks itself, so a multi worker run needs an import path it can
resolve inside each worker. This module is that import path. It reads the same
configuration scripts/serve.py was given, published as environment variables,
and builds the identical service, so a four worker deployment is four copies of
the one worker deployment rather than a second code path.

Importing this module builds a service, which loads a bundle and probes
backends. That is intentional. A worker that cannot construct a backend must die
at import rather than start and return errors.
"""

from __future__ import annotations

import os

from src.serving.service import ServiceConfig, build_service


def _flag(name: str, default: str = "0") -> bool:
    """Read a boolean environment flag."""
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def config_from_environment() -> ServiceConfig:
    """Rebuild the service configuration a parent scripts/serve.py published."""
    return ServiceConfig(
        bundle_dir=os.environ.get("ADRANK_BUNDLE_DIR", "results/serving/bundle"),
        artifact_dir=os.environ.get("ADRANK_ARTIFACT_DIR", "results"),
        model_name=os.environ.get("ADRANK_MODEL", "deepfm"),
        data_path=os.environ.get("ADRANK_DATA_PATH", "data/criteo.csv"),
        sample_size=int(os.environ.get("ADRANK_SAMPLE_SIZE", "100000")),
        synthetic=_flag("ADRANK_SYNTHETIC"),
        backend=os.environ.get("ADRANK_BACKEND") or None,
        allow_reduced_precision=_flag("ADRANK_ALLOW_REDUCED_PRECISION"),
        max_candidates=int(os.environ.get("ADRANK_MAX_CANDIDATES", "4096")),
        thread_pool_size=int(os.environ.get("ADRANK_THREADS", "8")),
        warmup=_flag("ADRANK_WARMUP", "1"),
        verbose=True,
    )


service = build_service(config_from_environment())
app = service.app
