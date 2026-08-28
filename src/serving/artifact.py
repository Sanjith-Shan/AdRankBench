"""The serving bundle, which is the one artifact both lanes load.

A deployable ranker is not a checkpoint. It is a checkpoint, the exported graph
that a fast runtime can actually execute, the fitted feature pipeline that turns
a raw request into the tensor that graph expects, the probability calibration
that turns a logit into something a bid can be computed from, and the provenance
that says which data all of those were fitted on. Ship four of those five and the
service still runs, which is the problem. This module builds all five together
and writes a manifest that names them, so the online service and the batch job
load one thing rather than assembling their own.

The bundle deliberately does not copy the large files. The exported graph stays
at results/deepfm.onnx and the weights stay at results/DeepFM.pt, which are the
exact paths scripts/run_inference_benchmark.py writes and reads. The manifest
records the sha256 of each one. That means the latency numbers in
docs/INFERENCE.md and the probabilities this service returns come from the same
bytes on disk, and a fingerprint mismatch at load time is loud rather than
silent.

Everything model shaped in here is delegated. Module construction, checkpoint
loading, on the spot training, and the ONNX export all come from
src.inference.export, and nothing in src/inference or src/data is modified. What
this module owns is the assembly, the manifest, and the fingerprints.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.loader import generate_synthetic, load_raw
from src.data.preprocess import FeaturePipeline
from src.data.split import temporal_split
from src.inference.export import (
    checkpoint_path_for,
    display_name,
    export_onnx,
    load_or_train_module,
    onnx_path_for,
)
from src.schema import Dataset, FeatureMeta
from src.serving.calibration import Calibrator, select_calibrator
from src.serving.features import (
    FEATURE_STATE_FILENAME,
    describe_state,
    load_feature_pipeline,
    read_feature_state,
    sha256_file,
    write_feature_state,
)
from src.train.config import SEED, get_config

# Where a bundle lives unless the caller says otherwise.
DEFAULT_BUNDLE_DIR: str = os.path.join("results", "serving", "bundle")

# The directory the exported graph and the checkpoint are read from, which is
# the same directory the inference benchmark writes them to.
DEFAULT_ARTIFACT_DIR: str = "results"

MANIFEST_FILENAME: str = "manifest.json"

# Bumped when the manifest layout changes in a way an older reader would
# misread. The loader refuses an unknown version rather than guessing.
BUNDLE_VERSION: int = 1


@dataclass
class ServingBundle:
    """A loaded bundle, ready to be handed to a scoring engine."""

    bundle_dir: str
    manifest: Dict[str, Any]
    pipeline: FeaturePipeline
    meta: FeatureMeta
    calibrator: Calibrator
    feature_state_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model", {}).get("name", "deepfm"))

    @property
    def onnx_path(self) -> str:
        return str(self.manifest.get("model", {}).get("onnx_path", ""))

    @property
    def checkpoint_path(self) -> str:
        return str(self.manifest.get("model", {}).get("checkpoint_path", ""))

    @property
    def model_config(self) -> Dict[str, Any]:
        return dict(self.manifest.get("model", {}).get("config", {}))

    def build_module(self):
        """Construct the eager module with the bundle's trained weights.

        This is what the PyTorch fallback backend runs and what the ONNX export
        would be regenerated from. It is built on demand rather than at load
        time, because a service that selected OpenVINO has no use for it and the
        weights are eighty odd megabytes.
        """
        import torch

        from src.inference.export import build_module

        module = build_module(self.model_name, self.meta, self.model_config)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        module.load_state_dict(state)
        module.eval()
        return module

    def describe(self) -> Dict[str, Any]:
        """Return the compact bundle description the health endpoint reports."""
        model = self.manifest.get("model", {})
        data = self.manifest.get("data", {})
        return {
            "bundle_dir": self.bundle_dir,
            "bundle_version": int(self.manifest.get("bundle_version", -1)),
            "built_at_utc": self.manifest.get("built_at_utc", ""),
            "model_name": display_name(self.model_name),
            "onnx_path": model.get("onnx_path", ""),
            "onnx_sha256": model.get("onnx_sha256", ""),
            "checkpoint_path": model.get("checkpoint_path", ""),
            "checkpoint_sha256": model.get("checkpoint_sha256", ""),
            "feature_pipeline_sha256": self.manifest.get("feature_pipeline", {}).get(
                "sha256", ""
            ),
            "feature_pipeline": self.feature_state_summary,
            "calibration": self.manifest.get("calibration", {}),
            "data": data,
        }


def bundle_paths(bundle_dir: str) -> Tuple[str, str]:
    """Return the manifest path and the feature state path for a bundle dir."""
    return (
        os.path.join(bundle_dir, MANIFEST_FILENAME),
        os.path.join(bundle_dir, FEATURE_STATE_FILENAME),
    )


def bundle_exists(bundle_dir: str) -> bool:
    """True when both bundle members are present on disk."""
    manifest_path, state_path = bundle_paths(bundle_dir)
    return os.path.exists(manifest_path) and os.path.exists(state_path)


def _load_frames(
    data_path: str, sample_size: int, synthetic: bool, seed: int
) -> Tuple[Any, Dict[str, Any]]:
    """Load the raw frame and return it with a provenance record.

    The fallback rule is the one the rest of the project uses. A real Criteo
    file is preferred, the synthetic generator is the fallback, and the record
    always says which one ran so no bundle is ambiguous about what it was fitted
    on.
    """
    use_real = (not synthetic) and os.path.exists(data_path)
    if use_real:
        frame = load_raw(data_path, sample_size=sample_size)
        stat = os.stat(data_path)
        provenance = {
            "source": "criteo",
            "path": os.path.abspath(data_path),
            "file_bytes": int(stat.st_size),
            "file_mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            "sample_size": int(sample_size),
            "seed": int(seed),
        }
    else:
        frame = generate_synthetic(sample_size, seed=seed)
        provenance = {
            "source": "synthetic",
            "path": "",
            "file_bytes": 0,
            "file_mtime_utc": "",
            "sample_size": int(sample_size),
            "seed": int(seed),
            "note": (
                "no real Criteo file was present, so the bundle was fitted on "
                "the synthetic generator. Every probability this bundle serves "
                "is a synthetic data probability and is labeled as one."
            ),
        }
    return frame, provenance


def build_bundle(
    bundle_dir: str = DEFAULT_BUNDLE_DIR,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    model_name: str = "deepfm",
    data_path: str = os.path.join("data", "criteo.csv"),
    sample_size: int = 100000,
    synthetic: bool = False,
    seed: int = SEED,
    export_batch: int = 1024,
    force_export: bool = False,
    verbose: bool = True,
) -> ServingBundle:
    """Fit, assemble, and write a serving bundle, then load it back.

    The order matters. The pipeline is fitted on the train split only, the
    module is loaded from the existing checkpoint or trained on that same split
    when there is none, the graph is exported from those weights, and the
    calibrator is fitted on the validation split. Nothing anywhere in the build
    touches the test split, which stays clean for the accuracy checks the rest
    of the project runs on it.
    """
    from src.train.trainer import set_seed

    set_seed(seed)
    started = time.time()

    frame, data_provenance = _load_frames(data_path, sample_size, synthetic, seed)
    train_df, val_df, test_df = temporal_split(frame)
    if verbose:
        print(
            f"loaded {len(frame)} rows from the {data_provenance['source']} source "
            f"and split them into {len(train_df)} train, {len(val_df)} val, and "
            f"{len(test_df)} test rows."
        )

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    train_ds = pipeline.transform(train_df)
    val_ds = pipeline.transform(val_df)
    meta = pipeline.meta
    if verbose:
        print(
            f"fitted the feature pipeline on train only. The featurized width is "
            f"{meta.n_numerical} numerical, {meta.n_cat} categorical, and "
            f"{meta.n_cross} cross fields."
        )

    checkpoint = checkpoint_path_for(model_name, artifact_dir)
    onnx_path = onnx_path_for(model_name, artifact_dir)
    config = get_config(model_name)
    module = load_or_train_module(
        model_name, meta, config, checkpoint, train_ds, val_ds, verbose=verbose
    )

    if force_export or not os.path.exists(onnx_path):
        export_onnx(module, meta, export_batch, onnx_path)
    elif verbose:
        print(f"reusing the exported graph at {onnx_path}.")

    calibration, validation = _score_validation(module, val_ds, verbose=verbose)

    os.makedirs(bundle_dir, exist_ok=True)
    manifest_path, state_path = bundle_paths(bundle_dir)
    state_sha = write_feature_state(pipeline, state_path)

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_on": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "build_seconds": round(time.time() - started, 3),
        "seed": int(seed),
        "model": {
            "name": model_name,
            "display_name": display_name(model_name),
            "config": config,
            "checkpoint_path": os.path.abspath(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "onnx_path": os.path.abspath(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "export_batch": int(export_batch),
        },
        "feature_meta": {
            "n_numerical": int(meta.n_numerical),
            "cat_vocab_sizes": [int(v) for v in meta.cat_vocab_sizes],
            "cross_vocab_sizes": [int(v) for v in meta.cross_vocab_sizes],
        },
        "feature_pipeline": {
            "file": FEATURE_STATE_FILENAME,
            "sha256": state_sha,
            "bytes": int(os.path.getsize(state_path)),
        },
        "calibration": calibration,
        "validation": validation,
        "data": {
            **data_provenance,
            "rows_total": int(len(frame)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "train_click_rate": float(np.mean(train_ds.label)),
            "val_click_rate": float(np.mean(val_ds.label)),
        },
    }

    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, manifest_path)
    if verbose:
        print(f"wrote the serving bundle to {bundle_dir}.")

    return load_bundle(bundle_dir, verify=True, verbose=verbose)


def _score_validation(
    module, val_ds: Dataset, verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Score the validation split in eager PyTorch, then calibrate and check it.

    Eager PyTorch is used on purpose rather than the fastest available backend.
    It is the accuracy reference every other runtime in this project is measured
    against, and a calibration fitted on one runtime's output has to be valid for
    all of them. The inference benchmark already reports that the cpu backends
    agree with eager PyTorch to four decimals of AUC, which is what makes one
    fitted calibrator safe to serve behind any of them.
    """
    from src.inference.backends import make_torch_runner
    from src.inference.export import dataset_arrays

    runner = make_torch_runner(module)
    numerical, cat = dataset_arrays(val_ds)
    chunks: List[np.ndarray] = []
    step = 4096
    for start in range(0, len(numerical), step):
        chunks.append(runner(numerical[start : start + step], cat[start : start + step]))
    probs = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)

    decision = select_calibrator(probs, val_ds.label)
    calibrator: Calibrator = decision["calibrator"]
    if verbose:
        print(f"calibration decision. {decision['reason']}.")

    calibration = {
        **calibrator.as_dict(),
        "applied": bool(decision["applied"]),
        "reason": str(decision["reason"]),
        "val_ece_before": float(decision["ece_before"]),
        "val_ece_after": float(decision["ece_after"]),
        "val_rows": int(len(val_ds)),
        "fitted_on": "validation split, never the test split",
    }

    # The validation block is the skew tripwire. A checkpoint that was trained
    # against a different feature pipeline than the one just fitted still loads
    # cleanly, because the tensor shapes depend on the bucket sizes rather than
    # on the data, and it still returns probabilities. What it cannot do is
    # rank. An area under the curve near one half at build time means the
    # weights and the transform do not belong to each other, and it is far
    # better to see that in the manifest than to discover it from a revenue
    # graph.
    validation = _validation_record(probs, val_ds.label, calibrator)
    if verbose:
        print(
            f"validation check on {validation['rows']} rows. AUC "
            f"{validation['auc']:.4f}, logloss {validation['logloss']:.4f}, "
            f"mean predicted {validation['mean_probability']:.4f} against a base "
            f"rate of {validation['base_rate']:.4f}."
        )
    if validation["auc"] < 0.55:
        print(
            "warning. The validation AUC of this bundle is close to chance, "
            "which means the checkpoint and the feature pipeline in it were "
            "almost certainly fitted on different data. Rebuild the bundle with "
            "the data the checkpoint was trained on, or delete the checkpoint so "
            "it is retrained on this split."
        )
    return calibration, validation


def _validation_record(
    probs: np.ndarray, labels: np.ndarray, calibrator: Calibrator
) -> Dict[str, Any]:
    """Return the validation metrics that go into the manifest."""
    from src.evaluation.metrics import compute_all

    calibrated = calibrator.apply(probs)
    metrics = compute_all(np.asarray(labels, dtype=np.float64), calibrated)
    return {
        "rows": int(len(labels)),
        "auc": float(metrics.get("auc", float("nan"))),
        "logloss": float(metrics.get("logloss", float("nan"))),
        "normalized_entropy": float(metrics.get("ne", float("nan"))),
        "mean_probability": float(np.mean(calibrated)) if len(calibrated) else float("nan"),
        "base_rate": float(np.mean(labels)) if len(labels) else float("nan"),
        "scored_with": "eager PyTorch on the cpu, the project accuracy reference",
    }


def load_bundle(
    bundle_dir: str = DEFAULT_BUNDLE_DIR, verify: bool = True, verbose: bool = True
) -> ServingBundle:
    """Load a bundle from disk, checking every fingerprint it recorded.

    Verification is on by default and it is the point of the manifest. A
    checkpoint or an exported graph that changed under a bundle means the
    feature statistics were fitted against different weights than the ones about
    to be served, which is the same skew this whole module exists to prevent,
    arriving from the other direction.
    """
    manifest_path, state_path = bundle_paths(bundle_dir)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"no serving bundle at {bundle_dir}. Build one with "
            "python scripts/serve.py --build-only before starting the service."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    version = int(manifest.get("bundle_version", -1))
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"the bundle at {bundle_dir} was written at version {version} and "
            f"this build reads version {BUNDLE_VERSION}. Rebuild it."
        )

    state = read_feature_state(state_path)
    pipeline = load_feature_pipeline(state_path)
    meta = pipeline.meta

    recorded = manifest.get("feature_meta", {})
    if recorded and (
        int(recorded.get("n_numerical", -1)) != int(meta.n_numerical)
        or list(recorded.get("cat_vocab_sizes", [])) != list(meta.cat_vocab_sizes)
        or list(recorded.get("cross_vocab_sizes", [])) != list(meta.cross_vocab_sizes)
    ):
        raise ValueError(
            "the feature meta recorded in the manifest does not match the meta "
            "the saved pipeline reconstructs. The bundle is inconsistent and "
            "must be rebuilt."
        )

    if verify:
        for label, path_key, sha_key in (
            ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
            ("exported graph", "onnx_path", "onnx_sha256"),
        ):
            path = str(manifest.get("model", {}).get(path_key, ""))
            expected = str(manifest.get("model", {}).get(sha_key, ""))
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"the bundle names a {label} at {path or 'an empty path'} and "
                    "it is not there. Rebuild the bundle."
                )
            actual = sha256_file(path)
            if expected and actual != expected:
                raise ValueError(
                    f"the {label} at {path} has sha256 {actual} and the bundle "
                    f"recorded {expected}. The artifact changed under the bundle, "
                    "so the fitted feature statistics no longer belong to these "
                    "weights. Rebuild the bundle."
                )
        state_sha = sha256_file(state_path)
        expected_state = str(manifest.get("feature_pipeline", {}).get("sha256", ""))
        if expected_state and state_sha != expected_state:
            raise ValueError(
                f"the feature pipeline file has sha256 {state_sha} and the "
                f"manifest recorded {expected_state}. Rebuild the bundle."
            )

    bundle = ServingBundle(
        bundle_dir=os.path.abspath(bundle_dir),
        manifest=manifest,
        pipeline=pipeline,
        meta=meta,
        calibrator=Calibrator.from_dict(manifest.get("calibration")),
        feature_state_summary=describe_state(state),
    )
    if verbose:
        print(
            f"loaded the serving bundle from {bundle_dir}, built at "
            f"{manifest.get('built_at_utc', 'an unrecorded time')} on the "
            f"{manifest.get('data', {}).get('source', 'unknown')} source."
        )
    return bundle


def load_or_build_bundle(
    bundle_dir: str = DEFAULT_BUNDLE_DIR, verbose: bool = True, **build_kwargs
) -> ServingBundle:
    """Load a bundle when one is present and build it when one is not."""
    if bundle_exists(bundle_dir):
        return load_bundle(bundle_dir, verify=True, verbose=verbose)
    if verbose:
        print(f"no bundle at {bundle_dir}. Building one now.")
    return build_bundle(bundle_dir=bundle_dir, verbose=verbose, **build_kwargs)
