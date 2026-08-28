"""Model construction, checkpoint handling, and ONNX export for the serving path.

Both the engine builder and the inference benchmark need the same three things.
They need a trained module, they need it exported to ONNX with a dynamic batch
axis, and they need the test rows sliced into batches in the exact layout the
module was trained on. Putting those here means the two scripts cannot drift
apart, which matters because a difference between the graph the engines were
built from and the graph the benchmark scores would silently invalidate every
comparison in the report.

The export uses the TorchScript exporter rather than the dynamo exporter. That
is a deliberate choice. TorchScript writes a single self contained ONNX file
with the weights inlined, while the dynamo path can emit a graph plus an
external data sidecar. One portable file is what ONNX Runtime, OpenVINO, and the
TensorRT parser all consume without extra handling, and it is what makes a
committed artifact meaningful.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from src.schema import Dataset, FeatureMeta

# The models this serving path knows how to build. Both are DLRM shaped, which
# is a large shared embedding table feeding a small perceptron, so both are
# interesting subjects for the memory bound question the analysis asks.
SUPPORTED_MODELS: Tuple[str, ...] = ("deepfm", "dcn")

# The display name each model uses in reports, file names, and engine names.
DISPLAY_NAMES: Dict[str, str] = {"deepfm": "DeepFM", "dcn": "DCN"}


def display_name(model_name: str) -> str:
    """Return the canonical display name for a model key."""
    return DISPLAY_NAMES.get(model_name.lower(), model_name)


def build_module(model_name: str, meta: FeatureMeta, config: Dict[str, Any]):
    """Construct an untrained module for one model key."""
    key = model_name.lower()
    if key == "deepfm":
        from src.models.deepfm import DeepFMModule

        return DeepFMModule(meta, config["embed_dim"], config["hidden"], config["dropout"])
    if key == "dcn":
        from src.models.dcn import DCNModule

        return DCNModule(
            meta,
            config["embed_dim"],
            config["cross_layers"],
            config["hidden"],
            config["dropout"],
        )
    raise KeyError(
        f"unknown model {model_name}. This serving path supports {list(SUPPORTED_MODELS)}."
    )


def load_or_train_module(
    model_name: str,
    meta: FeatureMeta,
    config: Dict[str, Any],
    checkpoint: str,
    train_ds: Dataset,
    val_ds: Dataset,
    verbose: bool = True,
):
    """Return an eval mode module on the cpu with trained weights.

    If the checkpoint exists it is loaded. If it does not exist the module is
    trained on the spot through the shared trainer and the weights are saved to
    the checkpoint path, so a later run loads instead of retraining. Training
    uses the same config and the same trainer as the main benchmark, so the
    weights that get served are the weights the benchmark reports.
    """
    import torch

    from src.train.trainer import train_torch_model

    module = build_module(model_name, meta, config)

    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        module.load_state_dict(state)
        if verbose:
            print(f"loaded {display_name(model_name)} weights from {checkpoint}.")
    else:
        if verbose:
            print(
                f"no checkpoint at {checkpoint}. Training "
                f"{display_name(model_name)} on cpu to create one."
            )
        module = train_torch_model(
            module, train_ds, val_ds, meta, config, device=torch.device("cpu")
        )
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save(module.state_dict(), checkpoint)
        if verbose:
            print(f"saved {display_name(model_name)} weights to {checkpoint}.")

    module = module.to("cpu")
    module.eval()
    return module


def export_onnx(module, meta: FeatureMeta, batch_size: int, onnx_path: str) -> str:
    """Export a ranking module to ONNX with a dynamic batch dimension.

    The dummy input matches the two model inputs. numerical is a dense float
    block of width n_numerical and cat is an integer block of width
    n_embed_fields whose values are valid field indices. The batch axis is
    marked dynamic so one exported graph serves any batch size at inference,
    which is what lets the same file feed a batch of one for online serving and
    a batch of four thousand for offline scoring.

    The exporter needs the onnx package listed in requirements.txt.
    """
    import torch

    module = module.eval()

    dummy_numerical = torch.randn(batch_size, meta.n_numerical, dtype=torch.float32)
    # Field indices stay inside the smallest field vocab so the dummy forward
    # pass through the embedding tables is always in range, whatever the meta
    # is. The actual values do not affect the exported graph shape.
    vocab_sizes = meta.embed_vocab_sizes()
    high = max(1, min(vocab_sizes)) if vocab_sizes else 1
    dummy_cat = torch.randint(0, high, (batch_size, meta.n_embed_fields), dtype=torch.int64)

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    try:
        with warnings.catch_warnings():
            # Hush the legacy exporter deprecation note. The TorchScript path is
            # chosen on purpose for the single file artifact it produces.
            warnings.simplefilter("ignore")
            torch.onnx.export(
                module,
                (dummy_numerical, dummy_cat),
                onnx_path,
                input_names=["numerical", "cat"],
                output_names=["logits"],
                dynamic_axes={
                    "numerical": {0: "batch"},
                    "cat": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
    except Exception as exc:  # noqa: BLE001 surface a clear, actionable message
        raise RuntimeError(
            "ONNX export failed. The export needs the onnx package. Install the "
            "inference dependencies with pip install -r requirements.txt and "
            f"rerun. Underlying error: {exc}"
        ) from exc
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"exported ONNX graph to {onnx_path} ({size_mb:.1f} MB).")
    return onnx_path


def onnx_path_for(model_name: str, output_dir: str) -> str:
    """Return the canonical ONNX path for one model."""
    return os.path.join(output_dir, f"{model_name.lower()}.onnx")


def checkpoint_path_for(model_name: str, output_dir: str) -> str:
    """Return the canonical checkpoint path for one model.

    DeepFM keeps the results/DeepFM.pt name the main benchmark already writes,
    so the inference path reuses the trained weights rather than training a
    second copy of the same model.
    """
    return os.path.join(output_dir, f"{display_name(model_name)}.pt")


def dataset_arrays(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (numerical, cat) arrays for a split in the model input layout.

    The cat block is the concatenation of the categorical fields and the cross
    fields, which is the layout the trainer feeds the module. Order is preserved
    so stacked predictions line up with the labels.
    """
    numerical = np.ascontiguousarray(ds.numerical, dtype=np.float32)
    cat = np.ascontiguousarray(
        np.concatenate([ds.categorical, ds.crosses], axis=1), dtype=np.int64
    )
    return numerical, cat


def make_batches(ds: Dataset, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Slice a split into a list of (numerical, cat) numpy batches."""
    numerical, cat = dataset_arrays(ds)
    batches: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(numerical), batch_size):
        end = start + batch_size
        batches.append(
            (
                np.ascontiguousarray(numerical[start:end]),
                np.ascontiguousarray(cat[start:end]),
            )
        )
    return batches


def model_size_bytes(module) -> int:
    """Return the size of the module parameters and buffers in bytes at fp32."""
    total = 0
    for param in module.parameters():
        total += int(param.numel()) * int(param.element_size())
    for buf in module.buffers():
        total += int(buf.numel()) * int(buf.element_size())
    return total
