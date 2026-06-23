"""Training utilities for the PyTorch ranking models.

This module holds the shared training loop and helpers that every neural model
in the benchmark relies on. The BaseModel wrappers in src.models build their
nn.Module and hand it to train_torch_model here, then use predict_torch for
inference. Keeping the loop in one place means every architecture trains under
the same optimizer, loss, early stopping, and device handling so the comparison
stays fair.

The loss is BCEWithLogitsLoss because the modules return raw logits with no
sigmoid. Early stopping watches validation logloss and the best weights are
restored at the end. Everything is seeded for reproducibility.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    def tqdm(iterable=None, **kwargs):
        """Fallback no op progress wrapper when tqdm is not installed."""
        return iterable if iterable is not None else []

from src.schema import Dataset, FeatureMeta


def set_seed(seed: int = 42) -> None:
    """Seed Python, numpy, and torch for reproducible runs.

    Also flips the cudnn flags toward deterministic behavior on a best effort
    basis. The mps backend has its own generator that respects torch.manual_seed
    so no extra handling is needed there.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best effort determinism on cuda. These are no ops on cpu and mps.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device from a preference string.

    With "auto" the order of preference is cuda, then mps, then cpu. Explicit
    values "cuda", "mps", and "cpu" are honored directly.
    """
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def _build_cat_tensor(ds: Dataset) -> np.ndarray:
    """Return the combined int64 field tensor, categorical then crosses.

    Deep models embed the concatenation of the categorical fields and the cross
    fields, so the loader stacks them into one (N, n_embed_fields) array.
    """
    return np.concatenate([ds.categorical, ds.crosses], axis=1).astype(np.int64)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """Build a DataLoader of (numerical, cat, label) tensors for one Dataset.

    The cat tensor is the concatenation of categorical and cross fields. Tensors
    are kept on cpu and moved to the device inside the training loop, which is
    the safe pattern for mps and pinned cpu memory.

    Note on num_workers. The CLAUDE spec suggests 4 workers, but we use 0 here.
    Zero workers avoids subprocess pickling and is safer across platforms, in
    particular on mps and for the small in memory datasets this benchmark uses.
    This is a deliberate deviation documented here.
    """
    numerical = torch.from_numpy(np.ascontiguousarray(ds.numerical, dtype=np.float32))
    cat = torch.from_numpy(_build_cat_tensor(ds))
    label = torch.from_numpy(np.ascontiguousarray(ds.label, dtype=np.float32))
    tensor_ds = TensorDataset(numerical, cat, label)
    return DataLoader(
        tensor_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def train_torch_model(
    module: nn.Module,
    train: Dataset,
    val: Dataset,
    meta: FeatureMeta,
    config: dict,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Train a logits producing nn.Module with early stopping on val logloss.

    The loop uses BCEWithLogitsLoss and Adam with lr and weight_decay from the
    config. The number of epochs comes from config["epochs"] and falls back to
    config["max_epochs"]. After each epoch the validation AUC and logloss are
    computed with sklearn. Training stops once val logloss has not improved for
    patience epochs and the best weights by val logloss are restored before the
    module is returned.

    Parameters
    ----------
    module : nn.Module
        Returns raw logits of shape (B,) from forward(numerical, cat).
    train, val : Dataset
        Featurized splits.
    meta : FeatureMeta
        Field metadata. Accepted for a uniform signature.
    config : dict
        Holds lr, weight_decay, batch_size, epochs or max_epochs, and patience.
    device : torch.device or None
        Where to train. Auto resolved when None.

    Returns
    -------
    nn.Module
        The same module with the best weights loaded.
    """
    if device is None:
        device = get_device("auto")

    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    batch_size = config.get("batch_size", 4096)
    patience = config.get("patience", 3)
    epochs = config.get("epochs", config.get("max_epochs", 20))

    module = module.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(train, batch_size=batch_size, shuffle=True, device=device)

    y_val = np.asarray(val.label, dtype=np.float32).reshape(-1)

    best_val_logloss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        module.train()
        running_loss = 0.0
        n_seen = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for numerical, cat, label in bar:
            numerical = numerical.to(device)
            cat = cat.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = module(numerical, cat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            bs = label.shape[0]
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(n_seen, 1)

        # Validation pass.
        val_probs = predict_torch(module, val, device, batch_size=max(batch_size, 8192))
        val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)
        val_logloss = float(log_loss(y_val, val_probs, labels=[0, 1]))
        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except ValueError:
            # Only one class present in val, AUC is undefined.
            val_auc = float("nan")

        print(
            f"epoch {epoch}: train_loss={train_loss:.5f}, "
            f"val_loss={val_logloss:.5f}, val_auc={val_auc:.5f}"
        )

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"early stopping at epoch {epoch}, "
                    f"best val_loss={best_val_logloss:.5f}"
                )
                break

    # Restore best weights.
    module.load_state_dict(best_state)
    module = module.to(device)
    return module


def predict_torch(
    module: nn.Module,
    data: Dataset,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run the module in eval mode and return click probabilities.

    Applies sigmoid to the raw logits and concatenates batches into one array.
    No shuffling so the output order matches the input rows.

    Returns
    -------
    np.ndarray
        Shape (N,) float in [0, 1].
    """
    module = module.to(device)
    module.eval()
    loader = make_loader(data, batch_size=batch_size, shuffle=False, device=device)

    chunks = []
    with torch.no_grad():
        for numerical, cat, _label in loader:
            numerical = numerical.to(device)
            cat = cat.to(device)
            logits = module(numerical, cat)
            probs = torch.sigmoid(logits)
            chunks.append(probs.detach().cpu().numpy())

    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32).reshape(-1)
