"""Base model interface and shared neural building blocks.

Every model in the benchmark implements BaseModel so the trainer and the
evaluation harness can treat them uniformly. The PyTorch models share the
EmbeddingLayer and MLP defined here to keep embedding sizing and field offset
logic consistent across architectures.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.schema import Dataset, FeatureMeta


class BaseModel:
    """Common interface for all ranking models.

    The fit method receives whole Dataset containers rather than raw X and y so
    that every model sees the same featurized inputs (numerical, categorical,
    frequency, crosses). predict_proba returns calibrated click probabilities.
    """

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        meta: FeatureMeta,
        config: dict,
    ) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, data: Dataset) -> np.ndarray:
        """Return a (N,) array of click probabilities in [0, 1]."""
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


# The neural building blocks live behind a lazy torch import so that importing
# the schema or the logistic baseline does not require torch to be present.
try:
    import torch
    import torch.nn as nn

    class EmbeddingLayer(nn.Module):
        """Shared embedding table for categorical and cross fields.

        Uses a single embedding table with per field offsets, the standard
        efficient pattern in CTR models. Returns both the dense second order
        embeddings and the first order (linear) weights used by the FM family.
        """

        def __init__(self, vocab_sizes: List[int], embed_dim: int):
            super().__init__()
            self.num_fields = len(vocab_sizes)
            self.embed_dim = embed_dim
            total = int(sum(vocab_sizes))
            self.embedding = nn.Embedding(total, embed_dim)
            self.linear = nn.Embedding(total, 1)
            offsets = np.array(
                [0] + list(np.cumsum(vocab_sizes)[:-1]), dtype=np.int64
            )
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.zeros_(self.linear.weight)

        def forward(self, cat: "torch.Tensor"):
            """cat is (B, num_fields) long. Returns (emb (B, F, k), lin (B, F))."""
            idx = cat + self.offsets.unsqueeze(0)
            emb = self.embedding(idx)
            lin = self.linear(idx).squeeze(-1)
            return emb, lin

    class MLP(nn.Module):
        """A stack of Linear, BatchNorm, ReLU, Dropout blocks.

        Outputs a representation of size hidden[-1]. Models attach their own
        final linear head so the same MLP can serve as a deep tower in DNN,
        DeepFM, and DCN.
        """

        def __init__(
            self,
            in_dim: int,
            hidden: List[int],
            dropout: float = 0.0,
            batchnorm: bool = True,
        ):
            super().__init__()
            layers: List[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            self.net = nn.Sequential(*layers)
            self.out_dim = prev

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
    EmbeddingLayer = None  # type: ignore
    MLP = None  # type: ignore
