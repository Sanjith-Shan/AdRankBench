"""DeepFM model.

DeepFM combines a factorization machine and a deep neural network that share a
single embedding table. The FM part models explicit low order interactions and
the deep part models high order ones, and because they share embeddings the
model learns one representation that serves both. The two outputs are summed at
the logit level.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel, MLP, EmbeddingLayer
from src.schema import Dataset, FeatureMeta
from src.train.trainer import predict_torch, train_torch_model


class DeepFMModule(nn.Module):
    """Wide factorization machine and a deep tower over a shared embedding.

    forward returns the sum of the FM logit and the deep logit as raw logits.
    """

    def __init__(self, meta: FeatureMeta, embed_dim: int, hidden, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fields = meta.n_embed_fields

        # Shared embedding used by both the FM part and the deep part.
        self.embedding = EmbeddingLayer(meta.embed_vocab_sizes(), embed_dim)

        # FM first order pieces.
        self.w0 = nn.Parameter(torch.zeros(1))
        self.numerical_linear = nn.Linear(meta.n_numerical, 1)

        # Deep part. Input is the flattened field embeddings plus numerical features.
        deep_in = self.n_fields * embed_dim + meta.n_numerical
        self.mlp = MLP(deep_in, list(hidden), dropout=dropout)
        self.deep_head = nn.Linear(self.mlp.out_dim, 1)

    def _fm_part(self, numerical: torch.Tensor, emb: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
        first_order = (
            self.w0
            + self.numerical_linear(numerical).squeeze(-1)
            + lin.sum(dim=1)
        )
        sum_emb = emb.sum(dim=1)
        sum_sq = (emb ** 2).sum(dim=1)
        second_order = 0.5 * (sum_emb ** 2 - sum_sq).sum(dim=1)
        return first_order + second_order

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        emb, lin = self.embedding(cat)

        fm_logit = self._fm_part(numerical, emb, lin)

        flat = emb.reshape(emb.size(0), self.n_fields * self.embed_dim)
        deep_in = torch.cat([flat, numerical], dim=1)
        deep_logit = self.deep_head(self.mlp(deep_in)).squeeze(-1)

        return fm_logit + deep_logit


class DeepFMModel(BaseModel):
    """BaseModel wrapper that trains a DeepFMModule through the shared trainer."""

    def __init__(self) -> None:
        self.module = None
        self.meta = None
        self.embed_dim = None
        self.device = None

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        meta: FeatureMeta,
        config: dict,
    ) -> "DeepFMModel":
        self.meta = meta
        self.embed_dim = config["embed_dim"]
        self.module = DeepFMModule(
            meta,
            self.embed_dim,
            config["hidden"],
            config["dropout"],
        )
        self.module = train_torch_model(self.module, train, val, meta, config)
        self.device = next(self.module.parameters()).device
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        return predict_torch(self.module, data, self.device)

    def get_name(self) -> str:
        return "DeepFM"
