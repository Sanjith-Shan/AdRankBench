"""Factorization Machine model.

The FM captures all pairwise feature interactions through low rank embeddings.
The second order term is computed with the O(kn) identity that rewrites the sum
over all pairs as half the difference between the square of the sum and the sum
of the squares of the field embeddings. This is what lets the FM model pairwise
signal that a plain linear model cannot.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel, EmbeddingLayer
from src.schema import Dataset, FeatureMeta
from src.train.trainer import predict_torch, train_torch_model


class FMModule(nn.Module):
    """First order plus second order factorization machine.

    forward receives the dense numerical block and the combined long tensor of
    categorical and cross field indices, and returns raw logits with no sigmoid.
    """

    def __init__(self, meta: FeatureMeta, embed_dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.numerical_linear = nn.Linear(meta.n_numerical, 1)
        self.embedding = EmbeddingLayer(meta.embed_vocab_sizes(), embed_dim)

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        emb, lin = self.embedding(cat)

        first_order = (
            self.w0
            + self.numerical_linear(numerical).squeeze(-1)
            + lin.sum(dim=1)
        )

        # O(kn) trick. sum over pairs equals half of (square of sum minus sum of squares).
        sum_emb = emb.sum(dim=1)
        sum_sq = (emb ** 2).sum(dim=1)
        second_order = 0.5 * (sum_emb ** 2 - sum_sq).sum(dim=1)

        return first_order + second_order


class FMModel(BaseModel):
    """BaseModel wrapper that trains an FMModule through the shared trainer."""

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
    ) -> "FMModel":
        self.meta = meta
        self.embed_dim = config["embed_dim"]
        self.module = FMModule(meta, self.embed_dim)
        self.module = train_torch_model(self.module, train, val, meta, config)
        self.device = next(self.module.parameters()).device
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        return predict_torch(self.module, data, self.device)

    def get_name(self) -> str:
        return "FM"
