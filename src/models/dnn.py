"""Plain deep neural network model.

This is the pure deep baseline. It embeds every sparse and cross field, flattens
those embeddings, concatenates the dense numerical block, and feeds the result
through an MLP with a single linear head. It carries no explicit interaction
structure, so comparing it against FM, DeepFM, and DCN shows how much the
explicit interaction terms add beyond a generic deep model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel, MLP, EmbeddingLayer
from src.schema import Dataset, FeatureMeta
from src.train.trainer import predict_torch, train_torch_model


class DNNModule(nn.Module):
    """Embedding plus MLP. forward returns raw logits with no sigmoid."""

    def __init__(self, meta: FeatureMeta, embed_dim: int, hidden, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fields = meta.n_embed_fields

        self.embedding = EmbeddingLayer(meta.embed_vocab_sizes(), embed_dim)

        in_dim = self.n_fields * embed_dim + meta.n_numerical
        self.mlp = MLP(in_dim, list(hidden), dropout=dropout)
        self.head = nn.Linear(self.mlp.out_dim, 1)

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        emb, _ = self.embedding(cat)
        flat = emb.reshape(emb.size(0), self.n_fields * self.embed_dim)
        x = torch.cat([flat, numerical], dim=1)
        return self.head(self.mlp(x)).squeeze(-1)


class DNNModel(BaseModel):
    """BaseModel wrapper that trains a DNNModule through the shared trainer."""

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
    ) -> "DNNModel":
        self.meta = meta
        self.embed_dim = config["embed_dim"]
        self.module = DNNModule(
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
        return "DNN"
