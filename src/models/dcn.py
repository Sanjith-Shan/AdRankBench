"""Deep and Cross Network model.

DCN runs a cross network and a deep network in parallel over the same input
vector, then concatenates their outputs into a final linear head. The cross
network applies explicit bounded degree feature crossing at each layer through
the standard update x_next = x0 * (x_l @ w) + b + x_l, which grows the
interaction order by one per layer while keeping the parameter count linear in
the input width.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel, MLP, EmbeddingLayer
from src.schema import Dataset, FeatureMeta
from src.train.trainer import predict_torch, train_torch_model


class CrossNetwork(nn.Module):
    """Stack of explicit feature cross layers.

    Each layer computes x_next = x0 * (x_l @ w) + b + x_l where x0 is the
    original input and x_l is the running representation. The output keeps the
    same width as the input.
    """

    def __init__(self, in_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(in_dim)) for _ in range(n_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(in_dim)) for _ in range(n_layers)]
        )
        for w in self.weights:
            nn.init.xavier_uniform_(w.view(1, -1))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            # x @ w is a per row scalar, broadcast back over x0.
            xw = (x * w).sum(dim=1, keepdim=True)
            x = x0 * xw + b + x
        return x


class DCNModule(nn.Module):
    """Parallel cross network and deep tower with a shared input vector.

    forward returns raw logits with no sigmoid.
    """

    def __init__(
        self,
        meta: FeatureMeta,
        embed_dim: int,
        cross_layers: int,
        hidden,
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fields = meta.n_embed_fields

        self.embedding = EmbeddingLayer(meta.embed_vocab_sizes(), embed_dim)

        in_dim = self.n_fields * embed_dim + meta.n_numerical
        self.cross = CrossNetwork(in_dim, cross_layers)
        self.mlp = MLP(in_dim, list(hidden), dropout=dropout)
        self.head = nn.Linear(in_dim + self.mlp.out_dim, 1)

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        emb, _ = self.embedding(cat)
        flat = emb.reshape(emb.size(0), self.n_fields * self.embed_dim)
        x0 = torch.cat([flat, numerical], dim=1)

        cross_out = self.cross(x0)
        deep_out = self.mlp(x0)
        combined = torch.cat([cross_out, deep_out], dim=1)

        return self.head(combined).squeeze(-1)


class DCNModel(BaseModel):
    """BaseModel wrapper that trains a DCNModule through the shared trainer."""

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
    ) -> "DCNModel":
        self.meta = meta
        self.embed_dim = config["embed_dim"]
        self.module = DCNModule(
            meta,
            self.embed_dim,
            config["cross_layers"],
            config["hidden"],
            config["dropout"],
        )
        self.module = train_torch_model(self.module, train, val, meta, config)
        self.device = next(self.module.parameters()).device
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        return predict_torch(self.module, data, self.device)

    def get_name(self) -> str:
        return "DCN"
