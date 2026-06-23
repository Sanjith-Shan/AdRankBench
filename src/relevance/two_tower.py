"""A two tower DSSM style semantic matching model for query to ad relevance.

The query tower and the ad tower are separate small MLPs that map a letter n gram
word hash vector into a shared embedding space. The relevance score is the cosine
similarity of the two embeddings scaled by a temperature. Cosine plus temperature
is the standard DSSM scoring choice and keeps the score bounded and easy to
calibrate.

The model is intentionally tiny so it trains in a few seconds on cpu. Training
uses pointwise BCE on (query, ad, label) pairs with the temperature scaled cosine
as the logit. No huggingface and no heavy dependencies, only torch and numpy.
"""

from __future__ import annotations

from typing import List

import numpy as np

import torch
import torch.nn as nn

from src.relevance.text import DEFAULT_BUCKETS, DEFAULT_NGRAM, batch_word_hash


def _set_seed(seed: int = 42) -> None:
    """Seed numpy and torch for reproducible weight init and shuffling."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tower(in_buckets: int, hidden: List[int], embed_dim: int) -> nn.Module:
    """Build a single tower MLP that ends in an embed_dim projection.

    Each hidden layer is Linear then ReLU. The final layer projects to embed_dim
    with no activation so the output can be l2 normalized for cosine scoring.
    """
    layers: List[nn.Module] = []
    prev = in_buckets
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, embed_dim))
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    """Two tower encoder with cosine similarity scoring.

    The query tower and the ad tower have the same shape but separate weights.
    Each maps a word hash vector to an embed_dim vector. The forward score is the
    temperature scaled cosine similarity, returned as a raw logit suitable for
    BCEWithLogitsLoss.
    """

    def __init__(
        self,
        in_buckets: int,
        embed_dim: int = 64,
        hidden: List[int] | None = None,
        temperature: float = 10.0,
        learn_temperature: bool = True,
    ):
        super().__init__()
        if hidden is None:
            hidden = [128]
        self.in_buckets = in_buckets
        self.embed_dim = embed_dim
        self.query_tower = _tower(in_buckets, hidden, embed_dim)
        self.ad_tower = _tower(in_buckets, hidden, embed_dim)
        # Temperature sharpens the cosine. A learned scalar lets the model pick
        # the right scale for BCE. It is kept positive by construction at use.
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)))

    def encode_query(self, q_vec: torch.Tensor) -> torch.Tensor:
        """Encode a batch of query word hash vectors to l2 normalized embeddings."""
        emb = self.query_tower(q_vec)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def encode_ad(self, a_vec: torch.Tensor) -> torch.Tensor:
        """Encode a batch of ad word hash vectors to l2 normalized embeddings."""
        emb = self.ad_tower(a_vec)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def forward(self, q_vec: torch.Tensor, a_vec: torch.Tensor) -> torch.Tensor:
        """Return temperature scaled cosine similarity logits, shape (B,)."""
        q = self.encode_query(q_vec)
        a = self.encode_ad(a_vec)
        cosine = (q * a).sum(dim=-1)
        return cosine * self.temperature


class TwoTowerModel:
    """Trainable wrapper around TwoTower for query to ad relevance.

    The wrapper owns the word hashing, the torch module, the training loop, and
    scoring. Text goes in as raw strings and the wrapper converts to word hash
    vectors. Training is pointwise BCE on labeled (query, ad) pairs. Scoring
    returns the sigmoid of the temperature scaled cosine so the output is a
    probability in [0, 1].
    """

    def __init__(
        self,
        n_buckets: int = DEFAULT_BUCKETS,
        ngram: int = DEFAULT_NGRAM,
        embed_dim: int = 64,
        hidden: List[int] | None = None,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 128,
        weight_decay: float = 1e-5,
        seed: int = 42,
    ):
        self.n_buckets = n_buckets
        self.ngram = ngram
        self.embed_dim = embed_dim
        self.hidden = hidden if hidden is not None else [128]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.module: TwoTower | None = None

    def _hash(self, texts: List[str]) -> np.ndarray:
        return batch_word_hash(texts, n_buckets=self.n_buckets, ngram=self.ngram)

    def fit(self, train_pairs: List[tuple]) -> "TwoTowerModel":
        """Train on a list of (query_text, ad_text, label) tuples.

        Uses pointwise BCE with the temperature scaled cosine as the logit. The
        whole training set is hashed once up front because the synthetic data is
        small. Returns self.
        """
        _set_seed(self.seed)

        queries = [p[0] for p in train_pairs]
        ads = [p[1] for p in train_pairs]
        labels = np.array([float(p[2]) for p in train_pairs], dtype=np.float32)

        q_mat = torch.from_numpy(self._hash(queries))
        a_mat = torch.from_numpy(self._hash(ads))
        y = torch.from_numpy(labels)

        self.module = TwoTower(
            in_buckets=self.n_buckets,
            embed_dim=self.embed_dim,
            hidden=self.hidden,
        )
        self.module.train()
        optimizer = torch.optim.Adam(
            self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        n = q_mat.shape[0]
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                idx_t = torch.from_numpy(idx.astype(np.int64))
                qb = q_mat.index_select(0, idx_t)
                ab = a_mat.index_select(0, idx_t)
                yb = y.index_select(0, idx_t)
                optimizer.zero_grad()
                logits = self.module(qb, ab)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        self.module.eval()
        return self

    def score_pairs(self, pairs: List[tuple]) -> np.ndarray:
        """Return relevance probabilities for (query_text, ad_text[, label]) pairs.

        The label, if present in each tuple, is ignored. Output is a (N,) float32
        array of probabilities in [0, 1].
        """
        if self.module is None:
            raise RuntimeError("TwoTowerModel must be fit before scoring.")
        queries = [p[0] for p in pairs]
        ads = [p[1] for p in pairs]
        return self._score(queries, ads)

    def _score(self, queries: List[str], ads: List[str]) -> np.ndarray:
        q_mat = torch.from_numpy(self._hash(queries))
        a_mat = torch.from_numpy(self._hash(ads))
        with torch.no_grad():
            logits = self.module(q_mat, a_mat)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().astype(np.float32)

    def rank(self, query: str, ads: List[str]) -> np.ndarray:
        """Score one query against a list of candidate ads.

        Returns a (len(ads),) float32 array of relevance probabilities aligned to
        the input ad order. Higher means more relevant.
        """
        if self.module is None:
            raise RuntimeError("TwoTowerModel must be fit before ranking.")
        return self._score([query] * len(ads), ads)
