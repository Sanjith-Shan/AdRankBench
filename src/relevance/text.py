"""Letter trigram word hashing for the two tower relevance model.

This is the text encoding from the original DSSM paper. Each token is wrapped
with a boundary marker and broken into character n grams. Each n gram is hashed
into a fixed size bucket space and the counts form a sparse bag of n grams
vector. This sidesteps the open vocabulary problem and needs no embeddings or
external tokenizer, so it stays tiny and fast on cpu.

The hashing uses md5 so that the same n gram maps to the same bucket across
runs and processes. The Python builtin hash is randomized per process and must
not be used here.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

# Boundary marker wrapped around every token so that prefixes and suffixes get
# their own n grams. The original DSSM uses '#word#'.
BOUNDARY = "#"

# Default bucket count for the word hash space. Large enough to keep collisions
# rare on the small synthetic vocabulary while staying cheap on cpu.
DEFAULT_BUCKETS = 20000

# Default character n gram size. Trigrams are the DSSM default.
DEFAULT_NGRAM = 3


def _stable_bucket(ngram: str, n_buckets: int) -> int:
    """Map a string n gram to a bucket index deterministically.

    Uses md5 of the utf-8 bytes so the result is identical across processes.
    """
    digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


def _token_ngrams(token: str, ngram: int) -> List[str]:
    """Return the list of character n grams for a single wrapped token.

    The token is wrapped with the boundary marker on both sides. If the wrapped
    token is shorter than the n gram size the whole wrapped token is emitted as
    one n gram so short tokens still contribute a feature.
    """
    wrapped = f"{BOUNDARY}{token}{BOUNDARY}"
    if len(wrapped) < ngram:
        return [wrapped]
    return [wrapped[i : i + ngram] for i in range(len(wrapped) - ngram + 1)]


def word_hash(
    text: str,
    n_buckets: int = DEFAULT_BUCKETS,
    ngram: int = DEFAULT_NGRAM,
) -> np.ndarray:
    """Encode text as a normalized letter n gram bag of words vector.

    The text is lowercased and split on whitespace. Each token is wrapped with a
    boundary marker and broken into character n grams. Every n gram is hashed
    into n_buckets and the counts are accumulated. The vector is l2 normalized so
    that cosine similarity downstream is well behaved. An empty or all
    whitespace text returns an all zero vector.

    Parameters
    ----------
    text : str
        The raw query or ad text.
    n_buckets : int
        Size of the hash bucket space and the output dimension.
    ngram : int
        Character n gram size.

    Returns
    -------
    np.ndarray
        A (n_buckets,) float32 vector.
    """
    vec = np.zeros(n_buckets, dtype=np.float32)
    if text is None:
        return vec
    tokens = text.lower().split()
    for token in tokens:
        for gram in _token_ngrams(token, ngram):
            vec[_stable_bucket(gram, n_buckets)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def batch_word_hash(
    texts: List[str],
    n_buckets: int = DEFAULT_BUCKETS,
    ngram: int = DEFAULT_NGRAM,
) -> np.ndarray:
    """Encode a list of texts into a (N, n_buckets) float32 matrix.

    Each row is the word hash of the corresponding text. This is a thin loop over
    word_hash and stays cheap because the vocabulary and bucket count are small.
    """
    n = len(texts)
    out = np.zeros((n, n_buckets), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = word_hash(text, n_buckets=n_buckets, ngram=ngram)
    return out
