"""Ranking metrics for the query to ad relevance task.

These are the standard retrieval metrics used to judge how well the model ranks a
relevant ad above irrelevant ones. A ranked_labels input is a list of relevance
labels ordered from the top ranked candidate to the bottom. A label of 1 means
relevant and 0 means not relevant. For a single query with one positive among
sampled negatives the ranked_labels list has exactly one entry equal to 1.

All functions are pure numpy and have no model dependency so they stay fast and
easy to test.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def recall_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Fraction of relevant items that appear in the top k.

    For a single positive query this is 1.0 when the positive is within the top k
    and 0.0 otherwise. With multiple relevants it is the count of relevants in the
    top k divided by the total number of relevants. Returns 0.0 when there are no
    relevant items.
    """
    labels = np.asarray(ranked_labels)
    total_relevant = float(labels.sum())
    if total_relevant == 0.0:
        return 0.0
    top = labels[:k]
    return float(top.sum()) / total_relevant


def mrr(ranked_lists: Sequence[Sequence[int]]) -> float:
    """Mean reciprocal rank over a list of ranked label lists.

    For each ranked list the reciprocal rank is 1 divided by the one based
    position of the first relevant item, or 0.0 when no relevant item is present.
    The mean is taken over all lists. An empty input returns 0.0.
    """
    if len(ranked_lists) == 0:
        return 0.0
    total = 0.0
    for labels in ranked_lists:
        arr = np.asarray(labels)
        hits = np.nonzero(arr > 0)[0]
        if hits.size > 0:
            first = int(hits[0]) + 1
            total += 1.0 / first
    return total / len(ranked_lists)


def _dcg(labels: np.ndarray, k: int) -> float:
    """Discounted cumulative gain of the top k labels.

    Gain is the raw label and the discount is 1 divided by log2 of the one based
    rank plus one.
    """
    top = labels[:k].astype(np.float64)
    if top.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, top.size + 2))
    return float(np.sum(top * discounts))


def ndcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Normalized discounted cumulative gain at k.

    The DCG of the given ranking is divided by the ideal DCG, which is the DCG of
    the same labels sorted in descending order. Returns 0.0 when the ideal DCG is
    zero, which happens when there are no relevant items.
    """
    labels = np.asarray(ranked_labels)
    ideal = np.sort(labels)[::-1]
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(labels, k) / idcg
