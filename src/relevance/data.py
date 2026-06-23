"""Synthetic query to ad relevance data with a hidden topic match signal.

This builds a tiny search ads style dataset that needs no downloads. There are a
handful of hidden topics. Each topic owns a small vocabulary of topical terms. A
query is a few terms sampled from one topic. An ad title is a few terms sampled
mostly from one topic with a little cross topic leakage to add realism. A
(query, ad) pair is labeled relevant when the query topic and the ad topic match.
A small amount of label noise is injected so the problem is not perfectly
separable.

The hidden topic is the only thing that drives relevance. The model never sees
the topic id. It only sees the raw text, so it has to recover the topic match
from the shared topical terms. The letter n gram word hashing in text.py turns
the shared terms into overlapping hash buckets, which is what lets the two tower
model learn a high cosine similarity for same topic pairs.

Everything is deterministic given the seed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Hidden topics and their term vocabularies. The terms are made up but topical so
# that within topic queries and ads share surface n grams.
TOPIC_VOCAB: Dict[str, List[str]] = {
    "travel": [
        "flight",
        "hotel",
        "vacation",
        "cruise",
        "resort",
        "airfare",
        "booking",
        "getaway",
        "beach",
        "itinerary",
    ],
    "finance": [
        "loan",
        "mortgage",
        "credit",
        "savings",
        "investment",
        "insurance",
        "refinance",
        "banking",
        "retirement",
        "brokerage",
    ],
    "tech": [
        "laptop",
        "smartphone",
        "software",
        "gpu",
        "cloud",
        "monitor",
        "keyboard",
        "router",
        "headphones",
        "tablet",
    ],
    "fashion": [
        "sneakers",
        "dress",
        "handbag",
        "jacket",
        "watch",
        "sunglasses",
        "jeans",
        "boots",
        "scarf",
        "perfume",
    ],
    "fitness": [
        "treadmill",
        "protein",
        "dumbbell",
        "yoga",
        "running",
        "workout",
        "cycling",
        "nutrition",
        "gym",
        "stretching",
    ],
}

TOPICS: List[str] = list(TOPIC_VOCAB.keys())


def _sample_terms(
    rng: np.random.Generator,
    topic: str,
    k: int,
    leak_prob: float = 0.0,
) -> List[str]:
    """Sample k terms for a piece of text from a topic.

    With probability leak_prob each term is drawn from a random other topic
    instead of the chosen one. This models the realistic case where ad copy
    sometimes mixes in off topic words.
    """
    terms: List[str] = []
    for _ in range(k):
        if leak_prob > 0.0 and rng.random() < leak_prob:
            other = TOPICS[rng.integers(0, len(TOPICS))]
            vocab = TOPIC_VOCAB[other]
        else:
            vocab = TOPIC_VOCAB[topic]
        terms.append(vocab[rng.integers(0, len(vocab))])
    return terms


def _make_text(rng: np.random.Generator, topic: str, k: int, leak_prob: float) -> str:
    return " ".join(_sample_terms(rng, topic, k, leak_prob))


def generate_query_ad_pairs(n_queries: int = 2000, seed: int = 42) -> dict:
    """Synthesize a small query to ad relevance dataset.

    Each query gets one matching positive ad from the same hidden topic and a set
    of sampled negative ads drawn from other topics. Pointwise labeled pairs are
    also produced for training, balanced between relevant and not relevant.

    Parameters
    ----------
    n_queries : int
        Number of distinct queries to generate before the train test split.
    seed : int
        Reproducibility seed.

    Returns
    -------
    dict
        A dictionary with the following keys.
        train_pairs and test_pairs are lists of (query_text, ad_text, label)
        tuples for pointwise training and evaluation.
        ranking is a list of dicts, one per test query, each with keys
        query (str), positive (str), negatives (list[str]). The positive ad
        shares the query topic and the negatives are from other topics. This is
        used to rank the positive against the negatives for retrieval metrics.
        topics is the list of hidden topic names for reference.
    """
    rng = np.random.default_rng(seed)

    # Number of negatives sampled per query for the ranking eval.
    n_negatives = 9
    # Probability that an ad mixes in an off topic term.
    ad_leak_prob = 0.2
    # Probability of flipping a pointwise label as injected noise.
    label_noise = 0.1
    # Fraction of queries used for the test split.
    test_frac = 0.2

    queries: List[dict] = []
    for _ in range(n_queries):
        topic = TOPICS[rng.integers(0, len(TOPICS))]
        q_len = int(rng.integers(2, 4))  # 2 or 3 terms per query
        query_text = _make_text(rng, topic, q_len, leak_prob=0.0)

        # Positive ad shares the topic.
        pos_len = int(rng.integers(3, 6))  # 3 to 5 terms per ad title
        positive = _make_text(rng, topic, pos_len, leak_prob=ad_leak_prob)

        # Negative ads come from other topics.
        other_topics = [t for t in TOPICS if t != topic]
        negatives: List[str] = []
        for _ in range(n_negatives):
            neg_topic = other_topics[rng.integers(0, len(other_topics))]
            neg_len = int(rng.integers(3, 6))
            negatives.append(_make_text(rng, neg_topic, neg_len, leak_prob=ad_leak_prob))

        queries.append(
            {
                "topic": topic,
                "query": query_text,
                "positive": positive,
                "negatives": negatives,
            }
        )

    # Positional train test split so the same seed always yields the same split.
    n_test = int(round(n_queries * test_frac))
    n_train = n_queries - n_test
    train_queries = queries[:n_train]
    test_queries = queries[n_train:]

    def _to_pointwise(query_set: List[dict]) -> List[Tuple[str, str, int]]:
        """Build balanced pointwise (query, ad, label) tuples with noise."""
        pairs: List[Tuple[str, str, int]] = []
        for q in query_set:
            # One positive pair.
            pos_label = 1
            if rng.random() < label_noise:
                pos_label = 0
            pairs.append((q["query"], q["positive"], pos_label))
            # One negative pair so the pointwise set is balanced.
            neg_ad = q["negatives"][rng.integers(0, len(q["negatives"]))]
            neg_label = 0
            if rng.random() < label_noise:
                neg_label = 1
            pairs.append((q["query"], neg_ad, neg_label))
        return pairs

    train_pairs = _to_pointwise(train_queries)
    test_pairs = _to_pointwise(test_queries)

    ranking = [
        {
            "query": q["query"],
            "positive": q["positive"],
            "negatives": list(q["negatives"]),
        }
        for q in test_queries
    ]

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "ranking": ranking,
        "topics": list(TOPICS),
    }
