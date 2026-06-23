#!/usr/bin/env python
"""Train and evaluate the two tower query to ad relevance model.

This script trains the DSSM style two tower model on synthetic query to ad data
and evaluates it as a small retrieval task. For every test query it ranks the one
true positive ad against several sampled negatives and reports Recall at 1,
Recall at 5, mean reciprocal rank, and normalized discounted cumulative gain at 5.

The relevance signal in the synthetic data is a hidden topic match. A query and
its positive ad share a topic, and the negatives come from other topics. The
model never sees the topic id. It only sees raw text encoded by letter n gram
word hashing, so it has to recover the topic match from shared surface n grams.

The results are printed and also written to results/relevance_report.md. The
whole run is small and fast on cpu.

Run from the repository root.
    python scripts/run_relevance.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from src.relevance.data import generate_query_ad_pairs
from src.relevance.metrics import mrr, ndcg_at_k, recall_at_k
from src.relevance.two_tower import TwoTowerModel

SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the relevance run."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the two tower relevance model."
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=2000,
        help="Number of synthetic queries to generate before the split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for the two tower model.",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where the relevance report is written.",
    )
    return parser.parse_args()


def ranked_labels_for_query(model: TwoTowerModel, item: dict) -> np.ndarray:
    """Rank the positive ad against the negatives and return ordered labels.

    The positive ad is concatenated with the negative ads into one candidate
    list. The model scores every candidate against the query. The candidates are
    sorted by score descending and the relevance labels are read off in that
    order. The positive has label 1 and every negative has label 0. Ties keep a
    stable order which is conservative for the positive.
    """
    candidates = [item["positive"]] + list(item["negatives"])
    labels = np.array([1] + [0] * len(item["negatives"]), dtype=np.int64)

    scores = np.asarray(model.rank(item["query"], candidates), dtype=np.float64)
    # Sort by score descending. Stable sort on the negated score keeps the input
    # order for ties which does not give the positive an unfair boost.
    order = np.argsort(-scores, kind="stable")
    return labels[order]


def evaluate(model: TwoTowerModel, ranking: list) -> dict:
    """Compute retrieval metrics over every test query.

    Returns a dict with recall_at_1, recall_at_5, mrr, and ndcg_at_5 averaged
    over the test queries.
    """
    ranked_lists = [ranked_labels_for_query(model, item) for item in ranking]

    r1 = float(np.mean([recall_at_k(r, 1) for r in ranked_lists])) if ranked_lists else 0.0
    r5 = float(np.mean([recall_at_k(r, 5) for r in ranked_lists])) if ranked_lists else 0.0
    mrr_score = float(mrr(ranked_lists))
    ndcg5 = float(np.mean([ndcg_at_k(r, 5) for r in ranked_lists])) if ranked_lists else 0.0

    return {
        "recall_at_1": r1,
        "recall_at_5": r5,
        "mrr": mrr_score,
        "ndcg_at_5": ndcg5,
        "n_queries": len(ranking),
    }


def write_report(metrics: dict, out_dir: str, train_time: float, n_train: int) -> str:
    """Write the relevance markdown report and return its path.

    The prose follows the project style rules. No em dashes. No semicolons. No
    mid sentence colons.
    """
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "relevance_report.md")

    lines = []
    lines.append("# Query to Ad Relevance Report")
    lines.append("")
    lines.append(
        "This report evaluates a two tower DSSM style semantic matching model on "
        "a synthetic search ads dataset. The model encodes the query and the ad "
        "with separate towers over letter n gram word hashing and scores a pair "
        "by the temperature scaled cosine similarity of the two embeddings."
    )
    lines.append("")
    lines.append(
        "The hidden relevance signal is a topic match. A query and its positive "
        "ad share a topic while the negatives come from other topics. The model "
        "never sees the topic id and has to recover the match from shared surface "
        "n grams. Each test query ranks its one positive ad against several "
        "sampled negatives."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"Training pairs. {n_train}")
    lines.append(f"Test queries. {metrics['n_queries']}")
    lines.append(f"Training time seconds. {train_time:.2f}")
    lines.append("")
    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Recall@1 | {metrics['recall_at_1']:.4f} |")
    lines.append(f"| Recall@5 | {metrics['recall_at_5']:.4f} |")
    lines.append(f"| MRR | {metrics['mrr']:.4f} |")
    lines.append(f"| NDCG@5 | {metrics['ndcg_at_5']:.4f} |")
    lines.append("")
    lines.append("## Reading The Numbers")
    lines.append("")
    lines.append(
        "Recall at 1 is the share of queries where the positive ad is ranked "
        "first. Recall at 5 is the share where it lands in the top five. MRR "
        "rewards placing the positive near the top and NDCG at 5 adds a position "
        "discount. Higher is better for all four. A model that learned the topic "
        "match well ranks the positive above the off topic negatives most of the "
        "time, so these numbers should sit well above the random baseline."
    )
    lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return report_path


def main() -> None:
    """Run the relevance experiment end to end."""
    args = parse_args()

    data = generate_query_ad_pairs(n_queries=args.n_queries, seed=SEED)
    train_pairs = data["train_pairs"]
    ranking = data["ranking"]
    print(
        f"generated {len(train_pairs)} train pairs and "
        f"{len(ranking)} test queries."
    )

    model = TwoTowerModel(epochs=args.epochs, seed=SEED)
    start = time.time()
    model.fit(train_pairs)
    train_time = time.time() - start
    print(f"trained two tower model in {train_time:.2f} seconds.")

    metrics = evaluate(model, ranking)
    print("\nretrieval metrics on the test queries.")
    print(f"  Recall@1 {metrics['recall_at_1']:.4f}")
    print(f"  Recall@5 {metrics['recall_at_5']:.4f}")
    print(f"  MRR      {metrics['mrr']:.4f}")
    print(f"  NDCG@5   {metrics['ndcg_at_5']:.4f}")

    report_path = write_report(metrics, args.output, train_time, len(train_pairs))
    print(f"\nwrote {report_path}.")
    print("relevance run complete.")


if __name__ == "__main__":
    main()
