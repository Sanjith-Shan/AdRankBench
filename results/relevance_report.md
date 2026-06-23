# Query to Ad Relevance Report

This report evaluates a two tower DSSM style semantic matching model on a synthetic search ads dataset. The model encodes the query and the ad with separate towers over letter n gram word hashing and scores a pair by the temperature scaled cosine similarity of the two embeddings.

The hidden relevance signal is a topic match. A query and its positive ad share a topic while the negatives come from other topics. The model never sees the topic id and has to recover the match from shared surface n grams. Each test query ranks its one positive ad against several sampled negatives.

## Setup

Training pairs. 3200
Test queries. 400
Training time seconds. 12.42

## Retrieval Metrics

| Metric | Value |
| --- | --- |
| Recall@1 | 0.7625 |
| Recall@5 | 0.9950 |
| MRR | 0.8601 |
| NDCG@5 | 0.8938 |

## Reading The Numbers

Recall at 1 is the share of queries where the positive ad is ranked first. Recall at 5 is the share where it lands in the top five. MRR rewards placing the positive near the top and NDCG at 5 adds a position discount. Higher is better for all four. A model that learned the topic match well ranks the positive above the off topic negatives most of the time, so these numbers should sit well above the random baseline.
