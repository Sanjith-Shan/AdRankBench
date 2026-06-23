# AdRankBench

### CTR Prediction Evaluation Framework for Ad Ranking Models

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

AdRankBench trains and evaluates five click through rate prediction architectures (Logistic Regression, FM, DeepFM, DCN, DNN) on the Criteo Display Advertising Challenge dataset. The framework implements production grade feature engineering with log transforms, hash encoding, frequency encoding, and explicit feature crosses. All models are evaluated with AUC, logloss, normalized entropy, group AUC, and calibration analysis using temporal train and test splits to prevent data leakage.

The project is built around the core problems of ad ranking systems. It covers ranking and retrieval, query understanding and relevance through a two tower DSSM model, probability calibration for ad pricing, and budget pacing through a feedback controller simulator. Each piece maps to a concrete skill that ad ranking and search ads teams care about.

## Results

The table below comes from a run on 100000 synthetic rows with seed 42. Reproduce it with `python scripts/run_benchmark.py --sample-size 100000`. The same table is mirrored into `results/benchmark_report.md`. Models are sorted by test AUC.

| Model | AUC | LogLoss | NE | RelaImpr | GAUC | ECE | Train s | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | 0.7789 | 0.4722 | 0.8141 | 0.1859 | 0.7646 | 0.0196 | 8.2 | 21.6M |
| DCN | 0.7715 | 0.5046 | 0.8702 | 0.1298 | 0.7581 | 0.0678 | 9.6 | 21.6M |
| DNN | 0.7708 | 0.4879 | 0.8412 | 0.1588 | 0.7571 | 0.0392 | 8.1 | 21.9M |
| FM | 0.7419 | 0.5024 | 0.8662 | 0.1338 | 0.7246 | 0.0174 | 13.4 | 21.4M |
| LogisticRegression | 0.6808 | 0.5365 | 0.9251 | 0.0749 | 0.6832 | 0.0102 | 0.2 | 53 |

DeepFM, DCN, DNN, and FM all beat the logistic regression baseline. DeepFM leads at 0.7789 AUC against 0.6808 for logistic regression, a lift of about 0.098 AUC and a drop in normalized entropy from 0.925 to 0.814. The synthetic data carries deliberate pairwise categorical interactions that a linear model cannot represent from raw frequency features, so the interaction aware models pull ahead. Logistic regression stays the best calibrated model by ECE, which is a useful reminder that ranking power and calibration are different axes. See the methodology notes below and the dedicated document in `docs/METHODOLOGY.md` for the full reasoning.

## Architecture

AdRankBench is a two stage pipeline. The first stage is feature engineering. Raw rows flow through a temporal split and then through a fit and transform feature pipeline that produces standardized numerical features, hash encoded categorical features, frequency encodings, and second order feature crosses. The second stage is model training and evaluation. The featurized splits are handed to each model, the trainer runs an early stopping loop, and the evaluation harness computes ranking and calibration metrics on the held out test split.

This mirrors how production ad ranking works. A retrieval stage built on something like BM25 or approximate nearest neighbor search narrows millions of candidates down to a shortlist. A ranking model in the FM or DeepFM or DCN family then scores that shortlist with rich feature interactions. AdRankBench focuses on the ranking stage and adds a separate two tower module that demonstrates the retrieval and relevance side.

## Models

- Logistic Regression. The linear baseline. It uses standardized numerical features plus frequency encoded categoricals. Every other model is expected to beat it.
- FM (Factorization Machine). Adds explicit second order feature interactions through low rank embeddings using the efficient O(kn) interaction formulation.
- DeepFM. Combines the FM component with a deep network over a shared embedding table so it captures both low order and high order interactions.
- DCN (Deep and Cross Network). Pairs an explicit cross network with a deep tower to compare explicit against implicit interaction modeling.
- DNN. A plain feedforward network over embeddings and numerical features. This is the throw a neural net at it baseline.

## Feature Engineering

Numerical features in ad data follow heavy tailed power law distributions, so the numerical pipeline clips negatives to zero, applies a log1p transform, and standardizes with statistics fit on train only. Missing values are filled with zero and a binary is_missing indicator is added per column, which turns missingness into its own signal rather than discarding it. The output width is 26, made of 13 transformed values and 13 indicators.

Categorical features have very high cardinality, so one hot encoding is not an option. The pipeline uses stable md5 based hash encoding into a fixed bucket space per field, which keeps memory bounded and stays deterministic across processes. It also produces a frequency encoding where each value is replaced by its normalized train frequency, which the logistic baseline consumes. Rare categories below a minimum count collapse to a shared rare token before hashing to reduce noise from one off values.

Feature crosses encode explicit second order interactions. The cross generator picks the top categorical columns by frequency variance, forms all pairwise combinations, and hashes each crossed value into its own bucket space. This is what production ad ranking systems do by hand, and it gives even the simpler models access to interaction signal.

## Evaluation

Every model is scored on the held out test split with several metrics, because no single number captures ad ranking quality.

- AUC. Area under the ROC curve. It measures pure ranking quality across the population.
- Logloss. Binary cross entropy. It is calibration aware and punishes confident wrong probabilities.
- Normalized Entropy (NE). Logloss divided by the entropy of the base click rate. A value below 1 means the model beats the constant base rate predictor. NE is robust to the overall click rate of the traffic, which is why it is the standard metric in the ad CTR literature.
- Relative Improvement (RelaImpr). The fractional reduction in cross entropy against a constant base rate predictor.
- GAUC (Group AUC). AUC computed within each impression group and averaged with impression weights. This is the production relevant view because a real auction ranks ads inside one user request or one query, not across the whole dataset. In real systems the group key is a user id or a query id. AdRankBench synthesizes impression groups for the test set so GAUC is computable offline.

Calibration is treated as a first class concern. The harness builds reliability curves by binning predictions into equal width buckets and comparing mean predicted probability against the observed click fraction. It also reports the Expected Calibration Error and overlays every model against the perfect calibration diagonal in a saved PNG. Calibration matters for ad pricing because bids and pacing decisions multiply the predicted click probability by a value, so a systematic bias in the probability turns directly into mispriced auctions. A model can rank well and still be miscalibrated, which is why calibration is reported alongside AUC.

## Search Ads Relevance

The relevance module is a two tower DSSM style semantic matching model for query to ad relevance. It lives in `src/relevance/` and runs through `scripts/run_relevance.py`. This is the retrieval and query understanding side of ad ranking, where the job is to match a short noisy query to relevant ad creatives.

Text is encoded with letter trigram word hashing, the original DSSM technique. Each token is wrapped with a boundary marker and broken into character n grams, and each n gram is hashed into a fixed bucket space to form a sparse bag of n grams vector. This sidesteps the open vocabulary problem and needs no external tokenizer or pretrained weights, so the whole module stays tiny and fast on cpu and brings no heavy dependencies.

The model has a separate query tower and ad tower. Each tower is a small MLP that maps the word hash vector into a shared embedding space. The relevance score is the cosine similarity of the two embeddings scaled by a temperature, which is the standard DSSM scoring choice and keeps the score bounded and easy to calibrate. Training is pointwise binary cross entropy on labeled query and ad pairs. The synthetic data hides a topic match signal, where a query and an ad are relevant when they share a hidden topic, so the model has to recover the topic from shared surface terms. Evaluation ranks the positive ad against sampled negatives per query and reports Recall@1, Recall@5, MRR, and NDCG@5. This demonstrates retrieval, relevance, and NLP query understanding in one compact module.

## Budget Pacing

The pacing module is a budget pacing simulator with a feedback controller. It lives in `src/pacing/` and runs through `scripts/run_pacing.py`. This maps to ads pacing and traffic control, where the job is to spend a daily campaign budget smoothly across the day rather than dumping it all in the morning.

The simulator synthesizes a realistic diurnal traffic curve where demand is low overnight, rises through the morning, and peaks in the early evening. It then replays a pacer against that curve slot by slot. Three pacers are compared. The PIDPacer uses a proportional integral derivative feedback loop that tracks an ideal spend trajectory and corrects when it drifts behind or ahead of plan. The AsapPacer spends as fast as possible and exhausts budget early, which is the failure mode smooth pacing exists to prevent. The ThrottlePacer is a simple proportional baseline that targets a flat per slot spend without feedback.

Each run reports budget utilization and a smoothness score, defined as the root mean squared error between the realized cumulative spend curve and the ideal traffic proportional curve. The script plots cumulative spend over the day against the ideal pacing line and saves it as a PNG. Smooth pacing matters because it avoids early budget exhaustion, keeps cost per mille stable, and gives the campaign broad time of day coverage instead of a narrow morning burst.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --sample-size 100000
```

The benchmark falls back to synthetic data automatically when no real Criteo file is present, so it runs end to end with no downloads. To use real data place a Criteo TSV or CSV at `data/criteo.csv` or pass `--data-path`. The download helper at `scripts/download_data.sh` tries to fetch a public Criteo sample and prints a message when it cannot, after which the benchmark uses synthetic data.

Run the role aligned extensions on their own.

```bash
python scripts/run_relevance.py
python scripts/run_pacing.py
```

All three entry points seed everything with seed 42 for reproducibility.

## Methodology Notes

AdRankBench uses a temporal positional split, not a random split. The data is time ordered, so train is the first 80 percent of rows, validation is the next 10 percent, and test is the last 10 percent. A random split would leak future impressions into training and inflate metrics in a way that never holds in production. Splitting by time matches how the model would actually be deployed, where it always predicts forward.

Normalized entropy and GAUC matter more than raw AUC for ad systems. Raw AUC measures population level ranking and is insensitive to the absolute click probability and to within request ordering. Ad pricing needs calibrated probabilities, which is what NE and logloss track, and ad auctions rank ads inside a single user request, which is what GAUC tracks. A model can win on raw AUC and still be the wrong choice for an auction. For the full reasoning on the split, on why NE and GAUC are the right headline metrics, and on the synthetic data design, see `docs/METHODOLOGY.md`.
