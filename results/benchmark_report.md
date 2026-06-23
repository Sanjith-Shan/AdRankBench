# AdRankBench Benchmark Report

This report compares click through rate models on a shared featurized dataset. Models are ranked by test AUC. All numbers come from the held out test split.

## Results

| Model | AUC | LogLoss | NE | RelaImpr | GAUC | ECE | Train s | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | 0.7872 | 0.4512 | 0.8130 | 0.1870 | 0.7873 | 0.0100 | 132.0263 | 21616509 |
| DCN | 0.7871 | 0.4508 | 0.8124 | 0.1876 | 0.7871 | 0.0052 | 140.1897 | 21620695 |
| DNN | 0.7863 | 0.4518 | 0.8141 | 0.1859 | 0.7864 | 0.0107 | 125.8154 | 21894881 |
| FM | 0.7828 | 0.4553 | 0.8205 | 0.1795 | 0.7829 | 0.0091 | 145.7100 | 21420028 |
| LogisticRegression | 0.7177 | 0.4983 | 0.8979 | 0.1021 | 0.7177 | 0.0057 | 11.4519 | 53 |

## Calibration

The reliability curves below overlay each model against the perfect calibration diagonal. A curve that hugs the diagonal is well calibrated.

![Calibration curves](calibration.png)

## Feature Importance

Top logistic regression coefficients by absolute magnitude. A larger magnitude means the feature pushes the predicted click probability more strongly. The sign shows the direction of that push.

| Feature | Coefficient |
| --- | --- |
| freq_C2 | 2.1069 |
| freq_C25 | -1.8654 |
| freq_C26 | 1.3529 |
| freq_C24 | -1.2621 |
| freq_C17 | 0.9193 |
| num_17 | -0.8946 |
| num_18 | -0.8522 |
| freq_C4 | -0.8510 |
| freq_C20 | 0.8260 |
| freq_C14 | -0.6730 |
| freq_C23 | 0.6315 |
| freq_C3 | -0.4029 |
| freq_C18 | -0.3758 |
| freq_C12 | -0.3689 |
| freq_C16 | -0.3686 |


## Key Observations

The strongest model by test AUC is DeepFM. It ranks impressions better than the logistic regression baseline which scored 0.7177 AUC. Ranking quality is what an ad auction cares about most because the auction orders candidates before pricing them.

The factorization machine family beats plain logistic regression because the synthetic clicks depend on pairwise interactions between categorical fields. Logistic regression only learns linear weights on single features, so it cannot capture the lift that appears when two category groups co occur. The FM second order term and the deep towers model those interactions directly, which is why they pull ahead on AUC and normalized entropy.

Calibration and normalized entropy round out the picture. A model can rank well and still be biased in absolute probability, so the calibration curve and the expected calibration error matter for bidding and pacing. Group AUC reflects within request ranking quality, which is closer to the live auction than dataset wide AUC. Reading these metrics together gives a fair comparison of the architectures.
