# Top-2 selection  (scores=scores_v1, lambda=0.7, delta=0.0001)

618 problems, K=10

| Method | nl4opt | mamo_easy | mamo_complex | complexor | industryor | optibench | optmath | Pooled |
|---|---|---|---|---|---|---|---|---|
| Acc@1 (SSC) | 91.0 | 93.0 | 53.0 | 50.0 | 35.0 | 56.0 | 49.0 | 62.5 |
| Random c2 given c1 | 91.3 | 93.3 | 55.0 | 54.3 | 36.7 | 59.8 | 55.4 | 64.9 |
| Random pair | 90.5 | 93.2 | 51.5 | 54.4 | 36.1 | 60.2 | 56.5 | 64.4 |
| Top-2 naive | 91.0 | 93.0 | 53.0 | 50.0 | 36.0 | 57.0 | 52.0 | 63.3 |
| MMR | 93.0 | 93.0 | 57.0 | 55.6 | 36.0 | 61.0 | 57.0 | 65.9 |
| Answer diversity | 91.0 | 94.0 | 56.0 | 55.6 | 38.0 | 58.0 | 58.0 | 65.5 |
| Hybrid (AnsDiv + MMR) | 93.0 | 94.0 | 59.0 | 55.6 | 39.0 | 62.0 | 61.0 | 67.6 |
| Pool coverage | 93.0 | 94.0 | 60.0 | 55.6 | 42.0 | 68.0 | 75.0 | 71.5 |

## Lambda sweep (pooled Acc@2)
| Rule | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MMR | 66.7 | 66.7 | 66.7 | 66.5 | 66.2 | 66.0 | 65.9 | 65.9 | 65.9 | 65.7 | 65.9 | 66.2 | 66.0 | 66.0 | 65.9 | 65.7 | 65.7 | 65.5 | 64.9 | 64.4 | 63.3 |
| Hybrid | 67.8 | 68.0 | 68.0 | 68.0 | 67.8 | 67.6 | 67.6 | 67.6 | 67.8 | 67.6 | 67.6 | 67.8 | 67.8 | 67.8 | 67.6 | 67.5 | 67.5 | 67.3 | 67.0 | 66.7 | 65.5 |

## Delta sweep (pooled Acc@2)
| Rule | 1e-06 | 0.001 | 0.01 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| Answer diversity | 65.5 | 65.5 | 65.7 | 65.9 | 66.2 |
| Hybrid | 67.6 | 67.6 | 67.5 | 67.3 | 67.5 |


## Diagnostics (pooled)
- recoverable problems (c1 wrong, pool holds a correct program): 56
- median SSC rank of the best correct candidate on those: 5.0
- mean number of correct candidates in those pools: 2.46
- naive c2 lands in c1's own clone group: 25 of 56
- c1 has no valid relaxation (answer diversity cannot fire): 0 of 56
- answer-diverse set empty, so Hybrid falls back to MMR: 20 of 56
- peers whose relaxed objective agrees with c1's within delta: 85.6% of 5229

