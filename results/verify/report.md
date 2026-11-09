# Verifier experiment (Sonnet 5.5, pointwise, solution-aware)

Benchmarks: mamo_complex, optmath; 200 problems x 10 candidates. Unparsed verdicts (counted as reject): 9.

## Verifier quality

| labels | scope | TPR | FPR | precision | AUROC(p) | ECE | no-correct pools with a false accept | clone clusters with mixed verdicts |
|---|---|---|---|---|---|---|---|---|
| orig | all | 0.87 | 0.45 | 0.62 | 0.79 | 0.194 | 56/65 | 95/234 |
| orig | mamo_complex | 0.98 | 0.60 | 0.58 | 0.81 | 0.244 | 38/40 | 34/112 |
| orig | optmath | 0.77 | 0.30 | 0.68 | 0.77 | 0.144 | 18/25 | 61/122 |
| sirl | all | 0.89 | 0.39 | 0.71 | 0.87 | 0.133 | 39/47 | 94/230 |
| sirl | mamo_complex | 0.99 | 0.51 | 0.74 | 0.94 | 0.126 | 21/22 | 33/108 |
| sirl | optmath | 0.77 | 0.30 | 0.68 | 0.77 | 0.144 | 18/25 | 61/122 |

## Policies

Baseline without verifier: return SSC top-1 always (0 calls).

### labels=orig, scope=all, accept threshold τ=0.7

| order | skip same-answer | budget | calls/problem | right % | wrong % | abstain % | accuracy when answered % |
|---|---|---|---|---|---|---|---|
| SSC top-1, no verifier | - | 0 | 0.00 | 51.0 | 49.0 | 0.0 | 51.0 |
| SSC | False | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| SSC | False | 3 | 1.71 | 46.5 | 25.0 | 28.5 | 65.0 |
| SSC | False | 10 | 3.29 | 51.5 | 31.5 | 17.0 | 62.0 |
| Cluster-first | False | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Cluster-first | False | 3 | 1.69 | 46.5 | 26.0 | 27.5 | 64.1 |
| Cluster-first | False | 10 | 3.19 | 51.5 | 31.5 | 17.0 | 62.0 |
| Cluster-first | True | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Cluster-first | True | 3 | 1.32 | 45.0 | 22.5 | 32.5 | 66.7 |
| Cluster-first | True | 10 | 1.62 | 45.5 | 22.5 | 32.0 | 66.9 |
| Hybrid λ=0.1 | False | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Hybrid λ=0.1 | False | 3 | 1.68 | 48.5 | 25.0 | 26.5 | 66.0 |
| Hybrid λ=0.1 | False | 10 | 3.12 | 52.0 | 31.0 | 17.0 | 62.7 |
| Hybrid λ=0.1 | True | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Hybrid λ=0.1 | True | 3 | 1.32 | 45.5 | 21.5 | 33.0 | 67.9 |
| Hybrid λ=0.1 | True | 10 | 1.62 | 45.5 | 22.0 | 32.5 | 67.4 |
| Hybrid λ=0.7 | False | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Hybrid λ=0.7 | False | 3 | 1.69 | 47.5 | 26.0 | 26.5 | 64.6 |
| Hybrid λ=0.7 | False | 10 | 3.18 | 51.5 | 31.5 | 17.0 | 62.0 |
| Hybrid λ=0.7 | True | 1 | 1.00 | 43.0 | 18.5 | 38.5 | 69.9 |
| Hybrid λ=0.7 | True | 3 | 1.33 | 45.0 | 21.5 | 33.5 | 67.7 |
| Hybrid λ=0.7 | True | 10 | 1.64 | 45.5 | 22.0 | 32.5 | 67.4 |
| Random | False | 1 | 1.00 | 37.9 | 17.6 | 44.4 | 68.3 |
| Random | False | 3 | 1.77 | 46.6 | 26.0 | 27.3 | 64.2 |
| Random | False | 10 | 3.28 | 52.0 | 31.0 | 17.0 | 62.7 |
| Random | True | 1 | 1.00 | 38.1 | 17.2 | 44.8 | 68.9 |
| Random | True | 3 | 1.42 | 43.9 | 21.2 | 34.9 | 67.4 |
| Random | True | 10 | 1.76 | 45.1 | 21.9 | 33.0 | 67.3 |

### labels=sirl, scope=all, accept threshold τ=0.7

| order | skip same-answer | budget | calls/problem | right % | wrong % | abstain % | accuracy when answered % |
|---|---|---|---|---|---|---|---|
| SSC top-1, no verifier | - | 0 | 0.00 | 57.4 | 42.6 | 0.0 | 57.4 |
| SSC | False | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| SSC | False | 3 | 1.69 | 54.3 | 17.8 | 27.9 | 75.4 |
| SSC | False | 10 | 3.25 | 60.9 | 22.3 | 16.8 | 73.2 |
| Cluster-first | False | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Cluster-first | False | 3 | 1.67 | 54.8 | 18.3 | 26.9 | 75.0 |
| Cluster-first | False | 10 | 3.13 | 60.9 | 22.3 | 16.8 | 73.2 |
| Cluster-first | True | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Cluster-first | True | 3 | 1.30 | 52.3 | 15.7 | 32.0 | 76.9 |
| Cluster-first | True | 10 | 1.56 | 52.8 | 15.7 | 31.5 | 77.0 |
| Hybrid λ=0.1 | False | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Hybrid λ=0.1 | False | 3 | 1.66 | 57.4 | 16.8 | 25.9 | 77.4 |
| Hybrid λ=0.1 | False | 10 | 3.08 | 61.4 | 21.8 | 16.8 | 73.8 |
| Hybrid λ=0.1 | True | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Hybrid λ=0.1 | True | 3 | 1.30 | 52.8 | 14.7 | 32.5 | 78.2 |
| Hybrid λ=0.1 | True | 10 | 1.56 | 52.8 | 15.2 | 32.0 | 77.6 |
| Hybrid λ=0.7 | False | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Hybrid λ=0.7 | False | 3 | 1.67 | 56.3 | 17.8 | 25.9 | 76.0 |
| Hybrid λ=0.7 | False | 10 | 3.12 | 60.9 | 22.3 | 16.8 | 73.2 |
| Hybrid λ=0.7 | True | 1 | 1.00 | 48.7 | 13.7 | 37.6 | 78.0 |
| Hybrid λ=0.7 | True | 3 | 1.31 | 52.3 | 14.7 | 33.0 | 78.0 |
| Hybrid λ=0.7 | True | 10 | 1.57 | 52.8 | 15.2 | 32.0 | 77.6 |
| Random | False | 1 | 1.00 | 43.9 | 12.2 | 44.0 | 78.3 |
| Random | False | 3 | 1.77 | 55.3 | 17.8 | 26.9 | 75.7 |
| Random | False | 10 | 3.26 | 61.3 | 21.9 | 16.8 | 73.6 |
| Random | True | 1 | 1.00 | 43.5 | 12.1 | 44.4 | 78.3 |
| Random | True | 3 | 1.41 | 51.9 | 14.2 | 33.9 | 78.5 |
| Random | True | 10 | 1.71 | 52.7 | 14.4 | 32.9 | 78.5 |

### labels=orig, scope=mamo_complex, accept threshold τ=0.7

| order | skip same-answer | budget | calls/problem | right % | wrong % | abstain % | accuracy when answered % |
|---|---|---|---|---|---|---|---|
| SSC top-1, no verifier | - | 0 | 0.00 | 53.0 | 47.0 | 0.0 | 53.0 |
| SSC | False | 10 | 2.28 | 52.0 | 38.0 | 10.0 | 57.8 |
| Cluster-first | False | 10 | 2.27 | 52.0 | 38.0 | 10.0 | 57.8 |
| Cluster-first | True | 10 | 1.38 | 51.0 | 30.0 | 19.0 | 63.0 |
| Hybrid λ=0.1 | False | 10 | 2.32 | 52.0 | 38.0 | 10.0 | 57.8 |
| Hybrid λ=0.1 | True | 10 | 1.38 | 51.0 | 30.0 | 19.0 | 63.0 |
| Hybrid λ=0.7 | False | 10 | 2.30 | 52.0 | 38.0 | 10.0 | 57.8 |
| Hybrid λ=0.7 | True | 10 | 1.39 | 51.0 | 30.0 | 19.0 | 63.0 |
| Random | False | 10 | 2.40 | 52.5 | 37.5 | 10.0 | 58.3 |
| Random | True | 10 | 1.55 | 50.1 | 29.3 | 20.7 | 63.1 |

### labels=orig, scope=optmath, accept threshold τ=0.7

| order | skip same-answer | budget | calls/problem | right % | wrong % | abstain % | accuracy when answered % |
|---|---|---|---|---|---|---|---|
| SSC top-1, no verifier | - | 0 | 0.00 | 49.0 | 51.0 | 0.0 | 49.0 |
| SSC | False | 10 | 4.31 | 51.0 | 25.0 | 24.0 | 67.1 |
| Cluster-first | False | 10 | 4.10 | 51.0 | 25.0 | 24.0 | 67.1 |
| Cluster-first | True | 10 | 1.87 | 40.0 | 15.0 | 45.0 | 72.7 |
| Hybrid λ=0.1 | False | 10 | 3.93 | 52.0 | 24.0 | 24.0 | 68.4 |
| Hybrid λ=0.1 | True | 10 | 1.86 | 40.0 | 14.0 | 46.0 | 74.1 |
| Hybrid λ=0.7 | False | 10 | 4.06 | 51.0 | 25.0 | 24.0 | 67.1 |
| Hybrid λ=0.7 | True | 10 | 1.88 | 40.0 | 14.0 | 46.0 | 74.1 |
| Random | False | 10 | 4.15 | 51.5 | 24.5 | 24.0 | 67.8 |
| Random | True | 10 | 1.97 | 39.8 | 14.1 | 46.2 | 73.9 |

### Effect of the accept threshold (orig labels, all, Hybrid λ=0.1 + skip, budget 10)

| τ | calls | right % | wrong % | abstain % | accuracy when answered % |
|---|---|---|---|---|---|
| 0.5 | 1.51 | 48.0 | 29.0 | 23.0 | 62.3 |
| 0.7 | 1.62 | 45.5 | 22.0 | 32.5 | 67.4 |
| 0.9 | 2.06 | 35.5 | 12.0 | 52.5 | 74.7 |
