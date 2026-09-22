# industryor  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 33.0 |
| Random expected | 32.6 |
| Zero-shot LLM judge | 30.0 |
| SC | 33.0 |
| SQ | 33.0 |
| TA | 29.0 |
| SC+SQ | 34.0 |
| SC+TA | 33.0 |
| SQ+TA | 29.0 |
| SSC_noLP | 34.0 |
| LP_only | 33.0 |
| Symbolic+LP-obj | 34.0 |
| Symbolic+LP-sol | 34.0 |
| SSC | 35.0 |
| Pool coverage | 42.0 |

McNemar SSC vs first position: b=3, c=5, p=0.7266 (exact)
Error decomposition: no correct candidate=58, selection failures=7, fail rate=16.7%
Discriminability (correct mean - incorrect mean): TA=+0.064, SQ=+0.037, SC=+0.228, lp_sol=+0.361, lp_obj=+0.314, SSC=+0.195
Clean candidates (no defects): 94.3%; LP relaxation optimal: 87.7%
