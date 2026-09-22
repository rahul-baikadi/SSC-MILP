# optibench  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 54.0 |
| Random expected | 55.6 |
| Zero-shot LLM judge | 56.0 |
| SC | 53.0 |
| SQ | 54.0 |
| TA | 59.0 |
| SC+SQ | 53.0 |
| SC+TA | 54.0 |
| SQ+TA | 59.0 |
| SSC_noLP | 55.0 |
| LP_only | 55.0 |
| Symbolic+LP-obj | 56.0 |
| Symbolic+LP-sol | 55.0 |
| SSC | 56.0 |
| Pool coverage | 68.0 |

McNemar SSC vs first position: b=4, c=6, p=0.7539 (exact)
Error decomposition: no correct candidate=32, selection failures=12, fail rate=17.6%
Discriminability (correct mean - incorrect mean): TA=+0.014, SQ=+0.006, SC=+0.057, lp_sol=+0.183, lp_obj=+0.194, SSC=+0.084
Clean candidates (no defects): 96.4%; LP relaxation optimal: 95.8%
