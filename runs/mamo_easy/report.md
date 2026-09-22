# mamo_easy  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 93.0 |
| Random expected | 91.6 |
| Zero-shot LLM judge | 93.0 |
| SC | 92.0 |
| SQ | 93.0 |
| TA | 92.0 |
| SC+SQ | 92.0 |
| SC+TA | 93.0 |
| SQ+TA | 92.0 |
| SSC_noLP | 93.0 |
| LP_only | 93.0 |
| Symbolic+LP-obj | 93.0 |
| Symbolic+LP-sol | 94.0 |
| SSC | 93.0 |
| Pool coverage | 94.0 |

McNemar SSC vs first position: b=0, c=0, p=1.0000 (none)
Error decomposition: no correct candidate=6, selection failures=1, fail rate=1.1%
Discriminability (correct mean - incorrect mean): TA=+0.048, SQ=+0.009, SC=+0.010, lp_sol=+0.084, lp_obj=+0.212, SSC=+0.066
Clean candidates (no defects): 99.4%; LP relaxation optimal: 99.7%
