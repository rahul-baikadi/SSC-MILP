# complexor  (n=18, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 55.6 |
| Random expected | 51.7 |
| SC | 55.6 |
| SQ | 55.6 |
| TA | 55.6 |
| SC+SQ | 55.6 |
| SC+TA | 50.0 |
| SQ+TA | 55.6 |
| SSC_noLP | 50.0 |
| LP_only | 50.0 |
| Symbolic+LP-obj | 50.0 |
| Symbolic+LP-sol | 50.0 |
| SSC | 50.0 |
| Pool coverage | 55.6 |

McNemar SSC vs first position: b=1, c=0, p=1.0000 (exact)
Error decomposition: no correct candidate=8, selection failures=1, fail rate=10.0%
Discriminability (correct mean - incorrect mean): TA=+0.044, SQ=+0.113, SC=+0.104, lp_sol=+0.222, lp_obj=+0.287, SSC=+0.139
Clean candidates (no defects): 88.3%; LP relaxation optimal: 86.1%
