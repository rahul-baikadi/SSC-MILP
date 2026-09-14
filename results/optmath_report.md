# optmath  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 46.0 |
| Random expected | 45.5 |
| SC | 42.0 |
| SQ | 46.0 |
| TA | 43.0 |
| SC+SQ | 42.0 |
| SC+TA | 42.0 |
| SQ+TA | 43.0 |
| SSC_noLP | 42.0 |
| LP_only | 50.0 |
| Symbolic+LP-obj | 48.0 |
| Symbolic+LP-sol | 49.0 |
| SSC | 49.0 |
| Pool coverage | 75.0 |

McNemar SSC vs first position: b=4, c=7, p=0.5488 (exact)
Error decomposition: no correct candidate=25, selection failures=26, fail rate=34.7%
Discriminability (correct mean - incorrect mean): TA=-0.001, SQ=+0.029, SC=+0.054, lp_sol=+0.182, lp_obj=+0.187, SSC=+0.080
Clean candidates (no defects): 95.6%; LP relaxation optimal: 91.9%
