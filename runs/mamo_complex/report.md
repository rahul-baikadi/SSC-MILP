# mamo_complex  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 45.0 |
| Random expected | 46.2 |
| Zero-shot LLM judge | 45.0 |
| SC | 48.0 |
| SQ | 46.0 |
| TA | 49.0 |
| SC+SQ | 48.0 |
| SC+TA | 48.0 |
| SQ+TA | 49.0 |
| SSC_noLP | 48.0 |
| LP_only | 51.0 |
| Symbolic+LP-obj | 53.0 |
| Symbolic+LP-sol | 48.0 |
| SSC | 53.0 |
| Pool coverage | 60.0 |

McNemar SSC vs first position: b=1, c=9, p=0.0215 (exact)
Error decomposition: no correct candidate=40, selection failures=7, fail rate=11.7%
Discriminability (correct mean - incorrect mean): TA=-0.048, SQ=+0.025, SC=-0.056, lp_sol=-0.185, lp_obj=+0.201, SSC=-0.030
Clean candidates (no defects): 95.9%; LP relaxation optimal: 93.0%
