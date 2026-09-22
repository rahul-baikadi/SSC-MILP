# nl4opt  (n=100, K=10)

| Method | Accuracy (%) |
|---|---|
| Random draw (first position) | 87.0 |
| Random expected | 88.2 |
| Zero-shot LLM judge | 89.0 |
| SC | 88.0 |
| SQ | 87.0 |
| TA | 90.0 |
| SC+SQ | 88.0 |
| SC+TA | 91.0 |
| SQ+TA | 90.0 |
| SSC_noLP | 91.0 |
| LP_only | 89.0 |
| Symbolic+LP-obj | 91.0 |
| Symbolic+LP-sol | 91.0 |
| SSC | 91.0 |
| Pool coverage | 93.0 |

McNemar SSC vs first position: b=0, c=4, p=0.1250 (exact)
Error decomposition: no correct candidate=7, selection failures=2, fail rate=2.2%
Discriminability (correct mean - incorrect mean): TA=+0.048, SQ=-0.001, SC=+0.008, lp_sol=+0.042, lp_obj=+0.085, SSC=+0.037
Clean candidates (no defects): 98.8%; LP relaxation optimal: 99.8%
