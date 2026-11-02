# Failure audit (scores_v1, Hybrid lambda=0.10 delta=0.001)

Generation failures (no correct candidate): **176**  
Hybrid selection failures: **22**

## Generation-failure causes by benchmark

| cause | nl4opt | mamo_easy | mamo_complex | complexor | industryor | optibench | optmath | total |
|---|---|---|---|---|---|---|---|---|
| wrong_consensus_>=6 | 2 | 3 | 12 | 5 | 12 | 17 | 7 | 58 |
| scattered_wrong | 0 | 0 | 1 | 0 | 21 | 2 | 7 | 31 |
| integrality_too_loose | 0 | 0 | 12 | 0 | 4 | 0 | 0 | 16 |
| near_miss_<1% | 2 | 3 | 3 | 0 | 1 | 4 | 1 | 14 |
| no_feasible_candidate | 0 | 0 | 1 | 2 | 5 | 1 | 5 | 14 |
| mostly_broken | 1 | 0 | 6 | 0 | 5 | 0 | 1 | 13 |
| gold_precision_artifact | 2 | 0 | 1 | 1 | 1 | 8 | 0 | 13 |
| integrality_too_strict | 0 | 0 | 4 | 0 | 2 | 0 | 3 | 9 |
| scale_x10^k | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 5 |
| gold_placeholder | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 |
| **total** | 7 | 6 | 40 | 8 | 58 | 32 | 25 | 176 |

## Candidate status inside no-coverage pools

- optimal (wrong value): 1435 (81.5%)
- infeasible: 237 (13.5%)
- unbounded: 22 (1.2%)
- error/timeout: 66 (3.8%)

## Can a no-coverage pool be detected without the gold answer? (AUROC for predicting 'no correct candidate')

| signal | direction | AUROC |
|---|---|---|
| max_SSC | low = risky | 0.572 |
| top_TA | low = risky | 0.548 |
| top_lp_cluster_mass | low = risky | 0.707 |
| largest_lp_cluster | low = risky | 0.709 |
| n_lp_clusters | high = risky | 0.620 |
| frac_lp_failed | high = risky | 0.617 |
| frac_fatal | high = risky | 0.545 |

### Regeneration trigger: top candidate's LP-objective cluster mass <= t

| t | pools flagged | of which no-coverage | no-coverage caught (recall) | covered pools flagged |
|---|---|---|---|---|
| 0.2 | 58 | 41 | 23.3% | 17 |
| 0.3 | 71 | 51 | 29.0% | 20 |
| 0.4 | 86 | 57 | 32.4% | 29 |
| 0.5 | 105 | 68 | 38.6% | 37 |
| 0.6 | 122 | 74 | 42.0% | 48 |

## Which diagnostics are actionable? Missing required TA feature vs. correctness (candidate level)

| feature | P(correct \| missing) | n missing | P(correct \| present) | n present | lift |
|---|---|---|---|---|---|
| objective:minimize | 0.000 | 82 | 0.570 | 4158 | +0.570 |
| objective:maximize | 0.171 | 70 | 0.689 | 1900 | +0.518 |
| var:binary | 0.000 | 61 | 0.461 | 1689 | +0.461 |
| var:integer | 0.378 | 407 | 0.775 | 2743 | +0.397 |
| constraint:upper_bound | 0.387 | 124 | 0.665 | 2226 | +0.278 |
| constraint:resource | 0.506 | 162 | 0.756 | 1038 | +0.250 |
| constraint:equality | 0.250 | 36 | 0.477 | 614 | +0.227 |
| constraint:demand | 0.434 | 346 | 0.638 | 1434 | +0.205 |
| constraint:lower_bound | 0.513 | 1688 | 0.692 | 2522 | +0.179 |
| constraint:routing | 0.177 | 158 | 0.260 | 262 | +0.082 |
| constraint:capacity | 0.526 | 574 | 0.588 | 2326 | +0.062 |
| constraint:equality_continuous | 0.500 | 30 | 0.560 | 50 | +0.060 |
| constraint:coverage | 0.525 | 160 | 0.573 | 260 | +0.048 |
| var:bounded | 0.595 | 1493 | 0.600 | 1957 | +0.006 |
| constraint:budget | 0.681 | 298 | 0.672 | 542 | -0.010 |
| constraint:assignment | 0.322 | 348 | 0.310 | 522 | -0.011 |
| constraint:flow_balance | 0.521 | 526 | 0.473 | 474 | -0.048 |
| constraint:linking | 0.631 | 848 | 0.383 | 892 | -0.247 |
| constraint:facility_linking | 0.605 | 516 | 0.182 | 44 | -0.423 |

## Static defects vs. correctness (candidate level)

| defect | P(correct) | n |
|---|---|---|
| (none) | 0.609 | 5963 |
| possible_nonlinearity | 0.435 | 131 |
| sandbox_construct_error | 0.000 | 73 |
| sandbox_timeout | 0.000 | 10 |
| syntax_error | 0.000 | 6 |
| unsupported_python_max_on_pyomo_expr | 0.500 | 2 |
| reserved_pyomo_component_name | 0.000 | 1 |

## Hybrid selection failures

| bench | pid | #correct | best correct SSC rank | c1 clone size | correct w/ same LP obj as c1 | correct w/ failed LP | SSC(c1)-SSC(best correct) | TA c1 | TA best correct |
|---|---|---|---|---|---|---|---|---|---|
| mamo_complex | mamo_complex_0048 | 6 | 3 | 1 | 0 | 0 | 0.0518 | 0.909 | 0.636 |
| industryor | industryor_0043 | 2 | 3 | 1 | 0 | 0 | 0.0527 | 0.874 | 0.653 |
| industryor | industryor_0080 | 1 | 8 | 3 | 1 | 0 | 0.2058 | 0.671 | 0.671 |
| optibench | optibench_0002 | 2 | 6 | 1 | 0 | 0 | 0.1825 | 0.782 | 0.762 |
| optibench | optibench_0027 | 2 | 3 | 3 | 0 | 0 | 0.0137 | 0.714 | 0.714 |
| optibench | optibench_0029 | 1 | 8 | 2 | 0 | 0 | 0.361 | 0.733 | 0.36 |
| optibench | optibench_0056 | 1 | 9 | 3 | 0 | 0 | 0.2072 | 0.859 | 0.718 |
| optibench | optibench_0080 | 3 | 5 | 1 | 0 | 1 | 0.1705 | 0.875 | 1.0 |
| optmath | optmath_0002 | 2 | 6 | 2 | 2 | 0 | 0.039 | 0.779 | 0.779 |
| optmath | optmath_0007 | 2 | 7 | 1 | 2 | 0 | 0.0385 | 1.0 | 0.875 |
| optmath | optmath_0012 | 1 | 7 | 1 | 1 | 0 | 0.0302 | 1.0 | 0.887 |
| optmath | optmath_0015 | 4 | 2 | 2 | 4 | 0 | 0.0 | 0.909 | 0.909 |
| optmath | optmath_0032 | 1 | 7 | 1 | 1 | 0 | 0.1261 | 0.9 | 0.9 |
| optmath | optmath_0043 | 1 | 6 | 1 | 1 | 0 | 0.0527 | 0.758 | 0.758 |
| optmath | optmath_0054 | 1 | 9 | 2 | 1 | 0 | 0.1173 | 0.8 | 0.8 |
| optmath | optmath_0058 | 4 | 5 | 1 | 4 | 0 | 0.0683 | 0.871 | 0.871 |
| optmath | optmath_0073 | 2 | 6 | 4 | 0 | 0 | 0.0305 | 0.922 | 0.922 |
| optmath | optmath_0077 | 3 | 4 | 2 | 3 | 0 | 0.0058 | 0.9 | 0.9 |
| optmath | optmath_0081 | 1 | 6 | 3 | 1 | 0 | 0.0178 | 0.9 | 0.9 |
| optmath | optmath_0089 | 2 | 2 | 1 | 2 | 0 | 0.0002 | 1.0 | 1.0 |
| optmath | optmath_0098 | 1 | 6 | 4 | 1 | 0 | 0.0502 | 1.0 | 1.0 |
| optmath | optmath_0100 | 1 | 3 | 1 | 1 | 0 | 0.0099 | 0.769 | 0.769 |
