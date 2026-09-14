"""LP-relaxation diagnostics — Section 3.4, Appendix A (Eqs. 15–16).

Given the per-candidate GLPK relaxation results of one pool:
  * LP-sol  r_i^sol : mean non-negative cosine between sorted, 30-dim
                      truncated / zero-padded value profiles over valid peers
  * LP-obj  r_i^obj : max(0, 1 - |z_i - m| / R), m = upper median of feasible
                      relaxed objectives, R = max - min (1 if < 1e-9)
Both are then divided by their positive pool maximum (0 if the max is 0).
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C


def profile(values: list[float], dim: int = C.LP_PROFILE_DIM) -> np.ndarray:
    v = sorted(float(x) for x in values)
    v = v[:dim] + [0.0] * max(0, dim - len(v))
    return np.asarray(v, dtype=float)


def _cos_plus(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0                      # zero-norm pairs contribute zero
    return max(0.0, float(a @ b) / (na * nb))


def lp_diagnostics(lp_results: list[dict], dim: int = C.LP_PROFILE_DIM) -> tuple[list[float], list[float]]:
    """lp_results[i] = {'status':..., 'objective':..., 'values': [...]}.
    Returns (lp_sol_hat, lp_obj_hat), both pool-normalised."""
    K = len(lp_results)
    valid = [i for i, r in enumerate(lp_results)
             if r.get("status") == "optimal" and r.get("values")]
    profiles = {i: profile(lp_results[i]["values"], dim) for i in valid}

    r_sol = [0.0] * K
    for i in valid:
        peers = [j for j in valid if j != i]
        if not peers:
            continue
        r_sol[i] = sum(_cos_plus(profiles[i], profiles[j]) for j in peers) / len(peers)

    feas = [i for i, r in enumerate(lp_results)
            if r.get("status") == "optimal" and r.get("objective") is not None
            and math.isfinite(r["objective"])]
    r_obj = [0.0] * K
    if feas:
        z = sorted(lp_results[i]["objective"] for i in feas)
        m = z[len(z) // 2]                          # upper middle order statistic
        R = z[-1] - z[0]
        if R < C.LP_RANGE_EPS:
            R = 1.0
        for i in feas:
            r_obj[i] = max(0.0, 1.0 - abs(lp_results[i]["objective"] - m) / R)

    def norm(x):
        mx = max(x) if x else 0.0
        return [v / mx if mx > 0 else 0.0 for v in x]

    return norm(r_sol), norm(r_obj)
