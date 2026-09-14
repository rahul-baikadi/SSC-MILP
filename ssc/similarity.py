"""Pairwise program similarity sim(M_i, M_j) in [0, 1] — Appendix E, Eqs. 17–22."""
from __future__ import annotations

from collections import Counter

from . import config as C
from .signature import ModelSig, VarSig


# ---------------------------------------------------------------- set / bag Jaccard
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def bag_jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den else 1.0


def h(a: float, b: float) -> float:
    """Eq. 21 helper: 1 - |a-b| / max(1, max(a,b))."""
    return max(0.0, 1.0 - abs(a - b) / max(1.0, max(a, b)))


def g(a: int, b: int) -> float:
    """Eq. 17 arity term: max(0, 1 - |a-a'|/3)."""
    return max(0.0, 1.0 - abs(a - b) / 3.0)


# ---------------------------------------------------------------- E.1 variables
def var_sim(v: VarSig, w: VarSig) -> float:
    if v.domain == w.domain:
        dom = 1.0
    elif v.domain == "unknown" or w.domain == "unknown":
        dom = C.VAR_UNKNOWN_DOMAIN_MATCH
    else:
        dom = 0.0
    W = C.VAR_SIM_WEIGHTS
    return (W["domain"] * dom + W["arity"] * g(v.arity, w.arity)
            + W["bounds"] * (1.0 if v.bounds == w.bounds else 0.0)
            + W["tokens"] * jaccard(v.tokens, w.tokens))


def var_collection_sim(A: list, B: list) -> float:
    """Greedy best-match alignment normalised by the larger collection."""
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    pairs = sorted(((var_sim(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)),
                   reverse=True)
    used_i, used_j, total = set(), set(), 0.0
    for s, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); total += s
        if len(used_i) == len(A) or len(used_j) == len(B):
            break
    return total / max(len(A), len(B))


# ---------------------------------------------------------------- E.2 constraints
def con_collection_sim(R: list, S: list) -> float:
    if not R and not S:
        return 1.0
    if not R or not S:
        return 0.0
    W = C.CON_SIM_WEIGHTS
    q_n = h(len(R), len(S))
    q_s = bag_jaccard(Counter(r.sense for r in R), Counter(s.sense for s in S))
    q_a = bag_jaccard(Counter(r.arity for r in R), Counter(s.arity for s in S))
    tR, tS = Counter(), Counter()
    vR, vS = Counter(), Counter()
    pR, pS = set(), set()
    for r in R:
        tR.update(r.terms); vR.update(r.var_shapes); pR |= r.params
    for s in S:
        tS.update(s.terms); vS.update(s.var_shapes); pS |= s.params
    q_t = bag_jaccard(tR, tS)
    q_v = bag_jaccard(vR, vS)
    q_p = jaccard(pR, pS)
    return (W["n"] * q_n + W["s"] * q_s + W["a"] * q_a + W["t"] * q_t
            + W["v"] * q_v + W["p"] * q_p)


# ---------------------------------------------------------------- E.3 objective
def obj_sim(o, p) -> float:
    if o is None and p is None:
        return 1.0
    if o is None or p is None:
        return 0.0
    if o.sense == "unknown" or p.sense == "unknown":
        s_dir = C.OBJ_DIRECTION_UNKNOWN
    else:
        s_dir = 1.0 if o.sense == p.sense else 0.0
    s_e = bag_jaccard(o.tokens, p.tokens)
    s_p = sum(w * bag_jaccard(o.profiles.get(k, Counter()), p.profiles.get(k, Counter()))
              for k, w in C.OBJ_PROFILE_WEIGHTS.items())
    W = C.OBJ_SIM_WEIGHTS
    return W["direction"] * s_dir + W["tokens"] * s_e + W["profile"] * s_p


# ---------------------------------------------------------------- E.4 parameter dependency
def dep_sim(m: ModelSig, n: ModelSig) -> float:
    W = C.DEP_SIM_WEIGHTS
    return (W["params"] * jaccard(m.param_tokens, n.param_tokens)
            + W["numbers"] * bag_jaccard(m.numeric_tokens, n.numeric_tokens))


# ---------------------------------------------------------------- E.5 shape
def shape_sim(m: ModelSig, n: ModelSig) -> float:
    W = C.SHAPE_SIM_WEIGHTS
    return (W["n_vars"] * h(len(m.vars), len(n.vars))
            + W["n_cons"] * h(len(m.rows), len(n.rows))
            + W["has_obj"] * (1.0 if m.has_obj == n.has_obj else 0.0)
            + W["domains"] * bag_jaccard(m.domains, n.domains)
            + W["senses"] * bag_jaccard(m.senses, n.senses))


# ---------------------------------------------------------------- E.6 overall
def program_sim(m: ModelSig, n: ModelSig) -> float:
    W = C.SIM_WEIGHTS
    s = (W["var"] * var_collection_sim(m.vars, n.vars)
         + W["obj"] * obj_sim(m.obj, n.obj)
         + W["con"] * con_collection_sim(m.rows, n.rows)
         + W["dep"] * dep_sim(m, n)
         + W["shape"] * shape_sim(m, n))
    return min(1.0, max(0.0, s))


def similarity_matrix(sigs: list[ModelSig]) -> list[list[float]]:
    K = len(sigs)
    S = [[1.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            S[i][j] = S[j][i] = program_sim(sigs[i], sigs[j])
    return S
