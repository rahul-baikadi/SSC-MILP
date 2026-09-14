"""The SSC selector (Section 3, Appendix A pseudocode) and its ablation
variants.  Everything here is deterministic given the pool."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .alignment import evidence_features, text_alignment
from .lp_diag import lp_diagnostics
from .quality import has_fatal, static_quality
from .sandbox import lp_relaxation, parse_candidate
from .signature import build_signature
from .similarity import similarity_matrix


# ---------------------------------------------------------------- clone groups
def clone_groups(S: list[list[float]], tau: float = C.CLONE_THRESHOLD) -> list[int]:
    """Union-find over sim >= tau.  Returns group id per candidate."""
    K = len(S)
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(K):
        for j in range(i + 1, K):
            if S[i][j] >= tau:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(K)]
    ids = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def structural_consistency(S, SQ, TA, groups) -> list[float]:
    """Eq. 11–12 / Eq. 23."""
    K = len(S)
    sizes = {}
    for gid in groups:
        sizes[gid] = sizes.get(gid, 0) + 1
    w = [SQ[j] * TA[j] / sizes[groups[j]] for j in range(K)]
    SC = []
    for i in range(K):
        num = sum(w[j] * S[i][j] for j in range(K) if j != i)
        den = sum(w[j] for j in range(K) if j != i)
        SC.append(num / den if den > 0 else 0.0)
    return SC


# ---------------------------------------------------------------- variants
def _argmax(xs: list[float]) -> int:
    best, bi = None, 0
    for i, x in enumerate(xs):
        if best is None or x > best + 1e-12:
            best, bi = x, i
    return bi


def symbolic_base(SC, SQ, TA, fatal, weights: dict | None = None) -> list[float]:
    """Eq. 8 with the fatal cap (Appendix F).  `weights` may omit signals
    (ablations); the remaining weights are renormalised."""
    W = dict(weights or C.SYMBOLIC_WEIGHTS)
    tot = sum(W.values())
    W = {k: v / tot for k, v in W.items()}
    out = []
    for i in range(len(SC)):
        b = W.get("SC", 0) * SC[i] + W.get("SQ", 0) * SQ[i] + W.get("TA", 0) * TA[i]
        if fatal[i]:
            b = min(b, SQ[i])
        out.append(b)
    return out


VARIANTS = ["SC", "SQ", "TA", "SC+SQ", "SC+TA", "SQ+TA", "SSC_noLP",
            "LP_only", "Symbolic+LP-obj", "Symbolic+LP-sol", "SSC"]


def variant_scores(SC, SQ, TA, fatal, lp_sol, lp_obj) -> dict[str, list[float]]:
    Wsym = C.SYMBOLIC_WEIGHTS
    Wf = C.FUSION_WEIGHTS
    B = symbolic_base(SC, SQ, TA, fatal)
    K = len(B)
    out = {
        "SC": symbolic_base(SC, SQ, TA, fatal, {"SC": 1}),
        "SQ": symbolic_base(SC, SQ, TA, fatal, {"SQ": 1}),
        "TA": symbolic_base(SC, SQ, TA, fatal, {"TA": 1}),
        "SC+SQ": symbolic_base(SC, SQ, TA, fatal, {"SC": Wsym["SC"], "SQ": Wsym["SQ"]}),
        "SC+TA": symbolic_base(SC, SQ, TA, fatal, {"SC": Wsym["SC"], "TA": Wsym["TA"]}),
        "SQ+TA": symbolic_base(SC, SQ, TA, fatal, {"SQ": Wsym["SQ"], "TA": Wsym["TA"]}),
        "SSC_noLP": B,
        "LP_only": [(Wf["LP_SOL"] * lp_sol[i] + Wf["LP_OBJ"] * lp_obj[i]) / (Wf["LP_SOL"] + Wf["LP_OBJ"])
                    for i in range(K)],
        "Symbolic+LP-obj": [(Wf["B"] * B[i] + Wf["LP_OBJ"] * lp_obj[i]) / (Wf["B"] + Wf["LP_OBJ"])
                            for i in range(K)],
        "Symbolic+LP-sol": [(Wf["B"] * B[i] + Wf["LP_SOL"] * lp_sol[i]) / (Wf["B"] + Wf["LP_SOL"])
                            for i in range(K)],
        "SSC": [Wf["B"] * B[i] + Wf["LP_SOL"] * lp_sol[i] + Wf["LP_OBJ"] * lp_obj[i] for i in range(K)],
    }
    return out


# ---------------------------------------------------------------- main entry
def score_pool(codes: list[str], requirements: list[dict], run_lp: bool = True,
               workers: int = 4, parsed: list[dict] | None = None,
               lp_results: list[dict] | None = None) -> dict:
    """Score one candidate pool.  Returns a JSON-serialisable dict with
    per-candidate signals, similarity matrix, variant scores and selections."""
    K = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        if parsed is None:
            parsed = list(ex.map(parse_candidate, codes))
        if run_lp and lp_results is None:
            lp_results = list(ex.map(lp_relaxation, codes))
    if not run_lp:
        lp_results = [{"status": "skipped", "objective": None, "values": []} for _ in codes]

    sigs = [build_signature(p) for p in parsed]
    defects = [p["defects"] for p in parsed]
    SQ = [static_quality(d) for d in defects]
    fatal = [has_fatal(d) for d in defects]
    TA = [text_alignment(s, requirements) for s in sigs]
    S = similarity_matrix(sigs)
    groups = clone_groups(S)
    SC = structural_consistency(S, SQ, TA, groups)
    lp_sol, lp_obj = lp_diagnostics(lp_results)
    scores = variant_scores(SC, SQ, TA, fatal, lp_sol, lp_obj)
    selected = {v: _argmax(sc) for v, sc in scores.items()}

    return {
        "K": K,
        "candidates": [{
            "defects": defects[i], "fatal": fatal[i], "SQ": SQ[i], "TA": TA[i], "SC": SC[i],
            "clone_group": groups[i], "lp_status": lp_results[i].get("status"),
            "lp_objective": lp_results[i].get("objective"),
            "lp_sol": lp_sol[i], "lp_obj": lp_obj[i],
            "evidence": sorted(evidence_features(sigs[i])),
            "n_vars": len(sigs[i].vars), "n_rows": len(sigs[i].rows),
            "obj_sense": sigs[i].obj.sense if sigs[i].obj else None,
            "SSC": scores["SSC"][i],
        } for i in range(K)],
        "similarity": S,
        "scores": scores,
        "selected": selected,
        "requirements": requirements,
    }
