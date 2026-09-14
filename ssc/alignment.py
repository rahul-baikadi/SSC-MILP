"""Text alignment TA(c_i, P) — Section 3.3 / Appendix L.

  TA = sum_{k: f_k in phi(M_i)} u_k / sum_k u_k                      (Eq. 14)

phi(M_i): the candidate's evidence set over the closed 21-token vocabulary,
derived from the parsed model without any LLM call.  The mapping from
constraint rows to semantic constraint types is rule-based pattern matching;
the paper does not publish the rules, so the ones below are a reconstruction
(see README).  A requirement is covered when *all* of its feature tokens are
present in the evidence set.
"""
from __future__ import annotations

from . import config as C
from .signature import ModelSig, RowSig, name_tokens

_HINTS = {
    "capacity": {"cap", "capacity", "limit", "max", "maximum", "upper"},
    "budget": {"budget", "cost", "spend", "spending", "money", "price", "fund", "funds", "dollar"},
    "resource": {"resource", "resources", "material", "materials", "labor", "labour", "hour",
                 "hours", "time", "worker", "workers", "machine", "machines", "ingredient",
                 "ingredients", "supply", "avail", "available", "availability", "staff", "space"},
    "demand": {"demand", "demands", "require", "required", "requirement", "requirements",
               "need", "needs", "min", "minimum", "least", "lower"},
    "coverage": {"cover", "covered", "coverage", "covering", "serve", "served"},
    "assignment": {"assign", "assigned", "assignment", "each", "one", "exactly"},
    "linking": {"link", "linking", "logic", "logical", "activation", "activate", "big", "bigm", "indicator"},
    "facility": {"facility", "facilities", "open", "opened", "site", "sites", "warehouse",
                 "warehouses", "plant", "plants", "depot", "depots", "location", "locations"},
    "flow": {"flow", "flows", "balance", "conservation", "conserve", "node", "nodes", "transship"},
    "inventory": {"inventory", "inventories", "stock", "storage", "period", "periods", "carry", "carryover"},
    "routing": {"route", "routes", "routing", "tour", "tours", "visit", "visits", "subtour",
                "mtz", "degree", "arc", "arcs", "edge", "edges", "vehicle", "vehicles"},
}


def _row_features(r: RowSig) -> set[str]:
    f: set[str] = set()
    toks = name_tokens(r.comp)
    doms = r.domains
    n_bin = sum(d == "binary" for d in doms)
    n_int = sum(d == "integer" for d in doms)
    n_cont = sum(d == "continuous" for d in doms)
    n = len(doms)
    rhs = r.rhs
    pos = any(c > 0 for c in r.coefs)
    neg = any(c < 0 for c in r.coefs)

    def hint(k):
        return bool(toks & _HINTS[k])

    if r.orig_sense == "eq":
        f.add("constraint:equality")
        if n > 0 and n_bin == n:
            f.add("constraint:equality_binary")
        elif n > 0 and n_bin == 0:
            f.add("constraint:equality_continuous")
    if r.orig_sense == "le":
        f.add("constraint:upper_bound")
    if r.orig_sense == "ge":
        f.add("constraint:lower_bound")
    if r.orig_sense == "range":
        f.add("constraint:upper_bound"); f.add("constraint:lower_bound")

    # assignment: sum of binaries == 1 (or <= 1 / >= 1 with unit coefs)
    if n > 0 and n_bin == n and all(abs(abs(c) - 1) < 1e-9 for c in r.coefs) \
            and rhs is not None and abs(abs(rhs) - 1) < 1e-9:
        f.add("constraint:assignment")
    elif hint("assignment") and n_bin > 0:
        f.add("constraint:assignment")

    # capacity / budget / resource: upper-bound style rows with positive rhs
    if r.orig_sense == "le":
        if (n >= 2 and rhs is not None and rhs < 0) or hint("capacity"):
            f.add("constraint:capacity")
        if hint("budget"):
            f.add("constraint:budget")
        if hint("resource") or (n >= 2 and rhs is not None and rhs < 0
                               and not all(abs(abs(c) - 1) < 1e-9 for c in r.coefs)):
            f.add("constraint:resource")
    # note: canonical form is >=, so an original <= row with positive original
    # rhs has canonical rhs < 0.

    # demand: lower-bound style rows with positive rhs
    if r.orig_sense == "ge" and ((rhs is not None and rhs > 0) or hint("demand")):
        f.add("constraint:demand")
    if r.orig_sense == "eq" and hint("demand"):
        f.add("constraint:demand")

    # coverage: sum of binaries >= 1 (or hint)
    if r.orig_sense == "ge" and n_bin == n and n > 0 and rhs is not None and rhs >= 1 - 1e-9:
        f.add("constraint:coverage")
    elif hint("coverage"):
        f.add("constraint:coverage")

    # linking: binary and non-binary variables in one row
    if n_bin > 0 and (n_int + n_cont) > 0:
        f.add("constraint:linking")
        if hint("facility"):
            f.add("constraint:facility_linking")
    elif hint("linking") and n_bin > 0:
        f.add("constraint:linking")
    if hint("facility") and n_bin > 0:
        f.add("constraint:facility_linking")

    # flow balance: equality with mixed-sign coefficients over >= 2 non-binary vars
    if r.orig_sense == "eq" and pos and neg and (n_int + n_cont) >= 2:
        f.add("constraint:flow_balance")
        if hint("inventory"):
            f.add("constraint:inventory_balance")
    if hint("flow"):
        f.add("constraint:flow_balance")
    if hint("inventory") and pos and neg:
        f.add("constraint:inventory_balance")

    # routing: degree-style rows on 2-index binaries, or hint
    if n_bin == n and n >= 2 and r.arity >= 1 and rhs is not None and abs(abs(rhs) - 1) < 1e-9 \
            and hint("routing"):
        f.add("constraint:routing")
    elif hint("routing") and n_bin > 0:
        f.add("constraint:routing")
    return f


def evidence_features(sig: ModelSig) -> set[str]:
    f: set[str] = set()
    if sig.obj is not None:
        if sig.obj.sense == "min":
            f.add("objective:minimize")
        elif sig.obj.sense == "max":
            f.add("objective:maximize")
    for v in sig.vars:
        if v.domain == "binary":
            f.add("var:binary")
        elif v.domain == "integer":
            f.add("var:integer")
        if v.ub is not None:
            f.add("var:bounded")
    for r in sig.rows:
        f |= _row_features(r)
    return {x for x in f if x in C.TA_VOCABULARY}


def text_alignment(sig: ModelSig, requirements: list[dict]) -> float:
    """requirements: [{'name': str, 'features': [tok,...], 'weight': float}, ...]"""
    if not requirements:
        return 0.0
    ev = evidence_features(sig)
    tot = sum(float(r["weight"]) for r in requirements)
    if tot <= 0:
        return 0.0
    hit = sum(float(r["weight"]) for r in requirements
              if r["features"] and all(t in ev for t in r["features"]))
    return hit / tot


def filter_requirements(raw: list) -> list[dict]:
    """Closed-vocabulary filtering and weight clamping (Appendix L)."""
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        feats = item.get("features", [])
        if isinstance(feats, str):
            feats = [feats]
        feats = [str(t).strip().lower() for t in feats if isinstance(t, (str,))]
        feats = [t for t in feats if t in C.TA_VOCABULARY]
        if not feats:
            continue
        try:
            w = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        w = min(C.TA_WEIGHT_MAX, max(C.TA_WEIGHT_MIN, w))
        out.append({"name": str(item.get("name", "req"))[:64], "features": feats, "weight": w})
    return out
