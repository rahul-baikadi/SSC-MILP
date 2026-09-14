"""Model signature M = (V, o, R) (Section 2, Eq. 7) built from the sandbox
worker's standard linear representation.

Canonicalisation conventions (Section 2):
  * maximisation objectives are mapped to minimisation (coefficients negated);
  * upper-bound rows (<=) are sign-flipped to lower-bound form (>=);
  * original senses are kept separately for the TA component;
  * numeric constants are magnitude-binned (floor(log10|x|)).
Token vocabularies are the reconstruction described in the README.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------- helpers
def magbin(x: float) -> str:
    if x is None:
        return "na"
    x = abs(float(x))
    if x == 0:
        return "0"
    return str(int(math.floor(math.log10(x))))


def signed_magbin(x: float) -> str:
    if x is None:
        return "na"
    return ("-" if x < 0 else "+") + magbin(x)


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name: str) -> set[str]:
    base = name.split("[", 1)[0]
    base = _CAMEL.sub("_", base)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", base) if t}


def bounds_expr(lb, ub) -> str:
    lo = "-inf" if lb is None else ("0" if lb == 0 else magbin(lb))
    hi = "inf" if ub is None else ("0" if ub == 0 else magbin(ub))
    return f"[{lo},{hi}]"


# ---------------------------------------------------------------- dataclasses
@dataclass
class VarSig:
    name: str
    domain: str            # binary | integer | continuous | unknown
    arity: int
    bounds: str
    tokens: set = field(default_factory=set)
    lb: float | None = None
    ub: float | None = None

    @property
    def shape(self) -> str:
        return f"{self.arity}:{self.domain}"


@dataclass
class ObjSig:
    present: bool
    sense: str                       # original: min | max | unknown
    tokens: Counter = field(default_factory=Counter)      # combined semantic tokens
    profiles: dict = field(default_factory=dict)          # name -> Counter
    n_terms: int = 0
    nonlinear: bool = False


@dataclass
class RowSig:
    comp: str
    sense: str                       # canonical: ge | eq | range | unknown
    orig_sense: str                  # le | ge | eq | range | unknown
    arity: int
    terms: Counter                   # linear-term tokens
    var_shapes: Counter
    magnitudes: Counter
    params: set
    profile: tuple
    n_terms: int
    rhs: float | None
    var_names: list                  # data-level var names (for TA rules)
    coefs: list                      # canonical coefficients
    domains: list                    # domain of each var in the row
    nonlinear: bool = False


@dataclass
class ModelSig:
    vars: list
    obj: ObjSig | None
    rows: list
    param_tokens: set
    numeric_tokens: Counter
    domains: Counter
    senses: Counter
    has_obj: bool
    constructed: bool
    defects: list

    @property
    def var_shapes_by_name(self) -> dict:
        return {v.name: v.shape for v in self.vars}


# ---------------------------------------------------------------- builder
def build_signature(parsed: dict) -> ModelSig:
    """parsed = output of ssc.sandbox.parse_candidate()."""
    model = parsed.get("model") or {}
    defects = list(parsed.get("defects", []))
    constructed = bool(model.get("variables") is not None and "n_var_data" in model)

    numeric = Counter()
    # ---- variables
    vars_: list[VarSig] = []
    shape_of: dict[str, str] = {}
    domain_of: dict[str, str] = {}
    for v in model.get("variables", []) or []:
        vs = VarSig(name=v["name"], domain=v["domain"], arity=int(v["arity"]),
                    bounds=bounds_expr(v.get("lb"), v.get("ub")),
                    tokens=name_tokens(v["name"]) | {f"dom:{v['domain']}"} |
                    {f"set:{s.lower()}" for s in v.get("index_sets", []) if s != "UnindexedComponent_set"},
                    lb=v.get("lb"), ub=v.get("ub"))
        vars_.append(vs)
        for b in (v.get("lb"), v.get("ub")):
            if b is not None and b != 0:
                numeric[magbin(b)] += 1
    for dname, meta in (model.get("var_meta") or {}).items():
        comp = meta["comp"]
        arity = next((x.arity for x in vars_ if x.name == comp), 0)
        shape_of[dname] = f"{arity}:{meta['domain']}"
        domain_of[dname] = meta["domain"]

    # ---- objective
    o = model.get("objective")
    obj = None
    if o is not None:
        sense = o.get("sense", "unknown")
        sgn = -1.0 if sense == "max" else 1.0
        terms = [(n, sgn * c) for n, c in o.get("terms", [])]
        const = sgn * (o.get("constant") or 0.0)
        tokens = Counter()
        prof = {k: Counter() for k in ("numbers", "operators", "var_shapes",
                                       "parameters", "linear_terms", "signed_numbers")}
        for n, c in terms:
            sh = shape_of.get(n, "?:unknown")
            prof["numbers"][magbin(c)] += 1
            prof["operators"]["+" if c >= 0 else "-"] += 1
            prof["var_shapes"][sh] += 1
            prof["linear_terms"][f"{sh}|{signed_magbin(c)}"] += 1
            prof["signed_numbers"][signed_magbin(c)] += 1
            numeric[magbin(c)] += 1
        if const != 0:
            prof["numbers"][magbin(const)] += 1
            prof["signed_numbers"][signed_magbin(const)] += 1
            numeric[magbin(const)] += 1
        for k, cnt in prof.items():
            for tok, m in cnt.items():
                tokens[f"{k}:{tok}"] += m
        obj = ObjSig(present=True, sense=sense, tokens=tokens, profiles=prof,
                     n_terms=len(terms), nonlinear=bool(o.get("nonlinear")))

    # ---- constraint rows
    rows: list[RowSig] = []
    senses = Counter()
    for r in model.get("constraints", []) or []:
        orig = r.get("sense", "unknown")
        terms = [(n, float(c)) for n, c in r.get("terms", [])]
        const = float(r.get("constant") or 0.0)
        lb, ub = r.get("lb"), r.get("ub")
        if orig == "le":
            canon, rhs = "ge", -(ub - const)
            terms = [(n, -c) for n, c in terms]
        elif orig == "ge":
            canon, rhs = "ge", lb - const
        elif orig == "eq":
            canon, rhs = "eq", (lb if lb is not None else ub) - const
        elif orig == "range":
            canon, rhs = "range", lb - const
        else:
            canon, rhs = "unknown", None
        tt, vsh, mags = Counter(), Counter(), Counter()
        for n, c in terms:
            sh = shape_of.get(n, "?:unknown")
            tt[f"{sh}|{signed_magbin(c)}"] += 1
            vsh[sh] += 1
            mags[magbin(c)] += 1
            numeric[magbin(c)] += 1
        if rhs is not None and rhs != 0:
            mags[f"rhs:{magbin(rhs)}"] += 1
            numeric[magbin(rhs)] += 1
        n_terms = len(terms)
        nbin = "1" if n_terms <= 1 else ("2" if n_terms == 2 else ("3-5" if n_terms <= 5 else "6+"))
        profile = (canon, nbin, rhs is not None and abs(rhs) > 1e-12,
                   len(set(vsh)), any(c < 0 for _, c in terms) and any(c > 0 for _, c in terms))
        rows.append(RowSig(comp=r["comp"], sense=canon, orig_sense=orig, arity=int(r.get("arity", 0)),
                           terms=tt, var_shapes=vsh, magnitudes=mags, params=set(),
                           profile=profile, n_terms=n_terms, rhs=rhs,
                           var_names=[n for n, _ in terms], coefs=[c for _, c in terms],
                           domains=[domain_of.get(n, "unknown") for n, _ in terms],
                           nonlinear=bool(r.get("nonlinear"))))
        senses[canon] += 1

    return ModelSig(vars=vars_, obj=obj, rows=rows, param_tokens=set(),
                    numeric_tokens=numeric,
                    domains=Counter(v.domain for v in vars_), senses=senses,
                    has_obj=obj is not None, constructed=constructed, defects=defects)


def empty_signature(defects: list[str]) -> ModelSig:
    return ModelSig(vars=[], obj=None, rows=[], param_tokens=set(), numeric_tokens=Counter(),
                    domains=Counter(), senses=Counter(), has_obj=False, constructed=False,
                    defects=list(defects))
