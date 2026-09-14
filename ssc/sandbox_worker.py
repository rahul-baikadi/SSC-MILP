"""Isolated worker: construct one candidate's Pyomo model and dump its standard
linear representation as JSON.  Never imports a solver.

Usage:  python -m ssc.sandbox_worker CODE_PATH OUT_JSON
Exit code is always 0 when JSON could be written; the JSON carries defects.
"""
from __future__ import annotations

import json
import math
import sys
import traceback


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x):
        return None
    if math.isinf(x):
        return None
    return x


def _domain_label(v) -> str:
    try:
        if v.is_binary():
            return "binary"
        if v.is_integer():
            return "integer"
        if v.is_continuous():
            return "continuous"
    except Exception:
        pass
    return "unknown"


def _index_arity(comp) -> int:
    try:
        d = comp.dim()
        if d is None:
            return -1     # unknown / non-finite dimension
        return int(d)
    except Exception:
        return -1


def _index_set_names(comp) -> list[str]:
    try:
        s = comp.index_set()
        if s is None:
            return []
        subs = getattr(s, "subsets", None)
        if subs is not None:
            return [str(x.name) for x in s.subsets()]
        return [str(s.name)]
    except Exception:
        return []


def _index_str(idx) -> str:
    if idx is None:
        return ""
    return str(idx)


def build_and_extract(code: str) -> dict:
    import pyomo.environ as pyo
    from pyomo.repn import generate_standard_repn

    out = {"defects": [], "variables": [], "var_meta": {}, "objective": None,
           "constraints": [], "n_var_data": 0, "n_con_data": 0,
           "component_names": []}
    ns: dict = {"__name__": "candidate"}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
    except SyntaxError:
        out["defects"].append("syntax_error")
        return out
    except ImportError:
        out["defects"].append("sandbox_import_error")
        out["error"] = traceback.format_exc(limit=3)
        return out
    except Exception:
        out["defects"].append("sandbox_construct_error")
        out["error"] = traceback.format_exc(limit=3)
        return out

    fn = ns.get("build_model")
    if not callable(fn):
        out["defects"].append("sandbox_construct_error")
        out["error"] = "build_model() not defined"
        return out
    try:
        try:
            model = fn(data=None)
        except TypeError:
            model = fn()
    except Exception:
        out["defects"].append("sandbox_construct_error")
        out["error"] = traceback.format_exc(limit=3)
        return out

    if not isinstance(model, pyo.Block):
        out["defects"].append("sandbox_no_model")
        return out
    try:
        if not model.is_constructed():
            model.construct()
    except Exception:
        out["defects"].append("sandbox_construct_error")
        out["error"] = traceback.format_exc(limit=3)
        return out

    out["component_names"] = [c.name for c in model.component_objects(descend_into=True)]

    # ---------------------------------------------------------- variables
    var_meta = {}
    for comp in model.component_objects(pyo.Var, active=True, descend_into=True):
        n = 0
        dom_counts: dict[str, int] = {}
        lbs, ubs = [], []
        for idx in comp:
            vd = comp[idx]
            n += 1
            d = _domain_label(vd)
            dom_counts[d] = dom_counts.get(d, 0) + 1
            lb, ub = _num(vd.lb), _num(vd.ub)
            lbs.append(lb); ubs.append(ub)
            var_meta[vd.name] = {"comp": comp.name, "domain": d, "lb": lb, "ub": ub}
        if n == 0:
            continue
        dom = max(dom_counts, key=dom_counts.get)
        if "unknown" in dom_counts:
            out["defects"].append("unknown_variable_domain")
        # bound expression: most common (lb, ub) pair
        pairs: dict[tuple, int] = {}
        for lb, ub in zip(lbs, ubs):
            pairs[(lb, ub)] = pairs.get((lb, ub), 0) + 1
        lb, ub = max(pairs, key=pairs.get)
        out["variables"].append({
            "name": comp.name, "arity": _index_arity(comp), "domain": dom,
            "lb": lb, "ub": ub, "n": n, "index_sets": _index_set_names(comp),
        })
        out["n_var_data"] += n
    out["var_meta"] = var_meta
    if out["n_var_data"] == 0:
        out["defects"].append("sandbox_no_variables")

    # ---------------------------------------------------------- objective
    objs = [o for o in model.component_objects(pyo.Objective, active=True, descend_into=True)]
    obj_data = []
    for o in objs:
        for idx in o:
            obj_data.append(o[idx])
    if not obj_data:
        out["defects"].append("sandbox_no_objective")
    else:
        od = obj_data[0]          # first active objective; extras ignored
        sense = "min" if od.sense == pyo.minimize else ("max" if od.sense == pyo.maximize else "unknown")
        if sense == "unknown":
            out["defects"].append("unknown_objective_direction")
        entry = {"name": od.parent_component().name, "sense": sense,
                 "constant": 0.0, "terms": [], "nonlinear": False, "n_objectives": len(obj_data)}
        try:
            repn = generate_standard_repn(od.expr, compute_values=True)
            entry["constant"] = _num(repn.constant) or 0.0
            entry["terms"] = [[v.name, _num(c) or 0.0] for v, c in zip(repn.linear_vars, repn.linear_coefs)]
            if not repn.is_linear():
                entry["nonlinear"] = True
                entry["nonlinear_vars"] = [v.name for v in repn.nonlinear_vars][:50]
                out["defects"].append("possible_nonlinearity")
        except Exception:
            entry["nonlinear"] = True
            entry["error"] = traceback.format_exc(limit=2)
            out["defects"].append("possible_nonlinearity")
        out["objective"] = entry

    # ---------------------------------------------------------- constraints
    for comp in model.component_objects(pyo.Constraint, active=True, descend_into=True):
        arity = _index_arity(comp)
        for idx in comp:
            cd = comp[idx]
            if not cd.active:
                continue
            row = {"comp": comp.name, "index": _index_str(idx), "arity": arity,
                   "lb": None, "ub": None, "terms": [], "constant": 0.0,
                   "nonlinear": False, "sense": "unknown"}
            try:
                lb = _num(pyo.value(cd.lower)) if cd.lower is not None else None
                ub = _num(pyo.value(cd.upper)) if cd.upper is not None else None
                row["lb"], row["ub"] = lb, ub
                if cd.equality:
                    row["sense"] = "eq"
                elif lb is not None and ub is not None:
                    row["sense"] = "range"
                elif ub is not None:
                    row["sense"] = "le"
                elif lb is not None:
                    row["sense"] = "ge"
                else:
                    row["sense"] = "unknown"
                repn = generate_standard_repn(cd.body, compute_values=True)
                row["constant"] = _num(repn.constant) or 0.0
                row["terms"] = [[v.name, _num(c) or 0.0] for v, c in zip(repn.linear_vars, repn.linear_coefs)]
                if not repn.is_linear():
                    row["nonlinear"] = True
                    out["defects"].append("possible_nonlinearity")
            except Exception:
                row["nonlinear"] = True
                row["sense"] = "unknown"
                row["error"] = traceback.format_exc(limit=2)
            if row["sense"] == "unknown":
                out["defects"].append("unknown_constraint_sense")
            out["constraints"].append(row)
            out["n_con_data"] += 1
    if out["n_con_data"] == 0:
        out["defects"].append("sandbox_no_constraints")

    out["defects"] = sorted(set(out["defects"]))
    return out


def main():
    code_path, out_path = sys.argv[1], sys.argv[2]
    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    try:
        res = build_and_extract(code)
    except BaseException:  # noqa: BLE001 - anything at all
        res = {"defects": ["sandbox_construct_error"], "error": traceback.format_exc(limit=3),
               "variables": [], "var_meta": {}, "objective": None, "constraints": [],
               "n_var_data": 0, "n_con_data": 0, "component_names": []}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
