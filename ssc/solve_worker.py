"""Isolated worker (offline): rebuild a candidate, solve with SCIP, return status,
objective and the non-zero variable values with readable names.
Usage: python -m ssc.solve_worker CODE_PATH OUT_JSON TIME_LIMIT_S"""
from __future__ import annotations

import json, math, os, sys, time, traceback


def main():
    code_path, out_path, tlim = sys.argv[1], sys.argv[2], float(sys.argv[3])
    res = {"status": "error", "objective": None, "vars": {}, "time": None, "detail": ""}
    t0 = time.time()
    try:
        import pyomo.environ as pyo
        from pyscipopt import Model as ScipModel
        code = open(code_path, encoding="utf-8").read()
        ns = {"__name__": "candidate"}
        exec(compile(code, "<candidate>", "exec"), ns)
        try:
            model = ns["build_model"](data=None)
        except TypeError:
            model = ns["build_model"]()
        objs = list(model.component_objects(pyo.Objective, active=True, descend_into=True))
        for o in objs[1:]:
            o.deactivate()
        lp = os.path.join(os.path.dirname(out_path), "model.lp")
        _, smap = model.write(lp, io_options={"symbolic_solver_labels": False})
        m = ScipModel(); m.hideOutput(); m.readProblem(lp)
        m.setParam("limits/time", tlim)
        m.optimize()
        st = m.getStatus()
        if st == "optimal":
            obj = float(m.getObjVal())
            if math.isfinite(obj):
                res["status"], res["objective"] = "optimal", obj
                byname = {v.name: m.getVal(v) for v in m.getVars()}
                sm = model.solutions.symbol_map[smap] if isinstance(smap, int) else smap
                for sym, ref in sm.bySymbol.items():
                    o_ = ref
                    if not hasattr(o_, "is_variable_type"):   # older pyomo stores weakrefs
                        try:
                            o_ = ref()
                        except Exception:  # noqa: BLE001
                            continue
                    if getattr(o_, "is_variable_type", lambda: False)():
                        val = byname.get(sym)
                        if val is not None and abs(val) > 1e-9:
                            res["vars"][o_.name] = round(val, 9)
            else:
                res["detail"] = "non-finite objective"
        elif st == "infeasible":
            res["status"] = "infeasible"
        elif st in ("unbounded", "inforunbd"):
            res["status"] = "unbounded"
        elif st == "timelimit":
            res["status"] = "timeout"
        else:
            res["detail"] = st
    except BaseException:  # noqa: BLE001
        res["detail"] = traceback.format_exc(limit=3)[-1500:]
    res["time"] = time.time() - t0
    json.dump(res, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
