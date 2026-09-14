"""Isolated worker: rebuild the candidate, relax integrality, solve with GLPK
under a time limit (Section 3.4 / Appendix A).  Emits the relaxed objective and
the vector of variable values.

Usage: python -m ssc.lp_worker CODE_PATH OUT_JSON TIME_LIMIT_S
"""
from __future__ import annotations

import json
import math
import sys
import traceback


def _rebuild(code: str):
    ns: dict = {"__name__": "candidate"}
    exec(compile(code, "<candidate>", "exec"), ns)
    fn = ns["build_model"]
    try:
        return fn(data=None)
    except TypeError:
        return fn()


def main():
    code_path, out_path, tlim = sys.argv[1], sys.argv[2], float(sys.argv[3])
    res = {"status": "error", "objective": None, "values": [], "detail": ""}
    try:
        import pyomo.environ as pyo
        from pyomo.opt import SolverStatus, TerminationCondition

        with open(code_path, encoding="utf-8") as f:
            code = f.read()
        model = _rebuild(code)
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(model)
        # keep exactly one active objective
        objs = [o for o in model.component_objects(pyo.Objective, active=True, descend_into=True)]
        for o in objs[1:]:
            o.deactivate()
        opt = pyo.SolverFactory("glpk")
        if not opt.available(exception_flag=False):
            res["detail"] = "glpk not available"
        else:
            opt.options["tmlim"] = int(max(1, tlim))
            r = opt.solve(model, tee=False, load_solutions=False)
            tc = r.solver.termination_condition
            if tc == TerminationCondition.optimal:
                model.solutions.load_from(r)
                vals = []
                for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
                    x = v.value
                    vals.append(0.0 if x is None else float(x))
                obj = float(pyo.value(objs[0].expr)) if objs else None
                if obj is None or math.isnan(obj) or math.isinf(obj):
                    res["status"] = "error"; res["detail"] = "bad objective"
                else:
                    res.update(status="optimal", objective=obj, values=vals)
            elif tc in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
                res["status"] = "infeasible"
            elif tc == TerminationCondition.unbounded:
                res["status"] = "unbounded"
            elif tc in (TerminationCondition.maxTimeLimit,):
                res["status"] = "timeout"
            else:
                res["status"] = "error"; res["detail"] = str(tc)
    except BaseException:  # noqa: BLE001
        res["status"] = "error"
        res["detail"] = traceback.format_exc(limit=3)[-1500:]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
