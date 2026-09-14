"""Isolated worker for OFFLINE EVALUATION ONLY: rebuild the candidate, write
it as an LP file, solve the integer program with SCIP (PySCIPOpt) under a
time limit, and report the objective.  Never used by the selector.

Usage: python -m ssc.scip_worker CODE_PATH OUT_JSON TIME_LIMIT_S
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
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
    res = {"status": "error", "objective": None, "time": None, "detail": ""}
    t0 = time.time()
    try:
        import pyomo.environ as pyo
        from pyscipopt import Model as ScipModel

        with open(code_path, encoding="utf-8") as f:
            code = f.read()
        model = _rebuild(code)
        objs = [o for o in model.component_objects(pyo.Objective, active=True, descend_into=True)]
        for o in objs[1:]:
            o.deactivate()
        lp_path = os.path.join(os.path.dirname(out_path), "model.lp")
        model.write(lp_path, io_options={"symbolic_solver_labels": False})
        m = ScipModel()
        m.hideOutput()
        m.readProblem(lp_path)
        m.setParam("limits/time", tlim)
        m.setParam("display/verblevel", 0)
        m.optimize()
        st = m.getStatus()
        if st == "optimal":
            obj = float(m.getObjVal())
            if math.isnan(obj) or math.isinf(obj):
                res["status"] = "error"; res["detail"] = "non-finite objective"
            else:
                res["status"] = "optimal"; res["objective"] = obj
        elif st == "infeasible":
            res["status"] = "infeasible"
        elif st in ("unbounded", "inforunbd"):
            res["status"] = "unbounded"
        elif st == "timelimit":
            res["status"] = "timeout"
        else:
            res["status"] = "error"; res["detail"] = st
    except BaseException:  # noqa: BLE001
        res["status"] = "error"
        res["detail"] = traceback.format_exc(limit=3)[-1500:]
    res["time"] = time.time() - t0
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
