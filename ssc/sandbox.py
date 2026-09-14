"""Subprocess drivers for the three isolated workers:
  * sandbox_worker  – construct model, extract standard linear repr (no solver)
  * lp_worker       – bounded GLPK solve of the continuous relaxation
  * scip_worker     – offline SCIP solve for correctness labels (evaluation only)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config
from .ast_checks import static_check

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_worker(module: str, code: str, timeout: float, extra_args: list[str] | None = None) -> dict:
    """Run `python -m ssc.<module> code.py out.json [extra]` in a temp dir.

    Returns the parsed JSON, or a dict with a single *_timeout /
    *_subprocess_error / *_bad_json defect (prefix chosen by the caller via
    the 'prefix' key it adds afterwards)."""
    with tempfile.TemporaryDirectory(prefix="ssc_") as td:
        code_path = os.path.join(td, "candidate.py")
        out_path = os.path.join(td, "out.json")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONHASHSEED"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        cmd = [sys.executable, "-m", f"ssc.{module}", code_path, out_path] + (extra_args or [])

        def _limits():
            try:
                import resource
                lim = int(os.environ.get("SSC_WORKER_MEM_BYTES", 6 * 1024 ** 3))
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except Exception:  # noqa: BLE001
                pass

        try:
            proc = subprocess.run(cmd, cwd=td, env=env, capture_output=True, text=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return {"_status": "timeout"}
        if not os.path.exists(out_path):
            return {"_status": "subprocess_error",
                    "stderr": (proc.stderr or "")[-2000:], "returncode": proc.returncode}
        try:
            with open(out_path, encoding="utf-8") as f:
                res = json.load(f)
        except Exception:
            return {"_status": "bad_json", "stderr": (proc.stderr or "")[-2000:]}
        res["_status"] = "ok"
        return res


def parse_candidate(code: str, timeout: float = config.SANDBOX_TIMEOUT_S) -> dict:
    """Static checks + isolated construction.  Result keys:
    defects (merged, sorted), static (dict), model (worker JSON or None)."""
    static = static_check(code)
    defects = set(static["defects"])
    model = None
    if "syntax_error" not in defects:
        res = _run_worker("sandbox_worker", code, timeout)
        st = res.pop("_status")
        if st == "ok":
            model = res
            defects |= set(res.get("defects", []))
        else:
            defects.add(f"sandbox_{st}")
            model = {"error": res.get("stderr", "")}
    return {"defects": sorted(defects), "static": static, "model": model}


def lp_relaxation(code: str) -> dict:
    """Bounded GLPK relaxation.  Keys: status ('optimal','infeasible',
    'unbounded','timeout','error'), objective, values (list[float])."""
    res = _run_worker("lp_worker", code, timeout=config.LP_WATCHDOG_S,
                      extra_args=[str(config.LP_TIME_LIMIT_S)])
    st = res.pop("_status")
    if st != "ok":
        return {"status": "timeout" if st == "timeout" else "error",
                "objective": None, "values": [], "detail": res.get("stderr", "")}
    return res


def scip_solve(code: str, time_limit: float = config.SCIP_TIME_LIMIT_S) -> dict:
    """Offline evaluation solve.  Keys: status, objective, time, detail."""
    res = _run_worker("scip_worker", code, timeout=time_limit + 60,
                      extra_args=[str(time_limit)])
    st = res.pop("_status")
    if st != "ok":
        return {"status": "timeout" if st == "timeout" else "error",
                "objective": None, "detail": res.get("stderr", "")}
    return res
