"""Offline evaluation (Section 2, Eq. 1; Section 4.3): solve every candidate
with SCIP under the frozen protocol and label it correct iff

    |v_i - v*| <= max(eps_rel * max(|v_i|, |v*|), eps_abs),   eps = 1e-4.

Candidates that fail to construct or return no finite objective are incorrect.
Also runs the selector (all variants) on each pool and stores the per-pool
score record, so that reports need only the eval/ and scores/ directories.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as C
from .sandbox import scip_solve
from .selector import score_pool


def is_correct(v, gold: float) -> bool:
    if v is None or gold is None:
        return False
    return abs(v - gold) <= max(C.EPS_REL * max(abs(v), abs(gold)), C.EPS_ABS)


def evaluate_pool(pool: dict, gold: float, workers: int = 4) -> dict:
    codes = [c["code"] for c in pool["candidates"]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(scip_solve, codes))
    out = []
    for r in res:
        out.append({"status": r["status"], "objective": r["objective"], "time": r.get("time"),
                    "correct": r["status"] == "optimal" and is_correct(r["objective"], gold),
                    "detail": (r.get("detail") or "")[-300:]})
    return {"pid": pool["pid"], "gold": gold, "candidates": out,
            "coverage": any(c["correct"] for c in out)}


def evaluate_benchmark(problems: list[dict], out_dir: str | Path, workers: int = 4,
                       pool_workers: int = 2, log=print) -> None:
    out_dir = Path(out_dir)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    gold = {p["pid"]: p["answer"] for p in problems}
    todo = [p for p in problems if (out_dir / "pools" / f"{p['pid']}.json").exists()
            and not (out_dir / "eval" / f"{p['pid']}.json").exists()]
    log(f"[evaluate] {len(todo)} pools to solve with SCIP")
    t0 = time.time()

    def _one(p):
        with open(out_dir / "pools" / f"{p['pid']}.json", encoding="utf-8") as fh:
            pool = json.load(fh)
        return evaluate_pool(pool, gold[p["pid"]], workers=workers)

    with ThreadPoolExecutor(max_workers=pool_workers) as ex:
        futs = {ex.submit(_one, p): p for p in todo}
        for n, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            r = f.result()
            with open(out_dir / "eval" / f"{p['pid']}.json", "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            nc = sum(c["correct"] for c in r["candidates"])
            log(f"[evaluate] {n}/{len(todo)} {p['pid']} correct={nc}/10 elapsed={time.time()-t0:.0f}s")


def score_benchmark(problems: list[dict], out_dir: str | Path, run_lp: bool = True,
                    workers: int = 4, pool_workers: int = 2, log=print) -> None:
    out_dir = Path(out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    todo = [p for p in problems if (out_dir / "pools" / f"{p['pid']}.json").exists()
            and not (out_dir / "scores" / f"{p['pid']}.json").exists()]
    log(f"[score] {len(todo)} pools to score")
    t0 = time.time()

    def _one(p):
        with open(out_dir / "pools" / f"{p['pid']}.json", encoding="utf-8") as fh:
            pool = json.load(fh)
        with open(out_dir / "requirements" / f"{p['pid']}.json", encoding="utf-8") as fh:
            reqs = json.load(fh)["requirements"]
        codes = [c["code"] for c in pool["candidates"]]
        r = score_pool(codes, reqs, run_lp=run_lp, workers=workers)
        r["pid"] = p["pid"]
        return r

    with ThreadPoolExecutor(max_workers=pool_workers) as ex:
        futs = {ex.submit(_one, p): p for p in todo}
        for n, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            r = f.result()
            with open(out_dir / "scores" / f"{p['pid']}.json", "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            log(f"[score] {n}/{len(todo)} {p['pid']} sel={r['selected']['SSC']} elapsed={time.time()-t0:.0f}s")
