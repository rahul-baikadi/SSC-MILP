"""Pointwise, solution-aware LLM verification of candidate models (gold-free).

  solve_benchmark   SCIP-solve every candidate once -> status, objective, non-zero vars
  verify_benchmark  one verifier call per candidate -> verdict, p_correct, errors, assumptions

Outputs (per benchmark):  runs/<bench>_verify/solutions/<pid>.json
                          runs/<bench>_verify/verdicts/<pid>.json
The reference answer is never read here.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .prompts_verify import SYSTEM_FULL, verifier_blocks

ROOT = Path(__file__).resolve().parents[1]


def solve_candidate(code: str, tlim: float = 60.0) -> dict:
    d = tempfile.mkdtemp(prefix="ssc_solve_")
    try:
        cp, op = Path(d) / "cand.py", Path(d) / "out.json"
        cp.write_text(code, encoding="utf-8")
        try:
            subprocess.run([sys.executable, "-m", "ssc.solve_worker", str(cp), str(op), str(tlim)],
                           cwd=str(ROOT), timeout=tlim + 60, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "objective": None, "vars": {}, "detail": "worker timeout"}
        if op.exists():
            return json.loads(op.read_text(encoding="utf-8"))
        return {"status": "error", "objective": None, "vars": {}, "detail": "no worker output"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def solve_benchmark(run: str, out: str, workers: int = 6, tlim: float = 60.0, log=print) -> None:
    src, dst = ROOT / "runs" / run / "pools", ROOT / "runs" / out / "solutions"
    dst.mkdir(parents=True, exist_ok=True)
    todo = [f for f in sorted(src.glob("*.json")) if not (dst / f.name).exists()]
    log(f"[solve] {run}: {len(todo)} pools")
    t0 = time.time()
    for n, f in enumerate(todo, 1):
        pool = json.loads(f.read_text(encoding="utf-8"))
        codes = [c["code"] for c in pool["candidates"]]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sols = list(ex.map(lambda c: solve_candidate(c, tlim), codes))
        tmp = dst / (f.name + ".tmp")
        tmp.write_text(json.dumps({"pid": pool["pid"], "solutions": sols}), encoding="utf-8")
        tmp.replace(dst / f.name)            # atomic: readers never see a half-written file
        if n % 10 == 0 or n == len(todo):
            log(f"[solve] {run} {n}/{len(todo)} {time.time() - t0:.0f}s")


def parse_verdict(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    obj = {}
    if s >= 0 and e > s:
        blob = text[s:e + 1]
        try:
            obj = json.loads(blob)
        except Exception:  # noqa: BLE001
            blob = re.sub(r",\s*([}\]])", r"\1", blob)
            try:
                obj = json.loads(blob)
            except Exception:  # noqa: BLE001
                obj = {}
    v = str(obj.get("verdict", "")).strip().lower()
    try:
        p = float(obj.get("p_correct"))
        p = min(max(p, 0.0), 1.0)
    except Exception:  # noqa: BLE001
        p = None
    ok = v in ("correct", "incorrect") and p is not None
    return {"parsed": ok, "verdict": v if ok else None, "accept": v == "correct" if ok else None,
            "p_correct": p, "errors": obj.get("errors") or [], "assumptions": obj.get("assumptions") or [],
            "reason": obj.get("reason", "")}


def redo_unparsed(client, run: str, out: str, max_tokens: int = 4000, workers: int = 8, log=print) -> None:
    """Re-run only the verdicts that could not be parsed (e.g. cut off at the token cap), with the
    same prompt and a larger cap. Each re-done entry records the cap it was produced with."""
    pools = ROOT / "runs" / run / "pools"
    sols_dir = ROOT / "runs" / out / "solutions"
    vdir = ROOT / "runs" / out / "verdicts"
    jobs = []
    for f in sorted(vdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for v in d["verdicts"]:
            if not v["parsed"]:
                jobs.append((f.name, v["index"]))
    log(f"[redo] {run}: {len(jobs)} unparsed verdicts, max_tokens={max_tokens}")
    cache = {}

    def _job(job):
        name, i = job
        if name not in cache:
            cache[name] = (json.loads((pools / name).read_text(encoding="utf-8")),
                           json.loads((sols_dir / name).read_text(encoding="utf-8"))["solutions"])
        pool, sols = cache[name]
        r = client.chat(SYSTEM_FULL, verifier_blocks(pool["question"], pool["candidates"][i]["code"], sols[i]),
                        max_tokens=max_tokens, tag=f"{pool['pid']}:verify:{i}")
        return name, i, dict(parse_verdict(r["content"]), index=i, raw=r["content"], usage=r.get("usage", {}),
                             max_tokens=max_tokens, stop_reason=r.get("stop_reason"))
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fu in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), 1):
            try:
                name, i, v = fu.result()
                results.setdefault(name, {})[i] = v
            except Exception as e:  # noqa: BLE001
                log(f"[redo] FAILED: {type(e).__name__}: {str(e)[:200]}")
            if k % 50 == 0 or k == len(jobs):
                log(f"[redo] {k}/{len(jobs)} cost=${client.cost():.2f} {time.time() - t0:.0f}s")
    for name, upd in results.items():
        d = json.loads((vdir / name).read_text(encoding="utf-8"))
        d["verdicts"] = [upd.get(v["index"], v) for v in d["verdicts"]]
        tmp = vdir / (name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(vdir / name)
    still = sum(1 for f in vdir.glob("*.json") for v in json.loads(f.read_text(encoding="utf-8"))["verdicts"] if not v["parsed"])
    log(f"[redo] {run}: done, still unparsed = {still}, cost=${client.cost():.2f}")
    (ROOT / "runs" / out / "redo_manifest.json").write_text(json.dumps(
        {"model": client.model, "max_tokens": max_tokens, "usage": client.usage, "cost_usd": round(client.cost(), 4),
         "still_unparsed": still, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")


def verify_benchmark(client, run: str, out: str, pids: list[str] | None = None, n: int | None = None,
                     problem_workers: int = 4, cand_workers: int = 5, max_tokens: int = 700, log=print) -> None:
    pools = ROOT / "runs" / run / "pools"
    sols_dir = ROOT / "runs" / out / "solutions"
    dst = ROOT / "runs" / out / "verdicts"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(pools.glob("*.json"))
    if pids:
        files = [f for f in files if f.stem in set(pids)]
    if n:
        files = files[:n]
    todo = [f for f in files if not (dst / f.name).exists() and (sols_dir / f.name).exists()]
    missing = sum(1 for f in files if not (sols_dir / f.name).exists())
    log(f"[verify] {run}: {len(todo)} pools, model={client.model}" + (f" ({missing} still unsolved, skipped)" if missing else ""))
    t0 = time.time()

    def _one(f):
        pool = json.loads(f.read_text(encoding="utf-8"))
        sols = json.loads((sols_dir / f.name).read_text(encoding="utf-8"))["solutions"]
        q = pool["question"]

        def _cand(i):
            r = client.chat(SYSTEM_FULL, verifier_blocks(q, pool["candidates"][i]["code"], sols[i]),
                            max_tokens=max_tokens, tag=f"{pool['pid']}:verify:{i}")
            return dict(parse_verdict(r["content"]), index=i, raw=r["content"], usage=r.get("usage", {}))
        # first call alone writes the cache prefix; the rest read it
        first = _cand(0)
        with ThreadPoolExecutor(max_workers=cand_workers) as ex:
            rest = list(ex.map(_cand, range(1, len(pool["candidates"]))))
        return pool["pid"], [first] + rest

    with ThreadPoolExecutor(max_workers=problem_workers) as ex:
        futs = {ex.submit(_one, f): f for f in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            f = futs[fu]
            try:
                pid, vs = fu.result()
            except Exception as e:  # noqa: BLE001
                log(f"[verify] {f.stem} FAILED: {type(e).__name__}: {str(e)[:200]}")
                continue
            (dst / f.name).write_text(json.dumps({"pid": pid, "model": client.model, "verdicts": vs}, indent=1),
                                      encoding="utf-8")
            log(f"[verify] {k}/{len(todo)} {pid} accept={sum(bool(v['accept']) for v in vs)}/{len(vs)} "
                f"parsed={sum(v['parsed'] for v in vs)} cost=${client.cost():.2f} {time.time() - t0:.0f}s")
    (ROOT / "runs" / out / "verify_manifest.json").write_text(json.dumps(
        {"model": client.model, "usage": client.usage, "cost_usd": round(client.cost(), 4),
         "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
