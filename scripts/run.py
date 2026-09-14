#!/usr/bin/env python3
"""SSC pipeline driver.

  python scripts/run.py generate  --benchmark nl4opt [--n 100]
  python scripts/run.py extract   --benchmark nl4opt
  python scripts/run.py score     --benchmark nl4opt [--no-lp]
  python scripts/run.py evaluate  --benchmark nl4opt
  python scripts/run.py judge     --benchmark nl4opt         (optional baseline)
  python scripts/run.py report    --benchmark nl4opt
  python scripts/run.py all       --benchmark nl4opt [--n 100] [--judge]
  python scripts/run.py check     (environment / solver / API sanity check)

Outputs go to runs/<benchmark>/{pools,requirements,scores,eval,judge}/ and
runs/<benchmark>/report.{json,md}.  Every stage is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ssc import config as C  # noqa: E402
from ssc.data import load_benchmark  # noqa: E402


def log(msg: str):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def get_llm(args):
    from ssc.llm import LLMClient
    return LLMClient(model=args.model, cache_dir=ROOT / "cache" / "llm")


def cmd_check(args):
    import shutil
    import pyomo.environ as pyo
    import pyomo.version
    log(f"python {sys.version.split()[0]}  pyomo {pyomo.version.version}")
    log(f"glpsol: {shutil.which('glpsol')}  glpk available: {pyo.SolverFactory('glpk').available(exception_flag=False)}")
    try:
        import pyscipopt
        log(f"pyscipopt {pyscipopt.__version__}")
    except Exception as e:  # noqa: BLE001
        log(f"pyscipopt MISSING: {e}")
    from ssc.sandbox import lp_relaxation, parse_candidate, scip_solve
    code = (ROOT / "tests" / "fixtures" / "ducks_ok.py").read_text()
    log(f"sandbox defects: {parse_candidate(code)['defects']}")
    log(f"lp: {lp_relaxation(code)['status']}  scip: {scip_solve(code)}")
    if not args.no_api:
        llm = get_llm(args)
        try:
            models = llm.list_models()
            log(f"API ok; models: {models}")
            r = llm.chat([{"role": "user", "content": "Reply with the single word OK."}],
                         temperature=0.0, max_tokens=5, tag="check")
            log(f"chat ok: {r['content']!r} model={r.get('model')} usage={r['usage']}")
        except Exception as e:  # noqa: BLE001
            log(f"API check FAILED: {e}")


def run_dir(args):
    return ROOT / "runs" / (args.run_name or args.benchmark)


def cmd_generate(args):
    from ssc.generate import generate_benchmark
    if args.no_templates:
        C.USE_TEMPLATES = False
    if args.repair_rounds is not None:
        C.REPAIR_ROUNDS = args.repair_rounds
    problems = load_benchmark(args.benchmark, args.n)
    generate_benchmark(get_llm(args), problems, run_dir(args),
                       problem_workers=args.problem_workers, recipe_workers=args.recipe_workers, log=log)


def cmd_extract(args):
    from ssc.extract import extract_benchmark
    problems = load_benchmark(args.benchmark, args.n)
    extract_benchmark(get_llm(args), problems, run_dir(args), workers=args.workers, log=log)


def cmd_score(args):
    from ssc.evaluate import score_benchmark
    problems = load_benchmark(args.benchmark, args.n)
    score_benchmark(problems, run_dir(args), run_lp=not args.no_lp,
                    workers=args.workers, pool_workers=args.pool_workers, log=log)


def cmd_evaluate(args):
    from ssc.evaluate import evaluate_benchmark
    problems = load_benchmark(args.benchmark, args.n)
    evaluate_benchmark(problems, run_dir(args), workers=args.workers,
                       pool_workers=args.pool_workers, log=log)


def cmd_judge(args):
    from ssc.judge import judge_benchmark
    problems = load_benchmark(args.benchmark, args.n)
    judge_benchmark(get_llm(args), problems, run_dir(args), workers=args.workers, log=log)


def cmd_report(args):
    from ssc.metrics import format_report, load_results, summarize
    problems = load_benchmark(args.benchmark, args.n)
    out = run_dir(args)
    rows = load_results(out, problems)
    summ = summarize(rows)
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    txt = format_report(args.benchmark, summ)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


def cmd_all(args):
    cmd_generate(args)
    cmd_extract(args)
    cmd_score(args)
    cmd_evaluate(args)
    if args.judge:
        cmd_judge(args)
    cmd_report(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "generate", "extract", "score", "evaluate", "judge", "report", "all"])
    ap.add_argument("--benchmark", default="nl4opt")
    ap.add_argument("--n", type=int, default=None, help="number of problems (default: benchmark eval size)")
    ap.add_argument("--model", default=os.environ.get("SSC_LLM_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--workers", type=int, default=4, help="threads inside a pool / API calls")
    ap.add_argument("--pool-workers", type=int, default=2, help="pools processed concurrently")
    ap.add_argument("--problem-workers", type=int, default=3)
    ap.add_argument("--recipe-workers", type=int, default=4)
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--run-name", default=None, help="output dir under runs/ (default: benchmark name)")
    ap.add_argument("--no-templates", action="store_true", help="generation control: no retrieved templates")
    ap.add_argument("--repair-rounds", type=int, default=None, help="generation control: override repair rounds")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
