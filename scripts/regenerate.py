"""Round-2 regeneration driver.

  python scripts/regenerate.py plan  --benchmark optmath --n 5          # no API: dump slots + prompts
  python scripts/regenerate.py run   --benchmark optmath --arm branch   # blind | feedback | branch
  python scripts/run.py evaluate --benchmark optmath --run-name optmath_r2_branch   # SCIP labels, new cands only
  python scripts/regenerate.py merge --benchmark optmath --arm branch   # union pools + eval
  python scripts/run.py score    --benchmark optmath --run-name optmath_r2_branch_union
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssc.data import load_benchmark  # noqa: E402
from ssc.prompts_regen import branch_messages, decision_messages, feedback_messages  # noqa: E402
from ssc import regenerate as R  # noqa: E402


def scores_dir(r1: Path, name: str | None) -> str:
    if name:
        return name
    return "scores" if (r1 / "scores").exists() else "scores_v1"


def cmd_plan(a):
    r1 = ROOT / "runs" / a.benchmark
    sd = scores_dir(r1, a.scores_dir)
    # gold-free: problems come from the round-1 pool files
    probs = [{"pid": f.stem, "question": json.load(open(f))["question"]}
             for f in sorted((r1 / "pools").glob("*.json"))]
    if a.pids:
        keep = set(a.pids.split(","))
        probs = [p for p in probs if p["pid"] in keep]
    probs = probs[: a.n]
    out = []
    for p in probs:
        pool = json.load(open(r1 / "pools" / f"{p['pid']}.json"))
        score = json.load(open(r1 / sd / f"{p['pid']}.json"))
        reqs = json.load(open(r1 / "requirements" / f"{p['pid']}.json"))["requirements"]
        slots = R.build_slots(pool, score, reqs)
        out.append(f"# {p['pid']}  trigger={R.triggered(slots['risk'])}  risk={json.dumps(slots['risk'])}\n")
        out.append("## Stage A prompt\n")
        for m in decision_messages(p["question"], slots):
            out.append(f"### [{m['role']}]\n{m['content']}\n")
        if a.show_feedback:
            out.append("## Feedback-arm prompt (user)\n" + feedback_messages(p["question"], slots)[1]["content"])
        demo = [{"decision": "domain_of_counts", "option": "NonNegativeIntegers", "evidence": "none"}]
        out.append("## Stage B prompt shape (example assignment)\n" + branch_messages(p["question"], demo)[1]["content"])
        out.append("\n---\n")
    dest = ROOT / "runs" / f"{a.benchmark}_r2_plan.md"
    dest.write_text("\n".join(out))
    print(f"wrote {dest}")


def cmd_run(a):
    from ssc.llm import LLMClient
    r1 = ROOT / "runs" / a.benchmark
    out = ROOT / "runs" / f"{a.benchmark}_r2_{a.arm}"
    llm = LLMClient(model=a.model)
    probs = load_benchmark(a.benchmark, a.n)
    R.regenerate_benchmark(llm, probs, r1, out, a.arm, scores_dir=scores_dir(r1, a.scores_dir), B=a.B,
                           problem_workers=a.problem_workers, workers=a.workers)


def cmd_merge(a):
    r1 = ROOT / "runs" / a.benchmark
    r2 = ROOT / "runs" / f"{a.benchmark}_r2_{a.arm}"
    R.merge_union(r1, r2, ROOT / "runs" / f"{a.benchmark}_r2_{a.arm}_union")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "merge"])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--arm", choices=["blind", "feedback", "branch"], default="branch")
    ap.add_argument("--B", type=int, default=R.B_DEFAULT)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pids", default=None)
    ap.add_argument("--scores-dir", default=None)
    ap.add_argument("--model", default=os.environ.get("SSC_LLM_MODEL", "deepseek-flash"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--problem-workers", type=int, default=3)
    ap.add_argument("--show-feedback", action="store_true")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
