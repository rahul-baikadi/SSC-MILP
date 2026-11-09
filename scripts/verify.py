"""Verifier driver.
  python scripts/verify.py solve  --benchmark mamo_complex
  python scripts/verify.py run    --benchmark mamo_complex --n 10      # pilot
  python scripts/verify.py run    --benchmark mamo_complex
Outputs go to runs/<bench>_verify/ (or --out)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssc import verify as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solve", "run", "redo"])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--run", default=None, help="pool run to verify (default = benchmark)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pids", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--problem-workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4000)
    a = ap.parse_args()
    run = a.run or a.benchmark
    out = a.out or f"{run}_verify"
    if a.cmd == "solve":
        V.solve_benchmark(run, out, workers=a.workers)
    elif a.cmd == "redo":
        from ssc.anthropic_client import AnthropicClient, DEFAULT_MODEL
        c = AnthropicClient(model=a.model or DEFAULT_MODEL)
        V.redo_unparsed(c, run, out, max_tokens=a.max_tokens, workers=a.workers)
        print(f"usage={c.usage} cost=${c.cost():.2f}")
    else:
        from ssc.anthropic_client import AnthropicClient, DEFAULT_MODEL
        c = AnthropicClient(model=a.model or DEFAULT_MODEL)
        V.verify_benchmark(c, run, out, pids=a.pids.split(",") if a.pids else None, n=a.n,
                           problem_workers=a.problem_workers)
        print(f"usage={c.usage} cost=${c.cost():.2f}")


if __name__ == "__main__":
    main()
