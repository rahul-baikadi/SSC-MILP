"""Top-2 selection study: keep SSC's argmax, add one diverse second candidate.

For every problem P in a benchmark the selector already commits to

    c_1 = argmax_{c in C(P)} SSC(c)

This script leaves c_1 untouched and asks how to spend one extra downstream
solve on a second candidate c_2.  Acc@2 counts a problem as solved when either
member of {c_1, c_2} is correct under the paper's tolerance test, which is
already materialised in runs/<bench>/eval/*.json.

Four rules, all reading cached features only (no LLM call, no MILP solve):

  Top-2 naive      second highest SSC
  MMR              lambda * SSC(c_i) - (1 - lambda) * sim(M_i, M_1)
  Answer diversity highest SSC whose relaxed objective disagrees with c_1's
  Hybrid           MMR restricted to the answer-diverse set, MMR as fallback

Usage:
    python scripts/eval_top2.py <repo_root> [scores_dir] [--delta D] [--md FILE]

    scores_dir defaults to "scores" (the corrected re-score).  Pass
    "scores_v1" to reproduce the pre-fix numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import OrderedDict

import numpy as np

BENCH = ["nl4opt", "mamo_easy", "mamo_complex", "complexor", "industryor", "optibench", "optmath"]
LAMBDAS = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
DELTA = 1e-4


# ----------------------------------------------------------------- loading
def load_benchmark(root: str, bench: str, scores_dir: str) -> list[dict]:
    """One record per problem: SSC vector, similarity matrix, clone groups,
    relaxed objectives, correctness labels, and the selector's own choice."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "runs", bench, scores_dir, "*.json"))):
        s = json.load(open(f))
        e = json.load(open(f.replace(os.sep + scores_dir + os.sep, os.sep + "eval" + os.sep)))
        out.append({
            "pid": s["pid"],
            "bench": bench,
            "K": len(s["candidates"]),
            "ssc": np.asarray(s["scores"]["SSC"], dtype=float),
            "sim": np.asarray(s["similarity"], dtype=float),
            "group": [c["clone_group"] for c in s["candidates"]],
            "z": [c["lp_objective"] if c["lp_status"] == "optimal" else None
                  for c in s["candidates"]],
            "correct": np.asarray([bool(c["correct"]) for c in e["candidates"]], dtype=bool),
            "c1": int(s["selected"]["SSC"]),
        })
    return out


# ----------------------------------------------------------------- helpers
def rel_gap(a: float, b: float) -> float:
    """Scale-free disagreement, same shape as the paper's correctness test."""
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def ssc_order(p: dict) -> list[int]:
    """Candidates by descending SSC; ties broken by index, as argmax does."""
    return sorted(range(p["K"]), key=lambda i: (-p["ssc"][i], i))


def diverse_set(p: dict, c1: int, delta: float) -> list[int]:
    """Candidates whose relaxed objective disagrees with c_1's by > delta.
    Empty when c_1 has no valid relaxation or every peer agrees with it."""
    z1 = p["z"][c1]
    if z1 is None:
        return []
    return [i for i in range(p["K"])
            if i != c1 and p["z"][i] is not None and rel_gap(p["z"][i], z1) > delta]


# ----------------------------------------------------------------- the rules
def pick_naive(p: dict, c1: int, **kw) -> int:
    return next(i for i in ssc_order(p) if i != c1)


def pick_mmr(p: dict, c1: int, lam: float = 0.7, **kw) -> int:
    cands = [i for i in range(p["K"]) if i != c1]
    return max(cands, key=lambda i: (lam * p["ssc"][i] - (1.0 - lam) * p["sim"][i][c1], -i))


def pick_ansdiv(p: dict, c1: int, delta: float = DELTA, **kw) -> int:
    A = diverse_set(p, c1, delta)
    if not A:
        return pick_naive(p, c1)
    return min(A, key=lambda i: (-p["ssc"][i], i))


def pick_hybrid(p: dict, c1: int, lam: float = 0.7, delta: float = DELTA, **kw) -> int:
    A = diverse_set(p, c1, delta)
    if not A:
        return pick_mmr(p, c1, lam=lam)
    return max(A, key=lambda i: (lam * p["ssc"][i] - (1.0 - lam) * p["sim"][i][c1], -i))


RULES = OrderedDict([
    ("Top-2 naive", pick_naive),
    ("MMR", pick_mmr),
    ("Answer diversity", pick_ansdiv),
    ("Hybrid (AnsDiv + MMR)", pick_hybrid),
])


# ----------------------------------------------------------------- metrics
def acc1(P: list[dict]) -> float:
    return 100.0 * np.mean([p["correct"][p["c1"]] for p in P])


def acc2(P: list[dict], rule, **kw) -> float:
    hits = [bool(p["correct"][p["c1"]] or p["correct"][rule(p, p["c1"], **kw)]) for p in P]
    return 100.0 * np.mean(hits)


def hit_vector(P: list[dict], rule, **kw) -> np.ndarray:
    return np.asarray([bool(p["correct"][p["c1"]] or p["correct"][rule(p, p["c1"], **kw)])
                       for p in P], dtype=bool)


def acc2_random_second(P: list[dict]) -> float:
    """c_1 from SSC, c_2 uniform over the remaining K-1 candidates."""
    tot = 0.0
    for p in P:
        if p["correct"][p["c1"]]:
            tot += 1.0
        else:
            tot += p["correct"].sum() / (p["K"] - 1)
    return 100.0 * tot / len(P)


def acc2_random_pair(P: list[dict]) -> float:
    """Both drawn uniformly without replacement; no selector at all."""
    tot = 0.0
    for p in P:
        K, m = p["K"], int(p["correct"].sum())
        tot += 1.0 - math.comb(K - m, 2) / math.comb(K, 2)
    return 100.0 * tot / len(P)


def coverage(P: list[dict]) -> float:
    return 100.0 * np.mean([p["correct"].any() for p in P])


# ----------------------------------------------------------------- LOBO
def lobo(data: dict, bench: str, rule, delta: float) -> tuple[float, float]:
    """Fit lambda on the other six benchmarks, report accuracy on this one."""
    train = [p for b, P in data.items() if b != bench for p in P]
    best = max(LAMBDAS, key=lambda l: (acc2(train, rule, lam=l, delta=delta), l))
    return best, acc2(data[bench], rule, lam=best, delta=delta)


# ----------------------------------------------------------------- report
def table(rows: list[list], header: list[str]) -> str:
    def fmt(v):
        return v if isinstance(v, str) else f"{v:.1f}"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("scores_dir", nargs="?", default="scores")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--md", default=None, help="also write the report to this file")
    a = ap.parse_args()

    data = OrderedDict((b, load_benchmark(a.root, b, a.scores_dir)) for b in BENCH)
    data = OrderedDict((b, P) for b, P in data.items() if P)
    if not data:
        raise SystemExit(f"no score files under {a.root}/runs/*/{a.scores_dir}/")
    pooled = [p for P in data.values() for p in P]
    benches = list(data)
    L = []

    def row(name, fn):
        return [name] + [fn(data[b]) for b in benches] + [fn(pooled)]

    header = ["Method"] + benches + ["Pooled"]
    rows = [
        row("Acc@1 (SSC)", acc1),
        row("Random c2 given c1", acc2_random_second),
        row("Random pair", acc2_random_pair),
    ]
    for name, rule in RULES.items():
        rows.append(row(name, lambda P, r=rule: acc2(P, r, lam=a.lam, delta=a.delta)))

    for name, rule in [("MMR (LOBO lambda)", pick_mmr),
                       ("Hybrid (LOBO lambda)", pick_hybrid)]:
        per = {b: lobo(data, b, rule, a.delta) for b in benches}
        pooled_acc = sum(per[b][1] * len(data[b]) for b in benches) / len(pooled)
        rows.append([name] + [per[b][1] for b in benches] + [pooled_acc])
        L.append((name, {b: per[b][0] for b in benches}))
    rows.append(row("Pool coverage", coverage))

    L.append(("header", None))
    out = [f"# Top-2 selection  (scores={a.scores_dir}, lambda={a.lam}, delta={a.delta:g})",
           "", f"{len(pooled)} problems, K={pooled[0]['K']}", "",
           table(rows, header), ""]

    for name, lams in L:
        if lams is not None:
            out.append(f"{name}: " + ", ".join(f"{b}={v}" for b, v in lams.items()))
    out.append("")

    # ---- lambda sensitivity, pooled
    out.append("## Lambda sweep (pooled Acc@2)")
    out.append(table(
        [["MMR"] + [acc2(pooled, pick_mmr, lam=l, delta=a.delta) for l in LAMBDAS],
         ["Hybrid"] + [acc2(pooled, pick_hybrid, lam=l, delta=a.delta) for l in LAMBDAS]],
        ["Rule"] + [f"{l:.1f}" for l in LAMBDAS]))
    out.append("")

    # ---- delta sensitivity, pooled
    deltas = [1e-6, 1e-4, 1e-2, 5e-2]
    out.append("## Delta sweep (pooled Acc@2)")
    out.append(table(
        [["Answer diversity"] + [acc2(pooled, pick_ansdiv, delta=d) for d in deltas],
         ["Hybrid"] + [acc2(pooled, pick_hybrid, lam=a.lam, delta=d) for d in deltas]],
        ["Rule"] + [f"{d:g}" for d in deltas]))
    out.append("")

    # ---- paired comparison against Acc@1 and against the naive rule
    out.append("## Paired counts (pooled)")
    base = np.asarray([bool(p["correct"][p["c1"]]) for p in pooled], dtype=bool)
    naive = hit_vector(pooled, pick_naive)
    pc = []
    for name, rule in RULES.items():
        h = hit_vector(pooled, rule, lam=a.lam, delta=a.delta)
        pc.append([name, int(h.sum() - base.sum()),
                   int((h & ~naive).sum()), int((~h & naive).sum())])
    out.append(table([[r[0], str(r[1]), str(r[2]), str(r[3])] for r in pc],
                     ["Rule", "problems gained over Acc@1", "wins vs naive", "losses vs naive"]))
    out.append("")

    # ---- why the rules can or cannot help
    rec = [p for p in pooled if not p["correct"][p["c1"]] and p["correct"].any()]
    tight = [rel_gap(p["z"][i], p["z"][p["c1"]]) <= a.delta
             for p in pooled for i in range(p["K"])
             if i != p["c1"] and p["z"][i] is not None and p["z"][p["c1"]] is not None]
    out.append("## Diagnostics (pooled)")
    out.append(f"- recoverable problems (c1 wrong, pool holds a correct program): {len(rec)}")
    if rec:
        med = np.median([next(r for r, i in enumerate(ssc_order(p)) if p["correct"][i])
                         for p in rec])
        out.append(f"- median SSC rank of the best correct candidate on those: {med:.1f}")
        out.append("- mean number of correct candidates in those pools: "
                   f"{np.mean([p['correct'].sum() for p in rec]):.2f}")
        same = sum(1 for p in rec if p["group"][pick_naive(p, p["c1"])] == p["group"][p["c1"]])
        out.append(f"- naive c2 lands in c1's own clone group: {same} of {len(rec)}")
        noz = sum(1 for p in rec if p["z"][p["c1"]] is None)
        out.append(f"- c1 has no valid relaxation (answer diversity cannot fire): {noz} of {len(rec)}")
        empt = sum(1 for p in rec if not diverse_set(p, p["c1"], a.delta))
        out.append(f"- answer-diverse set empty, so Hybrid falls back to MMR: {empt} of {len(rec)}")
    if tight:
        out.append(f"- peers whose relaxed objective agrees with c1's within delta: "
                   f"{100 * np.mean(tight):.1f}% of {len(tight)}")
    out.append("")

    text = "\n".join(out)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[written] {a.md}")


if __name__ == "__main__":
    main()
