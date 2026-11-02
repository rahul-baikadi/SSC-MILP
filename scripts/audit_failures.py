"""Audit of generation failures (no correct candidate in the K=10 pool) and
Hybrid top-2 selection failures, using only cached artifacts in runs/.

No LLM call and no MILP solve: reads pools/, eval/, scores_v1/, requirements/.

Usage:
    python scripts/audit_failures.py <repo_root> [--out results/audit]
Writes:
    <out>/audit_report.md       summary tables
    <out>/generation_failures.csv   one row per no-coverage problem
    <out>/selection_failures.csv    one row per Hybrid selection failure
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_top2 import load_benchmark, pick_hybrid, BENCH  # noqa: E402

TOL = 1e-4


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def is_close(a, b, eps=TOL):
    return abs(a - b) <= max(eps * max(abs(a), abs(b)), eps)


def answer_clusters(vals, eps=TOL):
    """Group finite objective values that agree under the correctness tolerance."""
    cl = []
    for i, v in vals:
        for c in cl:
            if is_close(v, c["v"], eps):
                c["idx"].append(i)
                break
        else:
            cl.append({"v": v, "idx": [i]})
    return sorted(cl, key=lambda c: -len(c["idx"]))


def gold_decimals(g):
    t = repr(float(g))
    if "e" in t:
        return 6
    frac = t.split(".")[1].rstrip("0")
    return len(frac)


def classify(gold, ev, sc):
    """Assign a single primary cause to a no-coverage problem (priority order)."""
    K = len(ev)
    opt = [(i, c["objective"]) for i, c in enumerate(ev)
           if c["status"] == "optimal" and c["objective"] is not None]
    statuses = Counter(c["status"] for c in ev)
    if gold <= -9999 or gold >= 1e12:
        return "gold_placeholder", {}
    if not opt:
        return "no_feasible_candidate", {}
    vals = [v for _, v in opt]
    best = min(vals, key=lambda v: rel(v, gold))
    info = {"best_rel_err": rel(best, gold)}
    # 1. gold equals the LP relaxation of some integer-typed candidate -> over-integerised
    for i, c in enumerate(sc):
        z = c["lp_objective"] if c["lp_status"] == "optimal" else None
        int_typed = "var:integer" in c["evidence"] or "var:binary" in c["evidence"]
        if z is not None and int_typed and is_close(z, gold, 1e-4):
            return "integrality_too_strict", info
    # 2. gold printed at <=2 decimals and some candidate rounds to it -> evaluation artifact
    d = gold_decimals(gold)
    if d <= 2 and any(abs(v - gold) <= 0.5 * 10 ** (-d) + 1e-9 for v in vals) and d > 0:
        return "gold_precision_artifact", info
    # 3. gold integral, candidates fractional and on the relaxation side of gold -> integrality too loose
    sense = Counter(c["obj_sense"] for c in sc).most_common(1)[0][0]
    if float(gold).is_integer():
        frac = [v for v in vals if abs(v - round(v)) > 1e-6]
        side = [v for v in frac if (v < gold if sense == "min" else v > gold)]
        if frac and len(side) >= len(vals) / 2:
            return "integrality_too_loose", info
    # 3. near-miss within 1% (constant / tolerance level)
    if any(rel(v, gold) < 1e-2 for v in vals):
        return "near_miss_<1%", info
    # 4. sign / sense flip
    if any(is_close(-v, gold) for v in vals if abs(gold) > 1e-9):
        return "sign_flip", info
    # 5. unit / scale error
    for v in vals:
        if abs(v) > 1e-9 and abs(gold) > 1e-9:
            r = abs(gold / v)
            k = math.log10(r)
            if abs(k - round(k)) < 1e-3 and round(k) != 0:
                return "scale_x10^k", info
    # 6. confident wrong consensus (>=6 of K agree on a wrong value) -> possible gold / reading issue
    cl = answer_clusters(opt)
    info["top_cluster"] = len(cl[0]["idx"])
    if len(cl[0]["idx"]) >= 6:
        return "wrong_consensus_>=6", info
    if statuses.get("infeasible", 0) + statuses.get("unbounded", 0) + statuses.get("error", 0) >= K // 2:
        return "mostly_broken", info
    return "scattered_wrong", info


def auroc(scores, labels):
    """P(score of positive > score of negative); ties count half."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gen_rows, sel_rows, pool_rows = [], [], []
    feat_stats = defaultdict(lambda: [0, 0, 0, 0])   # missing_correct, missing_total, present_correct, present_total
    defect_stats = defaultdict(lambda: [0, 0])        # correct, total

    for bench in BENCH:
        P = {p["pid"]: p for p in load_benchmark(a.root, bench, "scores_v1")}
        for f in sorted(glob.glob(os.path.join(a.root, "runs", bench, "eval", "*.json"))):
            e = json.load(open(f))
            pid = e["pid"]
            s = json.load(open(f.replace("/eval/", "/scores_v1/")))
            pool = json.load(open(f.replace("/eval/", "/pools/")))
            ev, sc, gold = e["candidates"], s["candidates"], e["gold"]
            req = s.get("requirements") or json.load(open(f.replace("/eval/", "/requirements/")))["requirements"]
            req_feats = sorted({ft for r in req for ft in r["features"]})
            corr = [bool(c["correct"]) for c in ev]

            # per-candidate: which required features are missing; which defects
            for c, ok in zip(sc, corr):
                have = set(c["evidence"])
                for ft in req_feats:
                    st = feat_stats[ft]
                    if ft in have:
                        st[2] += ok; st[3] += 1
                    else:
                        st[0] += ok; st[1] += 1
                for d in set(c["defects"]) or {"(none)"}:
                    defect_stats[d][0] += ok; defect_stats[d][1] += 1

            # pool-level solve-free signals
            ssc = np.array([c["SSC"] for c in sc])
            top = int(np.argmax(ssc))
            zs = [(i, c["lp_objective"]) for i, c in enumerate(sc)
                  if c["lp_status"] == "optimal" and c["lp_objective"] is not None]
            lpcl = answer_clusters(zs, 1e-3) if zs else []
            top_mass = 0
            for c in lpcl:
                if top in c["idx"]:
                    top_mass = len(c["idx"])
            pool_rows.append({
                "bench": bench, "pid": pid, "covered": any(corr),
                "max_SSC": float(ssc.max()),
                "top_TA": sc[top]["TA"],
                "top_lp_cluster_mass": top_mass / len(sc),
                "largest_lp_cluster": (len(lpcl[0]["idx"]) / len(sc)) if lpcl else 0.0,
                "n_lp_clusters": len(lpcl),
                "frac_lp_failed": 1 - len(zs) / len(sc),
                "frac_fatal": float(np.mean([c["fatal"] for c in sc])),
            })

            statuses = Counter(c["status"] for c in ev)
            if not any(corr):
                cause, info = classify(gold, ev, sc)
                opt = [(i, c["objective"]) for i, c in enumerate(ev) if c["status"] == "optimal"]
                cl = answer_clusters(opt) if opt else []
                missing_top = [ft for ft in req_feats if ft not in sc[top]["evidence"]]
                gen_rows.append({
                    "bench": bench, "pid": pid, "gold": gold, "cause": cause,
                    "best_rel_err": round(info.get("best_rel_err", float("nan")), 5),
                    "n_optimal": statuses.get("optimal", 0),
                    "n_infeasible": statuses.get("infeasible", 0),
                    "n_unbounded": statuses.get("unbounded", 0),
                    "n_error_timeout": statuses.get("error", 0) + statuses.get("timeout", 0),
                    "n_answer_clusters": len(cl),
                    "top_answer": cl[0]["v"] if cl else "",
                    "top_answer_votes": len(cl[0]["idx"]) if cl else 0,
                    "top_missing_req_features": ";".join(missing_top),
                    "question": pool["question"][:300].replace("\n", " "),
                })
            else:
                p = P[pid]
                c1 = p["c1"]
                c2 = pick_hybrid(p, c1, lam=a.lam, delta=a.delta)
                if not (corr[c1] or corr[c2]):
                    order = sorted(range(len(sc)), key=lambda i: (-ssc[i], i))
                    ranks = [order.index(i) + 1 for i in range(len(sc)) if corr[i]]
                    grp = Counter(c["clone_group"] for c in sc)
                    z1 = sc[c1]["lp_objective"] if sc[c1]["lp_status"] == "optimal" else None
                    corr_lp_agree = [
                        (sc[i]["lp_objective"] is not None and z1 is not None
                         and sc[i]["lp_status"] == "optimal"
                         and abs(sc[i]["lp_objective"] - z1) / max(abs(z1), abs(sc[i]["lp_objective"]), 1) <= a.delta)
                        for i in range(len(sc)) if corr[i]]
                    sel_rows.append({
                        "bench": bench, "pid": pid, "n_correct": sum(corr),
                        "best_correct_SSC_rank": min(ranks),
                        "c1_clone_size": grp[sc[c1]["clone_group"]],
                        "c1_status": ev[c1]["status"], "c2_status": ev[c2]["status"],
                        "correct_LP_same_as_c1": sum(corr_lp_agree),
                        "correct_LP_failed": sum(1 for i in range(len(sc)) if corr[i] and sc[i]["lp_status"] != "optimal"),
                        "c1_minus_bestcorrect_SSC": round(ssc[c1] - max(ssc[i] for i in range(len(sc)) if corr[i]), 4),
                        "c1_TA": round(sc[c1]["TA"], 3),
                        "bestcorrect_TA": round(max(sc[i]["TA"] for i in range(len(sc)) if corr[i]), 3),
                    })

    # ---------------- write CSVs
    for name, rows in [("generation_failures.csv", gen_rows), ("selection_failures.csv", sel_rows),
                       ("pool_signals.csv", pool_rows)]:
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # ---------------- report
    L = ["# Failure audit (scores_v1, Hybrid lambda=%.2f delta=%g)\n" % (a.lam, a.delta)]
    L.append(f"Generation failures (no correct candidate): **{len(gen_rows)}**  ")
    L.append(f"Hybrid selection failures: **{len(sel_rows)}**\n")

    L.append("## Generation-failure causes by benchmark\n")
    causes = sorted(Counter(r["cause"] for r in gen_rows).items(), key=lambda x: -x[1])
    cnames = [c for c, _ in causes]
    L.append("| cause | " + " | ".join(BENCH) + " | total |")
    L.append("|---" * (len(BENCH) + 2) + "|")
    for c in cnames:
        row = [sum(1 for r in gen_rows if r["cause"] == c and r["bench"] == b) for b in BENCH]
        L.append(f"| {c} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    tot = [sum(1 for r in gen_rows if r["bench"] == b) for b in BENCH]
    L.append("| **total** | " + " | ".join(map(str, tot)) + f" | {sum(tot)} |\n")

    L.append("## Candidate status inside no-coverage pools\n")
    agg = Counter()
    for r in gen_rows:
        agg["optimal (wrong value)"] += r["n_optimal"]; agg["infeasible"] += r["n_infeasible"]
        agg["unbounded"] += r["n_unbounded"]; agg["error/timeout"] += r["n_error_timeout"]
    n = sum(agg.values())
    for k, v in agg.items():
        L.append(f"- {k}: {v} ({100*v/n:.1f}%)")
    L.append("")

    L.append("## Can a no-coverage pool be detected without the gold answer? (AUROC for predicting 'no correct candidate')\n")
    y = [not r["covered"] for r in pool_rows]
    L.append("| signal | direction | AUROC |")
    L.append("|---|---|---|")
    for k, sign in [("max_SSC", -1), ("top_TA", -1), ("top_lp_cluster_mass", -1), ("largest_lp_cluster", -1),
                    ("n_lp_clusters", 1), ("frac_lp_failed", 1), ("frac_fatal", 1)]:
        L.append(f"| {k} | {'low = risky' if sign < 0 else 'high = risky'} | "
                 f"{auroc([sign * r[k] for r in pool_rows], y):.3f} |")
    L.append("")
    # simple trigger: top LP cluster mass threshold
    L.append("### Regeneration trigger: top candidate's LP-objective cluster mass <= t\n")
    L.append("| t | pools flagged | of which no-coverage | no-coverage caught (recall) | covered pools flagged |")
    L.append("|---|---|---|---|---|")
    nneg = sum(y)
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        fl = [r for r in pool_rows if r["top_lp_cluster_mass"] <= t]
        bad = sum(1 for r in fl if not r["covered"])
        L.append(f"| {t} | {len(fl)} | {bad} | {100*bad/nneg:.1f}% | {len(fl)-bad} |")
    L.append("")

    L.append("## Which diagnostics are actionable? Missing required TA feature vs. correctness (candidate level)\n")
    L.append("| feature | P(correct \\| missing) | n missing | P(correct \\| present) | n present | lift |")
    L.append("|---|---|---|---|---|---|")
    fr = []
    for ft, (mc, mt, pc, pt) in feat_stats.items():
        if mt >= 30 and pt >= 30:
            pm, pp = mc / mt, pc / pt
            fr.append((pp - pm, ft, pm, mt, pp, pt))
    for d, ft, pm, mt, pp, pt in sorted(fr, reverse=True):
        L.append(f"| {ft} | {pm:.3f} | {mt} | {pp:.3f} | {pt} | {d:+.3f} |")
    L.append("")
    L.append("## Static defects vs. correctness (candidate level)\n")
    L.append("| defect | P(correct) | n |")
    L.append("|---|---|---|")
    for d, (c, t) in sorted(defect_stats.items(), key=lambda x: -x[1][1]):
        L.append(f"| {d} | {c/t:.3f} | {t} |")
    L.append("")

    L.append("## Hybrid selection failures\n")
    L.append("| bench | pid | #correct | best correct SSC rank | c1 clone size | correct w/ same LP obj as c1 | correct w/ failed LP | SSC(c1)-SSC(best correct) | TA c1 | TA best correct |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sel_rows:
        L.append(f"| {r['bench']} | {r['pid']} | {r['n_correct']} | {r['best_correct_SSC_rank']} | {r['c1_clone_size']} | "
                 f"{r['correct_LP_same_as_c1']} | {r['correct_LP_failed']} | {r['c1_minus_bestcorrect_SSC']} | {r['c1_TA']} | {r['bestcorrect_TA']} |")
    open(os.path.join(a.out, "audit_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
