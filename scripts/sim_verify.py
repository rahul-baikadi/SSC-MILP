"""Offline analysis of verifier labels: verifier quality and verification policies.
python scripts/sim_verify.py --benchmarks mamo_complex optmath
Writes results/verify/report.md, results/verify/policies.csv, results/verify/*.png"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssc.eval_top2 import load_benchmark

def close(a, b):
    return a is not None and b is not None and abs(a - b) <= max(1e-4 * max(abs(a), abs(b)), 1e-4)

def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1.0)
def same(z1, z2): return z1 is not None and z2 is not None and rel(z1, z2) <= 1e-3

def load(benches):
    corr = json.load(open(ROOT / "results/labels/sirl_corrections.json"))["corrections"]
    P = []
    for b in benches:
        for p in load_benchmark(str(ROOT), b, "scores_v1"):
            pid = p["pid"]
            V = json.load(open(ROOT / "runs" / f"{b}_verify/verdicts/{pid}.json"))["verdicts"]
            ev = json.load(open(ROOT / "runs" / b / "eval" / f"{pid}.json"))
            objs = [c["objective"] if c["status"] == "optimal" else None for c in ev["candidates"]]
            lab = {"orig": [bool(x) for x in p["correct"]]}
            if pid in corr:
                g = corr[pid]
                lab["sirl"] = None if g is None else [close(o, g) for o in objs]
            else:
                lab["sirl"] = lab["orig"]
            P.append({"bench": b, "pid": pid, "ssc": np.asarray(p["ssc"]), "sim": np.asarray(p["sim"]), "z": p["z"],
                      "c1": int(p["c1"]), "acc": [bool(v["accept"]) for v in V],
                      "pc": [v["p_correct"] if v["p_correct"] is not None else 0.0 for v in V],
                      "parsed": [v["parsed"] for v in V], "lab": lab})
    return P

# ------------------------------------------------------------------ orders
def o_ssc(p): return sorted(range(len(p["ssc"])), key=lambda i: (-p["ssc"][i], i))
def o_cluster(p):
    f, r = [], []
    for i in o_ssc(p):
        (f if all(not same(p["z"][i], p["z"][j]) for j in f) else r).append(i)
    return f + r
def o_hybrid(p, lam):
    K = len(p["ssc"]); order = [p["c1"]]; rest = set(range(K)) - {p["c1"]}
    while rest:
        new = [i for i in rest if all(not same(p["z"][i], p["z"][j]) for j in order)]
        A = new if new else list(rest)
        i = max(A, key=lambda i: (lam * p["ssc"][i] - (1 - lam) * max(p["sim"][i][j] for j in order), -i))
        order.append(i); rest.remove(i)
    return order
ORDERS = {"SSC": o_ssc, "Cluster-first": o_cluster,
          "Hybrid λ=0.1": lambda p: o_hybrid(p, 0.1), "Hybrid λ=0.7": lambda p: o_hybrid(p, 0.7)}

def run_policy(p, order, correct, tau, b, skip_dup):
    """Walk the order; verify (cost 1); accept if verdict correct and p>=tau. Optionally skip candidates
    whose LP answer equals an already-rejected one. Returns (calls, outcome) outcome in {right, wrong, abstain}."""
    calls, rejected = 0, []
    for i in order:
        if calls >= b: break
        if skip_dup and any(same(p["z"][i], p["z"][j]) for j in rejected): continue
        calls += 1
        if p["acc"][i] and p["pc"][i] >= tau:
            return calls, ("right" if correct[i] else "wrong")
        rejected.append(i)
    return calls, "abstain"

def evaluate(P, labset, tau, b, order_name, skip_dup, rng=None):
    rows = []
    for p in P:
        correct = p["lab"][labset]
        if correct is None: continue
        if order_name == "Random":
            res = [run_policy(p, list(rng.permutation(len(correct))), correct, tau, b, skip_dup) for _ in range(50)]
            calls = np.mean([r[0] for r in res]); oc = {k: np.mean([r[1] == k for r in res]) for k in ("right", "wrong", "abstain")}
        else:
            c, o = run_policy(p, ORDERS[order_name](p), correct, tau, b, skip_dup)
            calls = c; oc = {k: float(o == k) for k in ("right", "wrong", "abstain")}
        rows.append((calls, oc))
    n = len(rows)
    calls = np.mean([r[0] for r in rows])
    right = np.mean([r[1]["right"] for r in rows]); wrong = np.mean([r[1]["wrong"] for r in rows]); abst = np.mean([r[1]["abstain"] for r in rows])
    answered = right + wrong
    return {"n": n, "calls": calls, "right": 100 * right, "wrong": 100 * wrong, "abstain": 100 * abst,
            "answered_acc": 100 * right / answered if answered else float("nan")}

def verifier_quality(P, labset):
    tp = fp = fn = tn = 0; scores, ys = [], []; nocov = nocov_fa = 0; mixed = clusters = 0
    for p in P:
        c = p["lab"][labset]
        if c is None: continue
        for a, pc, y in zip(p["acc"], p["pc"], c):
            tp += a and y; fn += (not a) and y; fp += a and (not y); tn += (not a) and (not y)
            scores.append(pc); ys.append(y)
        if not any(c):
            nocov += 1; nocov_fa += any(p["acc"])
        groups = {}
        for i, z in enumerate(p["z"]):
            if z is None: continue
            k = next((g for g in groups if same(z, g)), z); groups.setdefault(k, []).append(i)
        for g in groups.values():
            if len(g) > 1:
                clusters += 1; mixed += len({p["acc"][i] for i in g}) > 1
    s, y = np.array(scores), np.array(ys, bool)
    pos, neg = s[y], s[~y]
    auroc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    bins = np.linspace(0, 1, 11); ece = 0.0; rel_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.sum():
            ece += m.sum() / len(s) * abs(s[m].mean() - y[m].mean()); rel_rows.append((s[m].mean(), y[m].mean(), int(m.sum())))
    return {"TPR": tp / (tp + fn), "FPR": fp / (fp + tn), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "n_correct": tp + fn, "n_wrong": fp + tn, "AUROC": auroc, "ECE": ece,
            "nocov_pools": nocov, "nocov_false_accept": nocov_fa, "clone_clusters": clusters, "clone_mixed": mixed,
            "reliability": rel_rows}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmarks", nargs="+", default=["mamo_complex", "optmath"])
    a = ap.parse_args()
    out = ROOT / "results/verify"; out.mkdir(parents=True, exist_ok=True)
    P = load(a.benchmarks)
    rng = np.random.default_rng(0)
    L = ["# Verifier experiment (Sonnet 5.5, pointwise, solution-aware)\n",
         f"Benchmarks: {', '.join(a.benchmarks)}; {len(P)} problems x 10 candidates. "
         f"Unparsed verdicts (counted as reject): {sum(1 for p in P for x in p['parsed'] if not x)}.\n"]
    # verifier quality
    L.append("## Verifier quality\n")
    L.append("| labels | scope | TPR | FPR | precision | AUROC(p) | ECE | no-correct pools with a false accept | clone clusters with mixed verdicts |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    qual = {}
    for lab in ("orig", "sirl"):
        for scope in ["all"] + a.benchmarks:
            Q = verifier_quality([p for p in P if scope == "all" or p["bench"] == scope], lab)
            qual[(lab, scope)] = Q
            L.append(f"| {lab} | {scope} | {Q['TPR']:.2f} | {Q['FPR']:.2f} | {Q['precision']:.2f} | {Q['AUROC']:.2f} | {Q['ECE']:.3f} | "
                     f"{Q['nocov_false_accept']}/{Q['nocov_pools']} | {Q['clone_mixed']}/{Q['clone_clusters']} |")
    L.append("")
    # baselines without verifier
    L.append("## Policies\n")
    L.append("Baseline without verifier: return SSC top-1 always (0 calls).\n")
    rows = []
    for lab in ("orig", "sirl"):
        for scope in ["all"] + a.benchmarks:
            PP = [p for p in P if (scope == "all" or p["bench"] == scope) and p["lab"][lab] is not None]
            acc = 100 * np.mean([p["lab"][lab][p["c1"]] for p in PP])
            rows.append({"labels": lab, "scope": scope, "order": "SSC top-1, no verifier", "skip_dup": "-", "tau": "-", "budget": 0,
                         "n": len(PP), "calls": 0.0, "right": acc, "wrong": 100 - acc, "abstain": 0.0, "answered_acc": acc})
            for oname in list(ORDERS) + ["Random"]:
                for skip in (False, True):
                    if oname == "SSC" and skip: continue
                    for tau in (0.5, 0.7, 0.9):
                        for b in (1, 2, 3, 5, 10):
                            r = evaluate(PP, lab, tau, b, oname, skip, rng)
                            rows.append({"labels": lab, "scope": scope, "order": oname, "skip_dup": skip, "tau": tau, "budget": b, **r})
    with open(out / "policies.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def table(lab, scope, tau, budgets=(1, 3, 10)):
        L.append(f"### labels={lab}, scope={scope}, accept threshold τ={tau}\n")
        L.append("| order | skip same-answer | budget | calls/problem | right % | wrong % | abstain % | accuracy when answered % |")
        L.append("|---|---|---|---|---|---|---|---|")
        base = next(r for r in rows if r["labels"] == lab and r["scope"] == scope and r["budget"] == 0)
        L.append(f"| SSC top-1, no verifier | - | 0 | 0.00 | {base['right']:.1f} | {base['wrong']:.1f} | 0.0 | {base['answered_acc']:.1f} |")
        for r in rows:
            if r["labels"] == lab and r["scope"] == scope and r["tau"] == tau and r["budget"] in budgets:
                L.append(f"| {r['order']} | {r['skip_dup']} | {r['budget']} | {r['calls']:.2f} | {r['right']:.1f} | {r['wrong']:.1f} | "
                         f"{r['abstain']:.1f} | {r['answered_acc']:.1f} |")
        L.append("")
    for lab in ("orig", "sirl"):
        table(lab, "all", 0.7)
    for scope in a.benchmarks:
        table("orig", scope, 0.7, budgets=(10,))
    L.append("### Effect of the accept threshold (orig labels, all, Hybrid λ=0.1 + skip, budget 10)\n")
    L.append("| τ | calls | right % | wrong % | abstain % | accuracy when answered % |"); L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["labels"] == "orig" and r["scope"] == "all" and r["order"] == "Hybrid λ=0.1" and r["skip_dup"] is True and r["budget"] == 10:
            L.append(f"| {r['tau']} | {r['calls']:.2f} | {r['right']:.1f} | {r['wrong']:.1f} | {r['abstain']:.1f} | {r['answered_acc']:.1f} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items()} for k, v in qual.items()}, open(out / "verifier_quality.json", "w"), indent=1)
    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for oname, skip in [("SSC", False), ("Cluster-first", True), ("Hybrid λ=0.1", True), ("Random", True)]:
            pts = sorted((r["calls"], r["wrong"]) for r in rows if r["labels"] == "orig" and r["scope"] == "all"
                         and r["order"] == oname and r["skip_dup"] == skip and r["tau"] == 0.7)
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=oname + (" + skip" if skip else ""))
        base = next(r for r in rows if r["labels"] == "orig" and r["scope"] == "all" and r["budget"] == 0)
        ax.scatter([0], [base["wrong"]], color="k", zorder=5, label="SSC top-1, no verifier")
        ax.set_xlabel("verifier calls per problem"); ax.set_ylabel("wrong answers returned (%)")
        ax.set_title("Wrong-answer rate vs verification cost (τ=0.7)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "wrong_vs_calls.png", dpi=150)
        fig, ax = plt.subplots(figsize=(4.2, 4))
        R = qual[("orig", "all")]["reliability"]
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot([r[0] for r in R], [r[1] for r in R], marker="o")
        ax.set_xlabel("verifier p_correct"); ax.set_ylabel("fraction actually correct"); ax.set_title("Calibration (orig labels)")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)
    print("\n".join(L))

if __name__ == "__main__":
    main()
