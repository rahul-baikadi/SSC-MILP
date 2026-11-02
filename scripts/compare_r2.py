"""Round-1 vs round-2 (union) comparison for the regeneration pilot.
python scripts/compare_r2.py --benchmarks mamo_complex optmath"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssc import regenerate as R

ARMS = ["blind", "feedback", "branch"]

def load(run, sd="scores"):
    out = {}
    for f in sorted((ROOT / "runs" / run / sd).glob("*.json")):
        s = json.load(open(f)); e = json.load(open(ROOT / "runs" / run / "eval" / f.name))
        corr = [bool(c["correct"]) for c in e["candidates"]]
        out[s["pid"]] = {"s": s, "corr": corr, "sel": s["selected"]["SSC"]}
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmarks", nargs="+", default=["mamo_complex", "optmath"])
    a = ap.parse_args()
    rows, detail = [], {}
    for b in a.benchmarks:
        r1 = load(b)
        flag = {p for p, d in r1.items() if R.triggered(R.risk_features(d["s"]))}
        pids = sorted(r1)
        def stats(name, D, subset):
            P = [p for p in pids if p in subset and p in D]
            n = len(P)
            return {"bench": b, "arm": name, "n": n,
                    "acc1": 100 * np.mean([D[p]["corr"][D[p]["sel"]] for p in P]),
                    "cov": 100 * np.mean([any(D[p]["corr"]) for p in P])}
        for subset_name, subset in [("all", set(pids)), ("flagged", flag)]:
            rows.append(dict(stats("round1", r1, subset), subset=subset_name))
            for arm in ARMS:
                u = load(f"{b}_r2_{arm}_union")
                new = {}
                for p, d in u.items():
                    k1 = len(r1[p]["corr"])
                    new[p] = d["corr"][k1:]
                st = stats(arm, u, subset); st["subset"] = subset_name
                P = [p for p in pids if p in subset and p in u]
                st["new_only_cov"] = 100 * np.mean([any(new[p]) for p in P])
                st["recovered"] = sum(1 for p in P if not any(r1[p]["corr"]) and any(u[p]["corr"]))
                st["sel_lost"] = sum(1 for p in P if r1[p]["corr"][r1[p]["sel"]] and not u[p]["corr"][u[p]["sel"]])
                st["sel_gained"] = sum(1 for p in P if not r1[p]["corr"][r1[p]["sel"]] and u[p]["corr"][u[p]["sel"]])
                rows.append(st)
                if subset_name == "all":
                    detail[(b, arm)] = sorted(p for p in P if not any(r1[p]["corr"]) and any(u[p]["corr"]))
    # print
    hdr = f"{'bench':13s} {'subset':8s} {'arm':9s} {'n':>4s} {'Acc@1':>6s} {'Cov':>6s} {'newCov':>7s} {'recov':>5s} {'+sel':>5s} {'-sel':>5s}"
    print(hdr)
    for r in rows:
        print(f"{r['bench']:13s} {r['subset']:8s} {r['arm']:9s} {r['n']:4d} {r['acc1']:6.1f} {r['cov']:6.1f} "
              f"{r.get('new_only_cov', float('nan')):7.1f} {r.get('recovered', ''):>5} {r.get('sel_gained', ''):>5} {r.get('sel_lost', ''):>5}")
    # pooled
    print("\nPOOLED (all problems)")
    for arm in ["round1"] + ARMS:
        rr = [r for r in rows if r["arm"] == arm and r["subset"] == "all"]
        n = sum(r["n"] for r in rr)
        acc = sum(r["acc1"] * r["n"] for r in rr) / n; cov = sum(r["cov"] * r["n"] for r in rr) / n
        extra = ""
        if arm != "round1":
            extra = f" recovered={sum(r['recovered'] for r in rr)} sel+={sum(r['sel_gained'] for r in rr)} sel-={sum(r['sel_lost'] for r in rr)}"
        print(f"  {arm:9s} n={n} Acc@1={acc:.1f} Cov={cov:.1f}{extra}")
    json.dump({"rows": rows, "recovered": {f"{k[0]}|{k[1]}": v for k, v in detail.items()}},
              open(ROOT / "runs" / "r2_comparison.json", "w"), indent=1)

if __name__ == "__main__":
    main()
