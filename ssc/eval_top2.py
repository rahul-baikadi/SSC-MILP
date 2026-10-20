import json, glob, os, sys, math, random
import numpy as np

ROOT = sys.argv[1]
BENCH = ["nl4opt", "mamo_easy", "mamo_complex", "complexor", "industryor", "optibench", "optmath"]


def load(b):
    probs = []
    for f in sorted(glob.glob(f"{ROOT}/runs/{b}/scores_v1/*.json")):
        s = json.load(open(f))
        e = json.load(open(f.replace("scores_v1", "eval")))
        K = len(s["candidates"])
        probs.append(dict(
            ssc=np.array(s["scores"]["SSC"], float),
            S=np.array(s["similarity"], float),
            grp=[c["clone_group"] for c in s["candidates"]],
            lpz=[c["lp_objective"] if c["lp_status"] == "optimal" else None for c in s["candidates"]],
            truez=[c["objective"] for c in e["candidates"]],
            ok=np.array([bool(c["correct"]) for c in e["candidates"]]),
            sel=s["selected"]["SSC"], K=K))
    return probs


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def order(p):
    # descending SSC, stable on index (matches argmax tie-break)
    return sorted(range(p["K"]), key=lambda i: (-p["ssc"][i], i))


def pick_naive(p, c1):
    return next(i for i in order(p) if i != c1)


def pick_clone(p, c1):
    for i in order(p):
        if i != c1 and p["grp"][i] != p["grp"][c1]:
            return i
    return pick_naive(p, c1)


def pick_simthr(p, c1, t):
    for i in order(p):
        if i != c1 and p["S"][i][c1] < t:
            return i
    return pick_clone(p, c1)


def pick_mmr(p, c1, lam):
    best, bv = None, -1e9
    for i in range(p["K"]):
        if i == c1:
            continue
        v = lam * p["ssc"][i] - (1 - lam) * p["S"][i][c1]
        if v > bv + 1e-12:
            best, bv = i, v
    return best


def pick_ans(p, c1, delta=1e-4, key="lpz", fallback="clone"):
    z1 = p[key][c1]
    if z1 is not None:
        for i in order(p):
            if i != c1 and p[key][i] is not None and rel(p[key][i], z1) > delta:
                return i
    return pick_clone(p, c1) if fallback == "clone" else pick_naive(p, c1)


def acc2(probs, picker):
    hits = 0
    for p in probs:
        c1 = p["sel"]
        c2 = picker(p, c1)
        hits += bool(p["ok"][c1] or p["ok"][c2])
    return 100 * hits / len(probs)


def rand2_given_c1(probs):
    t = 0
    for p in probs:
        c1 = p["sel"]
        if p["ok"][c1]:
            t += 1
        else:
            t += p["ok"].sum() / (p["K"] - 1)
    return 100 * t / len(probs)


def rand_pair(probs):
    t = 0
    for p in probs:
        K, m = p["K"], int(p["ok"].sum())
        t += 1 - math.comb(K - m, 2) / math.comb(K, 2)
    return 100 * t / len(probs)


data = {b: load(b) for b in BENCH}
pooled = [p for b in BENCH for p in data[b]]
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]

methods = {
    "Top-2 naive": pick_naive,
    "Clone-excl (tau=0.95)": pick_clone,
    "Sim-excl (tau'=0.8)": lambda p, c: pick_simthr(p, c, 0.8),
    "MMR lam=0.7": lambda p, c: pick_mmr(p, c, 0.7),
    "MMR lam=0.5": lambda p, c: pick_mmr(p, c, 0.5),
    "AnsDiv LP-obj": pick_ans,
    "AnsDiv true-obj (oracle ceiling)": lambda p, c: pick_ans(p, c, key="truez"),
}

rows = []
hdr = ["Method"] + BENCH + ["Pooled"]
def row(name, f):
    return [name] + [f(data[b]) for b in BENCH] + [f(pooled)]

rows.append(row("Acc@1 SSC", lambda P: 100 * np.mean([p["ok"][p["sel"]] for p in P])))
rows.append(row("Random c2 | c1=SSC", rand2_given_c1))
rows.append(row("Random pair", rand_pair))
for n, m in methods.items():
    rows.append(row(n, lambda P, m=m: acc2(P, m)))

# LOBO lambda for MMR
def lobo_mmr(b):
    train = [p for bb in BENCH if bb != b for p in data[bb]]
    best = max(LAMS, key=lambda l: (acc2(train, lambda p, c: pick_mmr(p, c, l)), l))
    return best, acc2(data[b], lambda p, c: pick_mmr(p, c, best))
lo = {b: lobo_mmr(b) for b in BENCH}
lo_pooled = 100 * sum(lo[b][1] * len(data[b]) / 100 for b in BENCH) / len(pooled)
rows.append(["MMR (LOBO lam)"] + [lo[b][1] for b in BENCH] + [lo_pooled])
rows.append(row("Coverage", lambda P: 100 * np.mean([p["ok"].any() for p in P])))

def fmt(v):
    return v if isinstance(v, str) else f"{v:.1f}"
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(fmt(v) for v in r) + " |")
print("\nLOBO lambdas:", {b: lo[b][0] for b in BENCH})

print("\nMMR lambda sweep (pooled Acc@2):")
print({l: round(acc2(pooled, lambda p, c, l=l: pick_mmr(p, c, l)), 1) for l in LAMS})

# Diagnostics on recoverable failures
print("\nDiagnostics (pooled):")
rec = [p for p in pooled if not p["ok"][p["sel"]] and p["ok"].any()]
print("recoverable (c1 wrong, pool has correct):", len(rec))
def frac(cond):
    return sum(cond(p) for p in rec)
c1 = lambda p: p["sel"]
print(" naive c2 correct:", frac(lambda p: p["ok"][pick_naive(p, c1(p))]))
print(" naive c2 same clone group as c1:", frac(lambda p: p["grp"][pick_naive(p, c1(p))] == p["grp"][c1(p)]))
print(" naive c2 same LP obj as c1:", frac(lambda p: p["lpz"][c1(p)] is not None and p["lpz"][pick_naive(p, c1(p))] is not None and rel(p["lpz"][pick_naive(p, c1(p))], p["lpz"][c1(p)]) <= 1e-4))
print(" c1 LP obj invalid:", frac(lambda p: p["lpz"][c1(p)] is None))
print(" some correct cand has LP obj != c1 LP obj:", frac(lambda p: p["lpz"][c1(p)] is not None and any(p["ok"][i] and p["lpz"][i] is not None and rel(p["lpz"][i], p["lpz"][c1(p)]) > 1e-4 for i in range(p["K"]))))
print(" some correct cand in different clone group:", frac(lambda p: any(p["ok"][i] and p["grp"][i] != p["grp"][c1(p)] for i in range(p["K"]))))
print(" #correct in pool (mean):", round(np.mean([p["ok"].sum() for p in rec]), 2))
print(" rank of best correct cand by SSC (median):", np.median([next(r for r, i in enumerate(order(p)) if p["ok"][i]) for p in rec]))

# How often does c2 'waste' (c1 correct anyway)? n/a for acc@2. Check LP-tightness: LP obj == true obj for correct cands
tight = [rel(p["lpz"][i], p["truez"][i]) <= 1e-4 for p in pooled for i in range(p["K"]) if p["lpz"][i] is not None and p["truez"][i] is not None]
print("\nLP relaxation obj == MILP obj (all cands):", f"{100*np.mean(tight):.1f}% of {len(tight)}")
for b in BENCH:
    t = [rel(p["lpz"][i], p["truez"][i]) <= 1e-4 for p in data[b] for i in range(p["K"]) if p["lpz"][i] is not None and p["truez"][i] is not None]
    print(f"  {b}: {100*np.mean(t):.1f}%")

# per-benchmark delta sweep for AnsDiv
print("\nAnsDiv delta sweep (pooled):", {d: round(acc2(pooled, lambda p, c, d=d: pick_ans(p, c, delta=d)), 1) for d in [1e-6, 1e-4, 1e-2, 5e-2]})

def pick_hybrid(p, c1, lam, delta=1e-4):
    z1 = p["lpz"][c1]
    cand = [i for i in range(p["K"]) if i != c1 and p["lpz"][i] is not None and z1 is not None and rel(p["lpz"][i], z1) > delta]
    if not cand:
        return pick_mmr(p, c1, lam)
    return max(cand, key=lambda i: (lam * p["ssc"][i] - (1 - lam) * p["S"][i][c1], -i))
print("\nHybrid AnsDiv->MMR pooled:", {l: round(acc2(pooled, lambda p, c, l=l: pick_hybrid(p, c, l)), 1) for l in [0.3, 0.5, 0.7, 1.0]})
for l in [0.5,0.7]:
    print(l, [round(acc2(data[b], lambda p, c, l=l: pick_hybrid(p, c, l)),1) for b in BENCH])

# paired counts vs Acc@1 and vs naive
def hits(P, f): return np.array([bool(p["ok"][p["sel"]] or p["ok"][f(p, p["sel"])]) for p in P])
base = np.array([bool(p["ok"][p["sel"]]) for p in pooled])
for n, f in [("naive", pick_naive), ("clone", pick_clone), ("MMR.7", lambda p,c: pick_mmr(p,c,.7)), ("AnsDiv", pick_ans), ("Hybrid.7", lambda p,c: pick_hybrid(p,c,.7))]:
    h = hits(pooled, f); hn = hits(pooled, pick_naive)
    print(f"{n}: +{int(h.sum()-base.sum())} problems over Acc@1; vs naive: wins {int((h&~hn).sum())}, losses {int((~h&hn).sum())}")
