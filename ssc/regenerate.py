"""Round 2: diagnosis-guided regeneration (gold-free).

For each problem with a round-1 pool, score file and requirements:

  slots      pool summary (LP-relaxation clusters), two representative codes,
             signature diff, diagnostics (TA / SQ / LP / scale checks)
  arms
    blind     B fresh round-1-style samples (budget-matched control)
    feedback  B samples of the diff + diagnostics prompt (no explicit decisions)
    branch    Stage A (decisions JSON) -> top-B decision assignments that differ
              from the round-1 majority -> Stage B, one candidate per assignment

Every new candidate goes through the same repair loop as round 1.  Outputs:

  runs/<bench>_r2_<arm>/pools/<pid>.json     new candidates only (K = B)
  runs/<bench>_r2_<arm>/regen/<pid>.json     slots, Stage A output, assignments
  runs/<bench>_r2_<arm>_union/{pools,eval,requirements}/   after `merge`

The reference answer is never read here.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as C
from .ast_checks import repair_triggers
from .generate import _feedback, extract_code
from .llm import LLMClient
from .prompts import generation_messages, repair_messages
from .prompts_regen import branch_messages, decision_messages, feedback_messages
from .sandbox import parse_candidate
from .signature import build_signature
from .templates import retrieve_templates

B_DEFAULT = 4
LP_CLUSTER_TOL = 1e-3
CODE_CHARS = 6000
# TA features with positive lift in the round-1 audit (results/audit); linking-type
# requirements had negative lift and are never fed back.
FEEDBACK_FEATURES = {"objective:minimize", "objective:maximize", "var:binary", "var:integer",
                     "constraint:upper_bound", "constraint:resource", "constraint:equality",
                     "constraint:demand"}
FEATURE_TEXT = {
    "objective:minimize": "The objective may need to be MINIMISED; confirm the objective sense.",
    "objective:maximize": "The objective may need to be MAXIMISED; confirm the objective sense.",
    "var:binary": "No model uses binary (yes/no) variables; check whether any decision is a yes/no choice.",
    "var:integer": "Few models use integer variables; check whether counted quantities must be whole numbers.",
    "constraint:upper_bound": "An upper limit stated in the text may be missing.",
    "constraint:resource": "A resource-availability limit stated in the text may be missing.",
    "constraint:equality": "An equality requirement (exactly / must equal / balance) may be missing.",
    "constraint:demand": "A demand / minimum-requirement constraint may be missing.",
}


# ============================================================ round-1 analysis
def rel_gap(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def lp_clusters(score: dict) -> list[dict]:
    """Cluster candidates by LP-relaxation objective; largest cluster first,
    ties broken by best SSC inside the cluster."""
    cands = score["candidates"]
    cl: list[dict] = []
    for i, c in enumerate(cands):
        z = c["lp_objective"] if c["lp_status"] == "optimal" else None
        if z is None:
            continue
        for g in cl:
            if rel_gap(z, g["z"]) <= LP_CLUSTER_TOL:
                g["idx"].append(i)
                break
        else:
            cl.append({"z": z, "idx": [i]})
    for g in cl:
        g["best"] = max(g["idx"], key=lambda i: (cands[i]["SSC"], -i))
    return sorted(cl, key=lambda g: (-len(g["idx"]), -cands[g["best"]]["SSC"]))


def risk_features(score: dict) -> dict:
    """Gold-free pool features, also used later for the correctness probability."""
    cands = score["candidates"]
    K = len(cands)
    top = score["selected"]["SSC"]
    cl = lp_clusters(score)
    mass_top = next((len(g["idx"]) for g in cl if top in g["idx"]), 0) / K
    ssc = sorted((c["SSC"] for c in cands), reverse=True)
    return {
        "top_lp_cluster_mass": mass_top,
        "largest_lp_cluster": (len(cl[0]["idx"]) / K) if cl else 0.0,
        "n_lp_clusters": len(cl),
        "frac_lp_failed": sum(c["lp_status"] != "optimal" for c in cands) / K,
        "frac_fatal": sum(bool(c["fatal"]) for c in cands) / K,
        "max_SSC": ssc[0],
        "ssc_margin": ssc[0] - ssc[1] if K > 1 else 0.0,
        "top_TA": cands[top]["TA"],
        "top_SQ": cands[top]["SQ"],
    }


def triggered(feat: dict, t: float = 0.5) -> bool:
    """Deployment trigger (audit: AUROC 0.71).  The experiment regenerates every
    pool and applies this rule post hoc, so t can be studied without new calls."""
    return feat["top_lp_cluster_mass"] <= t or feat["frac_lp_failed"] >= 0.5


# ============================================================ signatures / diff
def summarize_model(parsed: dict) -> dict:
    sig = build_signature(parsed)
    comps = defaultdict(lambda: {"senses": Counter(), "n": 0, "rhs": []})
    for r in sig.rows:
        d = comps[r.comp]
        d["senses"][r.orig_sense] += 1
        d["n"] += 1
        if r.rhs is not None and len(d["rhs"]) < 3:
            # signature stores <= rows in canonical >= form (negated); show the written RHS
            d["rhs"].append(round(-r.rhs if r.orig_sense == "le" else r.rhs, 6) + 0.0)
    return {
        "constructed": sig.constructed,
        "vars": [(v.name, v.domain, v.arity) for v in sig.vars],
        "domains": dict(sig.domains),
        "sense": sig.obj.sense if sig.obj else None,
        "comps": {k: {"senses": dict(v["senses"]), "n": v["n"], "rhs": v["rhs"]} for k, v in comps.items()},
        "row_keys": Counter((r.orig_sense, None if r.rhs is None else round(abs(r.rhs), 4)) for r in sig.rows),
        "n_rows": len(sig.rows),
    }


def _fmt_comp(name: str, c: dict) -> str:
    s = "/".join(f"{k} x{v}" if v > 1 else k for k, v in c["senses"].items())
    rhs = ", ".join(str(x) for x in c["rhs"])
    return f"{name} [{s}{'; rhs ' + rhs if rhs else ''}]"


def signature_diff(A: dict, B: dict, la: str = "A", lb: str = "B") -> str:
    out = []
    if A["sense"] != B["sense"]:
        out.append(f"- Objective sense: {A['sense']} ({la}) vs {B['sense']} ({lb})")
    if A["domains"] != B["domains"]:
        fa = ", ".join(f"{n}:{d}" for n, d, _ in A["vars"])
        fb = ", ".join(f"{n}:{d}" for n, d, _ in B["vars"])
        out.append(f"- Variable domains: {la} {{{fa}}} vs {lb} {{{fb}}}")
    if len(A["vars"]) != len(B["vars"]):
        out.append(f"- Variable components: {len(A['vars'])} ({la}) vs {len(B['vars'])} ({lb})")
    onlyA = A["row_keys"] - B["row_keys"]
    onlyB = B["row_keys"] - A["row_keys"]

    def comps_with(M, keys):
        hit = []
        for name, c in M["comps"].items():
            if any(s in {k[0] for k in keys} for s in c["senses"]) and \
                    any(round(abs(r), 4) in {k[1] for k in keys} for r in c["rhs"]):
                hit.append(_fmt_comp(name, c))
        return hit
    if onlyA:
        out.append(f"- Constraints in {la} with no counterpart in {lb}: " + "; ".join(comps_with(A, onlyA)[:6]))
    if onlyB:
        out.append(f"- Constraints in {lb} with no counterpart in {la}: " + "; ".join(comps_with(B, onlyB)[:6]))
    if A["n_rows"] != B["n_rows"]:
        out.append(f"- Number of scalar constraint rows: {A['n_rows']} ({la}) vs {B['n_rows']} ({lb})")
    return "\n".join(out) or "- No structural difference detected; the groups differ only in coefficients or data values."


# ============================================================ slots
def build_slots(pool: dict, score: dict, requirements: list[dict], parsed: list[dict] | None = None) -> dict:
    cands = score["candidates"]
    K = len(cands)
    codes = [c["code"] for c in pool["candidates"]]
    cl = lp_clusters(score)
    n_fail = sum(c["lp_status"] not in ("optimal", "infeasible", "unbounded") for c in cands)
    n_inf = sum(c["lp_status"] == "infeasible" for c in cands)
    n_unb = sum(c["lp_status"] == "unbounded" for c in cands)

    # ---------------- pool summary
    letters = "ABCDEFGH"
    lines = [f"{K} models were built."]
    for j, g in enumerate(cl[:4]):
        lines.append(f"- Group {letters[j]}: {len(g['idx'])} model(s), LP-relaxation objective ~ {g['z']:.6g}")
    if len(cl) > 4:
        lines.append(f"- {len(cl) - 4} further singleton/small groups")
    if n_inf:
        lines.append(f"- {n_inf} model(s) have an infeasible LP relaxation (so the MILP is infeasible too)")
    if n_unb:
        lines.append(f"- {n_unb} model(s) have an unbounded LP relaxation")
    if n_fail:
        lines.append(f"- {n_fail} model(s) failed to construct or relax")
    pool_summary = "\n".join(lines)

    # ---------------- representatives + diff
    reps, diff = [], "- Only one group of models exists; no structural contrast is available."
    if parsed is None:
        idxs = sorted({g["best"] for g in cl[:2]})
        parsed = {i: parse_candidate(codes[i]) for i in idxs}
    if len(cl) >= 2:
        a, b = cl[0]["best"], cl[1]["best"]
        reps = [("Group A", codes[a][:CODE_CHARS]), ("Group B", codes[b][:CODE_CHARS])]
        diff = signature_diff(summarize_model(parsed[a]), summarize_model(parsed[b]))
    elif cl:
        reps = [("Group A", codes[cl[0]["best"]][:CODE_CHARS])]
    else:
        top = score["selected"]["SSC"]
        reps = [("Best-scored model", codes[top][:CODE_CHARS])]

    # ---------------- diagnostics
    diag = []
    senses = Counter(c["obj_sense"] for c in cands if c["obj_sense"])
    if len(senses) > 1:
        diag.append("- Models disagree on the objective sense ("
                    + ", ".join(f"{k}: {v}" for k, v in senses.items()) + "); confirm minimise vs maximise.")
    seen = set()
    for r in requirements:
        for ft in r["features"]:
            if ft not in FEEDBACK_FEATURES or ft in seen:
                continue
            m = sum(ft in c["evidence"] for c in cands) / K
            if m < 0.5:
                seen.add(ft)
                diag.append(f"- {FEATURE_TEXT[ft]} (requirement '{r['name']}'; present in {m:.0%} of models)")
    dc = Counter(d for c in cands for d in set(c["defects"]))
    if dc.get("possible_nonlinearity"):
        diag.append(f"- {dc['possible_nonlinearity']} model(s) may multiply or divide decision variables; "
                    "the model must be linear.")
    if n_inf:
        diag.append("- Some models are infeasible: check the direction and right-hand side of every limit, "
                    "and whether a 'can' / 'at most' sentence was encoded as a requirement.")
    if len(cl) >= 2 and cl[0]["z"] and cl[1]["z"]:
        r = abs(cl[0]["z"] / cl[1]["z"])
        k = math.log10(r) if r > 0 else 0
        if abs(k - round(k)) < 1e-3 and round(k) != 0:
            diag.append(f"- Groups A and B differ by a factor of 10^{round(abs(k))}: check the units/scale "
                        "of the data and of the quantity the question asks for.")
    diagnostics = "\n".join(diag) or "- No automatic check fired."
    return {"pool_summary": pool_summary, "representatives": reps, "signature_diff": diff,
            "diagnostics": diagnostics, "risk": risk_features(score)}


# ============================================================ stage A parsing
def parse_decisions(content: str, max_dec: int = 6) -> list[dict]:
    try:
        obj = json.loads(content)
    except Exception:  # noqa: BLE001
        s, e = content.find("{"), content.rfind("}")
        obj = json.loads(content[s:e + 1]) if s >= 0 < e else {}
    out = []
    for d in (obj.get("decisions") or [])[:max_dec]:
        opts = [str(o) for o in (d.get("options") or [])][:3]
        if len(opts) < 2:
            continue
        p = [float(x) for x in (d.get("plausibility") or [])][:len(opts)]
        if len(p) != len(opts) or sum(p) <= 0 or min(p) < 0:
            p = [1.0 / len(opts)] * len(opts)
        tot = sum(p)
        maj = d.get("majority_option")
        out.append({"decision": str(d.get("decision", "decision"))[:64],
                    "evidence": str(d.get("evidence", "none"))[:400],
                    "options": opts, "plausibility": [x / tot for x in p],
                    "majority_option": maj if maj in opts else None})
    return out


def top_assignments(decisions: list[dict], B: int) -> list[list[dict]]:
    """Top-B joint assignments by product plausibility, excluding the one that
    equals the round-1 majority on every decision whose majority is known."""
    if not decisions:
        return []
    combos = []
    for pick in itertools.product(*[range(len(d["options"])) for d in decisions]):
        logp = sum(math.log(max(decisions[j]["plausibility"][k], 1e-9)) for j, k in enumerate(pick))
        is_major = all(d["majority_option"] is None or d["options"][k] == d["majority_option"]
                       for d, k in zip(decisions, pick))
        if not is_major:
            combos.append((logp, pick))
    combos.sort(key=lambda x: -x[0])
    out = []
    for logp, pick in combos[:B]:
        out.append([{"decision": d["decision"], "option": d["options"][k], "evidence": d["evidence"],
                     "plausibility": d["plausibility"][k]} for d, k in zip(decisions, pick)])
    return out


# ============================================================ candidate creation
def _repair_loop(llm: LLMClient, problem: str, code: str, tag: str, templates: list[dict]) -> tuple[str, list, list]:
    history = []
    check = parse_candidate(code)
    for rnd in range(C.REPAIR_ROUNDS):
        trig = repair_triggers(check["defects"])
        if not trig:
            break
        rmsgs = repair_messages(problem, templates, 4, code, _feedback(check))
        r = llm.chat(rmsgs, temperature=C.REPAIR_TEMPERATURE, max_tokens=6000, tag=f"{tag}:repair{rnd}")
        code = extract_code(r["content"])
        history.append({"stage": f"repair{rnd}", "triggers": trig, "cached": r["cached"], "usage": r["usage"]})
        check = parse_candidate(code)
    return code, history, check["defects"]


def _make(llm, problem, msgs, temperature, tag, templates, meta) -> dict:
    t0 = time.time()
    r = llm.chat(msgs, temperature=temperature, max_tokens=6000, tag=tag)
    code = extract_code(r["content"])
    code, hist, defects = _repair_loop(llm, problem, code, tag, templates)
    return {"recipe": meta, "code": code,
            "history": [{"stage": "generate", "cached": r["cached"], "usage": r["usage"]}] + hist,
            "defects_at_generation": defects, "seconds": time.time() - t0}


def regenerate_problem(llm: LLMClient, problem: dict, pool: dict, score: dict, requirements: list[dict],
                       arm: str, B: int = B_DEFAULT, workers: int = 4) -> tuple[dict, dict]:
    pid, q = problem["pid"], problem["question"]
    templates = retrieve_templates(q) if C.USE_TEMPLATES else []
    record: dict = {"pid": pid, "arm": arm, "B": B}
    jobs = []  # (msgs, temperature, tag, meta)

    if arm == "blind":
        for b in range(B):
            v = b % len(C.TEMPERATURES)
            jobs.append((generation_messages(q, templates, v), C.TEMPERATURES[v],
                         f"{pid}:r2blind:v{v}:b{b}", {"arm": arm, "variant": v, "b": b}))
    else:
        slots = build_slots(pool, score, requirements)
        record["slots"] = {k: v for k, v in slots.items() if k != "representatives"}
        record["slots"]["representatives"] = [n for n, _ in slots["representatives"]]
        if arm == "feedback":
            msgs = feedback_messages(q, slots)
            for b in range(B):
                t = C.TEMPERATURES[b % len(C.TEMPERATURES)]
                jobs.append((msgs, t, f"{pid}:r2fb:b{b}", {"arm": arm, "b": b, "temperature": t}))
        elif arm == "branch":
            ra = llm.chat(decision_messages(q, slots), temperature=0.0, max_tokens=2000,
                          tag=f"{pid}:r2A", json_mode=True)
            try:
                decisions = parse_decisions(ra["content"])
            except Exception:  # noqa: BLE001
                decisions = []
            assigns = top_assignments(decisions, B)
            record.update({"stage_a_raw": ra["content"], "decisions": decisions, "assignments": assigns})
            for b, asg in enumerate(assigns):
                jobs.append((branch_messages(q, asg), 0.3, f"{pid}:r2B:b{b}",
                             {"arm": arm, "b": b, "assignment": asg}))
            # pad to B with the feedback prompt if Stage A produced fewer assignments
            for b in range(len(assigns), B):
                jobs.append((feedback_messages(q, slots), C.TEMPERATURES[b % 5], f"{pid}:r2Bpad:b{b}",
                             {"arm": arm, "b": b, "padded": True}))
        else:
            raise ValueError(arm)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_make, llm, q, m, t, tag, templates, meta) for m, t, tag, meta in jobs]
        cands = [f.result() for f in futs]
    for i, c in enumerate(cands):
        c["position"] = i
    new_pool = {"pid": pid, "question": q, "templates": [t["name"] for t in templates],
                "candidates": cands, "model": llm.model, "round": 2, "arm": arm,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return new_pool, record


def regenerate_benchmark(llm: LLMClient, problems: list[dict], r1_dir: str | Path, out_dir: str | Path,
                         arm: str, scores_dir: str = "scores", B: int = B_DEFAULT,
                         problem_workers: int = 3, workers: int = 4, log=print) -> None:
    r1, out = Path(r1_dir), Path(out_dir)
    (out / "pools").mkdir(parents=True, exist_ok=True)
    (out / "regen").mkdir(parents=True, exist_ok=True)
    todo = [p for p in problems if (r1 / "pools" / f"{p['pid']}.json").exists()
            and (r1 / scores_dir / f"{p['pid']}.json").exists()
            and not (out / "pools" / f"{p['pid']}.json").exists()]
    log(f"[regen:{arm}] {len(todo)} problems, B={B}")
    t0 = time.time()

    def _one(p):
        pool = json.load(open(r1 / "pools" / f"{p['pid']}.json"))
        score = json.load(open(r1 / scores_dir / f"{p['pid']}.json"))
        reqs = json.load(open(r1 / "requirements" / f"{p['pid']}.json"))["requirements"]
        return regenerate_problem(llm, p, pool, score, reqs, arm, B=B, workers=workers)

    with ThreadPoolExecutor(max_workers=problem_workers) as ex:
        futs = {ex.submit(_one, p): p for p in todo}
        for n, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                new_pool, rec = f.result()
            except Exception as e:  # noqa: BLE001
                log(f"[regen:{arm}] {p['pid']} FAILED: {e}")
                continue
            json.dump(new_pool, open(out / "pools" / f"{p['pid']}.json", "w"), indent=1)
            json.dump(rec, open(out / "regen" / f"{p['pid']}.json", "w"), indent=1)
            log(f"[regen:{arm}] {n}/{len(todo)} {p['pid']} usage={llm.usage} {time.time() - t0:.0f}s")
    json.dump({"model": llm.model, "arm": arm, "B": B, "usage": llm.usage,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out / "regen_manifest.json", "w"), indent=1)


# ============================================================ union
def merge_union(r1_dir: str | Path, r2_dir: str | Path, union_dir: str | Path, log=print) -> None:
    """Union pool = round-1 candidates followed by round-2 candidates.  Eval files
    are concatenated (round-1 SCIP labels reused), requirements copied."""
    r1, r2, u = Path(r1_dir), Path(r2_dir), Path(union_dir)
    for d in ("pools", "eval", "requirements"):
        (u / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((r2 / "pools").glob("*.json")):
        pid = f.stem
        p1, p2 = json.load(open(r1 / "pools" / f.name)), json.load(open(f))
        pool = dict(p1)
        pool["candidates"] = [dict(c, round=1) for c in p1["candidates"]] + \
                             [dict(c, round=2) for c in p2["candidates"]]
        json.dump(pool, open(u / "pools" / f.name, "w"), indent=1)
        (u / "requirements" / f.name).write_text((r1 / "requirements" / f.name).read_text())
        e1, e2 = r1 / "eval" / f.name, r2 / "eval" / f.name
        if e1.exists() and e2.exists():
            a, b = json.load(open(e1)), json.load(open(e2))
            ev = {"pid": pid, "gold": a["gold"], "candidates": a["candidates"] + b["candidates"]}
            ev["coverage"] = any(c["correct"] for c in ev["candidates"])
            json.dump(ev, open(u / "eval" / f.name, "w"), indent=1)
        n += 1
    log(f"[merge] {n} union pools -> {u}")
