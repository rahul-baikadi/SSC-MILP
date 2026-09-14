"""Metrics and report (Sections 4.3–5.5, Appendix J, M.5):
accuracy per selector variant, Random expected (Eq. 25), Random draw / first
position (Eq. 26), pool coverage (Eq. 4), error decomposition (Table 7),
signal discriminability (Table 6), McNemar test SSC vs first position."""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import binomtest, chi2

from .selector import VARIANTS


def load_results(out_dir: str | Path, problems: list[dict]) -> list[dict]:
    out_dir = Path(out_dir)
    rows = []
    for p in problems:
        e = out_dir / "eval" / f"{p['pid']}.json"
        s = out_dir / "scores" / f"{p['pid']}.json"
        if not (e.exists() and s.exists()):
            continue
        with open(e, encoding="utf-8") as fh:
            ev = json.load(fh)
        with open(s, encoding="utf-8") as fh:
            sc = json.load(fh)
        j = out_dir / "judge" / f"{p['pid']}.json"
        judge = None
        if j.exists():
            with open(j, encoding="utf-8") as fh:
                judge = json.load(fh).get("choice")
        rows.append({"pid": p["pid"], "correct": [c["correct"] for c in ev["candidates"]],
                     "selected": sc["selected"], "signals": sc["candidates"], "judge": judge})
    return rows


def mcnemar(b: int, c: int) -> dict:
    """b = first correct & SSC wrong, c = first wrong & SSC correct."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "method": "none"}
    if n < 25:
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        return {"b": b, "c": c, "p": float(p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = chi2.sf(stat, 1)
    return {"b": b, "c": c, "p": float(p), "chi2": float(stat), "method": "chi2_cc"}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    K = len(rows[0]["correct"])
    acc = {}
    for v in VARIANTS:
        acc[v] = sum(r["correct"][r["selected"][v]] for r in rows) / n
    first = sum(r["correct"][0] for r in rows) / n
    rand = sum(sum(r["correct"]) / K for r in rows) / n
    cov = sum(any(r["correct"]) for r in rows) / n
    judge_rows = [r for r in rows if r["judge"] is not None]
    judge_acc = (sum(r["correct"][r["judge"]] for r in judge_rows) / len(judge_rows)) if judge_rows else None

    b = sum(1 for r in rows if r["correct"][0] and not r["correct"][r["selected"]["SSC"]])
    c = sum(1 for r in rows if not r["correct"][0] and r["correct"][r["selected"]["SSC"]])
    no_cand = sum(1 for r in rows if not any(r["correct"]))
    sel_fail = sum(1 for r in rows if any(r["correct"]) and not r["correct"][r["selected"]["SSC"]])

    # Table 6 discriminability: mean signal on correct minus mean on incorrect
    disc = {}
    for sig in ("TA", "SQ", "SC", "lp_sol", "lp_obj", "SSC"):
        pos = [s[sig] for r in rows for s, ok in zip(r["signals"], r["correct"]) if ok]
        neg = [s[sig] for r in rows for s, ok in zip(r["signals"], r["correct"]) if not ok]
        disc[sig] = (sum(pos) / len(pos) if pos else float("nan")) - (sum(neg) / len(neg) if neg else float("nan"))

    return {"n": n, "K": K, "accuracy": acc, "first_position": first, "random_expected": rand,
            "coverage": cov, "judge": judge_acc,
            "mcnemar_SSC_vs_first": mcnemar(b, c),
            "error_decomposition": {"no_candidate": no_cand, "selection_fail": sel_fail,
                                    "fail_rate": sel_fail / (n - no_cand) if n > no_cand else float("nan")},
            "discriminability": disc,
            "clean_candidates": sum(1 for r in rows for s in r["signals"] if not s["defects"]) / (n * K),
            "lp_optimal_rate": sum(1 for r in rows for s in r["signals"] if s["lp_status"] == "optimal") / (n * K)}


def format_report(name: str, summ: dict) -> str:
    if summ.get("n", 0) == 0:
        return f"# {name}\nno results\n"
    L = [f"# {name}  (n={summ['n']}, K={summ['K']})", ""]
    L.append("| Method | Accuracy (%) |")
    L.append("|---|---|")
    L.append(f"| Random draw (first position) | {100*summ['first_position']:.1f} |")
    L.append(f"| Random expected | {100*summ['random_expected']:.1f} |")
    if summ["judge"] is not None:
        L.append(f"| Zero-shot LLM judge | {100*summ['judge']:.1f} |")
    for v in ("SC", "SQ", "TA", "SC+SQ", "SC+TA", "SQ+TA", "SSC_noLP", "LP_only",
              "Symbolic+LP-obj", "Symbolic+LP-sol", "SSC"):
        L.append(f"| {v} | {100*summ['accuracy'][v]:.1f} |")
    L.append(f"| Pool coverage | {100*summ['coverage']:.1f} |")
    m = summ["mcnemar_SSC_vs_first"]
    L += ["", f"McNemar SSC vs first position: b={m['b']}, c={m['c']}, p={m['p']:.4f} ({m['method']})"]
    e = summ["error_decomposition"]
    L.append(f"Error decomposition: no correct candidate={e['no_candidate']}, "
             f"selection failures={e['selection_fail']}, fail rate={100*e['fail_rate']:.1f}%")
    d = summ["discriminability"]
    L.append("Discriminability (correct mean - incorrect mean): " +
             ", ".join(f"{k}={v:+.3f}" for k, v in d.items() if not math.isnan(v)))
    L.append(f"Clean candidates (no defects): {100*summ['clean_candidates']:.1f}%; "
             f"LP relaxation optimal: {100*summ['lp_optimal_rate']:.1f}%")
    return "\n".join(L) + "\n"
