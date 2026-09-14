"""Benchmark loading.  Normalised test files (question / answer / index) are
those of the LLMOPT release (data/benchmarks/*.jsonl); their record counts
match Table 1 of the paper exactly (NL4OPT 230, MaMO-Easy 652, ...).  The
evaluation split is the first `n_eval` records in file order (Section 4.1)."""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = {
    "nl4opt": {"file": "data/benchmarks/nl4opt.jsonl", "n_eval": 100},
    "mamo_easy": {"file": "data/benchmarks/mamo_easy.jsonl", "n_eval": 100},
    "mamo_complex": {"file": "data/benchmarks/mamo_complex.jsonl", "n_eval": 100},
    "complexor": {"file": "data/benchmarks/complexor.jsonl", "n_eval": 18},
    "industryor": {"file": "data/benchmarks/industryor.jsonl", "n_eval": 100},
    "optibench": {"file": "data/benchmarks/optibench.jsonl", "n_eval": 100},
    "optmath": {"file": "data/benchmarks/optmath.jsonl", "n_eval": 100},
}


def _to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_benchmark(name: str, n: int | None = None) -> list[dict]:
    spec = BENCHMARKS[name]
    n = spec["n_eval"] if n is None else n
    rows = []
    with open(_ROOT / spec["file"], encoding="utf-8") as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("index", k)
            rows.append({
                "pid": f"{name}_{int(idx):04d}",
                "index": idx,
                "question": r["question"],
                "answer": _to_float(r.get("answer")),
                "answer_raw": r.get("answer"),
            })
            if len(rows) >= n:
                break
    missing = [r["pid"] for r in rows if r["answer"] is None]
    if missing:
        raise ValueError(f"non-numeric gold answers: {missing[:5]}")
    return rows
