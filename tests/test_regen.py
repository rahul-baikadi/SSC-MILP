"""Offline test of round-2 regeneration: all three arms + merge, mock LLM, real pool."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ssc import regenerate as R

ROOT = Path(__file__).resolve().parents[1]
PID, BENCH = "nl4opt_0017", "nl4opt"

STAGE_A = {"decisions": [
    {"decision": "total_units_direction", "evidence": "\"at least 300 units of chemicals can be used\"",
     "options": [">= 300", "<= 300"], "plausibility": [0.6, 0.4], "majority_option": ">= 300"},
    {"decision": "domain_of_units", "evidence": "none", "options": ["NonNegativeIntegers", "NonNegativeReals"],
     "plausibility": [0.5, 0.5], "majority_option": "NonNegativeIntegers"},
]}


class MockLLM:
    model, base_url = "mock", "mock"

    def __init__(self, code: str):
        self.code, self.calls, self.usage = code, [], {"calls": 0}

    def chat(self, messages, temperature=0.0, max_tokens=4096, tag="", json_mode=False):
        self.calls.append((tag, messages))
        self.usage["calls"] += 1
        content = json.dumps(STAGE_A) if json_mode else f"```python\n{self.code}\n```"
        return {"content": content, "cached": False, "usage": {}, "finish_reason": "stop"}


def _r1(tmp: Path) -> Path:
    r1 = tmp / BENCH
    for d in ("pools", "scores_v1", "requirements", "eval"):
        (r1 / d).mkdir(parents=True)
        shutil.copy(ROOT / "runs" / BENCH / d / f"{PID}.json", r1 / d / f"{PID}.json")
    return r1


def test_assignments_exclude_majority():
    dec = R.parse_decisions(json.dumps(STAGE_A))
    asg = R.top_assignments(dec, 4)
    assert len(asg) == 3                                  # 2x2 minus the majority combination
    majority = {(">= 300", "NonNegativeIntegers")}
    assert all(tuple(a["option"] for a in x) not in majority for x in asg)


def test_all_arms_and_merge(tmp_path):
    r1 = _r1(tmp_path)
    pool = json.load(open(r1 / "pools" / f"{PID}.json"))
    score = json.load(open(r1 / "scores_v1" / f"{PID}.json"))
    reqs = json.load(open(r1 / "requirements" / f"{PID}.json"))["requirements"]
    prob = {"pid": PID, "question": pool["question"]}
    code = pool["candidates"][0]["code"]
    for arm in ("blind", "feedback", "branch"):
        llm = MockLLM(code)
        new_pool, rec = R.regenerate_problem(llm, prob, pool, score, reqs, arm, B=4, workers=2)
        assert len(new_pool["candidates"]) == 4
        assert not any("gold" in json.dumps(m).lower() and "GOLD_ANSWER" not in json.dumps(m)
                       for _, m in llm.calls)
        if arm == "branch":
            assert len(rec["assignments"]) == 3
            assert sum(c["recipe"].get("padded", False) for c in new_pool["candidates"]) == 1
            assert any(t.endswith(":r2A") for t, _ in llm.calls)
        out = tmp_path / f"{BENCH}_r2_{arm}"
        (out / "pools").mkdir(parents=True)
        (out / "eval").mkdir(parents=True)
        json.dump(new_pool, open(out / "pools" / f"{PID}.json", "w"))
        ev = json.load(open(r1 / "eval" / f"{PID}.json"))
        json.dump({"pid": PID, "gold": ev["gold"], "candidates": ev["candidates"][:4]},
                  open(out / "eval" / f"{PID}.json", "w"))
        R.merge_union(r1, out, tmp_path / f"{BENCH}_r2_{arm}_union", log=lambda *_: None)
        u = json.load(open(tmp_path / f"{BENCH}_r2_{arm}_union" / "pools" / f"{PID}.json"))
        ue = json.load(open(tmp_path / f"{BENCH}_r2_{arm}_union" / "eval" / f"{PID}.json"))
        assert len(u["candidates"]) == 14 and len(ue["candidates"]) == 14
