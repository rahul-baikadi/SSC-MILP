"""Semantic requirement extraction (Appendix L): one LLM call per problem at
temperature 0, 1400-token cap, two retries, closed-vocabulary filtering.
If nothing valid remains the experiment must terminate (paper's rule)."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as C
from .alignment import filter_requirements
from .llm import LLMClient
from .prompts import extraction_messages

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse(content: str):
    txt = content.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE)
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        m = _JSON.search(txt)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        obj = obj.get("requirements", obj.get("items", []))
    return obj


def extract_requirements(llm: LLMClient, problem: dict) -> dict:
    msgs = extraction_messages(problem["question"])
    attempts = []
    for attempt in range(1 + C.TA_RETRIES):
        resp = llm.chat(msgs, temperature=0.0, max_tokens=C.TA_MAX_TOKENS,
                        tag=f"{problem['pid']}:extract:{attempt}", json_mode=True)
        raw = _parse(resp["content"])
        reqs = filter_requirements(raw) if raw is not None else []
        attempts.append({"attempt": attempt, "cached": resp["cached"], "n_valid": len(reqs)})
        if reqs:
            return {"pid": problem["pid"], "requirements": reqs, "attempts": attempts, "ok": True}
    return {"pid": problem["pid"], "requirements": [], "attempts": attempts, "ok": False}


def extract_benchmark(llm: LLMClient, problems: list[dict], out_dir: str | Path,
                      workers: int = 6, log=print) -> None:
    out_dir = Path(out_dir)
    (out_dir / "requirements").mkdir(parents=True, exist_ok=True)
    todo = [p for p in problems if not (out_dir / "requirements" / f"{p['pid']}.json").exists()]
    log(f"[extract] {len(problems)} problems, {len(todo)} to do")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_requirements, llm, p): p for p in todo}
        for f in as_completed(futs):
            p = futs[f]
            r = f.result()
            with open(out_dir / "requirements" / f"{p['pid']}.json", "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            if not r["ok"]:
                failed.append(p["pid"])
    log(f"[extract] done, usage={llm.usage}, failed={failed}")
    if failed:
        raise SystemExit(f"requirement extraction failed for {failed}; experiment terminates (Appendix L)")


def load_requirements(out_dir: str | Path, pid: str) -> list[dict]:
    with open(Path(out_dir) / "requirements" / f"{pid}.json", encoding="utf-8") as fh:
        return json.load(fh)["requirements"]
