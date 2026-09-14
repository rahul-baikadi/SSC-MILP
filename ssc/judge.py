"""Zero-shot listwise LLM judge baseline (Appendix K): one call per problem,
sees the problem and the ten candidates in pool order, no solver output, no
reference answer."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import LLMClient
from .prompts import judge_messages

_NUM = re.compile(r'"choice"\s*:\s*(\d+)')


def judge_pool(llm: LLMClient, pool: dict) -> dict:
    codes = [c["code"] for c in pool["candidates"]]
    msgs = judge_messages(pool["question"], codes)
    resp = llm.chat(msgs, temperature=0.0, max_tokens=400, tag=f"{pool['pid']}:judge")
    txt = resp["content"]
    choice = None
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.I))
        choice = int(obj.get("choice"))
    except Exception:  # noqa: BLE001
        m = _NUM.search(txt)
        if m:
            choice = int(m.group(1))
    if choice is None or not (0 <= choice < len(codes)):
        choice = 0
    return {"pid": pool["pid"], "choice": choice, "raw": txt[:500], "usage": resp["usage"]}


def judge_benchmark(llm: LLMClient, problems: list[dict], out_dir: str | Path,
                    workers: int = 4, log=print) -> None:
    out_dir = Path(out_dir)
    (out_dir / "judge").mkdir(parents=True, exist_ok=True)
    todo = [p for p in problems if (out_dir / "pools" / f"{p['pid']}.json").exists()
            and not (out_dir / "judge" / f"{p['pid']}.json").exists()]
    log(f"[judge] {len(todo)} pools")

    def _one(p):
        with open(out_dir / "pools" / f"{p['pid']}.json", encoding="utf-8") as fh:
            return judge_pool(llm, json.load(fh))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, p): p for p in todo}
        for f in as_completed(futs):
            p = futs[f]
            r = f.result()
            with open(out_dir / "judge" / f"{p['pid']}.json", "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
    log(f"[judge] done usage={llm.usage}")
