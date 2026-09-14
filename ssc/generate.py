"""Candidate generation pipeline (Section 3.5, Appendix M).

Per problem: retrieve templates -> K=10 candidates from five formulation
variants x two repeats on the cycled temperature schedule -> up to two repair
rounds driven by static + sandbox construction checks -> deterministic
per-problem shuffle of the ten recipes onto pool positions 0..9.
"""
from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as C
from .ast_checks import repair_triggers
from .llm import LLMClient
from .prompts import generation_messages, repair_messages
from .sandbox import parse_candidate
from .templates import retrieve_templates

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text or "")
    if blocks:
        # prefer the last block that defines build_model
        for b in reversed(blocks):
            if "build_model" in b:
                return b.strip("\n") + "\n"
        return blocks[-1].strip("\n") + "\n"
    return (text or "").strip() + "\n"


def recipes() -> list[dict]:
    """Ten (variant, temperature, repeat) recipes: variants cycled twice."""
    out = []
    for rep in range(2):
        for v in range(len(C.TEMPERATURES)):
            out.append({"variant": v, "temperature": C.TEMPERATURES[v], "repeat": rep})
    return out


def shuffled_positions(pid: str, seed: int = C.GENERATION_SEED) -> list[int]:
    """positions[p] = recipe index placed at pool position p."""
    order = list(range(C.K))
    random.Random(f"{seed}:{pid}").shuffle(order)
    return order


def _feedback(check: dict) -> dict:
    model = check.get("model") or {}
    fb = {"defects": check["defects"]}
    if model.get("error"):
        fb["error"] = str(model["error"])[-1500:]
    fb["counts"] = check.get("static", {}).get("counts")
    if model:
        fb["n_variables"] = model.get("n_var_data")
        fb["n_constraints"] = model.get("n_con_data")
        fb["has_objective"] = model.get("objective") is not None
    return fb


def generate_one(llm: LLMClient, problem: dict, recipe: dict, templates: list[dict]) -> dict:
    """Generate + repair a single candidate.  Returns the candidate record."""
    pid = problem["pid"]
    msgs = generation_messages(problem["question"], templates, recipe["variant"])
    tag = f"{pid}:gen:v{recipe['variant']}:r{recipe['repeat']}"
    t0 = time.time()
    resp = llm.chat(msgs, temperature=recipe["temperature"], max_tokens=6000, tag=tag)
    code = extract_code(resp["content"])
    history = [{"stage": "generate", "cached": resp["cached"], "usage": resp["usage"],
                "finish_reason": resp.get("finish_reason")}]
    check = parse_candidate(code)
    for rnd in range(C.REPAIR_ROUNDS):
        trig = repair_triggers(check["defects"])
        if not trig:
            break
        fb = _feedback(check)
        rmsgs = repair_messages(problem["question"], templates, recipe["variant"], code, fb)
        rresp = llm.chat(rmsgs, temperature=C.REPAIR_TEMPERATURE, max_tokens=6000,
                         tag=f"{tag}:repair{rnd}")
        code = extract_code(rresp["content"])
        history.append({"stage": f"repair{rnd}", "triggers": trig, "cached": rresp["cached"],
                        "usage": rresp["usage"], "finish_reason": rresp.get("finish_reason")})
        check = parse_candidate(code)
    return {"recipe": recipe, "code": code, "history": history,
            "defects_at_generation": check["defects"], "seconds": time.time() - t0}


def generate_pool(llm: LLMClient, problem: dict, workers: int = 4) -> dict:
    templates = retrieve_templates(problem["question"])
    recs = recipes()
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_one, llm, problem, r, templates): i for i, r in enumerate(recs)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    order = shuffled_positions(problem["pid"])
    candidates = []
    for pos, ridx in enumerate(order):
        c = dict(results[ridx])
        c["position"] = pos
        c["recipe_index"] = ridx
        candidates.append(c)
    return {"pid": problem["pid"], "question": problem["question"],
            "templates": [t["name"] for t in templates], "candidates": candidates,
            "model": llm.model, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def generate_benchmark(llm: LLMClient, problems: list[dict], out_dir: str | Path,
                       problem_workers: int = 3, recipe_workers: int = 4, log=print) -> None:
    out_dir = Path(out_dir)
    (out_dir / "pools").mkdir(parents=True, exist_ok=True)
    todo = [p for p in problems if not (out_dir / "pools" / f"{p['pid']}.json").exists()]
    log(f"[generate] {len(problems)} problems, {len(todo)} to do")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=problem_workers) as ex:
        futs = {ex.submit(generate_pool, llm, p, recipe_workers): p for p in todo}
        done = 0
        for f in as_completed(futs):
            p = futs[f]
            try:
                pool = f.result()
            except Exception as e:  # noqa: BLE001
                log(f"[generate] {p['pid']} FAILED: {e}")
                continue
            with open(out_dir / "pools" / f"{p['pid']}.json", "w", encoding="utf-8") as fh:
                json.dump(pool, fh, indent=1)
            done += 1
            n_ok = sum(1 for c in pool["candidates"] if not c["defects_at_generation"])
            log(f"[generate] {done}/{len(todo)} {p['pid']} clean={n_ok}/10 "
                f"usage={llm.usage} elapsed={time.time()-t0:.0f}s")
    manifest = {"model": llm.model, "base_url": llm.base_url, "K": C.K,
                "temperatures": C.TEMPERATURES, "repair_rounds": C.REPAIR_ROUNDS,
                "seed": C.GENERATION_SEED, "usage": llm.usage,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(out_dir / "generation_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
