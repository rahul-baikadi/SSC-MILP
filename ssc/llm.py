"""Thin OpenAI-compatible chat client (DeepSeek) with an on-disk cache so
that interrupted runs resume without re-billing, plus a mock backend for
offline tests.

Key resolution order: DEEPSEEK_API_KEY env var, then the file named by
DEEPSEEK_API_KEY_FILE, then ./deepseek_key / ../deepseek_key / ~/.deepseek_key.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = os.environ.get("SSC_LLM_MODEL", "deepseek-v4-flash")

_KEY_CANDIDATES = ["deepseek_key", "deepseek_key.txt", ".deepseek_key",
                   "../deepseek_key", "../deepseek_key.txt", "../.deepseek_key",
                   "~/.deepseek_key"]


def load_api_key() -> str | None:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    cands = []
    if os.environ.get("DEEPSEEK_API_KEY_FILE"):
        cands.append(os.environ["DEEPSEEK_API_KEY_FILE"])
    cands += _KEY_CANDIDATES
    for c in cands:
        p = Path(os.path.expanduser(c))
        if p.is_file():
            txt = p.read_text(encoding="utf-8").strip()
            # tolerate "DEEPSEEK_API_KEY=sk-..." or "export ..." formats
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line and not line.startswith("sk-"):
                    line = line.split("=", 1)[1].strip().strip('"').strip("'")
                return line
    return None


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 cache_dir: str | os.PathLike = "cache/llm", mock=None,
                 max_retries: int = 6, api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock                      # callable(messages, temperature, tag) -> str
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._client = None
        if mock is None:
            from openai import OpenAI
            key = api_key or load_api_key()
            if not key:
                raise RuntimeError("No DeepSeek API key found (DEEPSEEK_API_KEY / key file).")
            self._client = OpenAI(api_key=key, base_url=base_url, timeout=180.0, max_retries=0)

    # ------------------------------------------------------------ cache
    def _key(self, messages, temperature, max_tokens, tag) -> str:
        blob = json.dumps({"m": self.model, "msgs": messages, "t": temperature,
                           "mt": max_tokens, "tag": tag}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096,
             tag: str = "", json_mode: bool = False) -> dict:
        """Returns {'content': str, 'usage': {...}, 'cached': bool, 'model': str}.
        `tag` distinguishes otherwise-identical calls (e.g. repeat index)."""
        key = self._key(messages, temperature, max_tokens, tag)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            rec["cached"] = True
            with self._lock:
                self.usage["cached"] += 1
            return rec

        if self.mock is not None:
            content = self.mock(messages, temperature, tag)
            rec = {"content": content, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                   "model": "mock", "created": time.time()}
        else:
            rec = self._call(messages, temperature, max_tokens, json_mode)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += rec["usage"].get("prompt_tokens", 0)
            self.usage["completion_tokens"] += rec["usage"].get("completion_tokens", 0)
        rec["cached"] = False
        return rec

    def _call(self, messages, temperature, max_tokens, json_mode) -> dict:
        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                              max_tokens=max_tokens)
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                         "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
                return {"content": content, "usage": usage, "model": getattr(resp, "model", self.model),
                        "created": time.time(), "finish_reason": resp.choices[0].finish_reason}
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                # do not retry on auth / bad request
                if "401" in msg or "invalid_api_key" in msg or "Insufficient Balance" in msg or "402" in msg:
                    raise
                time.sleep(delay)
                delay = min(60.0, delay * 2)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def list_models(self) -> list[str]:
        if self._client is None:
            return ["mock"]
        return [m.id for m in self._client.models.list().data]
