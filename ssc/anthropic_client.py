"""Claude (Anthropic API) client for the verifier: prompt caching, retries,
on-disk response cache (resumable, no re-billing), usage/cost accounting.

Key resolution: ANTHROPIC_API_KEY env var, then the first existing file of
ANTHROPIC_API_KEY_FILE, ../anthropic_key, ../anthropic_key.txt, ./anthropic_key(.txt),
~/.anthropic_key.  The key is never printed or written anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("SSC_VERIFIER_MODEL", "claude-sonnet-5-5")
# USD per million tokens (input, output, cache-read, cache-write-5min); check current pricing.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.50),
          "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
          "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}
_ROOT = Path(__file__).resolve().parents[1]
_KEY_FILES = [_ROOT.parent / "anthropic_key", _ROOT.parent / "anthropic_key.txt",
              _ROOT / "anthropic_key", _ROOT / "anthropic_key.txt", Path("~/.anthropic_key").expanduser()]


def load_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    cands = ([Path(os.environ["ANTHROPIC_API_KEY_FILE"])] if os.environ.get("ANTHROPIC_API_KEY_FILE") else []) + _KEY_FILES
    for p in cands:
        if p.is_file():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip().strip('"').strip("'")
                if line.startswith("ANTHROPIC_API_KEY="):
                    line = line.split("=", 1)[1].strip().strip('"')
                if line and not line.startswith("#"):
                    return line
    return None


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None, mock=None,
                 max_retries: int = 6):
        self.model, self.mock, self.max_retries = model, mock, max_retries
        self.cache_dir = Path(cache_dir or _ROOT / "cache" / "anthropic")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0}
        self._client = None
        if mock is None:
            import anthropic
            key = load_key()
            if not key:
                raise RuntimeError("No Anthropic API key found (set ANTHROPIC_API_KEY or save ../anthropic_key).")
            self._client = anthropic.Anthropic(api_key=key, max_retries=0)

    def _key(self, system, blocks, max_tokens, tag):
        h = hashlib.sha256(json.dumps([self.model, system, blocks, max_tokens, tag], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def chat(self, system: str, blocks: list[dict], max_tokens: int = 600, tag: str = "") -> dict:
        """blocks: [{"text": str, "cache": bool}, ...] forming one user turn.
        A cache breakpoint is placed after the system prompt and after each block with cache=True."""
        path = self._key(system, blocks, max_tokens, tag)
        if path.exists():
            with self._lock:
                self.usage["cached"] += 1
            return json.loads(path.read_text(encoding="utf-8")) | {"cached": True}
        if self.mock is not None:
            out = {"content": self.mock(system, blocks, tag), "usage": {}}
        else:
            import anthropic
            content = [{"type": "text", "text": b["text"],
                        **({"cache_control": {"type": "ephemeral"}} if b.get("cache") else {})} for b in blocks]
            sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            delay = 2.0
            for attempt in range(self.max_retries + 1):
                try:
                    r = self._client.messages.create(model=self.model, max_tokens=max_tokens,
                                                     system=sys_blocks, messages=[{"role": "user", "content": content}])
                    break
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
                        anthropic.APITimeoutError) as e:
                    if attempt == self.max_retries:
                        raise
                    time.sleep(delay); delay = min(delay * 2, 60)
                except anthropic.APIStatusError as e:
                    if getattr(e, "status_code", 0) == 529 and attempt < self.max_retries:
                        time.sleep(delay); delay = min(delay * 2, 60); continue
                    raise
            u = r.usage
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                     "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
            out = {"content": "".join(b.text for b in r.content if getattr(b, "type", "") == "text"),
                   "usage": usage, "model": r.model, "stop_reason": r.stop_reason}
        with self._lock:
            self.usage["calls"] += 1
            for k, v in out.get("usage", {}).items():
                self.usage[k] = self.usage.get(k, 0) + (v or 0)
        path.write_text(json.dumps(out), encoding="utf-8")
        return out | {"cached": False}

    def cost(self) -> float:
        pi, po, pr, pw = PRICES.get(self.model, (0, 0, 0, 0))
        u = self.usage
        return (u["input_tokens"] * pi + u["output_tokens"] * po + u["cache_read_tokens"] * pr
                + u["cache_write_tokens"] * pw) / 1e6
