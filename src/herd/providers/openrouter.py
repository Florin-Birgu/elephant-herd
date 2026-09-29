"""OpenRouter client for Elephant Herd.

Two things this module exists to guarantee:

1. **Provider pinning.** The same model is served by ~30 hosts at cache-read prices
   spanning 10x, at quantizations from fp4 to bf16, and some with no caching at all.
   Unpinned routing therefore varies both cost and *recall quality* between calls,
   which would silently confound every measurement. Every request pins one host and
   forbids fallback.

2. **Cache verification.** The entire economic argument is that a resident corpus is
   re-read from cache. If caching silently stops working, cost rises ~30x with no
   error. Callers can demand a cache hit and get an exception instead of a bill.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 600  # iterated runs put N elephants on one host at once, and 300s
                       # was short enough to time out mid-generation on a busy round


class CacheMissError(RuntimeError):
    """A cache hit was required but the response shows the prefix was re-read."""


def load_api_key() -> str:
    """Environment first, then a gitignored .env at the repo root."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Export it, or put it in .env at the repo root. "
        "Note: ~/.zshrc is only read by interactive shells, so exporting there is not "
        "always enough."
    )


@dataclass
class Usage:
    """Token accounting for one call, normalised across providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    provider: str | None = None
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Hit the output cap. Critical: a truncated listener emits nothing and
        looks exactly like a silent one, which would be scored as a genuine
        silent-false-negative when it is really a budget bug."""
        return self.finish_reason == "length"

    @property
    def cache_hit(self) -> bool:
        return self.cached_tokens > 0

    @property
    def cache_hit_ratio(self) -> float:
        """Share of the prompt served from cache. The number to watch."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def summary(self) -> str:
        state = "HIT" if self.cache_hit else "MISS"
        return (
            f"{state} {self.cache_hit_ratio:6.1%} of {self.prompt_tokens:>9,} prompt tk"
            f" | out {self.completion_tokens:>5,}"
            f"{' TRUNC' if self.truncated else ''}"
            f" | ${self.cost_usd:.6f} | {self.latency_s:6.1f}s"
            f" | {self.provider or '?'}"
        )


@dataclass
class Response:
    text: str
    usage: Usage
    raw: dict = field(repr=False, default_factory=dict)


class OpenRouterClient:
    """Minimal, dependency-free client. urllib keeps the pilot install trivial."""

    def __init__(
        self,
        api_key: str | None = None,
        session_id: str = "herd",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or load_api_key()
        self.session_id = session_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_cost_usd = 0.0
        self.calls = 0

    def chat(
        self,
        model: str,
        messages: list[dict],
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        require_cache_hit: bool = False,
        min_cache_ratio: float = 0.5,
        reasoning: bool = True,
        reasoning_max_tokens: int | None = None,
    ) -> Response:
        """One chat completion.

        `provider` is a host slug from `/models/{id}/endpoints` (e.g. "deepseek",
        "alibaba"). Passing it forbids fallback: a rate-limited pinned host raises
        rather than silently rerouting to an uncached or differently-quantized one.

        `require_cache_hit` raises CacheMissError unless at least `min_cache_ratio`
        of the prompt came from cache. Use it on every call after a prefix is warm.

        `reasoning` defaults to True. An earlier version defaulted to False on the
        grounds that saying "<silent/>" costs ~202 output tokens with reasoning on and
        3 with it off - a 67x output saving. That reasoning was wrong twice over:
        output is ~1% of the bill (the cache read dominates), so the saving is ~$0.0001
        and therefore noise; and reasoning-off produced a confidently wrong answer on a
        gold question that reasoning-on answered correctly. Some models (GLM-5.3-flash)
        additionally reject reasoning-off with a 400.
        (`reasoning={"exclude": True}` is not a middle ground - it still generates and
        bills the tokens, merely hiding them from the response.)

        `reasoning_max_tokens` caps thinking on its own. Without it thinking and reply
        share `max_tokens`, and a long think can leave nothing for the reply, which
        comes back empty and looks like silence.
        """
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
            # Sticky routing: keeps turn 2+ on the host that holds the warm cache.
            "session_id": self.session_id,
            "reasoning": {"enabled": bool(reasoning)},
        }
        if reasoning and reasoning_max_tokens:
            body["reasoning"]["max_tokens"] = reasoning_max_tokens
        if provider:
            body["provider"] = {"order": [provider], "allow_fallbacks": False}

        raw = self._post(body)
        usage = self._parse_usage(raw)
        choices_ = raw.get("choices") or [{}]
        usage.finish_reason = choices_[0].get("finish_reason")
        self.total_cost_usd += usage.cost_usd
        self.calls += 1

        if require_cache_hit and usage.cache_hit_ratio < min_cache_ratio:
            raise CacheMissError(
                f"Expected >={min_cache_ratio:.0%} of the prompt from cache, got "
                f"{usage.cache_hit_ratio:.1%} ({usage.cached_tokens:,}/"
                f"{usage.prompt_tokens:,}). Aborting before this gets expensive. "
                f"Provider={usage.provider}."
            )

        choices = raw.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        return Response(text=text, usage=usage, raw=raw)

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "elephant-herd",
            },
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode())
                out["_latency_s"] = time.monotonic() - start
                if "error" in out and not out.get("choices"):
                    raise RuntimeError(f"API error: {out['error']}")
                return out
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:400]
                # Never fall back to another provider on 429 - that would break the
                # cache and change the quantization mid-experiment.
                # 520-524 are the host's gateway errors; one of them ended a whole benchmark run.
                if exc.code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) \
                        and attempt < self.max_retries - 1:
                    time.sleep(2**attempt * 2)
                    last = RuntimeError(f"HTTP {exc.code}: {detail}")
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            # IncompleteRead is the one that bit us: on long iterated runs the
            # connection drops mid-body, which is not a URLError and so used to abort
            # a whole run after hours of work. ConnectionError covers resets.
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                    http.client.RemoteDisconnected, ConnectionError) as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt * 2)
                    last = exc
                    continue
                raise
        raise RuntimeError(f"all {self.max_retries} attempts failed: {last}")

    @staticmethod
    def _parse_usage(raw: dict) -> Usage:
        u = raw.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        return Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            cached_tokens=details.get("cached_tokens", 0) or 0,
            cache_write_tokens=details.get("cache_write_tokens", 0) or 0,
            # OpenRouter reports actual charged cost, so we never guess from a
            # price table that may be stale.
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            or 0,
            cost_usd=float(u.get("cost", 0.0) or 0.0),
            latency_s=raw.get("_latency_s", 0.0),
            provider=raw.get("provider"),
        )
