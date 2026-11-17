"""Async client for Groq's OpenAI-compatible REST API.

Only what this app needs, with the resilience that matters for latency:

* one shared `httpx.AsyncClient` → TLS connections are kept alive and reused,
* on HTTP 429 / 401 the request moves **immediately** to another key; if every
  key is cooling down, important requests wait (briefly) for the first one to
  recover instead of failing,
* 5xx, timeouts and network errors are retried; 4xx client errors are not.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from .budget import RequestBudgets
from .key_pool import KeyPool

logger = logging.getLogger(__name__)
_RESET_PATTERN = re.compile(r"^(?:(\d+)m)?(?:([\d.]+)s)?$")
MAX_KEY_WAIT_S = 3.0  # longest a request waits for a rate-limited key to recover


class GroqError(Exception):
    def __init__(self, status: int, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.status = status  # upstream HTTP status, 502 for network errors, 504 for timeouts
        self.message = message
        self.retry_after_s = retry_after_s

    @classmethod
    def from_response(cls, response: httpx.Response) -> GroqError:
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = f"Groq returned HTTP {response.status_code}"
        return cls(response.status_code, message, parse_retry_after(response.headers))


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Seconds to wait, from Retry-After or Groq's x-ratelimit-reset-requests ("2.5s", "1m3s")."""
    try:
        seconds = float(headers.get("retry-after", ""))
        if seconds > 0:
            return seconds
    except ValueError:
        pass
    match = _RESET_PATTERN.match(headers.get("x-ratelimit-reset-requests", ""))
    if not match or not any(match.groups()):
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2) or 0)


class GroqClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        keys: KeyPool,
        base_url: str,
        max_attempts: int | None = None,
        budgets: RequestBudgets | None = None,
    ) -> None:
        self._http = http
        self._keys = keys
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts or 3
        self._budgets = budgets  # told about every 429 so the request budget matches Groq

    async def transcribe(
        self,
        wav: bytes,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
        timeout_s: float,
        may_wait: bool = True,
    ) -> dict[str, Any]:
        """POST /audio/transcriptions with verbose_json (we need per-segment confidence).

        `may_wait=False` for best-effort requests (live previews): fail fast on rate limits.
        """
        form: dict[str, str] = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        return await self._post(
            "/audio/transcriptions",
            timeout_s,
            may_wait=may_wait,
            model=model,
            data=form,
            files={"file": ("segment.wav", wav, "audio/wav")},
        )

    async def chat_completion(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        return await self._post("/chat/completions", timeout_s, json=payload)

    async def _post(
        self, path: str, timeout_s: float, *, may_wait: bool = True, model: str | None = None, **request: Any
    ) -> dict[str, Any]:
        last_error = GroqError(502, "Groq request failed")
        for attempt in range(self._max_attempts):
            if attempt:
                wait_s = self._keys.wait_time()
                if wait_s > 0 and not may_wait:
                    raise last_error
                if wait_s > 0:
                    await asyncio.sleep(min(wait_s, MAX_KEY_WAIT_S))
            key = self._keys.acquire()
            try:
                response = await self._http.post(
                    f"{self._base_url}{path}",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=timeout_s,
                    **request,
                )
            except httpx.TimeoutException:
                last_error = GroqError(504, "The AI service timed out")
                continue
            except httpx.TransportError:
                last_error = GroqError(502, "Could not reach the AI service")
                continue

            if response.is_success:
                return response.json()

            error = GroqError.from_response(response)
            if response.status_code == 429:
                logger.info("Groq rate limit on %s (retry after %ss)", model or path, error.retry_after_s)
                self._keys.mark_rate_limited(key, error.retry_after_s or 2.0)
                if self._budgets is not None and model:
                    self._budgets.rate_limited(model)
            elif response.status_code in (401, 403):
                self._keys.mark_invalid(key)
            elif response.status_code < 500:
                raise error  # bad input: another attempt will not help
            last_error = error
        raise last_error
