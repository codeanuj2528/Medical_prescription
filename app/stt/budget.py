"""Request budget: stay inside the provider's requests-per-minute limit.

Groq limits each model per key (free tier: ~20 requests/minute). Live text can
easily ask for more than that, and once Groq answers 429 the *final* text of an
utterance would fail too. So every request is counted against a token bucket
that mirrors the limit:

* optional requests (live previews, early passes, the speculative Hindi pass)
  only run if, afterwards, a reserve is still left for final text;
* final requests always run (the bucket may briefly go negative).

With many keys the bucket is large and nothing is ever skipped; with one key
the app quietly does less live work instead of failing.

The bucket is only our estimate of Groq's: after a restart, or with several
server instances, the real one may be emptier. So every 429 Groq returns
drains the matching bucket, and optional work pauses until it refills.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RequestBudget:
    def __init__(self, per_minute: float, clock: Callable[[], float] = time.monotonic) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self._capacity = float(per_minute)
        self._tokens = float(per_minute)
        self._per_second = per_minute / 60.0
        self._clock = clock
        self._updated = clock()

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def try_take(self, keep: float = 0.0) -> bool:
        """Take one request if at least `keep` requests remain afterwards."""
        self._refill()
        if self._tokens - 1 < keep:
            return False
        self._tokens -= 1
        return True

    def take(self) -> None:
        """A request that must run regardless (final text): always allowed."""
        self._refill()
        self._tokens -= 1

    def drain(self, amount: float) -> None:
        """Groq said 429: we over-estimated what is left — remove `amount` (never below empty)."""
        self._refill()
        self._tokens = min(self._tokens, max(0.0, self._tokens - amount))

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._per_second)
        self._updated = now


class RequestBudgets:
    """One budget per model, sized for the number of keys (Groq limits each model per key)."""

    def __init__(self, per_key_per_minute: float, keys: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_key_per_minute * keys
        self._per_key = per_key_per_minute
        self._clock = clock
        self._budgets: dict[str, RequestBudget] = {}

    def for_model(self, model: str) -> RequestBudget:
        if model not in self._budgets:
            self._budgets[model] = RequestBudget(self.per_minute, self._clock)
        return self._budgets[model]

    def rate_limited(self, model: str) -> None:
        """One key just hit its limit for this model: that key's share is gone for now."""
        self.for_model(model).drain(self._per_key)
