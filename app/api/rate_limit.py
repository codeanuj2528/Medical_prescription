"""In-memory sliding-window rate limiter.

On Vercel (Fluid Compute) one instance serves many concurrent requests, so
this caps abuse of the Groq keys per instance. It is a best-effort guard,
not a global quota — that would need a shared store such as Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str) -> float:
        """Record a hit. Returns 0 if allowed, otherwise seconds until the next slot frees up."""
        now = self._clock()
        hits = self._hits[client_id]
        while hits and hits[0] <= now - self._window_s:
            hits.popleft()
        if len(hits) >= self._limit:
            return hits[0] + self._window_s - now
        hits.append(now)
        if len(self._hits) > 5_000:
            self._prune(now)
        return 0.0

    def _prune(self, now: float) -> None:
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= now - self._window_s]
        for key in stale:
            del self._hits[key]
