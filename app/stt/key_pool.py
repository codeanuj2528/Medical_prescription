"""Round-robin pool of API keys with per-key cooldowns.

Parallel transcription multiplies the request rate, and Groq rate-limits per
key. Rotating through several keys raises the ceiling; a key that gets HTTP
429 sits out for its Retry-After while the others keep serving.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence


class KeyPool:
    INVALID_KEY_COOLDOWN_S = 600.0
    MIN_COOLDOWN_S = 0.5

    def __init__(self, keys: Sequence[str], clock: Callable[[], float] = time.monotonic) -> None:
        if not keys:
            raise ValueError("KeyPool needs at least one key")
        self._keys = list(keys)
        self._clock = clock
        self._cursor = 0
        self._cooling_until: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def acquire(self) -> str:
        """Next key that is not cooling down; if every key is, the one that recovers first."""
        now = self._clock()
        for offset in range(len(self._keys)):
            index = (self._cursor + offset) % len(self._keys)
            key = self._keys[index]
            if self._cooling_until.get(key, 0.0) <= now:
                self._cursor = (index + 1) % len(self._keys)
                return key
        return min(self._keys, key=lambda k: self._cooling_until.get(k, 0.0))

    def wait_time(self) -> float:
        """Seconds until at least one key is usable again (0 if one is free now)."""
        now = self._clock()
        return max(0.0, min(self._cooling_until.get(key, 0.0) for key in self._keys) - now)

    def mark_rate_limited(self, key: str, retry_after_s: float) -> None:
        self._cooling_until[key] = self._clock() + max(retry_after_s, self.MIN_COOLDOWN_S)

    def mark_invalid(self, key: str) -> None:
        self._cooling_until[key] = self._clock() + self.INVALID_KEY_COOLDOWN_S
