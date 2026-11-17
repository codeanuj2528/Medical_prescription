"""Request guards shared by the HTTP and WebSocket endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlparse


def client_id(headers: Mapping[str, str], fallback: str | None) -> str:
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or headers.get("x-real-ip") or fallback or "anonymous"


def origin_allowed(headers: Mapping[str, str], allowed_origins: Sequence[str]) -> bool:
    """Browsers always send Origin on WebSocket and cross-site requests.

    Accepting only our own origin stops other websites from spending our Groq
    quota through a visitor's browser. Requests without Origin (curl, the
    test scripts) are left to the rate limiter.
    """
    origin = headers.get("origin")
    if not origin:
        return True
    host = headers.get("x-forwarded-host") or headers.get("host")
    return urlparse(origin).netloc == host or origin in allowed_origins
