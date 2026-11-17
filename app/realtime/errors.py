"""User-facing wording for failures (never leaks internals)."""

from __future__ import annotations

from app.stt.groq_client import GroqError


def public_error_message(error: Exception) -> str:
    if isinstance(error, GroqError):
        if error.status == 429:
            return "The speech service is busy — retry in a moment."
        if error.status == 504:
            return "The speech service timed out."
        if error.status in (401, 403):
            return "The server's AI credentials were rejected."
        if 400 <= error.status < 500:
            return "The speech service could not process this audio."
        return "The speech service is unavailable right now."
    return "Transcription failed."
