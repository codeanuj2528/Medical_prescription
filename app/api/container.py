"""Composition root: builds the object graph once per process.

On Vercel (Fluid Compute) one warm instance serves many requests, so the
HTTP connection pool to Groq, the key pool's cooldowns and the rate limiters
are created once and shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, WebSocket

from app.config import Settings, get_settings
from app.stt.budget import RequestBudgets
from app.stt.groq_client import GroqClient
from app.stt.key_pool import KeyPool
from app.stt.service import TranscriptionService
from app.summary.service import SummaryService

from .rate_limit import SlidingWindowRateLimiter


@dataclass
class Container:
    settings: Settings
    http: httpx.AsyncClient
    transcription: TranscriptionService
    summary: SummaryService
    session_limiter: SlidingWindowRateLimiter
    summary_limiter: SlidingWindowRateLimiter

    async def aclose(self) -> None:
        await self.http.aclose()


def build_container(settings: Settings) -> Container:
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
    budgets = RequestBudgets(settings.groq_rpm_per_key, len(settings.groq_keys))
    groq = GroqClient(http, KeyPool(settings.groq_keys), settings.groq_base_url, budgets=budgets)
    return Container(
        settings=settings,
        http=http,
        transcription=TranscriptionService(
            groq,
            model=settings.stt_model,
            fast_model=settings.stt_fast_model,
            timeout_s=settings.stt_timeout_s,
            expected_languages=settings.stt_expected_languages,
            budgets=budgets,
        ),
        summary=SummaryService(groq, model=settings.llm_model, timeout_s=settings.llm_timeout_s),
        session_limiter=SlidingWindowRateLimiter(settings.sessions_per_minute, 60),
        summary_limiter=SlidingWindowRateLimiter(settings.summaries_per_minute, 60),
    )


def _container_for(app: FastAPI) -> Container:
    container = getattr(app.state, "container", None)
    if container is None:
        container = build_container(get_settings())
        app.state.container = container
    return container


def get_container(request: Request) -> Container:
    """FastAPI dependency for HTTP routes (overridable in tests)."""
    return _container_for(request.app)


def get_ws_container(websocket: WebSocket) -> Container:
    """FastAPI dependency for WebSocket routes (overridable in tests)."""
    return _container_for(websocket.app)
