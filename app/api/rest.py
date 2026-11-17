"""HTTP endpoints: health check and transcript summarisation."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app.config import ConfigError, get_settings
from app.summary.schemas import SummarizeRequest, SummarizeResponse

from .container import Container, get_container
from .errors import ApiError
from .security import client_id, origin_allowed

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> JSONResponse:
    """Deployment smoke test: confirms configuration without exposing secrets."""
    try:
        settings = get_settings()
    except ConfigError:
        return JSONResponse({"status": "misconfigured"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "sttModel": settings.stt_model,
            "sttFastModel": settings.stt_fast_model,
            "llmModel": settings.llm_model,
            "groqKeys": len(settings.groq_keys),
        }
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    body: SummarizeRequest,
    request: Request,
    response: Response,
    container: Annotated[Container, Depends(get_container)],
) -> SummarizeResponse:
    if not origin_allowed(request.headers, container.settings.allowed_origins):
        raise ApiError(403, "forbidden_origin", "Cross-origin requests are not allowed.")
    wait_s = container.summary_limiter.check(client_id(request.headers, request.client and request.client.host))
    if wait_s:
        raise ApiError(429, "rate_limited", "Too many summaries — wait a moment.", wait_s)

    result = await container.summary.summarize(body.transcript)
    response.headers["Server-Timing"] = f"llm;dur={result.latency_ms}"
    response.headers["Cache-Control"] = "no-store"
    return result
