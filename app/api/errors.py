"""Uniform JSON errors: {"error": {"code": ..., "message": ...}}."""

from __future__ import annotations

import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import ConfigError
from app.stt.groq_client import GroqError
from app.summary.service import SummaryError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An error whose message is safe to show to the user."""

    def __init__(self, status: int, code: str, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s


def error_response(error: ApiError) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if error.retry_after_s:
        headers["Retry-After"] = str(math.ceil(error.retry_after_s))
    return JSONResponse({"error": {"code": error.code, "message": error.message}}, error.status, headers)


def from_groq_error(error: GroqError) -> ApiError:
    if error.status == 429:
        return ApiError(429, "rate_limited", "The AI service is busy — try again shortly.", error.retry_after_s or 2)
    if error.status == 504:
        return ApiError(504, "upstream_timeout", "The AI service timed out.")
    if error.status in (401, 403):
        return ApiError(502, "upstream_auth", "The server's AI credentials were rejected.")
    if 400 <= error.status < 500:
        return ApiError(422, "upstream_rejected", f"The AI service rejected the request: {error.message}")
    return ApiError(502, "upstream_unavailable", "The AI service is unavailable right now.")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, error: ApiError) -> JSONResponse:
        return error_response(error)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", ())[1:])
        message = f"{field}: {first.get('msg', 'invalid value')}" if field else first.get("msg", "Invalid request")
        return error_response(ApiError(400, "invalid_request", message))

    @app.exception_handler(GroqError)
    async def _groq_error(_: Request, error: GroqError) -> JSONResponse:
        logger.warning("Groq error %s: %s", error.status, error.message)
        return error_response(from_groq_error(error))

    @app.exception_handler(SummaryError)
    async def _summary_error(_: Request, error: SummaryError) -> JSONResponse:
        logger.warning("Summary error: %s", error)
        return error_response(ApiError(502, "summary_failed", "The AI returned an unusable summary. Try again."))

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, error: ConfigError) -> JSONResponse:
        logger.error("Configuration error: %s", error)
        return error_response(ApiError(500, "server_misconfigured", "The server is missing its AI configuration."))
