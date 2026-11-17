"""FastAPI entry point (also the Vercel entry point: `app.main:app`)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import realtime, rest
from app.api.errors import register_error_handlers
from app.config import PUBLIC_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SECURITY_HEADERS = {
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    container = getattr(app.state, "container", None)
    if container is not None:
        await container.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MediScribe",
        summary="Live medical transcription with voice activity detection",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    register_error_handlers(app)
    app.include_router(rest.router)
    app.include_router(realtime.router)

    @app.middleware("http")
    async def security_headers(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    index = PUBLIC_DIR / "index.html"

    @app.get("/", include_in_schema=False)
    def home() -> Response:
        return FileResponse(index) if index.exists() else RedirectResponse("/index.html")

    # On Vercel, public/ is served by the CDN at the root paths (and must not be
    # mounted); locally, FastAPI serves it itself.
    if PUBLIC_DIR.is_dir() and not os.environ.get("VERCEL"):
        app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="static")
    return app


app = create_app()
