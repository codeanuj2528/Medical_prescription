"""WebSocket endpoint: live audio in, VAD telemetry and transcript chunks out."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.realtime.protocol import RetryCommand, StartCommand, StopCommand, parse_command
from app.realtime.session import StreamingSession

from .container import Container, get_ws_container
from .security import client_id, origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_MESSAGE_BYTES = 64_000  # ~2 s of 16 kHz PCM16; the browser sends ~60 ms at a time
POLICY_VIOLATION = 1008
TRY_AGAIN_LATER = 1013


class WebSocketSink:
    """Serialises outgoing messages: transcription tasks finish concurrently,
    but a WebSocket must be written by one sender at a time."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()
        self.closed = False

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._lock:
            try:
                await self._websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                self.closed = True


@router.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket, container: Annotated[Container, Depends(get_ws_container)]) -> None:
    settings = container.settings
    if not origin_allowed(websocket.headers, settings.allowed_origins):
        await websocket.close(code=POLICY_VIOLATION)
        return
    if container.session_limiter.check(client_id(websocket.headers, websocket.client and websocket.client.host)):
        await websocket.close(code=TRY_AGAIN_LATER)
        return

    await websocket.accept()
    sink = WebSocketSink(websocket)
    session: StreamingSession | None = None

    def new_session(command: StartCommand | None = None) -> StreamingSession:
        command = command or StartCommand(type="start")
        return StreamingSession(
            container.transcription,
            settings,
            sink.send,
            chunk_offset=command.chunk_offset,
            context=command.context,
            language=command.language_code(settings.stt_language),
        )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            audio = message.get("bytes")
            if audio is not None:
                if len(audio) > MAX_AUDIO_MESSAGE_BYTES:
                    await sink.send({"type": "error", "message": "Audio message too large."})
                    continue
                session = session or new_session()
                await session.feed(audio)
                continue

            try:
                command = parse_command(message.get("text") or "")
            except ValidationError:
                await sink.send({"type": "error", "message": "Unknown command."})
                continue

            if isinstance(command, StartCommand):
                if session:
                    await session.close()
                session = new_session(command)
                await sink.send({"type": "ready"})
            elif isinstance(command, StopCommand):
                if session:
                    await session.finish()
                await sink.send({"type": "stopped"})
            elif isinstance(command, RetryCommand) and session:
                await session.retry(command.id)
    except WebSocketDisconnect:
        pass
    finally:
        sink.closed = True
        if session:
            await session.close()
