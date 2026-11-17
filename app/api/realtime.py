"""WebSocket endpoint: live audio in, VAD telemetry and transcript chunks out."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.realtime.protocol import PingCommand, RetryCommand, StartCommand, StopCommand, parse_command
from app.realtime.session import StreamingSession

from .container import Container, get_ws_container
from .security import client_id, origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_MESSAGE_BYTES = 64_000  # ~2 s of 16 kHz PCM16; the browser sends 100 ms (replays: ≤ 1 s) at a time
POLICY_VIOLATION = 1008
TRY_AGAIN_LATER = 1013
ABNORMAL_CLOSURE = 1006  # the connection ended without a close frame


class ConnectionLog:
    """Logs every connection's lifetime and how it ended, per process.

    On a serverless platform this is how a dropped connection shows up in the
    logs: who closed it (close code), after how long, and on which instance.
    """

    def __init__(self) -> None:
        self.instance = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.open = 0

    def opened(self) -> float:
        self.open += 1
        logger.info("WebSocket opened on instance %s (%d open)", self.instance, self.open)
        return time.monotonic()

    def closed(self, since: float, code: int | None) -> None:
        self.open -= 1
        logger.info(
            "WebSocket closed after %.1f s with code %s on instance %s (%d open)",
            time.monotonic() - since,
            code if code is not None else ABNORMAL_CLOSURE,
            self.instance,
            self.open,
        )


connections = ConnectionLog()


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
    stopping: asyncio.Task[None] | None = None
    opened_at, close_code = connections.opened(), None

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
                close_code = message.get("code")
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
                # In the background, so this loop keeps answering pings while the
                # last transcriptions finish (the browser treats a silent server as gone).
                stopping = asyncio.create_task(_stop(session, sink))
            elif isinstance(command, RetryCommand) and session:
                await session.retry(command.id)
            elif isinstance(command, PingCommand):
                await sink.send({"type": "pong"})
    except WebSocketDisconnect as disconnect:
        close_code = disconnect.code
    finally:
        sink.closed = True
        connections.closed(opened_at, close_code)
        if stopping:
            stopping.cancel()
        if session:
            await session.close()


async def _stop(session: StreamingSession | None, sink: WebSocketSink) -> None:
    """Flush the last utterance, wait for every pending result, then say "stopped"."""
    if session:
        await session.finish()
    await sink.send({"type": "stopped"})
