"""WebSocket protocol between the browser and the server.

Browser → server
    binary frames  : raw 16 kHz mono PCM16 (little-endian), any length
    {"type": "start", "chunkOffset": 12, "context": "...transcript tail...", "language": "auto" | "en" | "hi"}
    {"type": "stop"}            flush the last utterance, wait for pending results
    {"type": "retry", "id": 7}  re-send a failed utterance to the recogniser
    {"type": "ping"}            heartbeat: proves the connection still works both ways

Server → browser
    {"type": "ready"}
    {"type": "vad", "frames": [[energyDb, thresholdDb, isSpeech, inUtterance], ...],
     "capturedMs": ..., "speechMs": ...}
    {"type": "chunk", "id": 3, "status": "pending", "startMs": ...}       an utterance has started
    {"type": "chunk", "id": 3, "status": "partial", "text": "..."}        live text while it is spoken
    {"type": "chunk", "id": 3, "status": "done", "text": "...", "language": "hi",
     "startMs": ..., "endMs": ..., "latencyMs": ..., "upstreamMs": ...}   final text
    {"type": "chunk", "id": 3, "status": "failed", "error": "..."}
    {"type": "stopped"}
    {"type": "pong"}
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class StartCommand(BaseModel):
    type: Literal["start"]
    chunk_offset: int = Field(default=0, ge=0, alias="chunkOffset")
    context: str = Field(default="", max_length=2_000)
    language: Literal["auto", "en", "hi"] | None = None  # None = server default

    def language_code(self, default: str | None) -> str | None:
        """ISO code to force, or None for auto-detect."""
        if self.language is None:
            return default
        return None if self.language == "auto" else self.language


class StopCommand(BaseModel):
    type: Literal["stop"]


class RetryCommand(BaseModel):
    type: Literal["retry"]
    id: int


class PingCommand(BaseModel):
    type: Literal["ping"]


AnyCommand = StartCommand | StopCommand | RetryCommand | PingCommand
Command = Annotated[AnyCommand, Field(discriminator="type")]
COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


def parse_command(text: str) -> AnyCommand:
    return COMMAND_ADAPTER.validate_json(text)
