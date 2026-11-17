"""One live consultation = one StreamingSession.

    PCM bytes ─▶ 20 ms frames ─▶ EnergyVad ─▶ SpeechSegmenter ─▶ utterance ─▶ final text
                                     │              │
                      VAD telemetry ─┘              └─ while it is still open: live text

Silence and noise stop at the segmenter; only speech reaches the recogniser.

Making it feel instant (like talking to a person):

* **Live previews** — while someone keeps talking, the utterance so far is
  re-transcribed about once a second (one quick request) and shown as live
  text, so words appear while they are still being spoken.
* **Commit at the pause** — the moment a short pause (60 ms) appears, two
  transcriptions of the utterance so far start together: the fast model's
  text goes on screen as soon as it returns (~250 ms), and the accurate
  model's text follows ~100–200 ms later and corrects it if they differ.
  Neither waits for the 600 ms hangover, which only decides where one
  utterance ends. If no more speech follows, the accurate result becomes the
  final text (no extra request); if they keep talking, the text simply grows.
* **Parallel** — every transcription is its own asyncio task; a semaphore
  caps final-quality requests, so a long utterance never blocks the next one.
  Results carry the utterance id and the browser keeps them in speech order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

from app.audio.segmenter import SpeechSegment, SpeechSegmenter
from app.audio.vad import EnergyVad, VadDecision
from app.audio.wav import encode_wav, pcm16_to_float
from app.config import Settings
from app.stt.service import TranscriptionResult

from .errors import public_error_message

logger = logging.getLogger(__name__)

EventSender = Callable[[dict[str, Any]], Awaitable[None]]
T = TypeVar("T")


class Transcriber(Protocol):
    async def transcribe(self, wav: bytes, context: str = "", language: str | None = None) -> TranscriptionResult: ...

    async def preview(self, wav: bytes, language: str | None = None) -> TranscriptionResult: ...


@dataclass
class LiveUtterance:
    """Bookkeeping for the utterance that is currently being spoken."""

    id: int
    start_ms: int
    requested: int = 0  # previews requested so far (eager passes included)
    shown: int = 0  # newest of them already shown — older results arriving late are ignored
    last_preview_at_ms: int = 0  # utterance length when the last quick preview was requested
    eager_task: asyncio.Task[TranscriptionResult] | None = None
    eager_voiced_ms: int = -1  # voiced audio the eager transcription covers
    preview_running: bool = False
    language: str | None = None  # language the last eager pass heard (keeps previews in one script)
    committed_voiced_ms: int = -1  # voiced audio covered by the text committed on screen at a pause
    committed_at: float = 0.0  # when that text was sent


class StreamingSession:
    def __init__(
        self,
        transcriber: Transcriber,
        settings: Settings,
        send: EventSender,
        *,
        chunk_offset: int = 0,
        context: str = "",
        language: str | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._settings = settings
        self._send = send
        self._language = language  # ISO code to force, or None = detect per utterance
        self._vad = EnergyVad(settings.vad, settings.audio)
        self._segmenter = SpeechSegmenter(settings.segmenter, settings.audio)
        self._parallel = asyncio.Semaphore(settings.stt_concurrency)
        self._buffer = bytearray()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._live: LiveUtterance | None = None
        self._texts: dict[int, str] = {}  # final text per utterance, used as Whisper context
        self._finished: set[int] = set()  # utterances whose final result was sent
        self._failed: dict[int, SpeechSegment] = {}  # kept for manual retry
        self._initial_context = context
        self._next_id = chunk_offset + 1
        self._last_voice_at = time.perf_counter()
        self._previews_paused_until_ms = 0
        self.captured_ms = 0
        self.speech_ms = 0

    # ── input ────────────────────────────────────────────────────────────────

    async def feed(self, data: bytes) -> None:
        """Accept raw PCM16 bytes of any length; analyse every complete 20 ms frame."""
        self._buffer.extend(data)
        frame_bytes = self._settings.audio.frame_bytes
        telemetry: list[list[float | int]] = []

        while len(self._buffer) >= frame_bytes:
            pcm = np.frombuffer(bytes(self._buffer[:frame_bytes]), dtype="<i2")
            del self._buffer[:frame_bytes]
            decision = self._vad.process(pcm16_to_float(pcm))
            if decision.is_speech:
                self._last_voice_at = time.perf_counter()
            segment = self._segmenter.push(pcm, decision.is_speech)
            self.captured_ms += self._settings.audio.frame_ms
            telemetry.append(self._telemetry(decision))
            if segment is not None:
                await self._finalize(segment)
            elif self._segmenter.in_speech:
                await self._track_live()

        if telemetry:
            await self._send(
                {"type": "vad", "frames": telemetry, "capturedMs": self.captured_ms, "speechMs": self.speech_ms}
            )

    async def finish(self) -> None:
        """Mic stopped: finalise the utterance in progress and wait for every pending result."""
        segment = self._segmenter.flush()
        if segment is not None:
            await self._finalize(segment)
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def retry(self, chunk_id: int) -> None:
        segment = self._failed.pop(chunk_id, None)
        if segment is None:
            return
        self._finished.discard(chunk_id)
        await self._send({"type": "chunk", "id": chunk_id, "status": "pending"})
        self._spawn(self._complete(chunk_id, segment, None, time.perf_counter()))

    async def close(self) -> None:
        """Connection gone: cancel outstanding work."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── live text while the utterance is open ───────────────────────────────

    async def _track_live(self) -> None:
        segmenter, live_settings = self._segmenter, self._settings.live
        if self._live is None:
            # Announce the utterance only once it is certain to be kept (not a cough).
            if segmenter.voiced_ms < self._settings.segmenter.min_voiced_ms:
                return
            self._live = await self._announce(segmenter.open_start_ms)
        if not live_settings.enabled:
            return

        live = self._live
        if segmenter.trailing_silence_ms >= live_settings.eager_pause_ms:
            if live.eager_voiced_ms != segmenter.voiced_ms:  # new speech since the last pause
                live.eager_voiced_ms = voiced = segmenter.voiced_ms
                pcm = segmenter.snapshot()
                self._spawn(self._fast_commit(live, pcm, voiced))  # on screen first
                live.eager_task = self._spawn(self._eager(live, pcm, voiced))  # accurate; becomes the final
        elif self._quick_preview_due(live):
            live.last_preview_at_ms = segmenter.open_length_ms
            live.preview_running = True
            self._spawn(self._quick_preview(live, segmenter.snapshot(), segmenter.voiced_ms))

    def _quick_preview_due(self, live: LiveUtterance) -> bool:
        length, settings = self._segmenter.open_length_ms, self._settings.live
        return (
            not live.preview_running
            and self._segmenter.trailing_silence_ms == 0
            and length >= settings.first_preview_ms
            and length - live.last_preview_at_ms >= settings.preview_every_ms
            and self.captured_ms >= self._previews_paused_until_ms
        )

    async def _eager(self, live: LiveUtterance, pcm: np.ndarray, voiced_ms: int) -> TranscriptionResult:
        """Final-quality transcription at a pause; shown now, and reused as the final if speech ended."""
        ticket = self._ticket(live)
        async with self._parallel:
            result = await self._transcriber.transcribe(
                self._wav(pcm), context=self._context_before(live.id), language=self._language
            )
        self._remember_language(live, result)
        if await self._show_partial(live, ticket, result.text, voiced_ms, stable=True):
            self._mark_committed(live, voiced_ms)
        return result

    async def _fast_commit(self, live: LiveUtterance, pcm: np.ndarray, voiced_ms: int) -> None:
        """Fast-model text for the speech so far, shown as soon as the speaker pauses."""
        ticket = self._ticket(live)
        try:
            result = await self._transcriber.preview(self._wav(pcm), language=self._language or live.language)
        except Exception as error:  # the accurate pass is still coming
            logger.info("Fast commit skipped: %s", error)
            return
        if await self._show_partial(live, ticket, result.text, voiced_ms, stable=True):
            self._mark_committed(live, voiced_ms)

    @staticmethod
    def _mark_committed(live: LiveUtterance, voiced_ms: int) -> None:
        """Remember when text for this much speech first reached the screen (for latency)."""
        if live.committed_voiced_ms != voiced_ms:
            live.committed_voiced_ms, live.committed_at = voiced_ms, time.perf_counter()

    async def _quick_preview(self, live: LiveUtterance, pcm: np.ndarray, voiced_ms: int) -> None:
        ticket = self._ticket(live)
        try:
            result = await self._transcriber.preview(self._wav(pcm), language=self._language or live.language)
        except Exception as error:  # previews are best-effort; back off and let finals have the quota
            logger.info("Live preview skipped: %s", error)
            self._previews_paused_until_ms = self.captured_ms + self._settings.live.preview_backoff_ms
            return
        finally:
            live.preview_running = False
        await self._show_partial(live, ticket, result.text, voiced_ms)

    async def _show_partial(
        self, live: LiveUtterance, ticket: int, text: str, voiced_ms: int, *, stable: bool = False
    ) -> bool:
        """Show live text unless it is stale, already final, or junk. Returns True if sent.

        `stable` = final-quality text committed at a pause (shown as normal text);
        quick previews are shown greyed and need a minimum amount of speech.
        """
        if ticket <= live.shown or live.id in self._finished or not any(ch.isalnum() for ch in text):
            return False
        if not stable and voiced_ms < self._settings.live.min_shown_voiced_ms:
            return False
        live.shown = ticket
        await self._send({"type": "chunk", "id": live.id, "status": "partial", "text": text, "stable": stable})
        return True

    @staticmethod
    def _remember_language(live: LiveUtterance, result: TranscriptionResult) -> None:
        # Only final-quality passes (which verify their detection) set it: a quick preview
        # that misheard the language must not lock later previews into that language.
        if result.text and result.language:
            live.language = result.language

    @staticmethod
    def _ticket(live: LiveUtterance) -> int:
        live.requested += 1
        return live.requested

    # ── final text ───────────────────────────────────────────────────────────

    async def _finalize(self, segment: SpeechSegment) -> None:
        live = self._live or await self._announce(segment.start_ms)
        self._live = None
        self.speech_ms += segment.duration_ms
        # The eager pass can stand in for the final only if no speech came after it.
        eager = live.eager_task if live.eager_voiced_ms == segment.voiced_ms else None
        self._spawn(self._complete(live.id, segment, eager, self._last_voice_at, live))

    async def _complete(
        self,
        chunk_id: int,
        segment: SpeechSegment,
        eager: asyncio.Task[TranscriptionResult] | None,
        speech_end: float,
        live: LiveUtterance | None = None,
    ) -> None:
        result = await self._reuse(eager)
        if result is None:
            try:
                async with self._parallel:
                    result = await self._transcriber.transcribe(
                        self._wav(segment.pcm), context=self._context_before(chunk_id), language=self._language
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # every failure becomes a "failed" chunk the user can retry
                logger.warning("Transcription of chunk %s failed: %s", chunk_id, error)
                self._failed[chunk_id] = segment
                self._finished.add(chunk_id)
                await self._send(
                    {"type": "chunk", "id": chunk_id, "status": "failed", "error": public_error_message(error)}
                )
                return

        self._texts[chunk_id] = result.text
        self._finished.add(chunk_id)
        # Honest, user-perceived latency: last spoken word → this text on screen. When the
        # text was already committed at the pause, that moment counts, not the hangover end.
        on_screen = time.perf_counter()
        if live is not None and eager is not None and live.committed_voiced_ms == segment.voiced_ms:
            on_screen = live.committed_at
        await self._send(
            {
                "type": "chunk",
                "id": chunk_id,
                "status": "done",
                "text": result.text,
                "language": result.language,
                "startMs": segment.start_ms,
                "endMs": segment.end_ms,
                "latencyMs": max(0, round((on_screen - speech_end) * 1000)),
                "upstreamMs": result.upstream_ms,
            }
        )

    @staticmethod
    async def _reuse(eager: asyncio.Task[TranscriptionResult] | None) -> TranscriptionResult | None:
        if eager is None:
            return None
        try:
            return await eager  # usually already finished by the time the hangover ends
        except asyncio.CancelledError:
            raise
        except Exception:
            return None  # fall back to a fresh final transcription

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _announce(self, start_ms: int) -> LiveUtterance:
        live = LiveUtterance(id=self._next_id, start_ms=start_ms)
        self._next_id += 1
        await self._send({"type": "chunk", "id": live.id, "status": "pending", "startMs": start_ms})
        return live

    def _context_before(self, chunk_id: int) -> str:
        """Transcript that precedes this utterance (only what has finished so far)."""
        earlier = [self._texts[i] for i in sorted(self._texts) if i < chunk_id and self._texts[i]]
        context = " ".join([self._initial_context, *earlier]).strip()
        return context[-self._settings.context_chars :]

    def _wav(self, pcm: np.ndarray) -> bytes:
        return encode_wav(pcm, self._settings.audio.sample_rate)

    def _spawn(self, coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        # Mark outcomes as observed: an unused eager pass may fail without anyone awaiting it.
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        return task

    def _telemetry(self, decision: VadDecision) -> list[float | int]:
        return [
            round(decision.energy_db, 1),
            round(decision.threshold_db, 1),
            int(decision.is_speech),
            int(self._segmenter.in_speech),
        ]
