"""One live consultation = one StreamingSession.

    PCM bytes ─▶ 20 ms frames ─▶ EnergyVad ─▶ SpeechSegmenter ─▶ utterance ─▶ final text
                                     │              │
                      VAD telemetry ─┘              └─ while it is still open: live text

Silence and noise stop at the segmenter; only speech reaches the recogniser.

Making it feel instant, with two lanes per utterance:

* **Fast lane** (fast model, one request, prompt-free) — at a 60 ms pause the
  speech so far is transcribed and put on screen (~250 ms later); during
  continuous speech it refreshes grey live text every ~0.7 s. One request at a
  time, at least 300 ms apart, so the request rate stays within Groq limits.
* **Accurate lane** (accurate model, language-aware) — at a 250 ms pause the
  speech so far gets its final-quality transcription. Its text replaces the
  fast text only while the speaker is still silent, and if no more speech
  follows before the 600 ms hangover ends, it becomes the final text with no
  extra request.
* **Parallel** — every transcription is its own asyncio task; a semaphore caps
  accurate requests, so a long utterance never blocks the next one. Results
  carry the utterance id and the browser keeps them in speech order.
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
from app.stt.service import BudgetExhausted, TranscriptionResult

from .errors import public_error_message

logger = logging.getLogger(__name__)

EventSender = Callable[[dict[str, Any]], Awaitable[None]]
T = TypeVar("T")

TELEMETRY_BATCH_FRAMES = 5


class Transcriber(Protocol):
    async def transcribe(
        self, wav: bytes, context: str = "", language: str | None = None, *, optional: bool = False
    ) -> TranscriptionResult: ...

    async def preview(
        self, wav: bytes, language: str | None = None, *, at_pause: bool = False
    ) -> TranscriptionResult: ...


@dataclass
class LiveUtterance:
    """Bookkeeping for the utterance that is currently being spoken."""

    id: int
    start_ms: int
    requested: int = 0  # transcriptions requested so far (both lanes)
    shown: int = 0  # newest of them already shown — older results arriving late are ignored
    fast_running: bool = False  # the fast lane runs one request at a time…
    last_fast_at_ms: int = -10_000  # …and at least `fast_min_gap_ms` apart (stream time)
    fast_voiced_ms: int = -1  # voiced audio covered by the last pause-time fast pass
    last_preview_at_ms: int = 0  # utterance length at the last grey preview
    eager_task: asyncio.Task[TranscriptionResult] | None = None  # the accurate lane's latest pass
    eager_voiced_ms: int = -1  # voiced audio that accurate pass covers
    language: str | None = None  # language the accurate lane heard (keeps fast text in one script)
    committed_voiced_ms: int = -1  # voiced audio covered by the text committed on screen at a pause
    committed_at: float = 0.0  # when that text was sent
    committed_text: str = ""  # the newest text on screen for that speech…
    committed_language: str | None = None  # …kept as the final's fallback when quota runs out


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
        self._telemetry_buffer: list[list[float | int]] = []
        self.captured_ms = 0
        self.speech_ms = 0

    # ── input ────────────────────────────────────────────────────────────────

    async def feed(self, data: bytes) -> None:
        """Accept raw PCM16 bytes of any length; analyse every complete 20 ms frame."""
        self._buffer.extend(data)
        frame_bytes = self._settings.audio.frame_bytes

        while len(self._buffer) >= frame_bytes:
            pcm = np.frombuffer(bytes(self._buffer[:frame_bytes]), dtype="<i2")
            del self._buffer[:frame_bytes]
            decision = self._vad.process(pcm16_to_float(pcm))
            if decision.is_speech:
                self._last_voice_at = time.perf_counter()
            segment = self._segmenter.push(pcm, decision.is_speech)
            self.captured_ms += self._settings.audio.frame_ms
            self._telemetry_buffer.append(self._telemetry(decision))
            if segment is not None:
                await self._finalize(segment)
            elif self._segmenter.in_speech:
                await self._track_live()

        if len(self._telemetry_buffer) >= TELEMETRY_BATCH_FRAMES:
            await self._flush_telemetry()

    async def _flush_telemetry(self) -> None:
        """VAD telemetry goes out ~10× a second (5 frames per message), not once per 20 ms frame."""
        if not self._telemetry_buffer:
            return
        frames, self._telemetry_buffer = self._telemetry_buffer, []
        await self._send({"type": "vad", "frames": frames, "capturedMs": self.captured_ms, "speechMs": self.speech_ms})

    async def finish(self) -> None:
        """Mic stopped: finalise the utterance in progress and wait for every pending result."""
        await self._flush_telemetry()
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

        live, silence, voiced = self._live, segmenter.trailing_silence_ms, segmenter.voiced_ms
        if silence >= live_settings.accurate_pause_ms and live.eager_voiced_ms != voiced:
            live.eager_voiced_ms = voiced
            live.eager_task = self._spawn(self._accurate_pass(live, segmenter.snapshot(), voiced))

        if not self._fast_lane_free(live):
            return
        if silence >= live_settings.fast_pause_ms:
            if live.fast_voiced_ms != voiced:  # new speech since the last pause-time fast pass
                live.fast_voiced_ms = voiced
                self._start_fast_pass(live, stable=True)
        elif self._grey_preview_due(live):
            live.last_preview_at_ms = segmenter.open_length_ms
            self._start_fast_pass(live, stable=False)

    def _fast_lane_free(self, live: LiveUtterance) -> bool:
        settings = self._settings.live
        return (
            not live.fast_running
            and self.captured_ms - live.last_fast_at_ms >= settings.fast_min_gap_ms
            and self.captured_ms >= self._previews_paused_until_ms
        )

    def _grey_preview_due(self, live: LiveUtterance) -> bool:
        length, settings = self._segmenter.open_length_ms, self._settings.live
        return length >= settings.first_preview_ms and length - live.last_preview_at_ms >= settings.preview_every_ms

    def _start_fast_pass(self, live: LiveUtterance, *, stable: bool) -> None:
        live.fast_running, live.last_fast_at_ms = True, self.captured_ms
        self._spawn(self._fast_pass(live, self._segmenter.snapshot(), self._segmenter.voiced_ms, stable))

    async def _fast_pass(self, live: LiveUtterance, pcm: np.ndarray, voiced_ms: int, stable: bool) -> None:
        """Fast lane: quick text for the speech so far (normal text at a pause, grey while speaking)."""
        ticket = self._ticket(live)
        try:
            result = await self._transcriber.preview(
                self._wav(pcm), language=self._language or live.language, at_pause=stable
            )
        except BudgetExhausted:
            return  # quota is short: skip quietly, the budget itself decides when to try again
        except Exception as error:  # a real failure (e.g. Groq 429): back off so the accurate lane keeps the quota
            logger.info("Fast pass skipped: %s", error)
            self._previews_paused_until_ms = self.captured_ms + self._settings.live.preview_backoff_ms
            return
        finally:
            live.fast_running = False
        if await self._show_partial(live, ticket, result.text, voiced_ms, stable=stable) and stable:
            self._mark_committed(live, voiced_ms, result)

    async def _accurate_pass(self, live: LiveUtterance, pcm: np.ndarray, voiced_ms: int) -> TranscriptionResult:
        """Accurate lane: final-quality text at a real pause; reused as the final if speech has ended."""
        ticket = self._ticket(live)
        async with self._parallel:
            # Optional: skipped when the request budget is short; the final is then made at the end.
            result = await self._transcriber.transcribe(
                self._wav(pcm), context=self._context_before(live.id), language=self._language, optional=True
            )
        self._remember_language(live, result)
        # Only while they are still silent: if speech resumed, this text is already incomplete.
        still_paused = self._live is live and self._segmenter.voiced_ms == voiced_ms
        if still_paused and await self._show_partial(live, ticket, result.text, voiced_ms, stable=True):
            self._mark_committed(live, voiced_ms, result)
        return result

    @staticmethod
    def _mark_committed(live: LiveUtterance, voiced_ms: int, result: TranscriptionResult) -> None:
        """Remember what text for this much speech is on screen, and when it first got there."""
        if live.committed_voiced_ms != voiced_ms:
            live.committed_voiced_ms, live.committed_at = voiced_ms, time.perf_counter()
        live.committed_text, live.committed_language = result.text, result.language

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
        error: Exception | None = None
        if result is None:
            result, error = await self._final_transcription(chunk_id, segment, live)
        if result is None:
            await self._fail(chunk_id, segment, error)
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

    async def _final_transcription(
        self, chunk_id: int, segment: SpeechSegment, live: LiveUtterance | None
    ) -> tuple[TranscriptionResult | None, Exception | None]:
        """Accurate text for the whole utterance.

        If text for exactly this speech is already on screen, the request is
        optional: when the quota is short (or Groq rate-limits it) it is skipped
        and the on-screen text becomes the final — no waiting, no failure.
        """
        on_screen = live is not None and live.committed_voiced_ms == segment.voiced_ms and bool(live.committed_text)
        try:
            async with self._parallel:
                result = await self._transcriber.transcribe(
                    self._wav(segment.pcm),
                    context=self._context_before(chunk_id),
                    language=self._language,
                    optional=on_screen,
                )
            return result, None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not on_screen or live is None:
                return None, error
            logger.info("Chunk %s keeps the text already on screen as final: %s", chunk_id, error)
            fallback = TranscriptionResult(
                text=live.committed_text, language=live.committed_language, audio_ms=segment.duration_ms, upstream_ms=0
            )
            return fallback, None

    async def _fail(self, chunk_id: int, segment: SpeechSegment, error: Exception | None) -> None:
        """Every failure becomes a "failed" chunk the user can retry."""
        logger.warning("Transcription of chunk %s failed: %s", chunk_id, error)
        self._failed[chunk_id] = segment
        self._finished.add(chunk_id)
        message = public_error_message(error) if error else "Transcription failed."
        await self._send({"type": "chunk", "id": chunk_id, "status": "failed", "error": message})

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
