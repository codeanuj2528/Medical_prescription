"""Speech segment (WAV bytes) → cleaned text."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .groq_client import GroqClient
from .hallucination import HallucinationFilter
from .language import LanguagePolicy, language_code

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 600  # Whisper's prompt window is 224 tokens; 600 chars stays inside it

# Whisper copies the *style* of its prompt. A short code-mixed example teaches it
# how Indian consultations are written: Hindi in Devanagari, English medical
# words in Latin script ("मुझे fever है") instead of transliterating them
# ("मेटफॉर्मिन"). It deliberately contains no drugs, doses or numbers: Whisper
# would copy those too (a prompt with "metformin 500 mg" added "mg" to speech
# that never said it).
CODE_MIXED_STYLE = "डॉक्टर साहब, मुझे fever और cough है, sugar की tablet भी लेता हूँ।"


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str | None  # ISO code of what was actually transcribed ("en", "hi", …)
    audio_ms: int
    upstream_ms: int  # time spent waiting on Groq (all attempts)


class TranscriptionService:
    def __init__(
        self,
        groq: GroqClient,
        *,
        model: str,
        fast_model: str | None = None,
        timeout_s: float,
        expected_languages: Sequence[str] = ("en", "hi"),
        hallucination_filter: HallucinationFilter | None = None,
    ) -> None:
        self._groq = groq
        self._model = model  # final text: the accurate model
        self._fast_model = fast_model or model  # live text: the fast model
        self._timeout_s = timeout_s
        self._policy = LanguagePolicy(expected_languages)
        self._filter = hallucination_filter or HallucinationFilter()

    async def transcribe(self, wav: bytes, context: str = "", language: str | None = None) -> TranscriptionResult:
        """`language`: ISO code to force, or None to detect it per utterance."""
        started = time.perf_counter()
        if language is None:
            raw, language, prompt = await self._transcribe_auto(wav)
        else:
            prompt = prompt_for(language, context)
            raw = await self._request(wav, language, prompt)

        return TranscriptionResult(
            text=self._filter.clean(raw, prompt=prompt),
            language=language or language_code(raw.get("language")),
            audio_ms=round(float(raw.get("duration") or 0) * 1000),
            upstream_ms=round((time.perf_counter() - started) * 1000),
        )

    async def preview(self, wav: bytes, language: str | None = None) -> TranscriptionResult:
        """Quick single-pass transcription with the fast model, for text shown on screen at once.

        One request, no speculative second pass: this text is replaced by the
        final transcription of the utterance moments later if they differ.
        """
        started = time.perf_counter()
        prompt = prompt_for(language, "") if language else None
        raw = await self._request(wav, language, prompt, model=self._fast_model)
        detected = language_code(raw.get("language"))
        # A preview "heard" in a language nobody here speaks (e.g. Japanese on half a
        # second of Hindi) is noise: show nothing rather than flash wrong text.
        plausible = language is not None or self._policy.resolve(raw.get("language")) == detected
        return TranscriptionResult(
            text=self._filter.clean(raw, prompt=prompt) if plausible else "",
            language=language or detected,
            audio_ms=round(float(raw.get("duration") or 0) * 1000),
            upstream_ms=round((time.perf_counter() - started) * 1000),
        )

    async def _transcribe_auto(self, wav: bytes) -> tuple[dict[str, Any], str | None, str | None]:
        """Speculative parallel decoding for auto-detect.

        Whisper writes English most faithfully with *no* prompt, and Hinglish
        most faithfully when told "Hindi" plus the code-mixed style prompt —
        but which one applies is only known after language detection. So both
        passes start at the same time; the plain pass's detection decides which
        answer is kept and the other is cancelled. Hindi utterances therefore
        cost no extra latency, English ones none at all.
        """
        plain = asyncio.ensure_future(self._request(wav, None, None))
        hindi = None
        if self._policy.expects("hi"):
            hindi = asyncio.ensure_future(self._request(wav, "hi", CODE_MIXED_STYLE))
        try:
            raw = await plain
        except BaseException:
            _discard(hindi)
            raise

        target = self._policy.resolve(raw.get("language"))
        if target == "hi" and hindi is not None:
            try:
                return await hindi, "hi", CODE_MIXED_STYLE
            except Exception as error:  # the plain pass is still a valid (if less polished) answer
                logger.warning("Hindi pass failed, keeping the auto-detect result: %s", error)
                return raw, "hi", None

        _discard(hindi)
        if target is None or target == language_code(raw.get("language")):
            return raw, target, None
        # Implausible detection (e.g. "Welsh" for a one-word English reply): redo it in the expected language.
        logger.info("Detected %r — re-transcribing as %r", raw.get("language"), target)
        prompt = prompt_for(target, "")
        return await self._request(wav, target, prompt), target, prompt

    async def _request(
        self, wav: bytes, language: str | None, prompt: str | None, *, model: str | None = None
    ) -> dict[str, Any]:
        return await self._groq.transcribe(
            wav, model=model or self._model, language=language, prompt=prompt or None, timeout_s=self._timeout_s
        )


def _discard(task: asyncio.Future[Any] | None) -> None:
    """Cancel a speculative request we no longer need (and silence its outcome)."""
    if task is None:
        return
    task.cancel()
    task.add_done_callback(lambda done: done.cancelled() or done.exception())


def prompt_for(language: str | None, context: str) -> str | None:
    """Which prompt to give Whisper for a *fixed* language.

    * English: the transcript tail (keeps names and spellings consistent).
    * Hindi: the code-mixed style example. An English transcript tail here
      would pull Hindi speech towards an English *translation*.
    """
    if language == "en":
        return build_prompt(context) or None
    return CODE_MIXED_STYLE


def build_prompt(context: str) -> str:
    """Whisper reads its prompt as "the text that came just before".

    Passing the transcript tail keeps names, drug spellings and style
    consistent across utterances that are transcribed independently (and in
    parallel). Returns "" when there is no context yet.
    """
    tail = " ".join(context.split())
    if len(tail) <= MAX_PROMPT_CHARS:
        return tail
    start = len(tail) - MAX_PROMPT_CHARS
    if tail[start - 1] != " ":  # the cut landed mid-word: skip to the next word
        next_space = tail.find(" ", start)
        start = next_space + 1 if next_space != -1 else start
    return tail[start:].lstrip()
