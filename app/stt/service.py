"""Speech segment (WAV bytes) → cleaned text."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .budget import RequestBudgets
from .groq_client import GroqClient, GroqError
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


FINAL_RESERVE = 0.5  # share of each model's per-minute budget kept free for final text…
PREVIEW_RESERVE = 0.7  # …and for text at pauses: grey while-speaking previews stop first when quota is short


class BudgetExhausted(Exception):
    """An optional request was skipped to keep quota for final text."""


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
        budgets: RequestBudgets | None = None,
        hallucination_filter: HallucinationFilter | None = None,
    ) -> None:
        self._groq = groq
        self._model = model  # final text: the accurate model
        self._fast_model = fast_model or model  # live text: the fast model
        self._timeout_s = timeout_s
        self._policy = LanguagePolicy(expected_languages)
        self._filter = hallucination_filter or HallucinationFilter()
        # Shared with the Groq client, which drains them on every 429; None = unlimited.
        self._budgets = budgets
        per_minute = budgets.per_minute if budgets else 0
        self._reserve = per_minute * FINAL_RESERVE
        self._preview_reserve = per_minute * PREVIEW_RESERVE

    async def transcribe(
        self, wav: bytes, context: str = "", language: str | None = None, *, optional: bool = False
    ) -> TranscriptionResult:
        """`language`: ISO code to force, or None to detect it per utterance.

        `optional=True` (early passes, or a final whose text is already on
        screen) raises BudgetExhausted instead of eating into the quota reserved
        for final text, and fails fast instead of waiting out a rate limit.

        A required final never waits for the accurate model's quota: if it is
        used up (by our count, or Groq answers 429) the fast model writes the
        final instead — the two models have separate per-minute limits.
        """
        started = time.perf_counter()
        model = self._fast_model if not optional and self._accurate_exhausted() else self._model
        self._spend(model, optional=optional)
        can_fall_back = not optional and model != self._fast_model
        try:
            raw, detected, prompt = await self._decode(
                wav, context, language, model, optional, may_wait=not optional and not can_fall_back
            )
        except GroqError as error:
            if not can_fall_back or error.status != 429:
                raise
            logger.info("%s is rate-limited: final text from %s", self._model, self._fast_model)
            self._spend(self._fast_model, optional=False)
            raw, detected, prompt = await self._decode(wav, context, language, self._fast_model, False, may_wait=True)

        return TranscriptionResult(
            text=self._filter.clean(raw, prompt=prompt),
            language=detected or language_code(raw.get("language")),
            audio_ms=round(float(raw.get("duration") or 0) * 1000),
            upstream_ms=round((time.perf_counter() - started) * 1000),
        )

    def _accurate_exhausted(self) -> bool:
        """By our count the accurate model has no request left this minute."""
        if self._budgets is None or self._fast_model == self._model:
            return False
        return self._budgets.for_model(self._model).available < 1

    async def _decode(
        self, wav: bytes, context: str, language: str | None, model: str, optional: bool, *, may_wait: bool
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """One transcription with `model`: auto-detect (speculative) or a fixed language."""
        if language is None:
            return await self._transcribe_auto(wav, model, optional, may_wait)
        prompt = prompt_for(language, context)
        return await self._request(wav, language, prompt, model=model, may_wait=may_wait), language, prompt

    async def preview(self, wav: bytes, language: str | None = None, *, at_pause: bool = False) -> TranscriptionResult:
        """Quick single-pass transcription with the fast model, for text shown on screen at once.

        One request, no speculative second pass: this text is replaced by the
        final transcription of the utterance moments later if they differ.
        `at_pause=False` (grey text while still speaking) is the first thing
        dropped when the request budget runs low.
        """
        started = time.perf_counter()
        keep = self._reserve if at_pause else self._preview_reserve
        self._spend(self._fast_model, optional=True, keep=keep)  # never eats the final-text reserve
        # No prompt here: on audio cut off mid-word Whisper tends to "finish the sentence"
        # with words from its prompt. The accurate final restores the code-mixed style.
        prompt = None
        raw = await self._request(wav, language, prompt, model=self._fast_model, may_wait=False)
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

    async def _transcribe_auto(
        self, wav: bytes, model: str, optional: bool, may_wait: bool
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """Speculative parallel decoding for auto-detect.

        Whisper writes English most faithfully with *no* prompt, and Hinglish
        most faithfully when told "Hindi" plus the code-mixed style prompt —
        but which one applies is only known after language detection. So both
        passes start at the same time; the plain pass's detection decides which
        answer is kept and the other is cancelled. Hindi utterances therefore
        cost no extra latency, English ones none at all.

        The speculative pass is optional: when the request budget is short it is
        skipped and Hindi keeps the plain pass's (Devanagari-only) text.
        """
        plain = asyncio.ensure_future(self._request(wav, None, None, model=model, may_wait=may_wait))
        hindi = None
        if self._policy.expects("hi") and self._try_spend(model):
            hindi = asyncio.ensure_future(self._request(wav, "hi", CODE_MIXED_STYLE, model=model, may_wait=False))
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
        self._spend(model, optional=optional)
        prompt = prompt_for(target, "")
        return await self._request(wav, target, prompt, model=model, may_wait=may_wait), target, prompt

    def _spend(self, model: str, *, optional: bool, keep: float | None = None) -> None:
        if self._budgets is None:
            return
        budget = self._budgets.for_model(model)
        if not optional:
            budget.take()
        elif not budget.try_take(keep=self._reserve if keep is None else keep):
            raise BudgetExhausted(model)

    def _try_spend(self, model: str) -> bool:
        return self._budgets is None or self._budgets.for_model(model).try_take(keep=self._reserve)

    async def _request(
        self,
        wav: bytes,
        language: str | None,
        prompt: str | None,
        *,
        model: str | None = None,
        may_wait: bool = True,
    ) -> dict[str, Any]:
        return await self._groq.transcribe(
            wav,
            model=model or self._model,
            language=language,
            prompt=prompt or None,
            timeout_s=self._timeout_s,
            may_wait=may_wait,
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
