"""Second line of defence after the VAD.

Whisper was trained on subtitled video, so on near-silence it sometimes
"hears" stock phrases ("Thank you.", "Thanks for watching") or loops. Any
noise that slips past the VAD must not become invented text in a medical
record, so each Whisper segment is checked against Whisper's own confidence
signals before its text is kept.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PHANTOM_PHRASES = frozenset(
    {
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "subtitles by the amara org community",
        "you",
        "bye",
    }
)


@dataclass(frozen=True)
class HallucinationThresholds:
    no_speech_prob: float = 0.6  # Whisper's own silence rule: likely no speech…
    min_avg_logprob: float = -1.0  # …and low confidence in the decoded text
    max_compression_ratio: float = 2.4  # text that compresses too well = repetition loop
    phantom_no_speech_prob: float = 0.1  # stock phrases dropped only when Whisper had doubts


class HallucinationFilter:
    def __init__(self, thresholds: HallucinationThresholds | None = None) -> None:
        self._t = thresholds or HallucinationThresholds()

    def clean(self, result: Mapping[str, Any], prompt: str | None = None) -> str:
        """Text of a verbose_json transcription with unreliable segments removed.

        `prompt` is what Whisper was primed with: on near-silence Whisper may
        simply repeat it, which must not end up in the transcript.
        """
        segments: Sequence[Mapping[str, Any]] = result.get("segments") or []
        if not segments:
            return _normalize_spaces(result.get("text", ""))

        kept = [s for s in segments if not self._is_non_speech(s) and not self._is_looping(s)]
        text = _normalize_spaces(" ".join(s.get("text", "") for s in kept))
        doubtful = self._whisper_doubted(kept)
        if doubtful and (self._is_phantom(text) or self._echoes_prompt(text, prompt)):
            return ""
        return text

    def _is_non_speech(self, segment: Mapping[str, Any]) -> bool:
        return (
            segment.get("no_speech_prob", 0.0) > self._t.no_speech_prob
            and segment.get("avg_logprob", 0.0) < self._t.min_avg_logprob
        )

    def _is_looping(self, segment: Mapping[str, Any]) -> bool:
        return segment.get("compression_ratio", 0.0) > self._t.max_compression_ratio

    def _whisper_doubted(self, segments: Sequence[Mapping[str, Any]]) -> bool:
        return any(s.get("no_speech_prob", 0.0) > self._t.phantom_no_speech_prob for s in segments)

    @staticmethod
    def _is_phantom(text: str) -> bool:
        key = re.sub(r"[^a-z ]+", " ", text.lower())
        return " ".join(key.split()) in PHANTOM_PHRASES

    @staticmethod
    def _echoes_prompt(text: str, prompt: str | None) -> bool:
        words = _words(text)
        return bool(prompt) and len(words.split()) >= 2 and words in _words(prompt or "")


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _words(text: str) -> str:
    """Lower-case words without punctuation, for echo comparison (keeps Devanagari intact)."""
    return " ".join(re.sub(r"[^\w\s\u0900-\u097F]", " ", text.lower()).split())
