"""Spoken-language handling for Whisper.

Forcing the wrong language is the #1 cause of "I said X, it wrote Y": with
`language="en"`, Hindi speech comes back as a garbled English *translation*.
So the default is auto-detect — and auto-detect gets a safety net, because on
short clips Whisper sometimes picks a language nobody in the room speaks.
"""

from __future__ import annotations

from collections.abc import Sequence

AUTO = "auto"

# Whisper's verbose_json reports the detected language by name.
_WHISPER_NAMES = {
    "english": "en",
    "hindi": "hi",
    "urdu": "ur",
    "marathi": "mr",
    "gujarati": "gu",
    "bengali": "bn",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "nepali": "ne",
}

# Same spoken language, different script: Whisper often labels Hindi speech "Urdu".
_SCRIPT_REDIRECTS = {"ur": "hi"}


def language_code(name: str | None) -> str | None:
    """'Hindi' → 'hi'. Unknown names are returned lower-cased; None stays None."""
    if not name:
        return None
    key = name.strip().lower()
    return _WHISPER_NAMES.get(key, key)


class LanguagePolicy:
    """Decides whether an auto-detected language is plausible for this clinic.

    `expected` lists the languages spoken in consultations (default en, hi).
    A detection outside it is re-transcribed with a forced language: the
    script-redirect target (Urdu → Hindi) if that is expected, otherwise the
    first expected language (short English words are the usual misfire).
    """

    def __init__(self, expected: Sequence[str]) -> None:
        if not expected:
            raise ValueError("LanguagePolicy needs at least one expected language")
        self._expected = tuple(expected)

    def expects(self, code: str) -> bool:
        return code in self._expected

    def resolve(self, detected_name: str | None) -> str | None:
        """The language this utterance should be treated as (None = unknown, keep as is)."""
        code = language_code(detected_name)
        return code if code is None or code in self._expected else self.retry_language(detected_name)

    def retry_language(self, detected_name: str | None) -> str | None:
        """Language to force for a second attempt, or None if the detection is fine."""
        code = language_code(detected_name)
        if code is None or code in self._expected:
            return None
        redirect = _SCRIPT_REDIRECTS.get(code)
        return redirect if redirect in self._expected else self._expected[0]
