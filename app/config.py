"""Application settings, read once from environment variables.

Every tunable number of the pipeline lives here, grouped by concern, so the
behaviour of the VAD / segmenter / STT can be changed without touching logic.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = PROJECT_ROOT / "public"

_GROQ_KEY_NAME = re.compile(r"^GROQ_API_KEY(_\d+)?$")


class ConfigError(RuntimeError):
    """The server cannot run with the current environment."""


@dataclass(frozen=True)
class AudioSettings:
    """Format of the stream the browser sends: 16 kHz mono PCM16 (Whisper's native rate)."""

    sample_rate: int = 16_000
    frame_ms: int = 20

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2  # int16


@dataclass(frozen=True)
class VadSettings:
    """Energy VAD with an adaptive noise floor. See app/audio/vad.py."""

    band_low_hz: float = 150.0  # speech band used for energy: drops hum and rumble…
    band_high_hz: float = 4_000.0  # …and hiss above the main speech formants
    calibration_ms: int = 300  # learn the room before deciding anything
    noise_window_ms: int = 3_000  # noise floor = quietest frame of the last 3 s
    noise_fall_rate: float = 0.3  # floor follows quieter frames quickly…
    noise_rise_rate: float = 0.04  # …and louder backgrounds slowly
    min_noise_floor_db: float = -85.0  # guard against digital silence
    onset_snr_db: float = 10.0  # dB above the floor needed to START speech
    offset_snr_db: float = 6.0  # dB above the floor needed to STAY in speech
    min_speech_db: float = -55.0  # absolute loudness floor for speech
    max_onset_flatness: float = 0.45  # noise-like (flat) spectra cannot start speech…
    flatness_smoothing: float = 0.4  # …judged on a ~3-frame moving average, not one lucky frame


@dataclass(frozen=True)
class SegmenterSettings:
    """Turns frame decisions into utterances. See app/audio/segmenter.py."""

    pre_roll_ms: int = 300  # audio kept before the onset (soft first consonant)
    onset_ms: int = 60  # consecutive speech needed to open an utterance (rejects clicks)
    hangover_ms: int = 600  # silence that ends an utterance; natural mid-sentence pauses are shorter
    post_roll_ms: int = 250  # trailing silence kept on the utterance
    soft_max_ms: int = 10_000  # beyond this, close at the next short pause…
    soft_hangover_ms: int = 250  # …of this length
    hard_max_ms: int = 15_000  # never send one huge, slow request
    min_voiced_ms: int = 200  # less voiced audio than this = cough/tap, dropped


@dataclass(frozen=True)
class LiveSettings:
    """Live text while someone is still talking. See app/realtime/session.py."""

    enabled: bool = True
    fast_pause_ms: int = 60  # fast lane: at a pause this long, put the speech so far on screen
    fast_min_gap_ms: int = 300  # fast lane: one request at a time, at least this far apart
    accurate_pause_ms: int = 250  # accurate lane: at a pause this long, final-quality transcription
    first_preview_ms: int = 900  # utterance length before the first grey preview (tiny clips → junk)
    preview_every_ms: int = 700  # during continuous speech, refresh the grey live text this often
    min_shown_voiced_ms: int = 500  # grey previews of less speech than this are not shown
    preview_backoff_ms: int = 3_000  # after a failed fast request (e.g. rate limit), pause the fast lane


@dataclass(frozen=True)
class Settings:
    groq_keys: tuple[str, ...]
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Final text uses the accurate model; text put on screen first uses the fast one
    # (~235 ms vs ~280 ms median on Groq, more on long clips) and is corrected if they differ.
    stt_model: str = "whisper-large-v3"
    stt_fast_model: str = "whisper-large-v3-turbo"
    llm_model: str = "openai/gpt-oss-120b"
    stt_language: str | None = None  # default when the browser doesn't choose; None = auto-detect
    stt_expected_languages: tuple[str, ...] = ("en", "hi")  # auto-detect outside these is re-checked
    stt_concurrency: int = 4  # utterances transcribed in parallel per session
    groq_rpm_per_key: int = 20  # Groq's requests/minute per model per key (free tier); budget scales with keys
    stt_timeout_s: float = 15.0
    llm_timeout_s: float = 45.0
    context_chars: int = 240  # transcript tail sent to Whisper as its prompt
    allowed_origins: tuple[str, ...] = ()
    sessions_per_minute: int = 30  # per client IP
    summaries_per_minute: int = 20  # per client IP
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    segmenter: SegmenterSettings = field(default_factory=SegmenterSettings)
    live: LiveSettings = field(default_factory=LiveSettings)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        keys = collect_groq_keys(env)
        if not keys:
            raise ConfigError("No Groq API key configured. Set GROQ_API_KEY (or GROQ_API_KEY_1..n / GROQ_API_KEYS).")
        language = env.get("STT_LANGUAGE", "auto").strip().lower()
        expected = tuple(code.strip().lower() for code in env.get("STT_EXPECTED_LANGUAGES", "en,hi").split(","))
        return cls(
            groq_keys=tuple(keys),
            groq_base_url=env.get("GROQ_BASE_URL", cls.groq_base_url).rstrip("/"),
            stt_model=env.get("GROQ_STT_MODEL", cls.stt_model),
            stt_fast_model=env.get("GROQ_STT_FAST_MODEL", cls.stt_fast_model),
            llm_model=env.get("GROQ_LLM_MODEL", cls.llm_model),
            stt_language=None if language in ("", "auto") else language,
            stt_expected_languages=tuple(code for code in expected if code) or cls.stt_expected_languages,
            stt_concurrency=int(env.get("STT_CONCURRENCY", cls.stt_concurrency)),
            groq_rpm_per_key=int(env.get("GROQ_RPM_PER_KEY", cls.groq_rpm_per_key)),
            allowed_origins=tuple(o.strip() for o in env.get("ALLOWED_ORIGINS", "").split(",") if o.strip()),
        )


def collect_groq_keys(env: Mapping[str, str]) -> list[str]:
    """Every Groq key from GROQ_API_KEYS (comma list), GROQ_API_KEY and GROQ_API_KEY_<n>.

    Quotes are stripped and duplicates removed, keeping first-seen order.
    """
    candidates = env.get("GROQ_API_KEYS", "").split(",")
    candidates += [value for name, value in env.items() if _GROQ_KEY_NAME.match(name)]
    cleaned = (key.strip().strip("\"'") for key in candidates)
    return list(dict.fromkeys(key for key in cleaned if key))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings for this process. Locally, .env.local / .env are loaded first."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env.local")
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass
    return Settings.from_env(os.environ)
