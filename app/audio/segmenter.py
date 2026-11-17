"""Speech segmenter — turns per-frame VAD decisions into whole utterances.

State machine::

    SILENCE ──speech──▶ ONSET ──3 speech frames in a row──▶ SPEECH ──450 ms silence──▶ emit utterance
       ▲                  │ (any silence: false start)          │
       └──────────────────┘                                     └── 15 s reached: emit and keep going

* Pre-roll: the last 300 ms before the onset are kept, so a soft first
  consonant that the VAD did not flag (the "s" in "six") is still sent.
* Hangover: pauses shorter than 450 ms stay inside the utterance, so a
  sentence is not chopped at every breath. Past 8 s the allowed pause shrinks
  to 200 ms, so long monologues still split at a natural point.
* Utterances with under 200 ms of voiced audio (cough, tap, door) are dropped
  here and never reach the speech recogniser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config import AudioSettings, SegmenterSettings


class _State(Enum):
    SILENCE = "silence"
    ONSET = "onset"
    SPEECH = "speech"


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start_ms: int  # stream time of the first sample (pre-roll included)
    end_ms: int
    pcm: np.ndarray  # int16 mono PCM, ready to wrap in a WAV header
    voiced_ms: int  # how much of it the VAD classed as speech
    reason: str  # "pause" | "max-length" | "flush"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class SpeechSegmenter:
    def __init__(self, settings: SegmenterSettings, audio: AudioSettings) -> None:
        def frames(ms: int) -> int:
            return max(1, math.ceil(ms / audio.frame_ms))

        self._frame_ms = audio.frame_ms
        self._pre_roll = frames(settings.pre_roll_ms)
        self._onset = frames(settings.onset_ms)
        self._hangover = frames(settings.hangover_ms)
        self._post_roll = min(frames(settings.post_roll_ms), self._hangover)
        self._soft_max = frames(settings.soft_max_ms)
        self._soft_hangover = frames(settings.soft_hangover_ms)
        self._hard_max = frames(settings.hard_max_ms)
        self._min_voiced = frames(settings.min_voiced_ms)
        self.discarded = 0  # utterances dropped as too short (for stats)
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._state is _State.SPEECH

    # ── the utterance in progress (used for live previews) ──────────────────

    @property
    def open_start_ms(self) -> int:
        return self._start_frame * self._frame_ms

    @property
    def open_length_ms(self) -> int:
        return len(self._frames) * self._frame_ms

    @property
    def voiced_ms(self) -> int:
        return self._voiced * self._frame_ms

    @property
    def trailing_silence_ms(self) -> int:
        return self._trailing_silence * self._frame_ms

    def snapshot(self) -> np.ndarray:
        """PCM of the utterance so far (a copy; the segmenter keeps going)."""
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype="<i2")

    def reset(self) -> None:
        self._state = _State.SILENCE
        self._pre_roll_frames: list[np.ndarray] = []
        self._onset_frames: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._start_frame = 0
        self._voiced = 0
        self._trailing_silence = 0
        self._frame_index = 0

    def push(self, frame: np.ndarray, is_speech: bool) -> SpeechSegment | None:
        """Feed one frame and its VAD decision. Returns an utterance when one ends."""
        if self._state is _State.SILENCE:
            segment = self._on_silence(frame, is_speech)
        elif self._state is _State.ONSET:
            segment = self._on_onset(frame, is_speech)
        else:
            segment = self._on_speech(frame, is_speech)
        self._frame_index += 1
        return segment

    def flush(self) -> SpeechSegment | None:
        """Close the utterance in progress right now (the mic was stopped)."""
        segment = self._close("flush") if self._state is _State.SPEECH else None
        self._onset_frames = []
        self._state = _State.SILENCE
        return segment

    # ── state handlers ───────────────────────────────────────────────────────

    def _on_silence(self, frame: np.ndarray, is_speech: bool) -> None:
        if not is_speech:
            self._remember(frame)
            return None
        self._onset_frames = [frame]
        self._state = _State.ONSET
        if len(self._onset_frames) >= self._onset:
            self._open()
        return None

    def _on_onset(self, frame: np.ndarray, is_speech: bool) -> None:
        if not is_speech:  # false start (click, tap): fold everything back into the pre-roll
            for pending in (*self._onset_frames, frame):
                self._remember(pending)
            self._onset_frames = []
            self._state = _State.SILENCE
            return None
        self._onset_frames.append(frame)
        if len(self._onset_frames) >= self._onset:
            self._open()
        return None

    def _on_speech(self, frame: np.ndarray, is_speech: bool) -> SpeechSegment | None:
        self._frames.append(frame)
        if is_speech:
            self._voiced += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        length = len(self._frames)
        required_silence = self._soft_hangover if length >= self._soft_max else self._hangover
        if self._trailing_silence >= required_silence:
            return self._close("pause")
        if length >= self._hard_max:
            return self._close("max-length")
        return None

    # ── transitions ──────────────────────────────────────────────────────────

    def _open(self) -> None:
        self._frames = [*self._pre_roll_frames, *self._onset_frames]
        self._start_frame = self._frame_index - len(self._frames) + 1
        self._voiced = len(self._onset_frames)
        self._trailing_silence = 0
        self._pre_roll_frames = []
        self._onset_frames = []
        self._state = _State.SPEECH

    def _close(self, reason: str) -> SpeechSegment | None:
        dropped = max(0, self._trailing_silence - self._post_roll)
        kept = self._frames[: len(self._frames) - dropped]
        trimmed = self._frames[len(self._frames) - dropped :] if dropped else []

        start_ms = self._start_frame * self._frame_ms
        segment = SpeechSegment(
            start_ms=start_ms,
            end_ms=start_ms + len(kept) * self._frame_ms,
            pcm=np.concatenate(kept),
            voiced_ms=self._voiced * self._frame_ms,
            reason=reason,
        )
        voiced_enough = self._voiced >= self._min_voiced

        self._frames = []
        self._voiced = 0
        self._trailing_silence = 0
        self._pre_roll_frames = trimmed[-self._pre_roll :]  # unsent silence seeds the next pre-roll
        self._state = _State.SILENCE

        if not voiced_enough:
            self.discarded += 1
            return None
        return segment

    def _remember(self, frame: np.ndarray) -> None:
        self._pre_roll_frames.append(frame)
        if len(self._pre_roll_frames) > self._pre_roll:
            self._pre_roll_frames.pop(0)
