"""Voice Activity Detection — decides, for every 20 ms frame, "speech or not".

Algorithm (energy VAD with an adaptive noise floor):

1. Measure the frame's band energy and spectral flatness (features.py).
2. Track the **noise floor** with *minimum statistics*: the floor follows the
   quietest frame of the last 3 s — falling fast, rising slowly. When a fan
   switches on, the floor rises within seconds and the fan stops counting as
   speech; speech itself never becomes the floor because it always has short
   quiet gaps between syllables.
3. SNR = energy − floor. Decide with **hysteresis**: starting speech needs
   SNR ≥ 10 dB *and* a non-flat (voice-like) spectrum; staying in speech needs
   only SNR ≥ 6 dB. Two thresholds stop the decision flickering at word edges.
   Flatness is averaged over the last few frames (exponential moving average)
   so a single noise frame that happens to look peaky cannot open speech.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config import AudioSettings, VadSettings

from .features import extract_features


@dataclass(frozen=True, slots=True)
class VadDecision:
    energy_db: float
    noise_floor_db: float
    threshold_db: float  # energy a frame must reach to count as speech right now
    snr_db: float
    flatness: float
    is_speech: bool


class EnergyVad:
    def __init__(self, settings: VadSettings, audio: AudioSettings) -> None:
        self._settings = settings
        self._sample_rate = audio.sample_rate
        self._calibration_frames = max(1, settings.calibration_ms // audio.frame_ms)
        self._history: deque[float] = deque(maxlen=max(1, settings.noise_window_ms // audio.frame_ms))
        self._frames_seen = 0
        self._noise_floor_db = settings.min_noise_floor_db
        self._flatness = 1.0  # smoothed spectral flatness; starts "noise-like"
        self._active = False

    @property
    def is_calibrating(self) -> bool:
        return self._frames_seen < self._calibration_frames

    def process(self, frame: np.ndarray) -> VadDecision:
        """frame: float32 samples in [-1, 1] (one 20 ms frame)."""
        s = self._settings
        features = extract_features(frame, self._sample_rate, s.band_low_hz, s.band_high_hz)
        self._update_noise_floor(features.energy_db)
        self._flatness += (features.flatness - self._flatness) * s.flatness_smoothing
        self._frames_seen += 1

        snr_db = features.energy_db - self._noise_floor_db
        self._active = not self.is_calibrating and self._classify(features.energy_db, snr_db, self._flatness)

        gate_db = s.offset_snr_db if self._active else s.onset_snr_db
        return VadDecision(
            energy_db=features.energy_db,
            noise_floor_db=self._noise_floor_db,
            threshold_db=max(self._noise_floor_db + gate_db, s.min_speech_db),
            snr_db=snr_db,
            flatness=self._flatness,
            is_speech=self._active,
        )

    def reset(self) -> None:
        self._history.clear()
        self._frames_seen = 0
        self._noise_floor_db = self._settings.min_noise_floor_db
        self._flatness = 1.0
        self._active = False

    def _classify(self, energy_db: float, snr_db: float, flatness: float) -> bool:
        s = self._settings
        if energy_db < s.min_speech_db:
            return False
        if self._active:  # already in speech: easier to stay
            return snr_db >= s.offset_snr_db
        return snr_db >= s.onset_snr_db and flatness <= s.max_onset_flatness

    def _update_noise_floor(self, energy_db: float) -> None:
        self._history.append(energy_db)
        target = max(min(self._history), self._settings.min_noise_floor_db)
        if self.is_calibrating:
            self._noise_floor_db = target  # still learning the room: snap to the quietest frame
            return
        rate = self._settings.noise_fall_rate if target < self._noise_floor_db else self._settings.noise_rise_rate
        self._noise_floor_db += (target - self._noise_floor_db) * rate
