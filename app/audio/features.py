"""Per-frame acoustic features used by the VAD.

Each 20 ms frame is reduced to two numbers:

* **band energy (dBFS)** — how loud the frame is *inside the speech band*
  (150 Hz – 4 kHz). Mains hum, desk rumble and high-frequency hiss fall
  outside the band and don't count.
* **spectral flatness** (0 … 1) — how noise-like the spectrum is. Voiced
  speech is peaky (harmonics + formants → close to 0); fans, rustle and
  white noise are flat (→ around 0.5 or more).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_POWER_EPSILON = 1e-10  # -100 dBFS: the value reported for digital silence


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    energy_db: float
    flatness: float


@lru_cache(maxsize=8)
def _hann_window(length: int) -> np.ndarray:
    return np.hanning(length).astype(np.float32)


@lru_cache(maxsize=8)
def _band_bins(length: int, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Boolean mask selecting the rFFT bins that fall inside [low_hz, high_hz]."""
    freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    return (freqs >= low_hz) & (freqs <= high_hz)


def power_to_db(power: float) -> float:
    return float(10.0 * np.log10(power + _POWER_EPSILON))


def band_power_spectrum(frame: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Power of each FFT bin inside the band, scaled so that a full-scale sine ≈ 0.5 (-3 dBFS)."""
    window = _hann_window(len(frame))
    spectrum = np.fft.rfft(frame * window)
    power = (np.abs(spectrum) ** 2) * (2.0 / (len(frame) * float(np.sum(window**2))))
    return power[_band_bins(len(frame), sample_rate, low_hz, high_hz)]


def spectral_flatness(band_power: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum (Wiener entropy)."""
    if band_power.size == 0:
        return 1.0
    safe = band_power + _POWER_EPSILON
    return float(np.exp(np.mean(np.log(safe))) / np.mean(safe))


def extract_features(frame: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> FrameFeatures:
    """frame: float32 samples in [-1, 1]."""
    band_power = band_power_spectrum(frame, sample_rate, low_hz, high_hz)
    return FrameFeatures(energy_db=power_to_db(float(np.sum(band_power))), flatness=spectral_flatness(band_power))
