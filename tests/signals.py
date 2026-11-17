"""Deterministic synthetic signals for audio tests (16 kHz float32)."""

from __future__ import annotations

import numpy as np

SR = 16_000
FRAME = 320  # 20 ms


def _samples(ms: float) -> int:
    return round(ms * SR / 1000)


def silence(ms: float) -> np.ndarray:
    return np.zeros(_samples(ms), dtype=np.float32)


def noise(ms: float, amplitude: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, _samples(ms)).astype(np.float32)


def voice(ms: float, amplitude: float = 0.25, f0: float = 140.0) -> np.ndarray:
    """Speech-like: harmonic series on a 140 Hz pitch with a 4 Hz syllable envelope."""
    t = np.arange(_samples(ms)) / SR
    harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 11))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (amplitude * 0.35 * harmonics * envelope).astype(np.float32)


def sine(ms: float, frequency: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(_samples(ms)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def mix(*parts: np.ndarray) -> np.ndarray:
    out = np.zeros(max(len(p) for p in parts), dtype=np.float32)
    for part in parts:
        out[: len(part)] += part
    return out


def frames(signal: np.ndarray) -> list[np.ndarray]:
    return [signal[i : i + FRAME] for i in range(0, len(signal) - FRAME + 1, FRAME)]


def to_pcm16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1, 1) * 32767).astype("<i2")


def consultation() -> np.ndarray:
    """Two utterances separated by pauses, over constant room noise (5.7 s)."""
    speech = concat(silence(1000), voice(1500), silence(1200), voice(1000), silence(1000))
    return mix(speech, noise(len(speech) / 16, 0.006))
