"""PCM ↔ WAV helpers (standard-library `wave`, no extra dependencies)."""

from __future__ import annotations

import io
import wave

import numpy as np


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0).astype("<i2")


def encode_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """int16 mono PCM → complete WAV file bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV (any channel count) → (int16 mono PCM, sample rate)."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype("<i2")
    return pcm, sample_rate


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resampler — good enough for preparing test/demo files."""
    if from_rate == to_rate:
        return pcm
    duration = len(pcm) / from_rate
    target_times = np.arange(int(duration * to_rate)) / to_rate
    source_times = np.arange(len(pcm)) / from_rate
    return np.interp(target_times, source_times, pcm.astype(np.float32)).astype("<i2")
