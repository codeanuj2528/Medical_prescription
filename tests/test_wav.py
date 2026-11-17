import io
import wave

import numpy as np

from app.audio.wav import decode_wav, encode_wav, float_to_pcm16, resample
from tests.signals import sine


def test_encodes_16khz_mono_pcm16():
    data = encode_wav(np.zeros(160, dtype=np.int16), 16_000)
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16_000, 160)
    assert len(data) == 44 + 320


def test_round_trip_is_lossless():
    pcm = float_to_pcm16(sine(50, 440, 0.8))
    decoded, rate = decode_wav(encode_wav(pcm, 16_000))
    assert rate == 16_000
    assert np.array_equal(decoded, pcm)


def test_float_to_pcm16_clips():
    assert float_to_pcm16(np.array([0, 1, -1, 2, -2], dtype=np.float32)).tolist() == [0, 32767, -32768, 32767, -32768]


def test_resample_keeps_duration():
    pcm = float_to_pcm16(sine(1000, 440))
    assert len(resample(pcm, 16_000, 48_000)) == 48_000
    assert len(resample(pcm, 16_000, 16_000)) == 16_000
