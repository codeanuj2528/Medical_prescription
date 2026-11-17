import numpy as np

from app.audio.features import extract_features
from tests.signals import SR, noise, sine, voice


def features(signal):
    return extract_features(signal[:320], SR, 150, 4000)


def test_full_scale_sine_is_about_minus_3_dbfs():
    assert abs(features(sine(20, 1000, 1.0)).energy_db - (-3.0)) < 1.0


def test_digital_silence_reports_the_floor():
    assert features(np.zeros(320, dtype=np.float32)).energy_db == -100.0


def test_out_of_band_hum_barely_counts():
    hum = features(sine(20, 50, 0.5)).energy_db
    in_band = features(sine(20, 1000, 0.5)).energy_db
    assert in_band - hum > 15


def test_flatness_separates_noise_from_voice():
    assert features(noise(20, 0.3)).flatness > 0.4
    assert features(voice(20)).flatness < 0.2
