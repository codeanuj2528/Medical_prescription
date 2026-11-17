from app.audio.vad import EnergyVad
from app.config import AudioSettings, VadSettings
from tests.signals import concat, frames, mix, noise, silence, voice

FRAMES_PER_SECOND = 50


def decide(signal):
    vad = EnergyVad(VadSettings(), AudioSettings())
    return [vad.process(frame) for frame in frames(signal)]


def speech_ratio(decisions):
    return sum(d.is_speech for d in decisions) / max(len(decisions), 1)


def at(seconds: float) -> int:
    return int(seconds * FRAMES_PER_SECOND)


def test_never_flags_digital_silence():
    assert speech_ratio(decide(silence(2000))) == 0


def test_ignores_steady_background_noise():
    assert speech_ratio(decide(noise(3000, 0.03))) < 0.02


def test_detects_speech_over_noise_and_releases_after_it():
    signal = mix(concat(silence(1000), voice(1500), silence(1500)), noise(4000, 0.005))
    decisions = decide(signal)
    assert speech_ratio(decisions[: at(1.0)]) == 0
    assert speech_ratio(decisions[at(1.1) : at(2.5)]) > 0.9
    assert speech_ratio(decisions[at(2.8) :]) == 0


def test_adapts_when_the_room_gets_louder():
    # A fan switching on (+20 dB) must be absorbed into the noise floor within seconds.
    decisions = decide(concat(noise(2000, 0.003, seed=1), noise(6000, 0.03, seed=2)))
    assert speech_ratio(decisions[-at(2) :]) == 0


def test_noise_burst_with_a_flat_spectrum_cannot_start_speech():
    decisions = decide(concat(noise(1000, 0.002), noise(600, 0.3, seed=3)))
    assert speech_ratio(decisions[at(1.0) :]) == 0


def test_threshold_sits_above_the_noise_floor():
    last = decide(noise(1000, 0.01))[-1]
    assert last.threshold_db > last.noise_floor_db
    assert not last.is_speech
