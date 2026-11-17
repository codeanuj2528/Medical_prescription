"""VAD + segmenter together, exactly as the streaming session uses them."""

from app.audio.segmenter import SpeechSegmenter
from app.audio.vad import EnergyVad
from app.config import AudioSettings, SegmenterSettings, VadSettings
from tests.signals import consultation, frames, noise, to_pcm16


def utterances(signal):
    vad = EnergyVad(VadSettings(), AudioSettings())
    segmenter = SpeechSegmenter(SegmenterSettings(), AudioSettings())
    found = []
    for frame in frames(signal):
        segment = segmenter.push(to_pcm16(frame), vad.process(frame).is_speech)
        if segment:
            found.append(segment)
    if last := segmenter.flush():
        found.append(last)
    return found


def test_finds_exactly_the_two_utterances():
    first, second = utterances(consultation())
    assert 600 <= first.start_ms <= 1000
    assert 2500 <= first.end_ms <= 2900
    assert 3300 <= second.start_ms <= 3700  # second utterance spans 3.7–4.7 s
    assert 4700 <= second.end_ms <= 5000


def test_silence_and_noise_never_reach_the_recogniser():
    signal = consultation()  # 2.5 s of speech inside 5.7 s of audio
    sent_ms = sum(s.duration_ms for s in utterances(signal))
    total_ms = len(signal) / 16
    assert sent_ms >= 2500  # all speech is sent (plus short pre/post-roll)…
    assert total_ms - sent_ms >= 2000  # …and most of the 3.2 s of silence is not


def test_noise_only_recording_sends_nothing():
    assert utterances(noise(5000, 0.02)) == []
