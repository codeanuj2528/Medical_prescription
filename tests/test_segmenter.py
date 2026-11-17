"""Segmenter state machine, driven with explicit speech/silence decisions.

Uses fixed settings (not the tunable defaults). In 20 ms frames: pre-roll 15, onset 3, hangover 23,
post-roll 10, soft max 400, soft hangover 10, hard max 750, min voiced 10.
"""

import numpy as np

from app.audio.segmenter import SpeechSegmenter
from app.config import AudioSettings, SegmenterSettings

FRAME = 4  # tiny frames; every sample holds its frame index so order can be checked
SETTINGS = SegmenterSettings(
    pre_roll_ms=300,
    onset_ms=60,
    hangover_ms=460,
    post_roll_ms=200,
    soft_max_ms=8_000,
    soft_hangover_ms=200,
    hard_max_ms=15_000,
    min_voiced_ms=200,
)


def run(pattern, flush=True):
    segmenter = SpeechSegmenter(SETTINGS, AudioSettings())
    segments, index = [], 0
    for is_speech, count in pattern:
        for _ in range(count):
            segment = segmenter.push(np.full(FRAME, index, dtype=np.int16), is_speech)
            if segment:
                segments.append(segment)
            index += 1
    if flush and (segment := segmenter.flush()):
        segments.append(segment)
    return segments, segmenter


def frame_ids(segment):
    return segment.pcm[::FRAME].tolist()


def test_emits_one_utterance_with_pre_roll_and_trimmed_post_roll():
    [segment], _ = run([(False, 20), (True, 30), (False, 40)])
    assert segment.start_ms == 5 * 20  # 20 silent frames minus 15 of pre-roll
    assert segment.end_ms == (20 + 30 + 10) * 20  # speech plus 10 frames of post-roll
    assert segment.voiced_ms == 30 * 20
    assert segment.reason == "pause"
    assert frame_ids(segment)[0] == 5


def test_bridges_a_pause_shorter_than_the_hangover():
    segments, _ = run([(True, 20), (False, 15), (True, 20), (False, 40)])
    assert len(segments) == 1


def test_splits_on_a_pause_longer_than_the_hangover():
    segments, _ = run([(True, 20), (False, 30), (True, 20), (False, 40)])
    assert len(segments) == 2
    assert segments[1].start_ms >= segments[0].end_ms


def test_ignores_clicks_shorter_than_the_onset():
    segments, segmenter = run([(False, 10), (True, 2), (False, 40)])
    assert segments == []
    assert segmenter.discarded == 0


def test_discards_blips_with_too_little_voiced_audio():
    segments, segmenter = run([(False, 10), (True, 6), (False, 40)])
    assert segments == []
    assert segmenter.discarded == 1


def test_hard_max_cuts_without_losing_or_duplicating_audio():
    segments, _ = run([(True, 800)])
    assert [s.reason for s in segments] == ["max-length", "flush"]
    assert segments[0].duration_ms == 15_000
    assert [i for s in segments for i in frame_ids(s)] == list(range(800))


def test_closes_on_a_short_pause_once_past_the_soft_max():
    segments, _ = run([(True, 450), (False, 12), (True, 20), (False, 40)])
    assert len(segments) == 2  # 12 silent frames < hangover, but ≥ soft hangover
    assert segments[0].reason == "pause"


def test_flush_sends_the_utterance_in_progress():
    segments, _ = run([(False, 5), (True, 25)])
    assert [s.reason for s in segments] == ["flush"]


def test_frames_stay_contiguous_and_ordered():
    segments, _ = run([(False, 20), (True, 20), (False, 30), (True, 20), (False, 40)])
    for segment in segments:
        ids = frame_ids(segment)
        assert ids == list(range(ids[0], ids[0] + len(ids)))
