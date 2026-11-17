"""StreamingSession: bytes in → VAD telemetry + parallel transcription events out."""

import asyncio
from dataclasses import replace

from app.realtime.session import StreamingSession
from app.stt.groq_client import GroqError
from app.stt.service import TranscriptionResult
from tests.signals import concat, consultation, mix, noise, silence, to_pcm16, voice

CHUNK_BYTES = 960 * 2  # 60 ms, like the browser


class FakeTranscriber:
    """Final-quality calls answer "utterance N"; quick previews answer "preview N"."""

    def __init__(self, delays=(), fail_full_calls=0):
        self.contexts, self.languages, self.preview_languages = [], [], []
        self.active = self.max_active = self.full_calls = self.previews = 0
        self._delays, self._fail_full_calls = list(delays), fail_full_calls

    async def transcribe(self, wav, context="", language=None):
        self.full_calls += 1
        call = self.full_calls
        self.contexts.append(context)
        self.languages.append(language)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self._delays[call - 1] if call <= len(self._delays) else 0)
            if call <= self._fail_full_calls:
                raise GroqError(503, "down")
            return TranscriptionResult(
                text=f"utterance {call}", language=language or "en", audio_ms=1000, upstream_ms=5
            )
        finally:
            self.active -= 1

    async def preview(self, wav, language=None):
        self.previews += 1
        self.preview_languages.append(language)
        return TranscriptionResult(
            text=f"preview {self.previews}", language=language or "en", audio_ms=500, upstream_ms=1
        )


async def stream(session, signal):
    data = to_pcm16(signal).tobytes()
    for start in range(0, len(data), CHUNK_BYTES):
        await session.feed(data[start : start + CHUNK_BYTES])
    await session.finish()


def make(settings, transcriber, **kwargs):
    """Session plus the events it sent; each event is stamped with the audio time it was sent at."""
    events = []
    holder = {}

    async def send(event):
        events.append({**event, "_at_ms": holder["session"].captured_ms})
        await asyncio.sleep(0)  # a real socket send yields to the event loop

    holder["session"] = StreamingSession(transcriber, settings, send, **kwargs)
    return holder["session"], events


def chunk_events(events, status):
    return [e for e in events if e["type"] == "chunk" and e["status"] == status]


def speech(*parts):
    """Speech/silence pattern over steady room noise."""
    signal = concat(*parts)
    return mix(signal, noise(len(signal) / 16, 0.004))


async def test_turns_speech_into_transcribed_chunks(settings):
    session, events = make(settings, FakeTranscriber())
    await stream(session, consultation())

    pending, done = chunk_events(events, "pending"), chunk_events(events, "done")
    assert [e["id"] for e in pending] == [1, 2]
    assert sorted(e["text"] for e in done) == ["utterance 1", "utterance 2"]
    assert all(e["latencyMs"] >= 0 for e in done)
    assert session.speech_ms < session.captured_ms * 0.7


async def test_reports_vad_telemetry_for_every_frame(settings):
    session, events = make(settings, FakeTranscriber())
    signal = consultation()
    await stream(session, signal)
    frames = [frame for e in events if e["type"] == "vad" for frame in e["frames"]]
    assert len(frames) == len(signal) // 320
    assert any(frame[3] for frame in frames)  # some frames inside an utterance
    assert len(frames[0]) == 4


async def test_transcribes_in_parallel_and_delivers_out_of_order(settings):
    # Five utterances; the first is slow, so later ones overtake it.
    signal = mix(
        concat(silence(600), *[concat(voice(700), silence(900)) for _ in range(5)]),
        noise(600 + 5 * 1600, 0.004),
    )
    transcriber = FakeTranscriber(delays=[0.3, 0.05, 0.05, 0.05, 0.05])
    session, events = make(settings, transcriber)
    data = to_pcm16(signal).tobytes()
    for start in range(0, len(data), CHUNK_BYTES):
        await session.feed(data[start : start + CHUNK_BYTES])
    await asyncio.sleep(0.15)  # let the fast ones finish while #1 is still running
    await session.finish()

    done_order = [e["id"] for e in chunk_events(events, "done")]
    assert len(done_order) == 5
    assert done_order[0] != 1  # #1 was overtaken
    assert 1 < transcriber.max_active <= settings.stt_concurrency


async def test_each_utterance_gets_the_preceding_transcript_as_context(settings):
    transcriber = FakeTranscriber()
    session, _ = make(settings, transcriber, context="Earlier text.")
    await stream(session, consultation())
    assert transcriber.contexts[0] == "Earlier text."
    assert transcriber.contexts[1] == "Earlier text. utterance 1"


async def test_chunk_offset_continues_numbering_after_a_reconnect(settings):
    session, events = make(settings, FakeTranscriber(), chunk_offset=12)
    await stream(session, consultation())
    assert [e["id"] for e in chunk_events(events, "pending")] == [13, 14]


async def test_failed_chunk_can_be_retried(settings):
    # Both the eager pass and the fallback final fail → the chunk is marked failed.
    session, events = make(settings, FakeTranscriber(fail_full_calls=2))
    await stream(session, concat(silence(600), voice(1000), silence(1000)))
    [failed] = chunk_events(events, "failed")
    assert "unavailable" in failed["error"]

    await session.retry(failed["id"])
    await session.finish()
    assert chunk_events(events, "done")[-1]["id"] == failed["id"]


async def test_silence_only_stream_sends_nothing(settings):
    transcriber = FakeTranscriber()
    session, events = make(settings, transcriber)
    await stream(session, noise(3000, 0.01))
    assert chunk_events(events, "pending") == []
    assert transcriber.contexts == []


async def test_session_language_is_used_for_every_utterance(settings):
    transcriber = FakeTranscriber()
    session, events = make(settings, transcriber, language="hi")
    await stream(session, consultation())
    assert transcriber.languages == ["hi", "hi"]
    assert {e["language"] for e in chunk_events(events, "done")} == {"hi"}


async def test_live_text_appears_while_still_speaking(settings):
    session, events = make(settings, FakeTranscriber())
    await stream(session, speech(silence(600), voice(3000), silence(1000)))
    partials = chunk_events(events, "partial")
    assert partials, "expected live text"
    assert partials[0]["_at_ms"] < 3600  # shown before the speaker stopped (speech ends at 3.6 s)
    assert chunk_events(events, "done")[0]["text"] == "utterance 1"


async def test_final_reuses_the_eager_pass_when_speech_has_ended(settings):
    transcriber = FakeTranscriber()
    session, events = make(settings, transcriber)
    await stream(session, speech(silence(600), voice(800), silence(1000)))
    assert transcriber.full_calls == 1  # the pause-time transcription became the final: no second request
    [done] = chunk_events(events, "done")
    assert done["text"] == "utterance 1"
    assert done["startMs"] < done["endMs"]


async def test_speech_after_a_short_pause_gets_a_fresh_final(settings):
    transcriber = FakeTranscriber()
    session, events = make(settings, transcriber)
    # 300 ms pause: long enough for an eager pass, too short to end the utterance.
    await stream(session, speech(silence(600), voice(800), silence(300), voice(800), silence(1000)))
    assert [e["id"] for e in chunk_events(events, "done")] == [1]
    assert chunk_events(events, "done")[0]["text"] == "utterance 2"  # covers both halves
    assert "utterance 1" in [e["text"] for e in chunk_events(events, "partial")]


async def test_live_text_can_be_switched_off(settings):
    off = replace(settings, live=replace(settings.live, enabled=False))
    transcriber = FakeTranscriber()
    session, events = make(off, transcriber)
    await stream(session, speech(silence(600), voice(3000), silence(1000)))
    assert chunk_events(events, "partial") == []
    assert transcriber.previews == 0
    assert transcriber.full_calls == 1


async def test_short_utterances_get_no_grey_preview_but_are_committed_at_the_pause(settings):
    session, events = make(settings, FakeTranscriber())
    await stream(session, speech(silence(600), voice(400), silence(1000)))  # "Haan." — too short to preview
    partials = chunk_events(events, "partial")
    assert all(e["stable"] for e in partials)  # no grey live text for half a second of speech
    assert [e["text"] for e in partials] == ["preview 1", "utterance 1"]  # fast text, then the accurate one
    assert [e["text"] for e in chunk_events(events, "done")] == ["utterance 1"]


async def test_text_is_committed_at_the_pause_before_the_utterance_closes(settings):
    session, events = make(settings, FakeTranscriber())
    await stream(session, speech(silence(600), voice(1500), silence(1000)))
    stable = [e for e in chunk_events(events, "partial") if e["stable"]]
    [done] = chunk_events(events, "done")
    speech_end_ms, hangover_ms = 2100, settings.segmenter.hangover_ms
    assert stable[0]["_at_ms"] < speech_end_ms + hangover_ms  # on screen before the hangover even ends
    assert stable[-1]["text"] == done["text"] == "utterance 1"  # the accurate text replaced the fast one
    assert events.index(stable[0]) < events.index(done)


async def test_previews_keep_the_language_already_heard_in_the_utterance(settings):
    class HindiTranscriber(FakeTranscriber):
        async def transcribe(self, wav, context="", language=None):
            result = await super().transcribe(wav, context, language)
            return replace(result, language="hi")

    transcriber = HindiTranscriber()
    session, _ = make(settings, transcriber)
    # A pause early on (eager pass hears Hindi), then a long stretch of speech (quick previews).
    await stream(session, speech(silence(600), voice(900), silence(250), voice(3000), silence(1000)))
    assert transcriber.preview_languages and transcriber.preview_languages[-1] == "hi"
