from app.stt.hallucination import HallucinationFilter
from app.stt.language import LanguagePolicy, language_code
from app.stt.service import CODE_MIXED_STYLE, MAX_PROMPT_CHARS, TranscriptionService, build_prompt


def segment(text, no_speech=0.01, logprob=-0.2, compression=1.2):
    return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob, "compression_ratio": compression}


class TestHallucinationFilter:
    clean = HallucinationFilter().clean

    def test_keeps_confident_speech(self):
        assert self.clean({"segments": [segment(" Fever for"), segment(" three days.")]}) == "Fever for three days."

    def test_drops_segments_whisper_thinks_are_silence(self):
        result = {"segments": [segment(" Take rest."), segment(" Thank you.", no_speech=0.8, logprob=-1.5)]}
        assert self.clean(result) == "Take rest."

    def test_drops_repetition_loops(self):
        assert self.clean({"segments": [segment(" the the the the the", compression=3.1)]}) == ""

    def test_drops_stock_phrases_only_when_whisper_had_doubts(self):
        assert self.clean({"segments": [segment(" Thank you.", no_speech=0.3)]}) == ""
        assert self.clean({"segments": [segment(" Thank you.", no_speech=0.01)]}) == "Thank you."

    def test_drops_an_echo_of_the_prompt_on_near_silence(self):
        echo = {"segments": [segment(" मुझे fever और cough है", no_speech=0.4)]}
        assert self.clean(echo, prompt=CODE_MIXED_STYLE) == ""

    def test_keeps_real_speech_that_happens_to_match_the_prompt(self):
        said = {"segments": [segment(" मुझे fever और cough है", no_speech=0.02)]}
        assert self.clean(said, prompt=CODE_MIXED_STYLE) == "मुझे fever और cough है"

    def test_falls_back_to_plain_text(self):
        assert self.clean({"text": "  hello   doctor "}) == "hello doctor"


def test_prompt_is_empty_without_context():
    assert build_prompt("   ") == ""


def test_prompt_keeps_the_tail_and_starts_on_a_word():
    prompt = build_prompt("word " * 400)
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert prompt.startswith("word")


class FakeGroq:
    """Answers by the requested language (auto passes run concurrently, so order isn't fixed)."""

    def __init__(self, by_language, fail=()):
        self.calls, self._by_language, self._fail = [], by_language, set(fail)

    async def transcribe(self, wav, **kwargs):
        self.calls.append(kwargs)
        language = kwargs["language"]
        if language in self._fail:
            raise RuntimeError("upstream error")
        return self._by_language[language]

    def languages(self):
        return sorted(str(call["language"]) for call in self.calls)


def whisper(text, language, duration=2.5):
    return {"text": text, "language": language, "duration": duration, "segments": [segment(f" {text}")]}


def service(groq, expected=("en", "hi")):
    return TranscriptionService(
        groq,
        model="whisper-large-v3",
        fast_model="whisper-large-v3-turbo",
        timeout_s=5,
        expected_languages=expected,
    )


async def test_fixed_english_sends_context_as_prompt():
    groq = FakeGroq({"en": whisper("Paracetamol 650 mg.", "English")})
    result = await service(groq).transcribe(b"wav", context="Patient has fever.", language="en")
    assert result.text == "Paracetamol 650 mg."
    assert result.language == "en"
    assert result.audio_ms == 2500
    assert groq.calls == [{**groq.calls[0], "language": "en", "prompt": "Patient has fever."}]


async def test_fixed_hindi_uses_the_code_mixed_style_prompt():
    groq = FakeGroq({"hi": whisper("मुझे fever है", "Hindi")})
    await service(groq).transcribe(b"wav", context="Patient has fever.", language="hi")
    assert groq.calls[0]["prompt"] == CODE_MIXED_STYLE


async def test_auto_english_keeps_the_unprompted_pass():
    groq = FakeGroq({None: whisper("Your throat looks red.", "English"), "hi": whisper("थ्रोट रेड", "Hindi")})
    result = await service(groq).transcribe(b"wav", context="earlier text")
    assert result.text == "Your throat looks red."
    assert result.language == "en"
    plain = next(call for call in groq.calls if call["language"] is None)
    assert plain["prompt"] is None  # no prompt → no paraphrasing, no language bias


async def test_auto_hindi_keeps_the_parallel_code_mixed_pass():
    groq = FakeGroq({None: whisper("मैं मेटफॉर्मिन लेता हूँ", "Hindi"), "hi": whisper("मैं metformin लेता हूँ", "Hindi")})
    result = await service(groq).transcribe(b"wav")
    assert result.text == "मैं metformin लेता हूँ"
    assert result.language == "hi"
    assert groq.languages() == ["None", "hi"]  # both passes ran, at the same time


async def test_hindi_detected_as_urdu_uses_the_hindi_pass():
    groq = FakeGroq({None: whisper("مجھے بخار ہے", "Urdu"), "hi": whisper("मुझे बुखार है", "Hindi")})
    result = await service(groq).transcribe(b"wav")
    assert result.text == "मुझे बुखार है"
    assert result.language == "hi"


async def test_implausible_language_is_redone_in_english():
    groq = FakeGroq({None: whisper("Ja.", "Welsh"), "hi": whisper("जा", "Hindi"), "en": whisper("Yes.", "English")})
    result = await service(groq).transcribe(b"wav")
    assert result.text == "Yes."
    assert result.language == "en"


async def test_failed_hindi_pass_falls_back_to_the_plain_pass():
    groq = FakeGroq({None: whisper("मुझे बुखार है", "Hindi")}, fail={"hi"})
    result = await service(groq).transcribe(b"wav")
    assert result.text == "मुझे बुखार है"


async def test_english_only_clinics_make_a_single_request():
    groq = FakeGroq({None: whisper("Hello doctor.", "English")})
    await service(groq, expected=("en",)).transcribe(b"wav")
    assert groq.languages() == ["None"]


def test_language_policy():
    policy = LanguagePolicy(("en", "hi"))
    assert policy.retry_language("English") is None
    assert policy.retry_language("hindi") is None
    assert policy.retry_language(None) is None
    assert policy.retry_language("Urdu") == "hi"
    assert policy.retry_language("Welsh") == "en"
    assert LanguagePolicy(("en",)).retry_language("Urdu") == "en"
    assert policy.resolve("Hindi") == "hi"
    assert policy.resolve("Urdu") == "hi"
    assert policy.resolve(None) is None
    assert language_code("Gujarati") == "gu"


async def test_preview_is_a_single_request_on_the_fast_model():
    groq = FakeGroq({None: whisper("Good morning", "English")})
    result = await service(groq).preview(b"wav")
    assert result.text == "Good morning"
    assert groq.languages() == ["None"]
    assert groq.calls[0]["model"] == "whisper-large-v3-turbo"


async def test_preview_never_sends_a_prompt_even_in_hindi():
    groq = FakeGroq({"hi": whisper("मुझे बुखार", "Hindi")})
    await service(groq).preview(b"wav", language="hi")
    assert groq.calls[0]["prompt"] is None  # no prompt words leaking into cut-off audio
    assert groq.calls[0]["may_wait"] is False  # best-effort: never queue behind rate limits


async def test_final_transcription_uses_the_accurate_model():
    groq = FakeGroq({"en": whisper("Good morning", "English")})
    await service(groq).transcribe(b"wav", language="en")
    assert groq.calls[0]["model"] == "whisper-large-v3"


async def test_preview_in_an_implausible_language_shows_nothing():
    groq = FakeGroq({None: whisper("はい", "Japanese")})
    assert (await service(groq).preview(b"wav")).text == ""
