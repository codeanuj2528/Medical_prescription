# MediScribe — live medical transcription with Voice Activity Detection

Captures live audio from the microphone, detects speech with a **Voice Activity
Detector written from scratch in Python/NumPy**, and sends **only the speech**
to the recogniser (Groq Whisper). Silence and background noise never leave the
server. Text appears **live while you are still speaking**, utterances are
transcribed **in parallel**, English / Hindi / Hinglish are handled
automatically, and an LLM turns the transcript into a structured clinical
summary.

```
Browser (thin client)                     Python / FastAPI (all the logic)                         Groq
─────────────────────                     ────────────────────────────────                         ────
mic ─▶ AudioWorklet ─▶ 16 kHz PCM16 ──WS──▶ 20 ms frames ─▶ EnergyVad ─▶ SpeechSegmenter ─▶ utterance
       (resample, 60 ms chunks)            │                  │              │                 │
                                           │      VAD telemetry (waveform)   │ silence/noise   ▼
UI ◀───── live text + final text ◀────────┴──────────────────────────────── │ dropped here   WAV ─▶ Whisper turbo + large-v3
                                                                             │                (parallel, key pool)
"Process with AI" ──HTTP──▶ /api/summarize ─▶ 2 extractors in parallel ───────────────────────────▶ gpt-oss-120b
```

## Measured results

Two synthetic doctor–patient consultations (room noise, keyboard clicks in the
pauses), streamed in real time over the WebSocket with
`scripts/simulate_stream.py`, both in **Auto** language mode:

| Metric | English (71 s) | Hinglish (39 s) |
|---|---|---|
| Utterances transcribed | 13 / 13, word-exact | 7 / 7, Hindi in Devanagari, English terms kept in English |
| Audio filtered by the VAD (never sent) | 11.7 s | 6.9 s |
| **Last spoken word → its text on screen** (8+ keys) | **≈ 305 ms** (p95 ≈ 345 ms) | **≈ 305 ms** (p95 ≈ 335 ms) |
| Same, **one free Groq key**, run back-to-back | ≈ 620 ms (p95 ≈ 950 ms), 0 failures | ≈ 400 ms (p95 ≈ 660 ms), 0 failures |
| Grey live text while still speaking | updates every ≈ 0.7 s | same |
| AI summary (2 parallel extractors) | ≈ 1.1–1.4 s | ≈ 1.2–1.5 s |

Example Hinglish output: *"हाँ, मुझे sugar है। मैं metformin 500 लेता हूँ। और
penicillin से allergy है।"* — the summary then lists Metformin 500 mg (current
medication, frequency left blank because it wasn't said) and a penicillin allergy.

About 60 ms of that is noticing the pause and ~240 ms is Groq's own Whisper
time — the practical floor for a non-streaming model. With a single free key
the request budget (below) trims live work so nothing ever fails. 112 tests pass
(`pytest`), `ruff` clean.

## How the VAD works

All in [`app/audio/`](app/audio). Every 20 ms frame (320 samples at 16 kHz):

1. **Features** ([`features.py`](app/audio/features.py)) — FFT of the
   Hann-windowed frame, keeping only the 150 Hz – 4 kHz speech band:
   * **band energy (dBFS)** — loudness where speech lives; mains hum, rumble
     and hiss outside the band don't count.
   * **spectral flatness** (0–1) — geometric/arithmetic mean of the power
     spectrum. Voiced speech is peaky (≈ 0.01), fans/rustle/white noise are flat
     (≈ 0.56). Measured on real speech: median 0.01 vs noise 0.57.
2. **Adaptive noise floor** ([`vad.py`](app/audio/vad.py)) — *minimum
   statistics*: the floor tracks the quietest frame of the last 3 s, falling
   fast and rising slowly. A fan switching on is absorbed within seconds;
   speech never becomes the floor because it always dips between syllables.
3. **Decision with hysteresis** — *start* speech needs SNR ≥ 10 dB **and** a
   voice-like (non-flat, 3-frame-averaged) spectrum; *stay* in speech needs
   only SNR ≥ 6 dB. Two thresholds stop flicker at word edges.
4. **Segmentation** ([`segmenter.py`](app/audio/segmenter.py)) — a state
   machine `SILENCE → ONSET → SPEECH`:
   * **onset 60 ms** — 3 speech frames in a row are needed (rejects clicks/taps);
   * **pre-roll 300 ms** — audio before the onset is kept, so soft first
     consonants ("s" in "six") aren't clipped;
   * **hangover 600 ms** — shorter, natural mid-sentence pauses stay inside the
     utterance (fragments are recognised much worse than whole sentences);
   * **soft max 10 s / hard max 15 s** — long monologues split at the next short
     pause, and never exceed 15 s (bounds request latency);
   * **min voiced 200 ms** — coughs and door knocks are dropped and never sent.

A second safety net, [`hallucination.py`](app/stt/hallucination.py), removes
Whisper segments that Whisper itself marks as non-speech (`no_speech_prob`,
`avg_logprob`), repetition loops (`compression_ratio`), and its well-known
phantom phrases ("Thank you.") on near-silence.

Every number above is a named setting in [`app/config.py`](app/config.py).

The browser's call-style noise suppression and echo cancellation are **off**
([`MicCapture.js`](public/js/core/MicCapture.js)): they swallow soft syllables
and word endings. The VAD already gates noise and Whisper is robust to it.

## Live text — "it answers as you speak"

[`session.py`](app/realtime/session.py) shows text before an utterance is over:

Two lanes per utterance:

* **Fast lane** — `whisper-large-v3-turbo`, one prompt-free request at a time,
  at least 300 ms apart. At a **60 ms pause** the speech so far is transcribed
  and put on screen ≈ 250 ms later; during continuous speech it refreshes grey
  live text every ≈ 0.7 s.
* **Accurate lane** — `whisper-large-v3`, language-aware. At a **250 ms pause**
  the speech so far gets its final-quality transcription; it replaces the fast
  text while the speaker is still silent. If no more speech follows before the
  600 ms **hangover** ends (the hangover only decides where an utterance ends,
  so natural mid-sentence pauses don't chop sentences), it becomes the final
  text with no extra request.
* Junk guards: no grey text for under 0.5 s of speech, previews "heard" in an
  unexpected language (e.g. Japanese on half a second of Hindi) are dropped,
  stale results never overwrite newer ones or the final.

## Request budget — works even with one free Groq key

Groq's free tier allows **20 requests per minute per model per key** (measured:
20 back-to-back requests, then HTTP 429 "try again in 3s").
[`stt/budget.py`](app/stt/budget.py) mirrors that with a token bucket per model,
sized by the number of keys:

* **final text always runs**; 50 % of each budget is kept for it;
* text at pauses runs while > 50 % is left; grey previews only while > 70 % is
  left, so they are the first thing dropped; the speculative Hindi pass needs
  spare budget too;
* every 429 from Groq drains the matching bucket (a restarted or second server
  instance can't know Groq's real count otherwise);
* if the accurate final can't run, the fast text already on screen becomes the
  final — slower or less polished, never failed.

With 8+ keys nothing is skipped; with one key the app quietly does less live
work, and a final whose accurate model is rate-limited is written by the fast
model instead of waiting. `GROQ_RPM_PER_KEY` adjusts the limit for paid tiers.

## Languages — English, Hindi, Hinglish

Forcing `language="en"` turns Hindi speech into a garbled English
*translation* ("I have 3 days since I have a glass of sugar"). So the default is
**Auto**, with a selector (Auto / English / Hindi) in the UI.
[`stt/service.py`](app/stt/service.py) and [`stt/language.py`](app/stt/language.py):

* **Speculative parallel decoding.** Whisper writes English most faithfully
  with *no* prompt, and Hinglish most faithfully when told "Hindi" plus a short
  code-mixed style prompt (*"मुझे fever है… metformin 500 mg"*), which keeps
  English medical words in Latin script instead of "मेटफॉर्मिन". The language is
  only known after detection, so **both passes run at the same time** and the
  detected language picks the answer; the other is cancelled — no added latency.
* **Detection safety net.** Hindi labelled as Urdu is redone as Hindi;
  a language nobody in the clinic speaks (e.g. "Welsh" for a one-word "Yes")
  is redone in English. Expected languages: `STT_EXPECTED_LANGUAGES` (default `en,hi`).
* **Models:** final text uses `whisper-large-v3` (more accurate on Hindi and
  Indian English); the text shown first uses `whisper-large-v3-turbo` (faster).
* **No examples to copy.** The style prompt contains no drugs, doses or numbers:
  an earlier prompt with "metformin 500 mg" made Whisper add "mg" to speech that
  never said it.

## Parallelism and latency

* **Parallel transcription** — each utterance becomes an `asyncio` task
  ([`session.py`](app/realtime/session.py)); a semaphore allows 4 at once. A
  long utterance never blocks the short one after it. Results carry their id
  and the browser slots them back into speech order.
* **Context without blocking** — in English sessions each request sends the
  transcript that has *finished so far* as Whisper's `prompt`, so names and
  spellings stay consistent across independently transcribed utterances.
* **Key pool** ([`key_pool.py`](app/stt/key_pool.py)) — Groq keys are rotated
  round-robin; a key that gets HTTP 429 sits out for its `Retry-After` while the
  others keep serving (immediate failover, no sleeping).
* **Parallel summary** ([`extractors.py`](app/summary/extractors.py)) — the
  summary is split into two halves requested concurrently with strict JSON
  schemas. Generation time scales with output tokens, so this roughly halves
  latency. If one half fails, the other is still returned (`partial: true`).
* **Wire format** — 16 kHz mono PCM16 (Whisper's native rate): 32 KB/s, no
  server-side resampling; one pooled HTTP/1.1 keep-alive client to Groq.
* **Function region** `bom1` (Mumbai) in `vercel.json` — closest to users in India.

## Project structure

```
app/
  config.py               all settings (audio, VAD, segmenter, models, limits)
  main.py                 FastAPI app factory (entry point for uvicorn and Vercel)
  audio/
    features.py           band energy + spectral flatness (NumPy FFT)
    vad.py                EnergyVad — adaptive noise floor + hysteresis
    segmenter.py          SpeechSegmenter — utterance state machine
    wav.py                PCM ↔ WAV, resampling
  stt/
    groq_client.py        async Groq client: key failover, retries, timeouts
    key_pool.py           round-robin keys with 429 cooldowns
    hallucination.py      drops Whisper's invented text (and echoes of its prompt)
    language.py           language codes + detection safety net
    budget.py             per-model request budget (token bucket synced with Groq 429s)
    service.py            TranscriptionService: speculative decoding, previews, prompts
  summary/
    schemas.py            Pydantic models (MedicalSummary, request/response)
    extractors.py         the two parallel extraction prompts + strict schemas
    service.py            SummaryService (asyncio.gather)
  realtime/
    protocol.py           WebSocket commands (Pydantic discriminated union)
    session.py            StreamingSession — VAD → utterances, live text, parallel finals
    errors.py             user-facing error wording
  api/
    container.py          composition root (dependency injection)
    realtime.py           WS /ws/transcribe
    rest.py               GET /api/health, POST /api/summarize
    errors.py             uniform JSON errors
    security.py           origin check, client id
    rate_limit.py         sliding-window limiter
public/                   static frontend (no build step)
  index.html, css/app.css
  worklets/pcm-capture.worklet.js   mic → 16 kHz PCM16 on the audio thread
  js/ScribeApp.js         UI controller (state + wiring only)
  js/core/                MicCapture, StreamClient (reconnect + replay), Recording, api
  js/ui/                  Waveform, TranscriptView, SummaryView, Toaster
scripts/simulate_stream.py  stream any WAV through the live pipeline + report
tests/                    112 pytest tests (VAD, segmenter, session, languages, API, …)
```

The browser does **no** audio analysis — it only captures, converts and
displays. That keeps the whole pipeline in Python, testable without a browser.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env.local      # then put your Groq key(s) in it
uv run uvicorn app.main:app --reload --reload-dir app --port 8000
```

Open <http://localhost:8000>, click **Start Mic**, and talk. (Microphone access
needs `localhost` or HTTPS.)

## Test

```bash
uv run pytest                    # 112 tests, ~2 s, no network needed
uv run ruff check . && uv run ruff format --check .
```

End-to-end against a running server with any 16-bit WAV file:

```bash
uv run python scripts/simulate_stream.py consultation.wav --summarize
uv run python scripts/simulate_stream.py hinglish.wav --language auto
```

It streams the file in real time exactly like the browser does and prints the
live text and final text of every utterance, the % of audio the VAD filtered
out, last-word → final-text latency (avg/p50/p95), the transcript and the AI summary.

## Deploy to Vercel

FastAPI is zero-config on Vercel: `app/main.py` exposes `app`, `public/` is
served from the CDN, and WebSockets run on Fluid Compute.

```bash
npm i -g vercel
vercel login
vercel link
vercel env add GROQ_API_KEYS production     # one key works; several (gsk_a,gsk_b,…) = faster live text
vercel deploy --prod
```

`vercel.json` pins the region, sets `maxDuration: 300` for the function and
adds security headers. `uvicorn[standard]` is a runtime dependency on purpose:
Vercel's Python runtime needs a WebSocket implementation installed.

**Dropped connections are expected on Vercel and handled.** A WebSocket ends at
the function's max duration (300 s on Hobby), and in testing Vercel's proxy also
dropped Python WebSocket connections without a close frame (more often at 50
messages/s than at 10 — hence 100 ms audio messages), with no error in the
function logs. The browser reconnects immediately, starts a fresh session with
the same chunk numbering and transcript context, and **replays the audio the old
connection never finished** (from the start of the first unfinished utterance,
held in the in-memory recording). Verified by restarting the server mid-stream:
every utterance transcribed once, in order, nothing lost.

## API

**WebSocket `/ws/transcribe`** — browser sends binary PCM16 (16 kHz mono) and:

```json
{"type": "start", "chunkOffset": 0, "context": "", "language": "auto"}
{"type": "stop"}
{"type": "retry", "id": 7}
```

Server sends `ready`, `vad` (per-frame `[energyDb, thresholdDb, isSpeech,
inUtterance]`), `chunk` (`pending` when speech starts → `partial` live text →
`done` with `text`, `language`, `latencyMs` — or `failed`), `stopped`, `error`.

**`POST /api/summarize`** `{"transcript": "..."}` →
`{"summary": {...}, "model": "...", "latency_ms": 1241, "partial": false}`

**`GET /api/health`** → models in use and number of keys (never the keys).
Interactive docs: `/api/docs`.

## Design decisions and trade-offs

* **Custom energy + flatness VAD instead of a neural VAD (Silero/WebRTC).** Zero
  heavy dependencies, every decision explainable and unit-tested, runs in
  microseconds per frame. A neural VAD would separate speech from *other
  speech-like* noise (TV, music) better; the `EnergyVad` class could be swapped
  for one behind the same `process(frame) → VadDecision` interface.
* **VAD on the server, not in the browser.** Raw audio goes to *our* server;
  only speech goes to the recogniser. All logic stays in one tested Python
  codebase, and the same code powers the simulator script.
* **WebSocket instead of uploading files.** Audio streams continuously in 100 ms
  messages (VAD telemetry comes back batched ~10× a second), so the server can
  preview, transcribe at pauses and finalise without waiting.
* **Speed costs requests.** With enough keys, live text uses ≈ 1–2 Whisper
  requests per second of speech; the request budget scales that down to fit
  whatever quota exists. Groq's free tier also caps ≈ 2,000 requests per key
  per day. Tune `LiveSettings` in `config.py` to trade speed for requests.
* **Safety.** Origin check on the API and WebSocket (other sites can't spend the
  Groq quota through a visitor's browser), per-IP rate limits, message size
  limits, uniform errors that never leak internals, audio kept only in memory.
* **Known limits.** Rate limits are per instance (a global quota needs Redis).
  There is no speaker diarisation, so the LLM infers who said what. AI output
  must be reviewed by a clinician.
