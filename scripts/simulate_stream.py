"""Stream a WAV file through the live pipeline exactly like the browser does.

Sends 20 ms PCM16 chunks over the WebSocket in real time, prints every
server event, then reports how much audio the VAD filtered out and the
transcription latency. Optionally asks for the AI summary afterwards.

    uv run python scripts/simulate_stream.py consultation.wav
    uv run python scripts/simulate_stream.py consultation.wav --speed 2 --summarize
    uv run python scripts/simulate_stream.py hindi.wav --language hi
    uv run python scripts/simulate_stream.py consultation.wav --url wss://<your-app>.vercel.app/ws/transcribe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.wav import decode_wav, resample

SAMPLE_RATE = 16_000
CHUNK_MS = 20  # same as the browser


@dataclass
class Report:
    started: float = field(default_factory=time.perf_counter)
    texts: dict[int, str] = field(default_factory=dict)
    latencies: list[int] = field(default_factory=list)
    partials: int = 0
    first_text: dict[int, str] = field(default_factory=dict)  # first text committed on screen per chunk
    corrected: int = 0  # chunks whose fast on-screen text the accurate final changed
    sent_ms: int = 0
    failed: int = 0
    speech_frames: int = 0
    frames: int = 0

    def stamp(self) -> str:
        return f"[{time.perf_counter() - self.started:6.2f}s]"


async def receive(ws: websockets.ClientConnection, report: Report) -> None:
    async for raw in ws:
        event = json.loads(raw)
        kind = event["type"]
        if kind == "vad":
            report.frames += len(event["frames"])
            report.speech_frames += sum(frame[2] for frame in event["frames"])
        elif kind == "chunk" and event["status"] == "pending" and "startMs" in event:
            print(f"{report.stamp()} chunk {event['id']:>2} speech started at {event['startMs'] / 1000:.1f} s")
        elif kind == "chunk" and event["status"] == "partial":
            report.partials += 1
            if event.get("stable"):
                report.first_text.setdefault(event["id"], event["text"])
            kind_label = "text  " if event.get("stable") else "live …"
            print(f"{report.stamp()} chunk {event['id']:>2} {kind_label} {event['text']}")
        elif kind == "chunk" and event["status"] == "done":
            report.sent_ms += event["endMs"] - event["startMs"]
            report.texts[event["id"]] = event["text"]
            report.corrected += report.first_text.get(event["id"], event["text"]) != event["text"]
            report.latencies.append(event["latencyMs"])
            lang = (event.get("language") or "?").upper()
            timing = f"{event['latencyMs']:4d} ms  {lang:>2}"
            print(f"{report.stamp()} chunk {event['id']:>2} done    {timing} │ {event['text']}")
        elif kind == "chunk" and event["status"] == "failed":
            report.failed += 1
            print(f"{report.stamp()} chunk {event['id']:>2} FAILED  {event['error']}")
        elif kind == "error":
            print(f"{report.stamp()} server error: {event['message']}")
        elif kind == "stopped":
            return


async def stream(path: Path, url: str, speed: float, language: str) -> Report:
    pcm, rate = decode_wav(path.read_bytes())
    pcm = resample(pcm, rate, SAMPLE_RATE)
    step = SAMPLE_RATE * CHUNK_MS // 1000
    report = Report()
    print(f"Streaming {path.name} ({len(pcm) / SAMPLE_RATE:.1f} s) to {url} at {speed}× real time\n")

    async with websockets.connect(url, max_size=2**22) as ws:
        await ws.send(json.dumps({"type": "start", "language": language}))
        receiver = asyncio.create_task(receive(ws, report))
        began = time.perf_counter()
        for index, start in enumerate(range(0, len(pcm), step)):
            await ws.send(pcm[start : start + step].tobytes())
            # Pace against the clock (not sleep-by-sleep) so the stream never drifts behind real time.
            due = began + (index + 1) * CHUNK_MS / 1000 / speed
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(receiver, timeout=60)

    total_ms = len(pcm) * 1000 // SAMPLE_RATE
    print("\n── Report ─────────────────────────────────────────────")
    print(f"audio captured     : {total_ms / 1000:.1f} s")
    print(f"sent to recogniser : {report.sent_ms / 1000:.1f} s ({100 * report.sent_ms / max(total_ms, 1):.0f}%)")
    print(f"filtered by VAD    : {(total_ms - report.sent_ms) / 1000:.1f} s of silence/noise")
    print(f"utterances         : {len(report.texts)} transcribed, {report.failed} failed")
    print(f"live text updates  : {report.partials} (text shown while still speaking)")
    print(f"fast text corrected: {report.corrected} of {len(report.texts)} by the accurate final")
    if report.latencies:
        ordered = sorted(report.latencies)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        avg, p50 = statistics.mean(ordered), statistics.median(ordered)
        print(f"last word → text   : avg {avg:.0f} ms · p50 {p50:.0f} ms · p95 {p95} ms  (on screen)")
    return report


async def summarize(url: str, transcript: str) -> None:
    http_url = url.replace("wss://", "https://").replace("ws://", "http://").replace("/ws/transcribe", "/api/summarize")
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(http_url, json={"transcript": transcript})
    print(f"\n── Summary ({(time.perf_counter() - started) * 1000:.0f} ms, HTTP {response.status_code}) ──")
    print(json.dumps(response.json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", type=Path, help="16-bit PCM WAV file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/transcribe")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    parser.add_argument("--language", choices=["auto", "en", "hi"], default="auto", help="spoken language")
    parser.add_argument("--summarize", action="store_true", help="also request the AI summary")
    args = parser.parse_args()

    report = asyncio.run(stream(args.wav, args.url, args.speed, args.language))
    transcript = " ".join(report.texts[i] for i in sorted(report.texts) if report.texts[i])
    print(f"\n── Transcript ─────────────────────────────────────────\n{transcript}")
    if args.summarize:
        asyncio.run(summarize(args.url, transcript))


if __name__ == "__main__":
    main()
