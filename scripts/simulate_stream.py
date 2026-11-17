"""Stream a WAV file through the live pipeline exactly like the browser does.

Sends 100 ms PCM16 messages over the WebSocket in real time, prints every
server event, then reports how much audio the VAD filtered out and the
latency. Like the browser, it survives dropped connections: it reconnects at
once and replays the audio whose transcription never came back, so a drop
costs a moment of delay, not words — and, like the browser, it pings every
second and treats a server that has said nothing for 3 s as a dropped
connection. Optionally asks for the AI summary.

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
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.wav import decode_wav, resample

SAMPLE_RATE = 16_000
SAMPLES_PER_MS = SAMPLE_RATE // 1000
CHUNK_MS = 100  # same as the browser
MAX_REPLAY_MS = 30_000  # same as the browser
MAX_RECONNECTS = 10
HEARTBEAT_S = 1.0  # same as the browser
SILENT_SERVER_S = 3.0


class Dropped(Exception):
    """The connection closed before the server said "stopped"."""


@dataclass
class Chunk:
    id: int
    start_ms: int  # position in the whole recording (not in one connection)
    end_ms: int | None = None
    status: str = "pending"
    text: str = ""
    first_text: str | None = None  # first text committed on screen
    latency_ms: int | None = None
    language: str | None = None


@dataclass
class Report:
    started: float = field(default_factory=time.perf_counter)
    chunks: dict[int, Chunk] = field(default_factory=dict)
    partials: int = 0
    failed: int = 0
    reconnects: int = 0
    replayed_ms: int = 0

    def stamp(self) -> str:
        return f"[{time.perf_counter() - self.started:6.2f}s]"

    def done(self) -> list[Chunk]:
        return sorted((c for c in self.chunks.values() if c.status == "done"), key=lambda c: c.start_ms)


class LiveStream:
    """One recording streamed in real time over as many connections as it takes."""

    def __init__(self, pcm: np.ndarray, url: str, speed: float, language: str, chunk_ms: int) -> None:
        self.pcm = pcm
        self.url = url
        self.speed = speed
        self.language = language
        self.step = SAMPLES_PER_MS * chunk_ms
        self.report = Report()
        self.base_ms = 0  # recording position where the current connection's audio starts
        self.last_id = 0
        self.began = 0.0
        self.last_heard = 0.0

    async def run(self) -> Report:
        self.began = time.perf_counter()
        sent = 0  # samples sent so far (the "mic" position)
        replay_from_ms: int | None = None
        while True:
            try:
                # ping_interval=None: browsers never send WebSocket pings, so neither does this script.
                async with websockets.connect(self.url, max_size=2**22, ping_interval=None, close_timeout=0.5) as ws:
                    self.base_ms = replay_from_ms or 0
                    start = {"type": "start", "language": self.language, "chunkOffset": self.last_id}
                    await ws.send(json.dumps(start))
                    receiver = asyncio.create_task(self.watch(ws))
                    try:
                        if replay_from_ms is not None:  # audio the dropped connection never finished
                            sent = self.captured()
                            await self.send_range(ws, replay_from_ms * SAMPLES_PER_MS, sent)
                        sent = await self.stream_live(ws, receiver, sent)
                        await ws.send(json.dumps({"type": "stop"}))
                        await asyncio.wait_for(receiver, timeout=60)
                        return self.report
                    finally:
                        receiver.cancel()
            except (Dropped, websockets.ConnectionClosed, OSError) as error:
                if self.report.reconnects >= MAX_RECONNECTS:
                    raise
                self.report.reconnects += 1
                replay_from_ms = self.prepare_replay()
                self.report.replayed_ms += max(0, self.captured() // SAMPLES_PER_MS - replay_from_ms)
                print(
                    f"{self.report.stamp()} connection dropped ({type(error).__name__}) — reconnecting, "
                    f"replaying from {replay_from_ms / 1000:.1f} s"
                )

    def captured(self) -> int:
        """Samples the mic has recorded by now (real time × speed)."""
        elapsed = (time.perf_counter() - self.began) * self.speed
        return min(len(self.pcm), int(elapsed * SAMPLE_RATE))

    async def send_range(self, ws: websockets.ClientConnection, start: int, end: int) -> None:
        for offset in range(start, end, self.step):
            await ws.send(self.pcm[offset : min(offset + self.step, end)].tobytes())

    async def stream_live(self, ws: websockets.ClientConnection, receiver: asyncio.Task, sent: int) -> int:
        while sent < len(self.pcm):
            if receiver.done():
                receiver.result()  # re-raises the drop
                raise Dropped
            await ws.send(self.pcm[sent : sent + self.step].tobytes())
            sent += self.step
            # Pace against the clock (not sleep-by-sleep) so the stream never drifts behind real time.
            due = self.began + sent / SAMPLE_RATE / self.speed
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
        return sent

    def prepare_replay(self) -> int:
        """Same rule as the browser: replay from the first unfinished utterance
        (or the end of the last finished one) and forget everything the replay
        will transcribe again."""
        chunks = list(self.report.chunks.values())
        unfinished = [c for c in chunks if c.status in ("pending", "partial")]
        ends = [c.end_ms for c in chunks if c.status == "done" and c.end_ms is not None]
        from_ms = min(c.start_ms for c in unfinished) if unfinished else max([self.base_ms, *ends])
        from_ms = max(from_ms, self.captured() // SAMPLES_PER_MS - MAX_REPLAY_MS, 0)
        for chunk in chunks:
            if chunk in unfinished or chunk.start_ms >= from_ms:
                del self.report.chunks[chunk.id]
        return from_ms

    async def watch(self, ws: websockets.ClientConnection) -> None:
        """Receive until "stopped"; meanwhile ping every second and give up on a silent server."""
        self.last_heard = time.perf_counter()
        receiving = asyncio.ensure_future(self.receive(ws))
        try:
            while not receiving.done():
                await asyncio.wait({receiving}, timeout=HEARTBEAT_S)
                if receiving.done():
                    break
                if time.perf_counter() - self.last_heard > SILENT_SERVER_S:
                    raise Dropped("the server went silent")
                await ws.send(json.dumps({"type": "ping"}))
            receiving.result()
        finally:
            receiving.cancel()

    async def receive(self, ws: websockets.ClientConnection) -> None:
        report = self.report
        async for raw in ws:
            self.last_heard = time.perf_counter()
            event = json.loads(raw)
            if event["type"] == "stopped":
                return
            if event["type"] == "error":
                print(f"{report.stamp()} server error: {event['message']}")
            if event["type"] == "chunk":
                self.on_chunk(event)
        raise Dropped

    def on_chunk(self, event: dict) -> None:
        report, status = self.report, event["status"]
        self.last_id = max(self.last_id, event["id"])
        chunk = report.chunks.get(event["id"])
        if chunk is None:
            chunk = report.chunks[event["id"]] = Chunk(event["id"], self.base_ms + event.get("startMs", 0))
            print(f"{report.stamp()} chunk {chunk.id:>2} speech started at {chunk.start_ms / 1000:.1f} s")
        chunk.status = status
        if status == "partial":
            report.partials += 1
            chunk.text = event["text"]
            if event.get("stable") and chunk.first_text is None:
                chunk.first_text = event["text"]
            label = "text  " if event.get("stable") else "live …"
            print(f"{report.stamp()} chunk {chunk.id:>2} {label} {event['text']}")
        elif status == "done":
            chunk.text, chunk.latency_ms = event["text"], event["latencyMs"]
            chunk.end_ms = self.base_ms + event["endMs"]
            chunk.language = event.get("language")
            timing = f"{chunk.latency_ms:4d} ms  {(chunk.language or '?').upper():>2}"
            print(f"{report.stamp()} chunk {chunk.id:>2} done    {timing} │ {chunk.text}")
        elif status == "failed":
            report.failed += 1
            print(f"{report.stamp()} chunk {chunk.id:>2} FAILED  {event['error']}")


def print_report(report: Report, total_ms: int) -> None:
    done = report.done()
    sent_ms = sum(c.end_ms - c.start_ms for c in done if c.end_ms is not None)
    corrected = sum(c.first_text is not None and c.first_text != c.text for c in done)
    print("\n── Report ─────────────────────────────────────────────")
    print(f"audio captured     : {total_ms / 1000:.1f} s")
    print(f"sent to recogniser : {sent_ms / 1000:.1f} s ({100 * sent_ms / max(total_ms, 1):.0f}%)")
    print(f"filtered by VAD    : {(total_ms - sent_ms) / 1000:.1f} s of silence/noise")
    print(f"utterances         : {len(done)} transcribed, {report.failed} failed")
    print(f"live text updates  : {report.partials} (text shown while still speaking)")
    print(f"fast text corrected: {corrected} of {len(done)} by the accurate final")
    print(f"dropped connections: {report.reconnects} (recovered by replaying {report.replayed_ms / 1000:.1f} s)")
    latencies = sorted(c.latency_ms for c in done if c.latency_ms is not None)
    if latencies:
        p95 = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
        avg, p50 = statistics.mean(latencies), statistics.median(latencies)
        print(f"last word → text   : avg {avg:.0f} ms · p50 {p50:.0f} ms · p95 {p95} ms  (on screen)")


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
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS, help="audio per WebSocket message (ms)")
    parser.add_argument("--summarize", action="store_true", help="also request the AI summary")
    args = parser.parse_args()

    pcm, rate = decode_wav(args.wav.read_bytes())
    pcm = resample(pcm, rate, SAMPLE_RATE)
    print(f"Streaming {args.wav.name} ({len(pcm) / SAMPLE_RATE:.1f} s) to {args.url} at {args.speed}× real time\n")
    report = asyncio.run(LiveStream(pcm, args.url, args.speed, args.language, args.chunk_ms).run())
    print_report(report, len(pcm) * 1000 // SAMPLE_RATE)

    transcript = " ".join(c.text for c in report.done() if c.text)
    print(f"\n── Transcript ─────────────────────────────────────────\n{transcript}")
    if args.summarize:
        asyncio.run(summarize(args.url, transcript))


if __name__ == "__main__":
    main()
