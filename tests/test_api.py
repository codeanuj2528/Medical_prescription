"""HTTP + WebSocket endpoints, with the Groq-backed services replaced by fakes."""

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import rest
from app.api.container import Container, get_container, get_ws_container
from app.api.rate_limit import SlidingWindowRateLimiter
from app.main import app
from app.stt.service import TranscriptionResult
from app.summary.schemas import MedicalSummary, SummarizeResponse
from tests.signals import consultation, to_pcm16


class FakeTranscription:
    def __init__(self):
        self.calls = 0
        self.languages = []

    async def transcribe(self, wav, context="", language=None, optional=False):
        self.calls += 1
        self.languages.append(language)
        return TranscriptionResult(text=f"utterance {self.calls}", language=language, audio_ms=1000, upstream_ms=3)

    async def preview(self, wav, language=None, at_pause=False):
        return TranscriptionResult(text="live preview", language=language, audio_ms=500, upstream_ms=1)


class FakeSummary:
    async def summarize(self, transcript):
        summary = MedicalSummary(chief_complaints=[f"from: {transcript[:10]}"])
        return SummarizeResponse(summary=summary, model="fake-llm", latency_ms=12, partial=False)


@pytest.fixture
def client(settings, monkeypatch):
    transcription = FakeTranscription()
    container = Container(
        settings=settings,
        http=httpx.AsyncClient(),
        transcription=transcription,
        summary=FakeSummary(),
        session_limiter=SlidingWindowRateLimiter(100, 60),
        summary_limiter=SlidingWindowRateLimiter(2, 60),
    )
    app.dependency_overrides[get_container] = lambda: container
    app.dependency_overrides[get_ws_container] = lambda: container
    monkeypatch.setattr(rest, "get_settings", lambda: settings)
    with TestClient(app) as test_client:
        test_client.transcription = transcription
        yield test_client
    app.dependency_overrides.clear()


def test_serves_the_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Live Medical Transcription" in response.text
    assert response.headers["permissions-policy"].startswith("microphone=(self)")
    assert client.get("/js/main.js").status_code == 200


def test_health_reports_configuration_without_secrets(client):
    body = client.get("/api/health").json()
    assert body == {
        "status": "ok",
        "sttModel": "whisper-large-v3",
        "sttFastModel": "whisper-large-v3-turbo",
        "sttRequestsPerMinute": 40,
        "llmModel": "openai/gpt-oss-120b",
        "groqKeys": 2,
    }


def test_summarize_returns_the_structured_summary(client):
    response = client.post("/api/summarize", json={"transcript": "Patient has fever for three days."})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-llm"
    assert body["summary"]["chief_complaints"] == ["from: Patient ha"]
    assert set(body["summary"]) >= {"patient", "medications", "follow_up", "clinical_impression"}


def test_summarize_validates_input(client):
    response = client.post("/api/summarize", json={"transcript": "too short"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_summarize_rejects_foreign_origins(client):
    response = client.post(
        "/api/summarize",
        json={"transcript": "Patient has fever for three days."},
        headers={"origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_summarize_is_rate_limited(client):
    body = {"transcript": "Patient has fever for three days."}
    assert [client.post("/api/summarize", json=body).status_code for _ in range(3)] == [200, 200, 429]


def test_websocket_streams_audio_and_returns_transcript(client):
    data = to_pcm16(consultation()).tobytes()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json() == {"type": "ready"}
        for start in range(0, len(data), 1920):
            ws.send_bytes(data[start : start + 1920])
        ws.send_json({"type": "stop"})

        events = []
        while (event := ws.receive_json())["type"] != "stopped":
            events.append(event)

    done = [e for e in events if e["type"] == "chunk" and e["status"] == "done"]
    assert sorted(e["text"] for e in done) == ["utterance 1", "utterance 2"]
    assert any(e["type"] == "vad" for e in events)


def stream_consultation(client, language):
    data = to_pcm16(consultation()).tobytes()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "language": language})
        ws.receive_json()
        for start in range(0, len(data), 1920):
            ws.send_bytes(data[start : start + 1920])
        ws.send_json({"type": "stop"})
        while ws.receive_json()["type"] != "stopped":
            pass
    return client.transcription.languages


def test_websocket_start_selects_the_language(client):
    assert stream_consultation(client, "hi") == ["hi", "hi"]


def test_websocket_auto_language_means_detect(client):
    assert stream_consultation(client, "auto") == [None, None]


def test_websocket_rejects_oversized_audio_messages(client):
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(b"\0" * 70_000)
        assert ws.receive_json() == {"type": "error", "message": "Audio message too large."}


def test_websocket_rejects_unknown_commands(client):
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "launch-rockets"})
        assert ws.receive_json() == {"type": "error", "message": "Unknown command."}


def test_websocket_rejects_foreign_origins(client):
    connect = client.websocket_connect("/ws/transcribe", headers={"origin": "https://evil.example"})
    with pytest.raises(WebSocketDisconnect) as closed, connect as ws:
        ws.receive_json()
    assert closed.value.code == 1008
