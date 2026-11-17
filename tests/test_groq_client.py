import httpx
import pytest

from app.stt.groq_client import GroqClient, GroqError, parse_retry_after
from app.stt.key_pool import KeyPool

BASE = "https://groq.test/openai/v1"


def client_with(handler, keys=("key-a", "key-b")):
    pool = KeyPool(keys)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqClient(http, pool, BASE), pool


async def test_rate_limited_key_fails_over_to_the_next_key():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        seen.append(key)
        if key == "key-a":
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"text": "ok"})

    client, pool = client_with(handler)
    result = await client.transcribe(b"wav", model="m", language="en", prompt=None, timeout_s=5)
    assert result == {"text": "ok"}
    assert seen == ["key-a", "key-b"]
    assert pool.acquire() == "key-b"  # key-a is still cooling down


async def test_client_errors_are_not_retried():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad audio"}})

    client, _ = client_with(handler)
    with pytest.raises(GroqError) as raised:
        await client.transcribe(b"wav", model="m", language=None, prompt=None, timeout_s=5)
    assert raised.value.status == 400
    assert raised.value.message == "bad audio"
    assert calls == 1


async def test_server_errors_are_retried():
    responses = iter([httpx.Response(503), httpx.Response(200, json={"choices": []})])
    client, _ = client_with(lambda request: next(responses))
    assert await client.chat_completion({"model": "m"}, timeout_s=5) == {"choices": []}


async def test_timeouts_become_504():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, _ = client_with(handler)
    with pytest.raises(GroqError) as raised:
        await client.chat_completion({}, timeout_s=0.1)
    assert raised.value.status == 504


async def test_transcription_request_shape():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(200, json={"text": ""})

    client, _ = client_with(handler)
    await client.transcribe(b"RIFFdata", model="whisper-large-v3-turbo", language="en", prompt="ctx", timeout_s=5)
    assert captured["url"] == f"{BASE}/audio/transcriptions"
    for part in (b"whisper-large-v3-turbo", b"verbose_json", b'name="language"', b"ctx", b"RIFFdata"):
        assert part in captured["body"]


async def test_waits_for_a_rate_limited_key_instead_of_failing():
    responses = iter(
        [
            httpx.Response(429, headers={"retry-after": "0.1"}, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json={"text": "ok"}),
        ]
    )
    client, _ = client_with(lambda request: next(responses), keys=("only-key",))
    result = await client.transcribe(b"wav", model="m", language=None, prompt=None, timeout_s=5)
    assert result == {"text": "ok"}


async def test_best_effort_requests_fail_fast_on_rate_limits():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"retry-after": "5"}, json={"error": {"message": "slow down"}})

    client, _ = client_with(handler, keys=("only-key",))
    with pytest.raises(GroqError) as raised:
        await client.transcribe(b"wav", model="m", language=None, prompt=None, timeout_s=5, may_wait=False)
    assert raised.value.status == 429
    assert calls == 1


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "2"}, 2.0),
        ({"x-ratelimit-reset-requests": "1m3s"}, 63.0),
        ({"x-ratelimit-reset-requests": "2.5s"}, 2.5),
        ({}, None),
    ],
)
def test_parse_retry_after(headers, expected):
    assert parse_retry_after(httpx.Headers(headers)) == expected
