import pytest

from app.stt.budget import RequestBudget, RequestBudgets
from app.stt.service import BudgetExhausted, TranscriptionService
from tests.test_stt_service import FakeGroq, whisper


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_optional_requests_stop_at_the_reserve():
    budget = RequestBudget(per_minute=4, clock=FakeClock())
    assert budget.try_take(keep=1.2)  # 3 left
    assert budget.try_take(keep=1.2)  # 2 left
    assert not budget.try_take(keep=1.2)  # would leave 1 < reserve


def test_required_requests_always_run_and_the_bucket_refills():
    clock = FakeClock()
    budget = RequestBudget(per_minute=60, clock=clock)  # one per second
    for _ in range(60):
        budget.take()
    budget.take()  # into debt: a final is never refused
    assert budget.available == pytest.approx(-1)
    clock.now += 3
    assert budget.available == pytest.approx(2)
    clock.now += 600
    assert budget.available == 60  # capped at one minute's worth


def limited_service(groq, per_minute):
    return TranscriptionService(
        groq,
        model="whisper-large-v3",
        fast_model="whisper-large-v3-turbo",
        timeout_s=5,
        budgets=RequestBudgets(per_minute, keys=1),
    )


async def test_live_previews_are_skipped_but_finals_still_run_when_the_budget_is_short():
    groq = FakeGroq({None: whisper("Good morning", "English"), "en": whisper("Good morning", "English")})
    service = limited_service(groq, per_minute=4)  # pause reserve 2, grey-preview reserve 2.8
    await service.preview(b"wav", at_pause=True)
    await service.preview(b"wav", at_pause=True)
    with pytest.raises(BudgetExhausted):
        await service.preview(b"wav", at_pause=True)
    for _ in range(5):  # finals ignore the budget
        assert (await service.transcribe(b"wav", language="en")).text == "Good morning"


async def test_optional_accurate_pass_is_skipped_when_short():
    groq = FakeGroq({None: whisper("Hello", "English")})
    service = limited_service(groq, per_minute=1)
    with pytest.raises(BudgetExhausted):
        await service.transcribe(b"wav", optional=True)
    assert groq.calls == []


async def test_speculative_hindi_pass_needs_spare_budget():
    groq = FakeGroq({None: whisper("मुझे बुखार है", "Hindi"), "hi": whisper("मुझे fever है", "Hindi")})
    service = limited_service(groq, per_minute=2)  # after the plain pass, nothing spare (reserve 1)
    result = await service.transcribe(b"wav")
    assert groq.languages() == ["None"]  # no second request
    assert result.text == "मुझे बुखार है"
    assert result.language == "hi"


async def test_grey_previews_stop_before_pause_text_does():
    groq = FakeGroq({None: whisper("Good morning", "English")})
    service = limited_service(groq, per_minute=10)  # grey reserve 7, pause reserve 5
    for _ in range(3):
        await service.preview(b"wav")  # 10 → 7 left
    with pytest.raises(BudgetExhausted):
        await service.preview(b"wav")  # would leave 6 < 7: grey preview refused
    await service.preview(b"wav", at_pause=True)  # text at a pause still allowed


def test_a_groq_429_drains_the_budget_by_one_keys_share():
    clock = FakeClock()
    one_key = RequestBudgets(20, keys=1, clock=clock)
    one_key.rate_limited("whisper-large-v3")
    assert one_key.for_model("whisper-large-v3").available == 0  # optional work stops at once
    assert one_key.for_model("whisper-large-v3-turbo").available == 20  # other models unaffected

    ten_keys = RequestBudgets(20, keys=10, clock=clock)
    ten_keys.rate_limited("whisper-large-v3")
    assert ten_keys.for_model("whisper-large-v3").available == 180  # only that key's share is gone


async def test_client_reports_429s_to_the_budget():
    import httpx

    from app.stt.groq_client import GroqClient
    from app.stt.key_pool import KeyPool

    responses = iter([httpx.Response(429, headers={"retry-after": "0.05"}), httpx.Response(200, json={"text": "ok"})])
    budgets = RequestBudgets(20, keys=1)
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses)))
    client = GroqClient(http, KeyPool(["k"]), "https://groq.test", budgets=budgets)
    await client.transcribe(b"wav", model="whisper-large-v3", language=None, prompt=None, timeout_s=5)
    assert budgets.for_model("whisper-large-v3").available < 1
