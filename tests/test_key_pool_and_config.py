import pytest

from app.api.rate_limit import SlidingWindowRateLimiter
from app.config import ConfigError, Settings, collect_groq_keys
from app.stt.key_pool import KeyPool


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_key_pool_round_robin():
    pool = KeyPool(["a", "b", "c"])
    assert [pool.acquire() for _ in range(4)] == ["a", "b", "c", "a"]


def test_key_pool_skips_cooling_keys_and_recovers():
    clock = FakeClock()
    pool = KeyPool(["a", "b"], clock=clock)
    pool.mark_rate_limited("a", 5)
    assert [pool.acquire() for _ in range(3)] == ["b", "b", "b"]
    clock.now += 6
    assert {pool.acquire(), pool.acquire()} == {"a", "b"}


def test_key_pool_reports_how_long_until_a_key_is_free():
    clock = FakeClock()
    pool = KeyPool(["a", "b"], clock=clock)
    assert pool.wait_time() == 0
    pool.mark_rate_limited("a", 10)
    assert pool.wait_time() == 0  # "b" is still free
    pool.mark_rate_limited("b", 4)
    assert pool.wait_time() == pytest.approx(4)


def test_key_pool_falls_back_to_the_key_that_recovers_first():
    clock = FakeClock()
    pool = KeyPool(["a", "b"], clock=clock)
    pool.mark_rate_limited("a", 10)
    pool.mark_rate_limited("b", 3)
    assert pool.acquire() == "b"


def test_collect_groq_keys_unquotes_and_dedupes():
    env = {
        "GROQ_API_KEYS": "k1, k2",
        "GROQ_API_KEY": '"k3"',
        "GROQ_API_KEY_1": "k1",
        "GROQ_API_KEY_9": "'k4'",
        "GROQ_MODEL": "not-a-key",
    }
    assert collect_groq_keys(env) == ["k1", "k2", "k3", "k4"]


def test_settings_require_a_key_and_support_auto_language():
    with pytest.raises(ConfigError):
        Settings.from_env({})
    defaults = Settings.from_env({"GROQ_API_KEY": "k"})
    assert defaults.stt_language is None  # auto-detect
    assert defaults.stt_model == "whisper-large-v3"
    assert defaults.stt_fast_model == "whisper-large-v3-turbo"
    assert defaults.stt_expected_languages == ("en", "hi")
    custom = Settings.from_env({"GROQ_API_KEY": "k", "STT_LANGUAGE": "EN", "STT_EXPECTED_LANGUAGES": "en, gu"})
    assert custom.stt_language == "en"
    assert custom.stt_expected_languages == ("en", "gu")


def test_rate_limiter_blocks_then_slides():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, 60, clock=clock)
    assert limiter.check("ip") == 0
    assert limiter.check("ip") == 0
    assert limiter.check("ip") == pytest.approx(60)
    assert limiter.check("other-ip") == 0
    clock.now += 61
    assert limiter.check("ip") == 0
