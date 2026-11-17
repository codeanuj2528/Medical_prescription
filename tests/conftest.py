from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(groq_keys=("test-key-1", "test-key-2"))
