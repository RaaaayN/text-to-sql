from pathlib import Path

import pytest
from pydantic import ValidationError

from text_to_sql.config import Settings


def test_defaults_are_safe_and_offline() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "fake"
    assert settings.max_repairs == 2
    assert settings.database_path == Path("data/demo.sqlite")


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(llm_provider="gemini", _env_file=None)


def test_limits_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError):
        Settings(max_repairs=3, _env_file=None)
