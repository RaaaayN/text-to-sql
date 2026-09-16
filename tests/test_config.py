from pathlib import Path

import pytest
from pydantic import ValidationError

from text_to_sql.config import Settings


def test_defaults_are_safe_and_offline() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "fake"
    assert settings.model_name == "gemini-3.1-flash-lite"
    assert settings.token_prices == (0.25, 1.50)
    assert settings.max_repairs == 2
    assert settings.database_path == Path("data/demo.sqlite")


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="TEXT_TO_SQL_API_KEY"):
        Settings(llm_provider="gemini", _env_file=None)


def test_accepts_short_and_legacy_api_key_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_API_KEY", "short-name")
    assert Settings(llm_provider="gemini", _env_file=None).resolved_api_key == "short-name"
    monkeypatch.delenv("TEXT_TO_SQL_API_KEY")
    monkeypatch.setenv("TEXT_TO_SQL_GEMINI_API_KEY", "legacy-name")
    assert Settings(llm_provider="gemini", _env_file=None).resolved_api_key == "legacy-name"


def test_unknown_gemini_model_requires_explicit_prices() -> None:
    with pytest.raises(ValidationError, match="unknown model pricing"):
        Settings(
            llm_provider="gemini",
            api_key="secret",
            model_name="gemini-future",
            _env_file=None,
        )

    settings = Settings(
        llm_provider="gemini",
        api_key="secret",
        model_name="gemini-future",
        input_token_price_per_million=1,
        output_token_price_per_million=2,
        _env_file=None,
    )
    assert settings.token_prices == (1, 2)


def test_limits_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError):
        Settings(max_repairs=3, _env_file=None)
