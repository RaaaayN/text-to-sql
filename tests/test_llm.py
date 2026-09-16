from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from text_to_sql.llm import (
    FakeLLMClient,
    GeminiLLMClient,
    LLMClient,
    LLMError,
    LLMOutputError,
    ScriptedLLMClient,
    ScriptExhaustedError,
    TokenPricing,
    TokenUsage,
    extract_sql,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"sql": " SELECT 1 "}, "SELECT 1"),
        ('{"sql":"SELECT name FROM users"}', "SELECT name FROM users"),
        ('```json\n{"sql":"SELECT 2"}\n```', "SELECT 2"),
    ],
)
def test_extract_sql_accepts_structured_outputs(payload: object, expected: str) -> None:
    assert extract_sql(payload) == expected


@pytest.mark.parametrize(
    "payload",
    ["SELECT 1", "{}", '{"sql": 1}', {"sql": ""}, [], None, "```sql\nSELECT 1\n```"],
)
def test_extract_sql_rejects_unstructured_or_invalid_outputs(payload: object) -> None:
    with pytest.raises(LLMOutputError):
        extract_sql(payload)


def test_scripted_client_is_deterministic_and_records_prompts() -> None:
    client = ScriptedLLMClient([{"sql": "SELECT 1"}, '{"sql":"SELECT 2"}'])

    assert isinstance(client, LLMClient)
    assert client.generate_sql("first").sql == "SELECT 1"
    assert client.generate_sql("repair").sql == "SELECT 2"
    assert client.prompts == ["first", "repair"]
    with pytest.raises(ScriptExhaustedError):
        client.generate_sql("one too many")


def test_scripted_client_can_raise_a_scripted_error() -> None:
    client = ScriptedLLMClient([TimeoutError("offline")])

    with pytest.raises(TimeoutError, match="offline"):
        client.generate_sql("prompt")


def test_fake_client_has_a_safe_default_response() -> None:
    result = FakeLLMClient().generate_sql("ignored")

    assert result.sql == "SELECT 1"
    assert result.model == "fake"


def test_token_cost_uses_provider_counts() -> None:
    usage = TokenUsage(input_tokens=1_250, output_tokens=250)
    pricing = TokenPricing(input_per_million=2.0, output_per_million=8.0)

    assert usage.total_tokens == 1_500
    assert pricing.cost_usd(usage) == pytest.approx(0.0045)


@pytest.mark.parametrize(
    "value",
    [
        lambda: TokenUsage(input_tokens=-1),
        lambda: TokenPricing(output_per_million=-0.1),
    ],
)
def test_usage_and_prices_cannot_be_negative(value: object) -> None:
    with pytest.raises(ValueError):
        value()  # type: ignore[operator]


@dataclass
class _Response:
    parsed: dict[str, str] | None
    text: str
    usage_metadata: object


class _Models:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_gemini_requests_json_and_captures_usage_cost_and_latency() -> None:
    response = _Response(
        parsed={"sql": "SELECT COUNT(*) FROM users"},
        text='{"sql":"SELECT COUNT(*) FROM users"}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
        ),
    )
    models = _Models(response)
    ticks = iter([10.0, 10.25])
    client = GeminiLLMClient(
        client=SimpleNamespace(models=models),
        model="gemini-test",
        pricing=TokenPricing(input_per_million=1, output_per_million=5),
        clock=lambda: next(ticks),
    )

    result = client.generate_sql("schema and question")

    assert result.sql == "SELECT COUNT(*) FROM users"
    assert result.usage == TokenUsage(input_tokens=100, output_tokens=25)
    assert result.cost_usd == pytest.approx(0.000225)
    assert result.latency_seconds == pytest.approx(0.25)
    assert models.calls[0]["model"] == "gemini-test"
    assert models.calls[0]["config"] == {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "OBJECT",
            "properties": {"sql": {"type": "STRING"}},
            "required": ["sql"],
        },
    }


def test_gemini_falls_back_to_json_text_and_camel_case_usage() -> None:
    response = {
        "text": '{"sql":"SELECT 42"}',
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
    }
    client = GeminiLLMClient(client=SimpleNamespace(models=_Models(response)))

    result = client.generate_sql("prompt")

    assert result.sql == "SELECT 42"
    assert result.usage.total_tokens == 10


def test_gemini_wraps_provider_failures_without_leaking_provider_type() -> None:
    client = GeminiLLMClient(
        client=SimpleNamespace(models=_Models(error=ConnectionError("network down")))
    )

    with pytest.raises(LLMError, match="Gemini request failed: network down"):
        client.generate_sql("prompt")


def test_gemini_requires_credentials_without_injected_client() -> None:
    with pytest.raises(ValueError, match="api_key"):
        GeminiLLMClient()
