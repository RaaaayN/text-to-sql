"""Small, provider-neutral LLM boundary for SQL generation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Base error raised by an LLM adapter."""


class LLMOutputError(LLMError):
    """Raised when the provider did not return the required structured output."""


class ScriptExhaustedError(LLMError):
    """Raised when a deterministic client receives more calls than responses."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Prices in US dollars per one million tokens."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("token prices cannot be negative")

    def cost_usd(self, usage: TokenUsage) -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class LLMResult:
    sql: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    raw_text: str = ""

    def __post_init__(self) -> None:
        if not self.sql.strip():
            raise ValueError("sql cannot be empty")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")


@runtime_checkable
class LLMClient(Protocol):
    """The only operation the generation pipeline needs from a provider."""

    def generate_sql(self, prompt: str) -> LLMResult:
        """Generate one SQL candidate from a fully rendered prompt."""


def extract_sql(payload: object) -> str:
    """Extract ``sql`` from a structured provider response.

    Mapping-like parsed responses are preferred.  JSON strings are accepted to
    support SDK versions that expose structured output through ``response.text``.
    Markdown fences are tolerated only when their contents are still valid JSON.
    """

    if not isinstance(payload, str | Mapping) and hasattr(payload, "sql"):
        payload = {"sql": payload.sql}

    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n")
            if first_newline == -1:
                raise LLMOutputError("structured response is an invalid code fence")
            text = text[first_newline + 1 : -3].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMOutputError("model response must be a JSON object with a sql field") from exc

    if not isinstance(payload, Mapping):
        raise LLMOutputError("model response must be a JSON object with a sql field")
    sql = payload.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise LLMOutputError("model response field 'sql' must be a non-empty string")
    return sql.strip()


ScriptItem = LLMResult | Mapping[str, object] | str | Exception


class ScriptedLLMClient:
    """Deterministic test double returning a finite sequence of responses."""

    def __init__(
        self,
        responses: Iterable[ScriptItem],
        *,
        model: str = "scripted",
        usage: TokenUsage | None = None,
        pricing: TokenPricing | None = None,
    ) -> None:
        self._responses = iter(responses)
        self.model = model
        self.usage = usage or TokenUsage()
        self.pricing = pricing or TokenPricing()
        self.prompts: list[str] = []

    def generate_sql(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        try:
            response = next(self._responses)
        except StopIteration as exc:
            raise ScriptExhaustedError("scripted LLM has no response left") from exc
        if isinstance(response, Exception):
            raise response
        if isinstance(response, LLMResult):
            return response

        sql = extract_sql(response)
        raw_text = response if isinstance(response, str) else json.dumps(response)
        return LLMResult(
            sql=sql,
            model=self.model,
            usage=self.usage,
            cost_usd=self.pricing.cost_usd(self.usage),
            raw_text=raw_text,
        )


class FakeLLMClient(ScriptedLLMClient):
    """Convenient one-response deterministic client for demos and smoke tests."""

    def __init__(
        self,
        sql: str = "SELECT 1",
        *,
        usage: TokenUsage | None = None,
        pricing: TokenPricing | None = None,
    ) -> None:
        super().__init__([{"sql": sql}], model="fake", usage=usage, pricing=pricing)


def _read_field(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _usage_from_response(response: object) -> TokenUsage:
    metadata = _read_field(response, "usage_metadata", "usageMetadata")
    if metadata is None:
        return TokenUsage()
    input_tokens = _read_field(
        metadata,
        "prompt_token_count",
        "promptTokenCount",
        "input_tokens",
        default=0,
    )
    output_tokens = _read_field(
        metadata,
        "candidates_token_count",
        "candidatesTokenCount",
        "output_tokens",
        default=0,
    )
    return TokenUsage(input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0))


class GeminiLLMClient:
    """Gemini adapter with a lazy optional dependency on ``google-genai``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        pricing: TokenPricing | None = None,
        client: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Gemini client is injected")
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise LLMError(
                    "Gemini support requires the optional dependency: "
                    "install text-to-sql[gemini]"
                ) from exc
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.pricing = pricing or TokenPricing()
        self._clock = clock

    def generate_sql(self, prompt: str) -> LLMResult:
        started = self._clock()
        try:
            models = self._client.models
            response = models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "OBJECT",
                        "properties": {"sql": {"type": "STRING"}},
                        "required": ["sql"],
                    },
                },
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        latency = self._clock() - started
        parsed = _read_field(response, "parsed")
        raw_text = str(_read_field(response, "text", default="") or "")
        sql = extract_sql(parsed if parsed is not None else raw_text)
        usage = _usage_from_response(response)
        return LLMResult(
            sql=sql,
            model=self.model,
            usage=usage,
            latency_seconds=latency,
            cost_usd=self.pricing.cost_usd(usage),
            raw_text=raw_text,
        )
