"""Runtime settings shared by the API and evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_TOKEN_PRICES_USD_PER_MILLION = {
    # Google Gemini Developer API standard paid tier, checked 2026-09-16.
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEXT_TO_SQL_", env_file=".env", extra="ignore"
    )

    database_path: Path = Path("data/demo.sqlite")
    llm_provider: Literal["fake", "gemini"] = "fake"
    model_name: str = "gemini-3.1-flash-lite"
    api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    max_repairs: int = Field(default=2, ge=0, le=2)
    max_rows: int = Field(default=500, ge=1, le=10_000)
    query_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    schema_top_k: int = Field(default=4, ge=1, le=20)
    schema_min_score: float = Field(default=0.2, ge=0)
    input_token_price_per_million: float | None = Field(default=None, ge=0)
    output_token_price_per_million: float | None = Field(default=None, ge=0)

    @property
    def resolved_api_key(self) -> str | None:
        return self.api_key or self.gemini_api_key

    @property
    def token_prices(self) -> tuple[float, float]:
        known_input, known_output = MODEL_TOKEN_PRICES_USD_PER_MILLION.get(
            self.model_name, (0.0, 0.0)
        )
        return (
            self.input_token_price_per_million
            if self.input_token_price_per_million is not None
            else known_input,
            self.output_token_price_per_million
            if self.output_token_price_per_million is not None
            else known_output,
        )

    @model_validator(mode="after")
    def require_provider_credentials(self) -> Settings:
        if self.llm_provider == "gemini" and not self.resolved_api_key:
            raise ValueError(
                "TEXT_TO_SQL_API_KEY (or legacy TEXT_TO_SQL_GEMINI_API_KEY) "
                "is required for the Gemini provider"
            )
        known_price = self.model_name in MODEL_TOKEN_PRICES_USD_PER_MILLION
        explicit_prices = (
            self.input_token_price_per_million is not None
            and self.output_token_price_per_million is not None
        )
        if self.llm_provider == "gemini" and not known_price and not explicit_prices:
            raise ValueError(
                "unknown model pricing: set TEXT_TO_SQL_INPUT_TOKEN_PRICE_PER_MILLION "
                "and TEXT_TO_SQL_OUTPUT_TOKEN_PRICE_PER_MILLION"
            )
        return self
