"""Runtime settings shared by the API and evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEXT_TO_SQL_", env_file=".env", extra="ignore"
    )

    database_path: Path = Path("data/demo.sqlite")
    llm_provider: Literal["fake", "gemini"] = "fake"
    model_name: str = "gemini-2.5-flash"
    gemini_api_key: str | None = Field(default=None, repr=False)
    max_repairs: int = Field(default=2, ge=0, le=2)
    max_rows: int = Field(default=500, ge=1, le=10_000)
    query_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    schema_top_k: int = Field(default=4, ge=1, le=20)
    schema_min_score: float = Field(default=0.2, ge=0)
    input_token_price_per_million: float = Field(default=0.0, ge=0)
    output_token_price_per_million: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def require_provider_credentials(self) -> Settings:
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("TEXT_TO_SQL_GEMINI_API_KEY is required for the Gemini provider")
        return self
