"""FastAPI surface for the single-turn text-to-SQL service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from itertools import repeat

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from text_to_sql.config import Settings
from text_to_sql.execution import SQLiteExecutor
from text_to_sql.llm import GeminiLLMClient, ScriptedLLMClient, TokenPricing
from text_to_sql.observability import ServiceMetrics
from text_to_sql.schema import SchemaResolver, introspect_sqlite
from text_to_sql.service import TextToSQLService


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class QueryResponse(BaseModel):
    status: str
    sql: str | None
    columns: list[str]
    rows: list[list[object]]
    reason: str | None
    attempts: int
    repairs: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_latency_seconds: float
    total_latency_seconds: float
    model: str | None
    truncated: bool


def build_service(settings: Settings, metrics: ServiceMetrics) -> TextToSQLService:
    schema = introspect_sqlite(settings.database_path, sample_rows=settings.schema_sample_rows)
    input_price, output_price = settings.token_prices
    pricing = TokenPricing(input_price, output_price)
    if settings.llm_provider == "gemini":
        llm = GeminiLLMClient(
            api_key=settings.resolved_api_key,
            model=settings.model_name,
            pricing=pricing,
        )
    else:
        # Offline smoke-test provider for the bundled demo database.
        llm = ScriptedLLMClient(
            repeat({"sql": "SELECT count(*) AS product_count FROM products"}),
            model="fake",
        )
    return TextToSQLService(
        resolver=SchemaResolver(
            schema,
            top_k=settings.schema_top_k,
            min_score=settings.schema_min_score,
        ),
        llm=llm,
        executor=SQLiteExecutor(
            settings.database_path,
            timeout_seconds=settings.query_timeout_seconds,
            max_rows=settings.max_rows,
        ),
        metrics=metrics,
        max_repairs=settings.max_repairs,
        max_rows=settings.max_rows,
    )


def create_app(
    *, settings: Settings | None = None, service: TextToSQLService | None = None
) -> FastAPI:
    resolved_settings = settings or Settings()
    metrics = service.metrics if service else ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or build_service(resolved_settings, metrics)
        yield

    application = FastAPI(
        title="Text to SQL",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.metrics = metrics

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/metrics", include_in_schema=False)
    def prometheus(request: Request) -> Response:
        return Response(
            content=request.app.state.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @application.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        try:
            outcome = request.app.state.service.ask(payload.question)
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            request.app.state.metrics.requests.labels(outcome="error").inc()
            raise HTTPException(status_code=500, detail="query processing failed") from exc
        result = outcome.result
        return QueryResponse(
            status=outcome.status,
            sql=outcome.sql,
            columns=list(result.columns) if result else [],
            rows=[list(row) for row in result.rows] if result else [],
            reason=outcome.reason,
            attempts=outcome.attempts,
            repairs=outcome.repairs,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cost_usd=outcome.cost_usd,
            model_latency_seconds=outcome.model_latency_seconds,
            total_latency_seconds=outcome.total_latency_seconds,
            model=outcome.model,
            truncated=result.truncated if result else False,
        )

    return application


app = create_app()
