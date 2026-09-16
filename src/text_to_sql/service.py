"""Single-turn text-to-SQL orchestration with a bounded repair loop."""

from __future__ import annotations

import time
from dataclasses import dataclass

from text_to_sql.execution import QueryExecutionError, QueryResult, SQLiteExecutor
from text_to_sql.llm import LLMClient, LLMError, LLMResult
from text_to_sql.observability import ServiceMetrics
from text_to_sql.prompts import generation_prompt, repair_prompt
from text_to_sql.schema import SchemaResolver, render_schema_prompt
from text_to_sql.sql_guard import SQLValidationError, validate_sql


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    status: str
    sql: str | None
    result: QueryResult | None
    reason: str | None
    attempts: int
    repairs: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_latency_seconds: float
    total_latency_seconds: float
    model: str | None


class TextToSQLService:
    def __init__(
        self,
        *,
        resolver: SchemaResolver,
        llm: LLMClient,
        executor: SQLiteExecutor,
        metrics: ServiceMetrics,
        max_repairs: int = 2,
        max_rows: int = 500,
    ) -> None:
        if not 0 <= max_repairs <= 2:
            raise ValueError("max_repairs must be between zero and two")
        self.resolver = resolver
        self.llm = llm
        self.executor = executor
        self.metrics = metrics
        self.max_repairs = max_repairs
        self.max_rows = max_rows

    def ask(self, question: str, *, evidence: str = "") -> QueryOutcome:
        started = time.perf_counter()
        resolved = self.resolver.resolve(question)
        if resolved.refused:
            return self._refusal(
                started=started,
                reason=resolved.reason or "schema does not cover the question",
            )

        rendered_schema = render_schema_prompt(resolved)
        prompt = generation_prompt(question=question, schema=rendered_schema, evidence=evidence)
        usage = _UsageAccumulator()
        failed_sql = ""
        last_error = ""

        for repair_count in range(self.max_repairs + 1):
            if repair_count:
                prompt = repair_prompt(
                    question=question,
                    schema=rendered_schema,
                    failed_sql=failed_sql,
                    database_error=last_error,
                    attempt=repair_count,
                )
                self.metrics.repairs.inc()
            try:
                generated = self.llm.generate_sql(prompt)
            except LLMError as exc:
                return self._refusal(
                    started=started,
                    reason=f"model error: {exc}",
                    attempts=repair_count + 1,
                    repairs=repair_count,
                    usage=usage,
                )

            usage.add(generated)
            self.metrics.observe_usage(generated.usage, generated.cost_usd)
            failed_sql = generated.sql
            try:
                validated = validate_sql(generated.sql, limit=self.max_rows)
            except SQLValidationError as exc:
                return self._refusal(
                    started=started,
                    reason=f"unsafe SQL refused: {exc}",
                    sql=generated.sql,
                    attempts=repair_count + 1,
                    repairs=repair_count,
                    usage=usage,
                )

            try:
                result = self.executor.execute(validated.bounded)
            except QueryExecutionError as exc:
                last_error = str(exc)
                if repair_count < self.max_repairs:
                    continue
                return self._refusal(
                    started=started,
                    reason=f"repair budget exhausted: {last_error}",
                    sql=validated.normalized,
                    attempts=repair_count + 1,
                    repairs=repair_count,
                    usage=usage,
                )

            elapsed = time.perf_counter() - started
            self.metrics.requests.labels(outcome="success").inc()
            self.metrics.latency.observe(elapsed)
            return QueryOutcome(
                status="success",
                sql=validated.normalized,
                result=result,
                reason=None,
                attempts=repair_count + 1,
                repairs=repair_count,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                model_latency_seconds=usage.latency_seconds,
                total_latency_seconds=elapsed,
                model=usage.model,
            )

        raise AssertionError("bounded loop did not return")

    def _refusal(
        self,
        *,
        started: float,
        reason: str,
        sql: str | None = None,
        attempts: int = 0,
        repairs: int = 0,
        usage: _UsageAccumulator | None = None,
    ) -> QueryOutcome:
        elapsed = time.perf_counter() - started
        usage = usage or _UsageAccumulator()
        self.metrics.requests.labels(outcome="refused").inc()
        self.metrics.latency.observe(elapsed)
        return QueryOutcome(
            status="refused",
            sql=sql,
            result=None,
            reason=reason,
            attempts=attempts,
            repairs=repairs,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            model_latency_seconds=usage.latency_seconds,
            total_latency_seconds=elapsed,
            model=usage.model,
        )


@dataclass
class _UsageAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    model: str | None = None

    def add(self, result: LLMResult) -> None:
        self.input_tokens += result.usage.input_tokens
        self.output_tokens += result.usage.output_tokens
        self.cost_usd += result.cost_usd
        self.latency_seconds += result.latency_seconds
        self.model = result.model

