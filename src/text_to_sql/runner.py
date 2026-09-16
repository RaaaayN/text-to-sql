"""Execution-based benchmark runner for the four declared ablations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from prometheus_client import CollectorRegistry

from text_to_sql.benchmark import BenchmarkExample
from text_to_sql.evaluation import (
    EvaluationRecord,
    build_ablation_report,
    compare_results,
    write_json,
    write_jsonl,
)
from text_to_sql.evaluation import (
    QueryResult as EvaluationQueryResult,
)
from text_to_sql.execution import QueryExecutionError, SQLiteExecutor
from text_to_sql.llm import LLMClient
from text_to_sql.observability import ServiceMetrics
from text_to_sql.schema import (
    DatabaseSchema,
    ResolvedSchema,
    SchemaResolver,
    introspect_sqlite,
)
from text_to_sql.service import TextToSQLService
from text_to_sql.sql_guard import SQLValidationError, validate_sql


@dataclass(frozen=True, slots=True)
class AblationConfiguration:
    name: str
    schema_resolution: bool
    repair: bool


STANDARD_ABLATIONS = (
    AblationConfiguration("direct", schema_resolution=False, repair=False),
    AblationConfiguration("schema_only", schema_resolution=True, repair=False),
    AblationConfiguration("repair_only", schema_resolution=False, repair=True),
    AblationConfiguration("full", schema_resolution=True, repair=True),
)


class FullSchemaResolver:
    """Resolver used by ablations that deliberately send the full schema."""

    def __init__(self, schema: DatabaseSchema) -> None:
        self.schema = schema

    def resolve(self, question: str) -> ResolvedSchema:
        del question
        selected = {
            table.name: tuple(column.name for column in table.columns)
            for table in self.schema.tables
        }
        scores = {table.name: 0.0 for table in self.schema.tables}
        return ResolvedSchema(self.schema, selected, scores)


def bird_database_path(root: str | Path, database_id: str) -> Path:
    """Resolve either BIRD's nested layout or a flat fixture layout."""

    root_path = Path(root)
    candidates = (
        root_path / database_id / f"{database_id}.sqlite",
        root_path / f"{database_id}.sqlite",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"database {database_id!r} not found below {root_path}")


def evaluate_configuration(
    examples: Sequence[BenchmarkExample],
    *,
    database_root: str | Path,
    configuration: AblationConfiguration,
    llm: LLMClient,
    max_rows: int = 1_000,
    timeout_seconds: float = 30.0,
) -> tuple[EvaluationRecord, ...]:
    """Evaluate one configuration, preserving one auditable record per example."""

    services: dict[str, TextToSQLService] = {}
    executors: dict[str, SQLiteExecutor] = {}
    records: list[EvaluationRecord] = []
    for example in examples:
        if example.database_id not in services:
            path = bird_database_path(database_root, example.database_id)
            schema = introspect_sqlite(path)
            resolver = (
                SchemaResolver(schema)
                if configuration.schema_resolution
                else FullSchemaResolver(schema)
            )
            executor = SQLiteExecutor(path, timeout_seconds=timeout_seconds, max_rows=max_rows)
            services[example.database_id] = TextToSQLService(
                resolver=resolver,  # type: ignore[arg-type]
                llm=llm,
                executor=executor,
                metrics=ServiceMetrics(CollectorRegistry()),
                max_repairs=2 if configuration.repair else 0,
                max_rows=max_rows,
            )
            executors[example.database_id] = executor

        outcome = services[example.database_id].ask(
            example.question,
            evidence=example.evidence,
        )
        correct = False
        error = outcome.reason
        if outcome.result is not None:
            try:
                reference = validate_sql(example.reference_sql, limit=max_rows)
                expected = executors[example.database_id].execute(reference.bounded)
                comparison = compare_results(
                    EvaluationQueryResult(outcome.result.rows, outcome.result.columns),
                    EvaluationQueryResult(expected.rows, expected.columns),
                    example.reference_sql,
                )
                correct = comparison.equal
                error = comparison.reason
            except (SQLValidationError, QueryExecutionError) as exc:
                error = f"reference execution failed: {exc}"

        records.append(
            EvaluationRecord(
                example_id=example.example_id,
                configuration=configuration.name,
                question=example.question,
                generated_sql=outcome.sql,
                reference_sql=example.reference_sql,
                correct=correct,
                attempts=outcome.attempts,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                latency_seconds=outcome.total_latency_seconds,
                cost_usd=outcome.cost_usd,
                refused=outcome.status == "refused",
                error=error,
            )
        )
    return tuple(records)


def run_ablation_suite(
    examples: Sequence[BenchmarkExample],
    *,
    database_root: str | Path,
    llm_factory: Callable[[AblationConfiguration], LLMClient],
    output_directory: str | Path,
    model_version: str,
    seed: int = 20260916,
    configurations: Sequence[AblationConfiguration] = STANDARD_ABLATIONS,
    pricing_usd_per_million: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Run, journal, summarize and atomically publish a complete ablation suite."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records_by_configuration: dict[str, tuple[EvaluationRecord, ...]] = {}
    for configuration in configurations:
        records = evaluate_configuration(
            examples,
            database_root=database_root,
            configuration=configuration,
            llm=llm_factory(configuration),
        )
        records_by_configuration[configuration.name] = records
        write_jsonl(output / f"{configuration.name}.jsonl", records)
    report = build_ablation_report(
        records_by_configuration,
        baseline="direct",
        model_version=model_version,
        seed=seed,
    )
    if pricing_usd_per_million is not None:
        report["pricing"] = {
            "basis": "standard_paid_tier_list_price",
            "currency": "USD",
            "input_per_million_tokens": pricing_usd_per_million[0],
            "output_per_million_tokens_including_thinking": pricing_usd_per_million[1],
        }
    write_json(output / "evaluation-report.json", report)
    return report
