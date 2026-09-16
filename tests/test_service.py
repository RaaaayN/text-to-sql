from __future__ import annotations

import sqlite3
from pathlib import Path

from prometheus_client import CollectorRegistry

from text_to_sql.execution import SQLiteExecutor
from text_to_sql.llm import ScriptedLLMClient, TokenPricing, TokenUsage
from text_to_sql.observability import ServiceMetrics
from text_to_sql.schema import SchemaResolver, introspect_sqlite
from text_to_sql.service import TextToSQLService


def _service(path: Path, responses: list[dict[str, str]], max_repairs: int = 2):
    llm = ScriptedLLMClient(
        responses,
        usage=TokenUsage(100, 20),
        pricing=TokenPricing(1.0, 2.0),
    )
    metrics = ServiceMetrics(CollectorRegistry())
    return (
        TextToSQLService(
            resolver=SchemaResolver(introspect_sqlite(path)),
            llm=llm,
            executor=SQLiteExecutor(path),
            metrics=metrics,
            max_repairs=max_repairs,
            max_rows=10,
        ),
        llm,
        metrics,
    )


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany("INSERT INTO products(name) VALUES (?)", [("pen",), ("book",)])
    return path


def test_success_returns_result_and_usage(tmp_path: Path) -> None:
    service, _, metrics = _service(
        _database(tmp_path), [{"sql": "SELECT name FROM products ORDER BY id"}]
    )

    outcome = service.ask("List product names")

    assert outcome.status == "success"
    assert outcome.result is not None
    assert outcome.result.rows == (("pen",), ("book",))
    assert outcome.attempts == 1
    assert outcome.cost_usd == 0.00014
    assert 'outcome="success"' in metrics.render().decode()


def test_repairs_database_error_within_budget(tmp_path: Path) -> None:
    service, llm, _ = _service(
        _database(tmp_path),
        [
            {"sql": "SELECT missing FROM products"},
            {"sql": "SELECT count(*) FROM products"},
        ],
    )

    outcome = service.ask("Count products")

    assert outcome.status == "success"
    assert outcome.repairs == 1
    assert outcome.attempts == 2
    assert outcome.input_tokens == 200
    assert "no such column" in llm.prompts[1]


def test_refuses_unsafe_sql_without_repair(tmp_path: Path) -> None:
    service, _, _ = _service(_database(tmp_path), [{"sql": "DELETE FROM products"}])

    outcome = service.ask("List products")

    assert outcome.status == "refused"
    assert outcome.repairs == 0
    assert "unsafe SQL" in (outcome.reason or "")


def test_refuses_question_without_schema_evidence_before_model_call(tmp_path: Path) -> None:
    service, llm, _ = _service(_database(tmp_path), [{"sql": "SELECT 1"}])

    outcome = service.ask("Will it rain tomorrow?")

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert llm.prompts == []


def test_stops_after_repair_budget(tmp_path: Path) -> None:
    service, _, _ = _service(
        _database(tmp_path),
        [{"sql": "SELECT nope FROM products"}] * 3,
    )

    outcome = service.ask("List products")

    assert outcome.status == "refused"
    assert outcome.attempts == 3
    assert outcome.repairs == 2
    assert "budget exhausted" in (outcome.reason or "")
