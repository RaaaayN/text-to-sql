from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from text_to_sql.api import create_app
from text_to_sql.config import Settings
from text_to_sql.execution import SQLiteExecutor
from text_to_sql.llm import ScriptedLLMClient
from text_to_sql.observability import ServiceMetrics
from text_to_sql.schema import SchemaResolver, introspect_sqlite
from text_to_sql.service import TextToSQLService


def _app(tmp_path: Path):
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany("INSERT INTO products(name) VALUES (?)", [("pen",), ("book",)])
    metrics = ServiceMetrics(CollectorRegistry())
    service = TextToSQLService(
        resolver=SchemaResolver(introspect_sqlite(database)),
        llm=ScriptedLLMClient([{"sql": "SELECT name FROM products ORDER BY id"}]),
        executor=SQLiteExecutor(database),
        metrics=metrics,
    )
    return create_app(settings=Settings(database_path=database, _env_file=None), service=service)


def test_query_endpoint_runs_complete_pipeline(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/query", json={"question": "List product names"})

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["rows"] == [["pen"], ["book"]]


def test_metrics_and_health_are_exposed(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        client.post("/query", json={"question": "List product names"})
        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "text_to_sql_requests_total" in metrics.text


def test_empty_question_is_rejected(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
