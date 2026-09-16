from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from text_to_sql.execution import QueryExecutionError, SQLiteExecutor


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO products(name) VALUES (?)", [("pen",), ("book",), ("bag",)]
        )
    return path


def test_executes_select_and_returns_metadata(database: Path) -> None:
    result = SQLiteExecutor(database).execute("SELECT id, name FROM products ORDER BY id")

    assert result.columns == ("id", "name")
    assert result.rows == ((1, "pen"), (2, "book"), (3, "bag"))
    assert result.duration_seconds >= 0
    assert result.truncated is False


def test_truncates_results_at_budget(database: Path) -> None:
    result = SQLiteExecutor(database, max_rows=2).execute("SELECT * FROM products")

    assert len(result.rows) == 2
    assert result.truncated is True


def test_engine_rejects_writes(database: Path) -> None:
    with pytest.raises(QueryExecutionError):
        SQLiteExecutor(database).execute("DELETE FROM products")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM products").fetchone() == (3,)


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"

    with pytest.raises(QueryExecutionError, match="does not exist"):
        SQLiteExecutor(missing).execute("SELECT 1")

    assert not missing.exists()
