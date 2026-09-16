from __future__ import annotations

import sqlite3

import pytest

from text_to_sql.sql_guard import SQLValidationError, enforce_read_only, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, name FROM users ORDER BY id",
        "WITH active AS (SELECT * FROM users WHERE active = 1) SELECT name FROM active",
        "SELECT id FROM users UNION SELECT id FROM admins",
    ],
)
def test_accepts_read_only_queries(sql: str) -> None:
    validated = validate_sql(sql, limit=25)

    assert validated.normalized
    assert validated.bounded.endswith('AS "_text_to_sql_result" LIMIT 25')


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO users(name) VALUES ('Mallory')",
        "UPDATE users SET active = 0",
        "DELETE FROM users",
        "CREATE TABLE stolen(data TEXT)",
        "DROP TABLE users",
        "ALTER TABLE users ADD COLUMN secret TEXT",
        "PRAGMA table_info(users)",
        "ATTACH DATABASE '/tmp/other.db' AS other",
        "DETACH DATABASE other",
        "VACUUM",
        "VALUES (1)",
    ],
)
def test_rejects_non_select_statements(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_rejects_multiple_statements_even_if_each_is_select() -> None:
    with pytest.raises(SQLValidationError, match="exactly one"):
        enforce_read_only("SELECT 1; SELECT 2")


@pytest.mark.parametrize("sql", ["", "  ", "SELECT FROM", "SELECT 1; DELETE FROM users"])
def test_rejects_empty_malformed_or_mixed_sql(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_rejects_unterminated_string_as_validation_error() -> None:
    with pytest.raises(SQLValidationError, match="could not be parsed"):
        validate_sql("SELECT * FROM players WHERE work_rate = 'high")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_sql("SELECT 1", limit=limit)  # type: ignore[arg-type]


def test_outer_limit_caps_results_without_changing_projection_or_order() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE numbers (value INTEGER)")
    connection.executemany("INSERT INTO numbers VALUES (?)", [(3,), (1,), (2,)])

    rows = connection.execute(
        enforce_read_only("SELECT value FROM numbers ORDER BY value DESC", limit=2)
    ).fetchall()

    assert rows == [(3,), (2,)]


def test_outer_limit_respects_a_stricter_model_limit() -> None:
    connection = sqlite3.connect(":memory:")
    rows = connection.execute(
        enforce_read_only("SELECT 1 AS value UNION ALL SELECT 2 LIMIT 1", limit=10)
    ).fetchall()

    assert rows == [(1,)]
