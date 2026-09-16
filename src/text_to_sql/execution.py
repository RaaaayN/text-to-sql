"""Bounded, read-only SQLite query execution."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    duration_seconds: float
    truncated: bool = False


class QueryExecutionError(RuntimeError):
    """Raised when SQLite rejects or interrupts a query."""


class QueryTimeoutError(QueryExecutionError):
    """Raised when a query exceeds its execution budget."""


class SQLiteExecutor:
    """Execute queries through a fresh SQLite read-only connection."""

    def __init__(
        self,
        database: str | Path,
        *,
        timeout_seconds: float = 5.0,
        max_rows: int = 1_000,
        progress_steps: int = 1_000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.database = Path(database).resolve()
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self.progress_steps = progress_steps

    def execute(self, sql: str) -> QueryResult:
        if not self.database.is_file():
            raise QueryExecutionError(f"database does not exist: {self.database}")

        started = time.perf_counter()
        deadline = started + self.timeout_seconds
        timed_out = False

        def interrupt_when_expired() -> int:
            nonlocal timed_out
            timed_out = time.perf_counter() >= deadline
            return int(timed_out)

        uri = f"file:{self.database.as_posix()}?mode=ro&immutable=1"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.set_progress_handler(interrupt_when_expired, self.progress_steps)
                cursor = connection.execute(sql)
                columns = tuple(item[0] for item in cursor.description or ())
                fetched = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            if timed_out:
                raise QueryTimeoutError(
                    f"query exceeded {self.timeout_seconds:g}s execution budget"
                ) from exc
            raise QueryExecutionError(str(exc)) from exc

        duration = time.perf_counter() - started
        return QueryResult(
            columns=columns,
            rows=tuple(tuple(row) for row in fetched[: self.max_rows]),
            duration_seconds=duration,
            truncated=len(fetched) > self.max_rows,
        )

