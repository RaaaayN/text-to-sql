"""Syntactic guardrails for model-generated SQL.

The database connection must still be opened read-only.  This module is the
second, application-level layer: it accepts one query, proves that its parsed
syntax tree is read-only, and caps the number of rows returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp, parse
from sqlglot.errors import ParseError


class SQLValidationError(ValueError):
    """Raised when generated SQL is not a single read-only query."""


@dataclass(frozen=True, slots=True)
class ValidatedSQL:
    """A validated query and the bounded statement safe to execute."""

    original: str
    normalized: str
    bounded: str
    limit: int


# Keep this explicit.  It documents the threat model and prevents a future
# sqlglot Query subclass from becoming executable merely because it parses.
_FORBIDDEN_NODE_NAMES = (
    "Alter",
    "Attach",
    "Command",
    "Copy",
    "Create",
    "Delete",
    "Detach",
    "Drop",
    "Execute",
    "Grant",
    "Insert",
    "Into",
    "LoadData",
    "Merge",
    "Pragma",
    "Set",
    "Transaction",
    "TruncateTable",
    "Update",
    "Use",
)


def _forbidden_types() -> tuple[type[exp.Expression], ...]:
    return tuple(
        node_type
        for name in _FORBIDDEN_NODE_NAMES
        if isinstance((node_type := getattr(exp, name, None)), type)
    )


def validate_sql(
    sql: str,
    *,
    dialect: str = "sqlite",
    limit: int = 1_000,
) -> ValidatedSQL:
    """Validate and bound a single SELECT (optionally introduced by a CTE).

    The row cap is applied through an outer query.  This preserves the model's
    projection, ordering and any stricter inner limit while guaranteeing that
    no more than ``limit`` rows reach the application.
    """

    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("SQL must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    try:
        statements = [statement for statement in parse(sql, read=dialect) if statement]
    except (ParseError, ValueError) as exc:
        raise SQLValidationError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise SQLValidationError("exactly one SQL statement is required")

    statement = statements[0]
    # SELECT, WITH ... SELECT, UNION/INTERSECT/EXCEPT of SELECTs all derive
    # from Query.  VALUES alone is intentionally rejected: the public contract
    # is a SELECT query, not merely any expression that happens to be harmless.
    if not isinstance(statement, exp.Query) or not any(statement.find_all(exp.Select)):
        raise SQLValidationError("only SELECT queries are allowed")

    forbidden_types = _forbidden_types()
    forbidden = next(
        (node for node in statement.walk() if isinstance(node, forbidden_types)),
        None,
    )
    if forbidden is not None:
        raise SQLValidationError(
            f"read-only query required; found {forbidden.__class__.__name__.upper()}"
        )

    normalized = statement.sql(dialect=dialect, pretty=False)
    bounded = f'SELECT * FROM ({normalized}) AS "_text_to_sql_result" LIMIT {limit}'
    return ValidatedSQL(
        original=sql,
        normalized=normalized,
        bounded=bounded,
        limit=limit,
    )


def enforce_read_only(sql: str, *, dialect: str = "sqlite", limit: int = 1_000) -> str:
    """Return executable, row-bounded SQL or raise :class:`SQLValidationError`."""

    return validate_sql(sql, dialect=dialect, limit=limit).bounded

