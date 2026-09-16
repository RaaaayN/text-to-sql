"""Execution-based evaluation utilities for text-to-SQL experiments.

The comparator intentionally ignores result-column order: each row is represented
as a sorted bag of typed values.  Row duplicates are preserved.  Row order is
ignored unless the reference SQL contains an ``ORDER BY`` token pair outside a
comment or quoted literal.  Consequently aliases and driver-specific column names
cannot affect execution accuracy.

This definition follows the benchmark use case, where the value table is the
oracle.  Applications for which the association between a value and a named
output column is meaningful should use a stricter, domain-specific comparator.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryResult:
    """Materialized SQL result.

    ``columns`` is retained for logging and diagnostics.  It does not participate
    in equality because SQL aliases are not part of execution accuracy.
    """

    rows: Sequence[Sequence[Any]]
    columns: Sequence[str] = ()


@dataclass(frozen=True)
class ComparisonResult:
    equal: bool
    ordered: bool
    reason: str | None = None


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    """One auditable benchmark prediction and its service measurements."""

    example_id: str
    configuration: str
    question: str
    generated_sql: str | None
    reference_sql: str
    correct: bool
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    refused: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("attempts", "input_tokens", "output_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("latency_seconds", "cost_usd"):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reference_has_order_by(sql: str) -> bool:
    """Return whether SQL contains ORDER BY outside comments and quoted text."""

    tokens: list[str] = []
    current: list[str] = []
    i = 0
    state = "code"
    quote_end = ""

    def flush() -> None:
        if current:
            tokens.append("".join(current).casefold())
            current.clear()

    while i < len(sql):
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if state == "line_comment":
            if char in "\r\n":
                state = "code"
            i += 1
            continue
        if state == "block_comment":
            if char == "*" and nxt == "/":
                state = "code"
                i += 2
            else:
                i += 1
            continue
        if state == "quoted":
            if char == quote_end:
                # SQL escapes quote characters by doubling them.
                if quote_end in {"'", '"', "`"} and nxt == quote_end:
                    i += 2
                    continue
                state = "code"
            i += 1
            continue

        if char == "-" and nxt == "-":
            flush()
            state = "line_comment"
            i += 2
        elif char == "/" and nxt == "*":
            flush()
            state = "block_comment"
            i += 2
        elif char in {"'", '"', "`", "["}:
            flush()
            state = "quoted"
            quote_end = "]" if char == "[" else char
            i += 1
        elif char.isalnum() or char == "_":
            current.append(char)
            i += 1
        else:
            flush()
            i += 1
    flush()
    return any(
        left == "order" and right == "by" for left, right in zip(tokens, tokens[1:], strict=False)
    )


def compare_results(
    predicted: QueryResult | Sequence[Sequence[Any]],
    reference: QueryResult | Sequence[Sequence[Any]],
    reference_sql: str,
) -> ComparisonResult:
    """Compare two materialized results using execution-accuracy semantics."""

    predicted_rows = _rows(predicted)
    reference_rows = _rows(reference)
    ordered = reference_has_order_by(reference_sql)

    if any(len(row) != len(reference_rows[0]) for row in reference_rows[1:]):
        return ComparisonResult(False, ordered, "reference result has inconsistent row widths")
    if reference_rows:
        width = len(reference_rows[0])
        if any(len(row) != width for row in predicted_rows):
            return ComparisonResult(False, ordered, "result column counts differ")
    elif predicted_rows:
        return ComparisonResult(False, ordered, "result row counts differ")

    actual = [_canonical_row(row) for row in predicted_rows]
    expected = [_canonical_row(row) for row in reference_rows]
    if len(actual) != len(expected):
        return ComparisonResult(False, ordered, "result row counts differ")
    if not ordered:
        actual.sort(key=_stable_key)
        expected.sort(key=_stable_key)
    if actual != expected:
        qualifier = "ordered " if ordered else ""
        return ComparisonResult(False, ordered, f"{qualifier}result values differ")
    return ComparisonResult(True, ordered)


def bootstrap_accuracy(
    outcomes: Sequence[bool | int],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Compute a deterministic percentile bootstrap interval for accuracy."""

    values = _binary_values(outcomes)
    samples = _bootstrap_means(values, resamples=resamples, seed=seed)
    alpha = 1.0 - _validate_bootstrap(confidence, resamples)
    return ConfidenceInterval(
        estimate=sum(values) / len(values),
        low=_quantile(samples, alpha / 2),
        high=_quantile(samples, 1 - alpha / 2),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def bootstrap_accuracy_difference(
    candidate: Sequence[bool | int],
    baseline: Sequence[bool | int],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Paired bootstrap interval for candidate accuracy minus baseline accuracy."""

    candidate_values = _binary_values(candidate)
    baseline_values = _binary_values(baseline)
    if len(candidate_values) != len(baseline_values):
        raise ValueError("paired outcomes must have the same length")
    differences = [a - b for a, b in zip(candidate_values, baseline_values, strict=True)]
    samples = _bootstrap_means(differences, resamples=resamples, seed=seed)
    alpha = 1.0 - _validate_bootstrap(confidence, resamples)
    return ConfidenceInterval(
        estimate=sum(differences) / len(differences),
        low=_quantile(samples, alpha / 2),
        high=_quantile(samples, 1 - alpha / 2),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def summarize_configuration(
    records: Sequence[EvaluationRecord],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate accuracy, cost, latency, repair, and refusal metrics."""

    if not records:
        raise ValueError("at least one evaluation record is required")
    configurations = {record.configuration for record in records}
    if len(configurations) != 1:
        raise ValueError("records must belong to one configuration")
    accuracy = bootstrap_accuracy(
        [record.correct for record in records],
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    latency = sorted(record.latency_seconds for record in records)
    cost = sorted(record.cost_usd for record in records)
    repaired = sum(record.correct and record.attempts > 1 for record in records)
    return {
        "configuration": records[0].configuration,
        "examples": len(records),
        "execution_accuracy": accuracy.to_dict(),
        "cost_usd": {"mean": sum(cost) / len(cost), "p95": _quantile(cost, 0.95)},
        "latency_seconds": {
            "p50": _quantile(latency, 0.50),
            "p95": _quantile(latency, 0.95),
        },
        "repair_rate": repaired / len(records),
        "refusal_rate": sum(record.refused for record in records) / len(records),
        "mean_attempts": sum(record.attempts for record in records) / len(records),
        "tokens": {
            "input": sum(record.input_tokens for record in records),
            "output": sum(record.output_tokens for record in records),
        },
    }


def build_ablation_report(
    records_by_configuration: Mapping[str, Sequence[EvaluationRecord]],
    *,
    baseline: str,
    model_version: str,
    seed: int = 0,
    confidence: float = 0.95,
    resamples: int = 10_000,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready report with paired deltas against ``baseline``.

    Records are aligned by ``example_id`` before paired bootstrapping, so input
    mapping order never changes the result.
    """

    if baseline not in records_by_configuration:
        raise ValueError(f"unknown baseline configuration: {baseline}")
    if not model_version:
        raise ValueError("model_version is required for reproducibility")
    indexed: dict[str, dict[str, EvaluationRecord]] = {}
    for configuration, records in records_by_configuration.items():
        by_id = {record.example_id: record for record in records}
        if len(by_id) != len(records):
            raise ValueError(f"duplicate example_id in configuration {configuration!r}")
        if any(record.configuration != configuration for record in records):
            raise ValueError(f"record configuration mismatch for {configuration!r}")
        indexed[configuration] = by_id
    baseline_ids = set(indexed[baseline])
    if not baseline_ids:
        raise ValueError("at least one evaluation record is required")
    for configuration, by_id in indexed.items():
        if set(by_id) != baseline_ids:
            raise ValueError(f"configuration {configuration!r} does not contain the same examples")

    ordered_ids = sorted(baseline_ids)
    baseline_outcomes = [indexed[baseline][key].correct for key in ordered_ids]
    summaries: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for configuration in sorted(records_by_configuration):
        ordered_records = [indexed[configuration][key] for key in ordered_ids]
        summaries[configuration] = summarize_configuration(
            ordered_records, confidence=confidence, resamples=resamples, seed=seed
        )
        if configuration != baseline:
            deltas[configuration] = bootstrap_accuracy_difference(
                [record.correct for record in ordered_records],
                baseline_outcomes,
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            ).to_dict()

    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return {
        "schema_version": 1,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "model_version": model_version,
        "seed": seed,
        "confidence": confidence,
        "bootstrap_resamples": resamples,
        "examples": len(ordered_ids),
        "baseline": baseline,
        "configurations": summaries,
        "accuracy_deltas_vs_baseline": deltas,
    }


def write_jsonl(path: str | Path, records: Iterable[EvaluationRecord | Mapping[str, Any]]) -> None:
    """Write an evaluation journal as UTF-8 JSON Lines, atomically."""

    destination = Path(path)
    lines = (
        json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) for record in records
    )
    _atomic_write(destination, "".join(f"{line}\n" for line in lines))


def write_json(path: str | Path, payload: Any) -> None:
    """Write a stable, human-readable JSON artifact atomically."""

    content = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(path), content)


def _rows(result: QueryResult | Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    raw_rows = result.rows if isinstance(result, QueryResult) else result
    rows = [tuple(row) for row in raw_rows]
    if (
        isinstance(result, QueryResult)
        and result.columns
        and any(len(row) != len(result.columns) for row in rows)
    ):
        raise ValueError("row width does not match QueryResult.columns")
    return rows


def _canonical_row(row: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted((_canonical_value(value) for value in row), key=_stable_key))


def _canonical_value(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", 0.0 if value == 0 else value)
    if isinstance(value, Decimal):
        return ("decimal", str(value.normalize()))
    if isinstance(value, bytes | bytearray | memoryview):
        return ("bytes", bytes(value).hex())
    if isinstance(value, date | datetime):
        return (type(value).__name__, value.isoformat())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        items = sorted(
            ((_canonical_value(key), _canonical_value(item)) for key, item in value.items()),
            key=_stable_key,
        )
        return ("mapping", tuple(items))
    if isinstance(value, list | tuple):
        return ("sequence", tuple(_canonical_value(item) for item in value))
    return (f"object:{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _stable_key(value: Any) -> str:
    return repr(value)


def _binary_values(values: Sequence[bool | int]) -> list[int]:
    if not values:
        raise ValueError("at least one outcome is required")
    if any(value not in (False, True, 0, 1) for value in values):
        raise ValueError("outcomes must be binary")
    return [int(value) for value in values]


def _validate_bootstrap(confidence: float, resamples: int) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    return confidence


def _bootstrap_means(values: Sequence[int], *, resamples: int, seed: int) -> list[float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    size = len(values)
    samples = [
        sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(resamples)
    ]
    samples.sort()
    return samples


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(destination)
