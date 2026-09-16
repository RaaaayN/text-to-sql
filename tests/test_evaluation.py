import json
from datetime import UTC, datetime

import pytest

from text_to_sql.evaluation import (
    EvaluationRecord,
    QueryResult,
    bootstrap_accuracy,
    bootstrap_accuracy_difference,
    build_ablation_report,
    compare_results,
    reference_has_order_by,
    summarize_configuration,
    write_json,
    write_jsonl,
)


def make_record(
    example_id: str,
    configuration: str,
    correct: bool,
    *,
    attempts: int = 1,
    refused: bool = False,
) -> EvaluationRecord:
    return EvaluationRecord(
        example_id=example_id,
        configuration=configuration,
        question=f"question {example_id}",
        generated_sql="SELECT 1",
        reference_sql="SELECT 1",
        correct=correct,
        attempts=attempts,
        input_tokens=10,
        output_tokens=2,
        latency_seconds=float(int(example_id) + 1),
        cost_usd=0.01,
        refused=refused,
    )


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT * FROM t ORDER\n BY x", True),
        ("select row_number() over (order by x) from t", True),
        ("SELECT 'order by' AS text", False),
        ('SELECT "order by" FROM t', False),
        ("SELECT * FROM t -- ORDER BY x", False),
        ("SELECT * FROM t /* ORDER BY x */", False),
        ("SELECT [order by] FROM t", False),
        ("SELECT order_id, byline FROM t", False),
    ],
)
def test_reference_has_order_by_lexes_sql(sql, expected):
    assert reference_has_order_by(sql) is expected


def test_unordered_rows_and_columns_compare_equal_and_preserve_duplicates():
    predicted = QueryResult(columns=("name", "id"), rows=(("Ada", 1), ("Lin", 2), ("Ada", 1)))
    reference = QueryResult(columns=("id", "name"), rows=((1, "Ada"), (1, "Ada"), (2, "Lin")))

    assert compare_results(predicted, reference, "SELECT id, name FROM people").equal
    assert not compare_results(predicted.rows[:-1], reference.rows, "SELECT * FROM people").equal


def test_order_by_makes_row_order_significant_but_not_column_order():
    predicted = (("Ada", 1), ("Lin", 2))
    same_order = ((1, "Ada"), (2, "Lin"))
    reversed_rows = tuple(reversed(same_order))

    result = compare_results(predicted, same_order, "SELECT id, name FROM people ORDER BY id")
    assert result.equal and result.ordered
    assert not compare_results(predicted, reversed_rows, "SELECT * FROM people ORDER BY id").equal


def test_comparison_is_type_aware_and_handles_special_values():
    assert not compare_results(((True,),), ((1,),), "SELECT value").equal
    assert compare_results(((float("nan"), -0.0),), ((float("nan"), 0.0),), "SELECT a, b").equal


def test_comparison_validates_shape():
    result = compare_results(((1, 2),), ((1,),), "SELECT 1")
    assert not result.equal
    assert result.reason == "result column counts differ"
    with pytest.raises(ValueError, match="row width"):
        compare_results(QueryResult(rows=((1,),), columns=("a", "b")), ((1,),), "SELECT 1")


def test_bootstrap_is_deterministic_and_estimate_is_exact():
    first = bootstrap_accuracy([1, 0, 1, 1], resamples=500, seed=42)
    second = bootstrap_accuracy([1, 0, 1, 1], resamples=500, seed=42)

    assert first == second
    assert first.estimate == 0.75
    assert first.low <= first.estimate <= first.high


def test_paired_bootstrap_difference_and_validation():
    interval = bootstrap_accuracy_difference(
        [True, True, True, False], [False, True, False, False], resamples=500, seed=9
    )
    assert interval.estimate == 0.5
    assert interval.low <= interval.estimate <= interval.high
    with pytest.raises(ValueError, match="same length"):
        bootstrap_accuracy_difference([True], [True, False])
    with pytest.raises(ValueError, match="binary"):
        bootstrap_accuracy([2])


def test_summary_contains_product_metrics():
    records = [
        make_record("0", "full", False, refused=True),
        make_record("1", "full", True, attempts=2),
        make_record("2", "full", True),
    ]
    summary = summarize_configuration(records, resamples=100, seed=3)

    assert summary["execution_accuracy"]["estimate"] == pytest.approx(2 / 3)
    assert summary["repair_rate"] == pytest.approx(1 / 3)
    assert summary["refusal_rate"] == pytest.approx(1 / 3)
    assert summary["tokens"] == {"input": 30, "output": 6}
    assert summary["cost_usd"]["mean"] == pytest.approx(0.01)
    assert summary["latency_seconds"]["p50"] == 2.0


def test_ablation_report_aligns_examples_by_id_and_is_reproducible():
    baseline = [make_record("0", "direct", False), make_record("1", "direct", True)]
    full = [make_record("1", "full", True), make_record("0", "full", True)]
    generated_at = datetime(2026, 9, 16, tzinfo=UTC)

    report = build_ablation_report(
        {"full": full, "direct": baseline},
        baseline="direct",
        model_version="gemini-test",
        seed=7,
        resamples=200,
        generated_at=generated_at,
    )

    assert report["examples"] == 2
    assert report["generated_at"] == "2026-09-16T00:00:00+00:00"
    assert report["accuracy_deltas_vs_baseline"]["full"]["estimate"] == 0.5
    assert report["configurations"]["full"]["execution_accuracy"]["estimate"] == 1.0


def test_ablation_report_rejects_unpaired_or_duplicate_examples():
    with pytest.raises(ValueError, match="same examples"):
        build_ablation_report(
            {
                "direct": [make_record("0", "direct", True)],
                "full": [make_record("1", "full", True)],
            },
            baseline="direct",
            model_version="model",
            resamples=10,
        )
    duplicate = [make_record("0", "direct", True), make_record("0", "direct", False)]
    with pytest.raises(ValueError, match="duplicate"):
        build_ablation_report(
            {"direct": duplicate}, baseline="direct", model_version="model", resamples=10
        )


def test_json_and_jsonl_writers_create_parseable_artifacts(tmp_path):
    record = make_record("0", "full", True)
    journal = tmp_path / "nested" / "journal.jsonl"
    report_path = tmp_path / "report.json"

    write_jsonl(journal, [record])
    write_json(report_path, {"generated_at": datetime(2026, 9, 16), "record": record})

    assert json.loads(journal.read_text(encoding="utf-8"))["question"] == "question 0"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["generated_at"] == "2026-09-16T00:00:00"
    assert report["record"]["correct"] is True


def test_record_rejects_invalid_measurements():
    with pytest.raises(ValueError, match="latency_seconds"):
        EvaluationRecord("x", "full", "q", None, "SELECT 1", False, 0, latency_seconds=-1)
