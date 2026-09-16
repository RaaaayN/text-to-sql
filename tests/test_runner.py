from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from text_to_sql.benchmark import BenchmarkExample
from text_to_sql.llm import ScriptedLLMClient
from text_to_sql.runner import (
    AblationConfiguration,
    bird_database_path,
    evaluate_configuration,
)


@pytest.fixture
def bird_root(tmp_path: Path) -> Path:
    directory = tmp_path / "shop"
    directory.mkdir()
    database = directory / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany("INSERT INTO products(name) VALUES (?)", [("pen",), ("book",)])
    return tmp_path


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="1",
        database_id="shop",
        question="How many products are there?",
        reference_sql="SELECT count(*) FROM products",
    )


def test_resolves_nested_bird_database(bird_root: Path) -> None:
    assert bird_database_path(bird_root, "shop") == bird_root / "shop" / "shop.sqlite"


def test_evaluates_prediction_against_reference_execution(bird_root: Path) -> None:
    records = evaluate_configuration(
        [_example()],
        database_root=bird_root,
        configuration=AblationConfiguration("direct", False, False),
        llm=ScriptedLLMClient([{"sql": "SELECT count(id) FROM products"}]),
    )

    assert len(records) == 1
    assert records[0].correct is True
    assert records[0].configuration == "direct"


def test_repair_ablation_records_attempts(bird_root: Path) -> None:
    records = evaluate_configuration(
        [_example()],
        database_root=bird_root,
        configuration=AblationConfiguration("full", True, True),
        llm=ScriptedLLMClient(
            [
                {"sql": "SELECT missing FROM products"},
                {"sql": "SELECT count(*) FROM products"},
            ]
        ),
    )

    assert records[0].correct is True
    assert records[0].attempts == 2


def test_missing_database_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="shop"):
        bird_database_path(tmp_path, "shop")
