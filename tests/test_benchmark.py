import json
from pathlib import Path

import pytest

from text_to_sql.benchmark import freeze_split, load_bird


def test_loads_bird_fields_and_freezes_source(tmp_path: Path) -> None:
    source = tmp_path / "mini_dev.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": 42,
                    "db_id": "shop",
                    "question": "How many products?",
                    "SQL": "SELECT count(*) FROM products",
                    "evidence": "product means row",
                }
            ]
        ),
        encoding="utf-8",
    )

    frozen = freeze_split(source, name="mini-dev", seed=20260916)
    destination = tmp_path / "manifest.json"
    frozen.write(destination)

    assert frozen.examples[0].example_id == "42"
    assert frozen.examples[0].database_id == "shop"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert len(payload["source_sha256"]) == 64


def test_rejects_incomplete_records(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text('[{"question": "hello"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="db_id"):
        load_bird(source)
