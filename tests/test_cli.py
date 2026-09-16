import json
import sqlite3
from pathlib import Path

import pytest

from text_to_sql.cli import main


def test_init_demo_creates_expected_database(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite"

    assert main(["init-demo", "--database", str(database)]) == 0

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM products").fetchone() == (3,)


def test_init_demo_refuses_to_overwrite(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite"
    main(["init-demo", "--database", str(database)])

    with pytest.raises(FileExistsError, match="--force"):
        main(["init-demo", "--database", str(database)])


def test_freeze_split_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "bird.json"
    destination = tmp_path / "manifest.json"
    source.write_text(
        json.dumps([{"db_id": "shop", "question": "Count?", "SQL": "SELECT 1"}]),
        encoding="utf-8",
    )

    assert main(["freeze-split", str(source), str(destination)]) == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["count"] == 1
