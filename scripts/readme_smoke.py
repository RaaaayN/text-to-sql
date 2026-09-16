"""Run the documented offline path as an integration smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from text_to_sql.api import create_app
from text_to_sql.config import Settings
from text_to_sql.demo import create_demo_database


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = create_demo_database(Path(directory) / "demo.sqlite")
        app = create_app(settings=Settings(database_path=database, _env_file=None))
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "How many products are there?"})
            response.raise_for_status()
            payload = response.json()
        if payload["status"] != "success" or payload["rows"] != [[3]]:
            raise RuntimeError(f"unexpected README result: {payload}")
        print("README smoke test passed: product_count=3")


if __name__ == "__main__":
    main()

