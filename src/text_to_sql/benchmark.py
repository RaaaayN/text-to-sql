"""BIRD benchmark loading and immutable split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    database_id: str
    question: str
    reference_sql: str
    evidence: str = ""


@dataclass(frozen=True)
class FrozenSplit:
    name: str
    seed: int
    source_sha256: str
    examples: tuple[BenchmarkExample, ...]

    def write(self, destination: str | Path) -> None:
        payload = {
            "name": self.name,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "count": len(self.examples),
            "examples": [asdict(example) for example in self.examples],
        }
        Path(destination).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def load_bird(path: str | Path) -> tuple[BenchmarkExample, ...]:
    source = Path(path)
    records: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("BIRD input must be a JSON array")

    examples: list[BenchmarkExample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        try:
            database_id = str(record["db_id"])
            question = str(record["question"])
            reference_sql = str(record["SQL"])
        except KeyError as exc:
            raise ValueError(f"record {index} is missing {exc.args[0]!r}") from exc
        raw_id = str(record.get("question_id", index))
        example_id = raw_id if raw_id not in seen_ids else f"{raw_id}:{index}"
        seen_ids.add(raw_id)
        examples.append(
            BenchmarkExample(
                example_id=example_id,
                database_id=database_id,
                question=question,
                reference_sql=reference_sql,
                evidence=str(record.get("evidence", "")),
            )
        )
    return tuple(examples)


def freeze_split(path: str | Path, *, name: str, seed: int) -> FrozenSplit:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return FrozenSplit(
        name=name,
        seed=seed,
        source_sha256=digest,
        examples=load_bird(source),
    )
