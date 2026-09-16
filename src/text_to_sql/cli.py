"""Command-line utilities for local operation and benchmark hygiene."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from text_to_sql.benchmark import freeze_split
from text_to_sql.demo import create_demo_database


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-to-sql")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("init-demo", help="create the offline demonstration database")
    demo.add_argument("--database", type=Path, default=Path("data/demo.sqlite"))
    demo.add_argument("--force", action="store_true")

    freeze = commands.add_parser("freeze-split", help="write a reproducible BIRD manifest")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("destination", type=Path)
    freeze.add_argument("--name", default="mini-dev")
    freeze.add_argument("--seed", type=int, default=20260916)

    serve = commands.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-demo":
        path = create_demo_database(args.database, force=args.force)
        print(f"Created demo database: {path}")
        return 0
    if args.command == "freeze-split":
        frozen = freeze_split(args.source, name=args.name, seed=args.seed)
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        frozen.write(args.destination)
        print(f"Frozen {len(frozen.examples)} examples: {args.destination}")
        return 0
    if args.command == "serve":
        import uvicorn

        uvicorn.run("text_to_sql.api:app", host=args.host, port=args.port)
        return 0
    raise AssertionError("unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

