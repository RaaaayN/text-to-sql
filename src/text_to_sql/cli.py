"""Command-line utilities for local operation and benchmark hygiene."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import repeat
from pathlib import Path

from text_to_sql.benchmark import freeze_split, load_bird
from text_to_sql.config import Settings
from text_to_sql.demo import create_demo_database
from text_to_sql.llm import GeminiLLMClient, ScriptedLLMClient, TokenPricing
from text_to_sql.runner import run_ablation_suite


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

    evaluate = commands.add_parser("evaluate", help="run the four BIRD ablations")
    evaluate.add_argument("source", type=Path, help="BIRD JSON split")
    evaluate.add_argument("database_root", type=Path)
    evaluate.add_argument("output_directory", type=Path)
    evaluate.add_argument(
        "--final",
        action="store_true",
        help="enforce final-run safeguards (Gemini, >=300 examples, no overwrite)",
    )
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
    if args.command == "evaluate":
        settings = Settings()
        examples = load_bird(args.source)
        report_path = args.output_directory / "evaluation-report.json"
        if args.final:
            if len(examples) < 300:
                raise ValueError("a final evaluation requires at least 300 examples")
            if settings.llm_provider != "gemini":
                raise ValueError("a final evaluation requires TEXT_TO_SQL_LLM_PROVIDER=gemini")
            if report_path.exists():
                raise FileExistsError("final report already exists; refusing to overwrite it")

        pricing = TokenPricing(
            settings.input_token_price_per_million,
            settings.output_token_price_per_million,
        )

        def llm_factory(_configuration):
            if settings.llm_provider == "gemini":
                return GeminiLLMClient(
                    api_key=settings.gemini_api_key,
                    model=settings.model_name,
                    pricing=pricing,
                )
            return ScriptedLLMClient(
                repeat({"sql": "SELECT count(*) AS product_count FROM products"}),
                model="fake",
            )

        report = run_ablation_suite(
            examples,
            database_root=args.database_root,
            llm_factory=llm_factory,
            output_directory=args.output_directory,
            model_version=settings.model_name if settings.llm_provider == "gemini" else "fake",
        )
        print(f"Evaluated {report['examples']} examples: {report_path}")
        return 0
    raise AssertionError("unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
