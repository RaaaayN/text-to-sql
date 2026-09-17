# Text-to-SQL

Ask a question in plain English and get back a safe, executable SQLite query.

This project is a deliberately small Text-to-SQL service built around Gemini. It retrieves the
useful parts of the schema, asks the model for structured SQL, validates the result with
`sqlglot`, and only then runs it against a read-only database. If SQLite rejects the query, the
service can ask the model to repair it, with a hard limit of two attempts.

The full pipeline was evaluated on the 500 SQLite examples in BIRD Mini-Dev. It reached **68.4%
execution accuracy**, compared with **56.0%** for direct prompting, while lowering the average
model cost from **$0.00096 to $0.00075 per question**.

## How it works

```text
Question
   │
   ▼
Retrieve relevant tables and columns
   │
   ▼
Generate structured SQL with Gemini
   │
   ▼
Parse and validate with sqlglot
   │
   ▼
Run on read-only SQLite ── error ──► Repair (up to 2 attempts)
```

The system is single-turn and deterministic by design. It is not an autonomous agent: each
request follows the same bounded path, which makes failures easier to understand and results
easier to reproduce.

### Why retrieve the schema first?

Sending an entire schema to the model works surprisingly well as a baseline, but it gets noisy
and expensive as the database grows. The retrieval step keeps only the tables and columns that
look relevant to the question.

That step accounts for most of the measured gain:

```text
Direct prompting       56.0%
        │
        ├── schema retrieval
        ▼
Retrieval only         65.2%
        │
        ├── validation and repair
        ▼
Full pipeline          68.4%
```

## Results

The benchmark used `gemini-3.1-flash-lite` at `temperature=0`. Accuracy is based on executed
results rather than string equality between SQL queries. Confidence intervals are calculated
with 10,000 paired bootstrap resamples.

| Configuration | Execution accuracy | Change vs direct | Average cost | p95 latency |
|---|---:|---:|---:|---:|
| Direct LLM | 56.0% | — | $0.00096 | 1.37 s |
| Schema retrieval | 65.2% | +9.2 pts | $0.00072 | 1.34 s |
| Repair only | 56.8% | +0.8 pts | $0.00097 | 1.55 s |
| **Full pipeline** | **68.4%** | **+12.4 pts** | **$0.00075** | **1.48 s** |

The full run improves accuracy by 12.4 percentage points and cuts average inference cost by
roughly 22%. The [versioned evaluation report](artifacts/evaluation-report.json) contains the
confidence intervals, token counts, refusals, repair attempts, costs, and latency measurements.

These are benchmark results, not a claim about performance on production data warehouses. See
[Limitations](#limitations) for the main caveats.

## Quick start

You will need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

Install the project and create the demo database:

```bash
uv sync --frozen --extra dev
uv run text-to-sql init-demo
```

Start the API:

```bash
uv run text-to-sql serve
```

Then, from another terminal, ask it a question:

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"How many products are there?"}'
```

A successful response looks like this:

```json
{
  "sql": "SELECT COUNT(*) AS product_count FROM products",
  "columns": ["product_count"],
  "rows": [[42]],
  "metadata": {
    "latency_ms": 812,
    "repair_attempts": 0
  }
}
```

To check the README flow without starting a server or making a network request, run:

```bash
uv run python scripts/readme_smoke.py
```

The service can also run in Docker:

```bash
docker build -t text-to-sql .
docker run --rm -p 8000:8000 text-to-sql
```

## Using Gemini

The default `fake` provider is only there for offline development against the demo database. For
real generation, install the Gemini extra and set the provider credentials:

```bash
uv sync --frozen --extra dev --extra gemini

export TEXT_TO_SQL_LLM_PROVIDER=gemini
export TEXT_TO_SQL_MODEL_NAME=gemini-3.1-flash-lite
export TEXT_TO_SQL_API_KEY=...
```

Before running a full evaluation, you can make one measured request:

```bash
uv run --extra gemini text-to-sql smoke
```

The command reports input and output tokens, reasoning tokens, model latency, total latency, and
estimated inference cost. The remaining settings are documented in `.env.example`.

## Safety and observability

Generated SQL is never passed straight to SQLite. Every query is parsed and inspected first.
The validator rejects write operations such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, and `ALTER`,
as well as multiple statements and references outside the known schema.

Accepted queries still run through a read-only SQLite connection, with a timeout and a limit on
the number of returned rows. Questions that cannot be answered from the available schema produce
an explicit refusal instead of a best-effort query.

Prometheus-compatible metrics are available at:

```bash
curl http://127.0.0.1:8000/metrics
```

They cover request outcomes, token use, estimated cost, model and end-to-end latency, refusals,
and repair attempts. Evaluation runs also write a JSON report with per-example predictions and
execution results.

## Running an evaluation

The evaluator compares result sets, not SQL text. Duplicate rows are preserved; row order only
matters when the reference query contains `ORDER BY`, and column order is normalized.

For a development run:

```bash
uv run text-to-sql evaluate \
  mini_dev.json \
  databases \
  artifacts/dev-run
```

The `--final` mode adds guardrails against accidentally using the fake provider, an incomplete or
mutable benchmark split, or an output directory that already contains a final report:

```bash
TEXT_TO_SQL_LLM_PROVIDER=gemini \
TEXT_TO_SQL_API_KEY=... \
uv run text-to-sql evaluate \
  mini_dev.json \
  databases \
  artifacts/final-run \
  --final
```

Freeze a benchmark split before the final run with:

```bash
uv run text-to-sql freeze-split \
  mini_dev.json \
  artifacts/mini-dev-manifest.json
```

Development data is used for retrieval and prompt tuning; the frozen Mini-Dev split is reserved
for the final evaluation. Bootstrap resampling uses a fixed seed so the reported intervals can be
reproduced.

## Scope

The repository focuses on the core Text-to-SQL path: retrieval, generation, validation,
read-only execution, bounded repair, and measurement. It intentionally leaves out fine-tuning,
multi-turn conversations, autonomous agents, vector databases, chart generation, a web UI, and
multi-provider routing.

Keeping that scope narrow is useful here: each component can be removed and measured as a clean
ablation instead of disappearing inside a larger agent stack.

## Limitations

BIRD is a benchmark, not a production warehouse. Execution accuracy can also mark two
semantically different queries as equivalent when they happen to return the same result on one
database.

The current implementation targets SQLite and has not been evaluated on large enterprise
schemas, PostgreSQL, Snowflake, or BigQuery. Retrieval also depends on clues in table names,
column names, and representative values, so vague schemas and unseen synonyms remain difficult.

In short, the benchmark measures this pipeline under controlled conditions. It should not be
read as a general claim of production-level Text-to-SQL reliability.

## Stack

Python 3.11 · FastAPI · Gemini · sqlglot · SQLite · Prometheus · pytest · Docker · uv
