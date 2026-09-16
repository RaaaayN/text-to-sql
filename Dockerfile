FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TEXT_TO_SQL_DATABASE_PATH=/app/data/demo.sqlite

WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.9
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN uv run text-to-sql init-demo

EXPOSE 8000
CMD ["uv", "run", "text-to-sql", "serve", "--host", "0.0.0.0", "--port", "8000"]

