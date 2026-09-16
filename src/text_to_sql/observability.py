"""Service metrics kept separate from offline evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, input_price_per_million: float, output_price_per_million: float) -> float:
        return (
            self.input_tokens * input_price_per_million
            + self.output_tokens * output_price_per_million
        ) / 1_000_000


class ServiceMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "text_to_sql_requests_total",
            "Questions handled by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.repairs = Counter(
            "text_to_sql_repairs_total",
            "SQL repair attempts.",
            registry=self.registry,
        )
        self.tokens = Counter(
            "text_to_sql_tokens_total",
            "Model tokens used by direction.",
            ("direction",),
            registry=self.registry,
        )
        self.cost = Counter(
            "text_to_sql_cost_usd_total",
            "Estimated model cost in US dollars.",
            registry=self.registry,
        )
        self.latency = Histogram(
            "text_to_sql_request_duration_seconds",
            "End-to-end service latency.",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )

    def observe_usage(self, usage: TokenUsage, cost_usd: float) -> None:
        self.tokens.labels(direction="input").inc(usage.input_tokens)
        self.tokens.labels(direction="output").inc(usage.output_tokens)
        self.cost.inc(cost_usd)

    def render(self) -> bytes:
        return generate_latest(self.registry)

