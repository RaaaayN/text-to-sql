from prometheus_client import CollectorRegistry

from text_to_sql.observability import ServiceMetrics, TokenUsage


def test_cost_uses_provider_token_counters() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)

    assert usage.cost_usd(0.10, 0.40) == 0.30


def test_metrics_include_outcome_usage_and_cost() -> None:
    metrics = ServiceMetrics(CollectorRegistry())
    metrics.requests.labels(outcome="success").inc()
    metrics.observe_usage(TokenUsage(12, 3), 0.0001)

    payload = metrics.render().decode()

    assert 'text_to_sql_requests_total{outcome="success"} 1.0' in payload
    assert 'text_to_sql_tokens_total{direction="input"} 12.0' in payload
    assert "text_to_sql_cost_usd_total 0.0001" in payload
