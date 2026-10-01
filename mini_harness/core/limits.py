from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Guardrails. Token/cost budgets are per Session (cumulative) and checked before each model call."""

    max_turns: int = 30
    max_total_tokens: int = 200_000
    max_cost_usd: float | None = None  # requires Pricing on the Agent
    wall_clock_s: float = 300.0  # hard limit per run, enforced by the supervisor
    max_tool_concurrency: int = 8
    repeat_call_threshold: int = 3  # Nth identical consecutive turn -> warn; N+1th -> stop
