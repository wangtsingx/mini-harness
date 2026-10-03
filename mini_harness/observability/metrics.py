"""Metrics are DERIVED from spans (single source of truth): no second instrumentation to keep in sync."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mini_harness.observability.tracer import Span


@dataclass(frozen=True)
class RunMetrics:
    trace_id: int
    status: str
    reason: str | None  # completed | max_turns | budget_exceeded | repeated_calls | max_tokens | cancelled | failed
    duration_s: float
    turns: int
    model_calls: int
    model_time_s: float
    ttft_avg_s: float | None
    tool_calls: int
    tool_errors: int
    tool_time_s: float  # sum of tool durations (parallel tools overlap, so this can exceed wall time)
    retries: int  # model-call retries + tool re-attempts
    compactions: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None

    @property
    def tool_error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls else 0.0

    @property
    def cache_hit_ratio(self) -> float:
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.model_time_s if self.model_time_s else 0.0


def summarize(spans: Sequence[Span], trace_id: int | None = None) -> RunMetrics:
    """Metrics of one run (default: the most recent)."""
    runs = [s for s in spans if s.kind == "run"]
    if not runs:
        raise ValueError("no run span in trace")
    run = runs[-1] if trace_id is None else next(s for s in runs if s.trace_id == trace_id)
    mine = [s for s in spans if s.trace_id == run.trace_id]
    models = [s for s in mine if s.kind == "model"]
    tools = [s for s in mine if s.kind == "tool"]
    ttfts = [s.attrs["ttft_s"] for s in models if s.attrs.get("ttft_s") is not None]
    a = run.attrs
    return RunMetrics(
        trace_id=run.trace_id,
        status=run.status,
        reason=a.get("reason"),
        duration_s=run.duration_s or 0.0,
        turns=sum(1 for s in mine if s.kind == "turn"),
        model_calls=len(models),
        model_time_s=sum(s.duration_s or 0.0 for s in models),
        ttft_avg_s=sum(ttfts) / len(ttfts) if ttfts else None,
        tool_calls=len(tools),
        tool_errors=sum(1 for s in tools if s.attrs.get("is_error")),
        tool_time_s=sum(s.duration_s or 0.0 for s in tools),
        retries=sum(s.attrs.get("retries", 0) for s in models)
        + sum(max(s.attrs.get("attempts", 1) - 1, 0) for s in tools),
        compactions=sum(1 for s in mine if s.kind == "compaction"),
        input_tokens=sum(s.attrs.get("input_tokens", 0) for s in models),
        output_tokens=sum(s.attrs.get("output_tokens", 0) for s in models),
        cache_read_tokens=sum(s.attrs.get("cache_read_tokens", 0) for s in models),
        cache_write_tokens=sum(s.attrs.get("cache_write_tokens", 0) for s in models),
        cost_usd=a.get("cost_usd"),
    )


def combine(runs: Sequence[RunMetrics]) -> RunMetrics:
    """Merge the runs of one multi-prompt conversation into a single record (sums; status/reason from the last)."""
    if not runs:
        raise ValueError("nothing to combine")
    last = runs[-1]
    calls = sum(r.model_calls for r in runs)
    ttft = [(r.ttft_avg_s, r.model_calls) for r in runs if r.ttft_avg_s is not None]
    costs = [r.cost_usd for r in runs if r.cost_usd is not None]
    return RunMetrics(
        trace_id=last.trace_id,
        status=next((r.status for r in runs if r.status != "ok"), "ok"),
        reason=last.reason,
        duration_s=sum(r.duration_s for r in runs),
        turns=sum(r.turns for r in runs),
        model_calls=calls,
        model_time_s=sum(r.model_time_s for r in runs),
        ttft_avg_s=sum(v * w for v, w in ttft) / sum(w for _, w in ttft) if ttft else None,
        tool_calls=sum(r.tool_calls for r in runs),
        tool_errors=sum(r.tool_errors for r in runs),
        tool_time_s=sum(r.tool_time_s for r in runs),
        retries=sum(r.retries for r in runs),
        compactions=sum(r.compactions for r in runs),
        input_tokens=sum(r.input_tokens for r in runs),
        output_tokens=sum(r.output_tokens for r in runs),
        cache_read_tokens=sum(r.cache_read_tokens for r in runs),
        cache_write_tokens=sum(r.cache_write_tokens for r in runs),
        cost_usd=sum(costs) if costs else None,
    )


def summarize_all(spans: Sequence[Span]) -> list[RunMetrics]:
    return [summarize(spans, s.trace_id) for s in spans if s.kind == "run"]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100); None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def aggregate(runs: Sequence[RunMetrics]) -> dict[str, Any]:
    """Fleet-level view over many runs."""
    if not runs:
        return {"runs": 0}
    tool_calls = sum(r.tool_calls for r in runs)
    in_total = sum(r.input_tokens + r.cache_read_tokens + r.cache_write_tokens for r in runs)
    ttfts = [r.ttft_avg_s for r in runs if r.ttft_avg_s is not None]
    costs = [r.cost_usd for r in runs if r.cost_usd is not None]
    return {
        "runs": len(runs),
        "completion_rate": sum(r.reason == "completed" for r in runs) / len(runs),  # finished normally, not "correct"
        "repeated_calls_rate": sum(r.reason == "repeated_calls" for r in runs) / len(runs),
        "duration_p50_s": percentile([r.duration_s for r in runs], 50),
        "duration_p95_s": percentile([r.duration_s for r in runs], 95),
        "ttft_p50_s": percentile(ttfts, 50),
        "turns_avg": sum(r.turns for r in runs) / len(runs),
        "tool_error_rate": sum(r.tool_errors for r in runs) / tool_calls if tool_calls else 0.0,
        "retries_total": sum(r.retries for r in runs),
        "compactions_total": sum(r.compactions for r in runs),
        "tokens_in": sum(r.input_tokens for r in runs),
        "tokens_out": sum(r.output_tokens for r in runs),
        "cache_hit_ratio": sum(r.cache_read_tokens for r in runs) / in_total if in_total else 0.0,
        "cost_usd_total": sum(costs) if costs else None,
    }
