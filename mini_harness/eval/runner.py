"""Offline eval: run cases concurrently (optionally several times each), score them, attribute failures, and compare
against a saved baseline so a PR can be judged on success rate, cost and latency - not on a feeling."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_harness.core.session import Session
from mini_harness.eval.attribution import FIXES, Attribution, attribute
from mini_harness.eval.checks import Checker, Score, as_score, final_text
from mini_harness.eval.replay import RunResult, drive
from mini_harness.observability.metrics import RunMetrics, aggregate, combine, percentile, summarize_all
from mini_harness.observability.tracer import Tracer
from mini_harness.sdk import Agent

BASELINE_VERSION = 1


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Treats runs as independent, which is optimistic when repeats
    of the same case are correlated: more distinct cases tighten the estimate more than more repeats do."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str | Sequence[str]  # several prompts = a multi-turn conversation on one session
    check: Checker
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)  # free-form per-case settings for your agent factory


# Build a fresh Agent for each run. Receiving the case lets you vary tools/system/model per case if you need to.
EvalAgentFactory = Callable[[EvalCase, Tracer], Agent]


@dataclass(frozen=True)
class RunReport:
    case_id: str
    repeat: int
    passed: bool
    score: Score
    attribution: Attribution | None
    metrics: RunMetrics | None
    error: str | None
    final: str
    duration_s: float


@dataclass(frozen=True)
class CaseReport:
    id: str
    tags: tuple[str, ...]
    runs: tuple[RunReport, ...]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.runs) / len(self.runs)

    @property
    def flaky(self) -> bool:
        return 0 < self.pass_rate < 1


@dataclass(frozen=True)
class Cluster:
    key: str  # layer/subtype
    layer: str
    count: int
    case_ids: tuple[str, ...]
    confidence: str  # most common confidence among members
    sample_reason: str
    fix: str


@dataclass(frozen=True)
class Comparison:
    success_before: float
    success_after: float
    regressions: tuple[tuple[str, float, float], ...]  # (case, before, after)
    improvements: tuple[tuple[str, float, float], ...]
    new_cases: tuple[str, ...]
    removed_cases: tuple[str, ...]
    cost_change_pct: float | None
    p95_change_pct: float | None

    @property
    def ok(self) -> bool:
        return not self.regressions

    def render(self) -> str:
        lines = [
            f"success {self.success_before:.1%} -> {self.success_after:.1%}  ({'OK' if self.ok else 'REGRESSION'})"
        ]
        for cid, b, a in self.regressions:
            lines.append(f"  regressed: {cid}  {b:.0%} -> {a:.0%}")
        for cid, b, a in self.improvements:
            lines.append(f"  improved:  {cid}  {b:.0%} -> {a:.0%}")
        if self.new_cases:
            lines.append(f"  new cases: {', '.join(self.new_cases)}")
        if self.removed_cases:
            lines.append(f"  removed:   {', '.join(self.removed_cases)}")
        for label, pct in (("cost", self.cost_change_pct), ("p95 latency", self.p95_change_pct)):
            if pct is not None:
                lines.append(f"  {label}: {pct:+.1f}%")
        return "\n".join(lines)


def _pct(before: float | None, after: float | None) -> float | None:
    return None if not before or after is None else (after - before) / before * 100


@dataclass(frozen=True)
class EvalReport:
    cases: tuple[CaseReport, ...]
    repeats: int = 1
    wall_s: float = 0.0

    # ------------------------------------------------------------ numbers
    @property
    def runs(self) -> list[RunReport]:
        return [r for c in self.cases for r in c.runs]

    @property
    def success_rate(self) -> float:
        runs = self.runs
        return sum(r.passed for r in runs) / len(runs) if runs else 0.0

    def success_ci(self) -> tuple[float, float]:
        runs = self.runs
        return wilson_interval(sum(r.passed for r in runs), len(runs))

    @property
    def flaky_cases(self) -> list[str]:
        return [c.id for c in self.cases if c.flaky]

    def summary(self) -> dict[str, Any]:
        runs = self.runs
        durations = [r.duration_s for r in runs]
        agg = aggregate([r.metrics for r in runs if r.metrics])
        costs = [r.metrics.cost_usd for r in runs if r.metrics and r.metrics.cost_usd is not None]
        return {
            "cases": len(self.cases),
            "runs": len(runs),
            "success_rate": self.success_rate,
            "success_ci95": self.success_ci(),
            "cases_fully_passing": sum(c.pass_rate == 1 for c in self.cases),
            "flaky_cases": self.flaky_cases,
            "duration_p50_s": percentile(durations, 50),
            "duration_p95_s": percentile(durations, 95),
            "turns_avg": agg.get("turns_avg"),
            "tokens_in": agg.get("tokens_in", 0),
            "tokens_out": agg.get("tokens_out", 0),
            "cache_hit_ratio": agg.get("cache_hit_ratio", 0.0),
            "tool_error_rate": agg.get("tool_error_rate", 0.0),
            "retries_total": agg.get("retries_total", 0),
            "cost_usd_total": sum(costs) if costs else None,
        }

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """tag -> (passed runs, total runs)"""
        out: dict[str, list[int]] = {}
        for c in self.cases:
            for tag in c.tags:
                acc = out.setdefault(tag, [0, 0])
                acc[0] += sum(r.passed for r in c.runs)
                acc[1] += len(c.runs)
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    # ------------------------------------------------------------ failures
    def clusters(self) -> list[Cluster]:
        groups: dict[str, list[RunReport]] = {}
        for r in self.runs:
            if r.attribution is not None:
                groups.setdefault(r.attribution.key, []).append(r)
        out = []
        for key, members in groups.items():
            attrs = [m.attribution for m in members if m.attribution]
            conf = Counter(a.confidence for a in attrs).most_common(1)[0][0]
            out.append(
                Cluster(
                    key,
                    attrs[0].layer,
                    len(members),
                    tuple(sorted({m.case_id for m in members})),
                    conf,
                    attrs[0].reason,
                    FIXES[attrs[0].layer],
                )  # fmt: skip
            )
        return sorted(out, key=lambda c: (-c.count, c.key))

    def layer_counts(self) -> dict[str, int]:
        return dict(Counter(r.attribution.layer for r in self.runs if r.attribution))

    # ------------------------------------------------------------ output
    def render(self) -> str:
        s = self.summary()
        passed_runs = sum(r.passed for r in self.runs)
        lines = [
            f"EVAL  {s['cases']} cases x {self.repeats} run(s)   success {s['success_rate']:.1%} "
            f"({passed_runs}/{s['runs']} runs)   fully passing {s['cases_fully_passing']}/{s['cases']}   "
            f"flaky {len(s['flaky_cases'])}",
        ]
        lat = (
            f"p50 {s['duration_p50_s']:.2f}s  p95 {s['duration_p95_s']:.2f}s"
            if s["duration_p50_s"] is not None
            else "n/a"
        )
        cost = f"${s['cost_usd_total']:.4f}" if s["cost_usd_total"] is not None else "n/a"
        turns = f"{s['turns_avg']:.1f}" if s["turns_avg"] is not None else "n/a"
        lines.append(
            f"latency {lat} | turns avg {turns} | tokens in/out {s['tokens_in']}/{s['tokens_out']} "
            f"| cache hit {s['cache_hit_ratio']:.0%} | tool errors {s['tool_error_rate']:.0%} | cost {cost}"
        )
        if self.by_tag():
            lines.append("by tag: " + "   ".join(f"{t} {p}/{n}" for t, (p, n) in self.by_tag().items()))
        clusters = self.clusters()
        if clusters:
            lines.append(f"\nFAILURES - {len(clusters)} cluster(s) by likely cause")
            for c in clusters:
                lines += [
                    f"  [{c.key}] x{c.count}  confidence: {c.confidence}",
                    f"      cases: {', '.join(c.case_ids)}",
                    f"      why:   {c.sample_reason}",
                    f"      fix:   {c.fix}",
                ]
        if s["flaky_cases"]:
            rates = {c.id: f"{c.pass_rate:.0%}" for c in self.cases if c.flaky}
            lines.append("\nFLAKY (passed on some repeats only): " + ", ".join(f"{k} ({v})" for k, v in rates.items()))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        s = self.summary()
        return {
            "version": BASELINE_VERSION,
            "summary": {k: s[k] for k in ("success_rate", "runs", "duration_p50_s", "duration_p95_s", "turns_avg",
                                           "tokens_in", "tokens_out", "cost_usd_total")},
            "cases": {
                c.id: {
                    "pass_rate": c.pass_rate, "runs": len(c.runs), "tags": list(c.tags),
                    "causes": sorted({r.attribution.key for r in c.runs if r.attribution}),
                }
                for c in self.cases
            },
        }  # fmt: skip

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def compare(self, baseline: dict[str, Any] | EvalReport | str | Path, *, tolerance: float = 0.0) -> Comparison:
        """Judge this report against a saved baseline. A case regresses when its pass rate drops by more than
        `tolerance` (use e.g. 0.34 with repeats=3 to ignore a single flaky run)."""
        if isinstance(baseline, EvalReport):
            base = baseline.to_dict()
        elif isinstance(baseline, dict):
            base = baseline
        else:
            base = json.loads(Path(baseline).read_text(encoding="utf-8"))
        old, new = base["cases"], self.to_dict()["cases"]
        regress, improve = [], []
        for cid in sorted(old.keys() & new.keys()):
            b, a = old[cid]["pass_rate"], new[cid]["pass_rate"]
            if a < b - tolerance:
                regress.append((cid, b, a))
            elif a > b + tolerance:
                improve.append((cid, b, a))
        now = self.summary()
        return Comparison(
            base["summary"]["success_rate"], self.success_rate, tuple(regress), tuple(improve),
            tuple(sorted(new.keys() - old.keys())), tuple(sorted(old.keys() - new.keys())),
            _pct(base["summary"].get("cost_usd_total"), now["cost_usd_total"]),
            _pct(base["summary"].get("duration_p95_s"), now["duration_p95_s"]),
        )  # fmt: skip


class EvalRunner:
    def __init__(
        self,
        make_agent: EvalAgentFactory,
        cases: Sequence[EvalCase],
        *,
        concurrency: int = 4,
        repeats: int = 1,
    ) -> None:
        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case ids")
        if concurrency < 1 or repeats < 1:
            raise ValueError("concurrency and repeats must be >= 1")
        self._make_agent = make_agent
        self._cases = list(cases)
        self._concurrency = concurrency
        self._repeats = repeats

    async def run(self) -> EvalReport:
        sem = asyncio.Semaphore(self._concurrency)

        async def one(case: EvalCase, repeat: int) -> RunReport:
            async with sem:
                return await self._run_one(case, repeat)

        t0 = time.perf_counter()
        reports = await asyncio.gather(*(one(c, r) for c in self._cases for r in range(self._repeats)))
        grouped = [CaseReport(c.id, c.tags, tuple(r for r in reports if r.case_id == c.id)) for c in self._cases]
        return EvalReport(tuple(grouped), self._repeats, time.perf_counter() - t0)

    async def _run_one(self, case: EvalCase, repeat: int) -> RunReport:
        tracer = Tracer()
        t0 = time.perf_counter()
        prompts = [case.prompt] if isinstance(case.prompt, str) else list(case.prompt)
        try:
            result = await drive(self._make_agent(case, tracer), prompts)
        except Exception as e:  # noqa: BLE001 - a broken factory is a failed run, not a crashed eval
            result = RunResult([], e, Session())
        duration = time.perf_counter() - t0

        try:
            raw = case.check(result)
            score = as_score(await raw if inspect.isawaitable(raw) else raw)
        except Exception as e:  # noqa: BLE001
            score = Score(False, f"checker crashed: {type(e).__name__}: {e}", "checker_error")
        if result.error is not None and score.kind != "checker_error":
            score = Score(False, f"run raised {type(result.error).__name__}: {result.error}", "other", score.facts)

        runs = summarize_all(tracer.spans)  # a multi-prompt case has one run per prompt: count them all
        metrics = combine(runs) if runs else None  # None when the run never started (e.g. factory crashed)
        return RunReport(
            case.id, repeat, score.passed, score, attribute(result, score), metrics,
            None if result.error is None else f"{type(result.error).__name__}: {result.error}",
            final_text(result)[:300], duration,
        )  # fmt: skip
