"""Run the eval suite against a REAL model and write a self-contained, honest results folder.

  ANTHROPIC_API_KEY=... uv run python -m evals.run_real --model <model-id> --repeats 3 \
        --price-in <usd/Mtok> --price-out <usd/Mtok> --price-cache-read <usd/Mtok> --price-cache-write <usd/Mtok>

Prices are NOT built in (they change and differ per model): pass the current ones from your provider's pricing page,
or cost shows as n/a. Nothing in the output is invented: every number comes from the runs, and a --dry-run (scripted
oracle, no API) is stamped as such and never produces resume lines.

A/B ablations (same suite, one switch):  --no-cache   --no-compaction   --no-injection-defense   --system-file F
Compare any run to an earlier one:       --baseline evals/results/<run>/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from evals.fixtures import Truth, build_workspace
from evals.oracle import oracle_provider
from evals.suite import SYSTEM, SYSTEM_NO_DEFENSE, build_cases
from evals.tools import build_eval_registry
from mini_harness import Agent, ContextConfig, Limits, Pricing, Tracer
from mini_harness.core.compaction import ExtractiveCompactor
from mini_harness.eval import EvalCase, EvalReport, EvalRunner, wilson_interval
from mini_harness.providers.base import Provider
from mini_harness.providers.retry import RetryingProvider

DRY_BANNER = "DRY RUN: scripted oracle answers, NOT a model evaluation. These numbers say nothing about any model."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", help="model id (required unless --dry-run/--list)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint")
    ap.add_argument("--name", help="label for this run (default: model + timestamp)")
    ap.add_argument("--out", default="evals/results", help="parent directory for results")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", help="comma-separated case ids and/or tags")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=180.0, help="wall-clock seconds per run")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--context-window", type=int, default=200_000, help="your model's real window")
    for p in ("in", "out", "cache-read", "cache-write"):
        ap.add_argument(f"--price-{p}", type=float, help=f"USD per million {p.replace('-', ' ')} tokens")
    ap.add_argument("--no-cache", action="store_true", help="ablation: no prompt-cache breakpoints")
    ap.add_argument("--no-compaction", action="store_true", help="ablation: never compact the context")
    ap.add_argument("--no-injection-defense", action="store_true", help="ablation: drop the data-not-instructions line")
    ap.add_argument("--system-file", help="use this file as the system prompt")
    ap.add_argument("--baseline", help="report.json of an earlier run to compare against")
    ap.add_argument("--tolerance", type=float, default=0.34, help="per-case pass-rate drop tolerated as noise")
    ap.add_argument("--fail-on-regression", action="store_true")
    ap.add_argument("--workspace", help="where to build the fixture workspace (default: a temp dir)")
    ap.add_argument("--dry-run", action="store_true", help="no API: scripted oracle model (validates the pipeline)")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    return ap.parse_args(argv)


def select(cases: list[EvalCase], only: str | None) -> list[EvalCase]:
    if not only:
        return cases
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    return [c for c in cases if c.id in wanted or wanted & set(c.tags)]


def make_provider(args: argparse.Namespace) -> tuple[Provider, Any]:
    if args.provider == "openai":
        from mini_harness.providers.openai_compat import DEFAULT_BASE_URL, OpenAICompatProvider

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY is not set")
        inner = OpenAICompatProvider(key, base_url=args.base_url or DEFAULT_BASE_URL)
    else:
        from mini_harness.providers.anthropic import AnthropicProvider

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ANTHROPIC_API_KEY is not set (use --dry-run to exercise the pipeline without a key)")
        inner = AnthropicProvider(key)
    return RetryingProvider(inner), inner


def pricing_from(args: argparse.Namespace) -> Pricing | None:
    if args.price_in is None or args.price_out is None:
        return None
    return Pricing(args.price_in, args.price_out, args.price_cache_read or 0.0, args.price_cache_write or 0.0)


def system_prompt(args: argparse.Namespace) -> str:
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8")
    return SYSTEM_NO_DEFENSE if args.no_injection_defense else SYSTEM


# ---------------------------------------------------------------- output
def run_record(r: Any) -> dict[str, Any]:
    m = r.metrics
    return {
        "case": r.case_id, "repeat": r.repeat, "passed": r.passed, "reason": r.score.reason, "kind": r.score.kind,
        "cause": r.attribution.key if r.attribution else None,
        "confidence": r.attribution.confidence if r.attribution else None,
        "error": r.error, "final": r.final, "duration_s": round(r.duration_s, 3),
        "turns": m.turns if m else None, "tool_calls": m.tool_calls if m else None,
        "tool_errors": m.tool_errors if m else None, "compactions": m.compactions if m else None,
        "input_tokens": m.input_tokens if m else None, "output_tokens": m.output_tokens if m else None,
        "cache_read_tokens": m.cache_read_tokens if m else None, "cost_usd": m.cost_usd if m else None,
    }  # fmt: skip


def significance_note(base: dict[str, Any], report: EvalReport) -> tuple[bool, str]:
    n0 = base["summary"]["runs"]
    lo0, hi0 = wilson_interval(round(base["summary"]["success_rate"] * n0), n0)
    lo1, hi1 = report.success_ci()
    overlap = not (lo1 > hi0 or lo0 > hi1)
    note = (
        f"baseline CI95 [{lo0:.0%}, {hi0:.0%}] vs now [{lo1:.0%}, {hi1:.0%}]: "
        + ("the intervals OVERLAP, so the difference may be noise; don't claim an improvement from this alone."
           if overlap else "the intervals do not overlap: the difference is unlikely to be noise.")
    )  # fmt: skip
    return (not overlap), note


def summary_markdown(
    report: EvalReport, args: argparse.Namespace, meta: dict[str, Any], cmp: Any, sig: tuple[bool, str] | None
) -> str:
    s = report.summary()
    lo, hi = s["success_ci95"]
    dry = args.dry_run
    L = [f"# Eval run: {meta['name']}", ""]
    if dry:
        L += [f"> **{DRY_BANNER}**", ""]
    L += [
        f"- date (UTC): {meta['date']}  |  model: `{meta['model']}` ({meta['provider']})  |  harness: mini-harness {meta['harness']}",
        f"- {s['cases']} cases x {report.repeats} repeats = {s['runs']} runs, concurrency {args.concurrency}, max_turns {args.max_turns}",
        f"- switches: cache={'off' if args.no_cache else 'on'}, compaction={'off' if args.no_compaction else 'on'}, "
        f"injection_defense={'off' if args.no_injection_defense else 'on'}, context_window={args.context_window}",
        "", "## Headline", "",
        f"- success rate: **{s['success_rate']:.1%}** (95% CI {lo:.0%}-{hi:.0%}, {sum(r.passed for r in report.runs)}/{s['runs']} runs); "
        f"cases fully passing {s['cases_fully_passing']}/{s['cases']}; flaky: {', '.join(s['flaky_cases']) or 'none'}",
        f"- latency per run: p50 {s['duration_p50_s']:.1f}s, p95 {s['duration_p95_s']:.1f}s; avg turns {s['turns_avg']:.1f}",
        f"- tokens in/out: {s['tokens_in']}/{s['tokens_out']}; cache hit ratio {s['cache_hit_ratio']:.0%}; tool error rate {s['tool_error_rate']:.0%}",
    ]  # fmt: skip
    cost = s["cost_usd_total"]
    L.append(
        f"- cost: ${cost:.4f} total, ${cost / s['runs']:.4f} per run"
        if cost is not None
        else "- cost: n/a (pass --price-in/--price-out)"
    )
    tags = report.by_tag()
    L += ["", "## By tag", ""] + [f"- {t}: {p}/{n} ({p / n:.0%})" for t, (p, n) in tags.items()]
    comp_runs = [r for r in report.runs if r.metrics and r.metrics.compactions]
    ctx_runs = [r for r in report.runs if "context" in next(c.tags for c in report.cases if c.id == r.case_id)]
    L += ["", "## Context compaction", "",
          f"- runs with at least one compaction: {len(comp_runs)}/{len(report.runs)}; "
          f"context-recall runs passed: {sum(r.passed for r in ctx_runs)}/{len(ctx_runs)}"]  # fmt: skip
    clusters = report.clusters()
    L += ["", "## Failures by likely cause", ""]
    L += [
        f"- `{c.key}` x{c.count} ({c.confidence}): {', '.join(c.case_ids)} - {c.sample_reason}" for c in clusters
    ] or ["- none"]
    if cmp is not None:
        L += ["", "## Versus baseline", "", "```", cmp.render(), "```"]
        if sig:
            L.append(f"\n{sig[1]}")
    L += ["", "## Using these numbers on a resume", ""]
    if dry:
        L.append("Not available for a dry run.")
    else:
        L.append(
            f"- 在 {s['cases']} 个任务 x {report.repeats} 次重复（共 {s['runs']} 次运行）的自建评测集上，`{meta['model']}` + 自研 Harness 的成功率为 "
            f"{s['success_rate']:.0%}（95% 置信区间 {lo:.0%}-{hi:.0%}）；p95 延迟 {s['duration_p95_s']:.1f}s"
            + (f"；每次运行平均成本 ${cost / s['runs']:.4f}" if cost is not None else "")
        )
        if cmp is not None and sig is not None:
            if sig[0] and cmp.success_after > cmp.success_before:
                L.append(f"- 相对基线，成功率 {cmp.success_before:.0%} -> {cmp.success_after:.0%}（置信区间不重叠）")
            if cmp.cost_change_pct is not None:
                L.append(f"- 相对基线，成本变化 {cmp.cost_change_pct:+.1f}%，p95 延迟变化 "
                         f"{cmp.p95_change_pct:+.1f}%" if cmp.p95_change_pct is not None else f"- 相对基线，成本变化 {cmp.cost_change_pct:+.1f}%")  # fmt: skip
        L.append("- 样本量很小（几十次运行）：只引用上面有置信区间支撑的结论，并如实写明评测集是自建的。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
async def amain(args: argparse.Namespace) -> int:
    ws = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="harness-eval-ws-"))
    truth: Truth = build_workspace(ws)
    cases = select(build_cases(truth), args.only)
    if args.list:
        for c in cases:
            print(f"{c.id:<22} {','.join(c.tags):<12} turns={1 if isinstance(c.prompt, str) else len(c.prompt)}")
        return 0
    if not cases:
        sys.exit("no cases selected")
    if not args.dry_run and not args.model:
        sys.exit("--model is required (or use --dry-run)")

    provider: Provider | None = None
    inner: Any = None
    if not args.dry_run:
        provider, inner = make_provider(args)
    system = system_prompt(args)
    pricing = pricing_from(args)
    tracers: list[tuple[str, Tracer]] = []

    def make_agent(case: EvalCase, tracer: Tracer) -> Agent:
        tracers.append((case.id, tracer))
        ctx = ContextConfig(
            context_window=case.meta.get("context_window", args.context_window),
            enabled=not args.no_compaction, prompt_caching=not args.no_cache,
        )  # fmt: skip
        prov = oracle_provider(case.id, truth) if args.dry_run else provider
        return Agent(
            prov, build_eval_registry(ws), model=args.model or "oracle", system=system, tracer=tracer, pricing=pricing,
            limits=Limits(max_turns=args.max_turns, wall_clock_s=args.timeout), context=ctx, max_tokens=args.max_tokens,
            compactor=ExtractiveCompactor() if args.dry_run else None,
        )  # fmt: skip

    try:
        report = await EvalRunner(make_agent, cases, concurrency=args.concurrency, repeats=args.repeats).run()
    finally:
        if inner is not None:
            await inner.aclose()

    stamp = datetime.now(UTC)
    name = args.name or f"{'dryrun' if args.dry_run else (args.model or 'run')}-{stamp:%Y%m%d-%H%M%S}"
    out = Path(args.out) / name.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "date": stamp.isoformat(timespec="seconds"), "model": args.model or "oracle",
            "provider": "scripted" if args.dry_run else args.provider, "harness": metadata.version("mini-harness")}  # fmt: skip

    cmp = sig = None
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        cmp = report.compare(base, tolerance=args.tolerance)
        sig = significance_note(base, report)

    doc = report.to_dict()
    doc["meta"] = {**meta, "dry_run": args.dry_run, "args": {k: v for k, v in vars(args).items() if "key" not in k}}
    (out / "report.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    text = (f"{DRY_BANNER}\n\n" if args.dry_run else "") + report.render()
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "runs.jsonl").write_text(
        "\n".join(json.dumps(run_record(r), ensure_ascii=False) for r in report.runs) + "\n", encoding="utf-8"
    )
    with (out / "traces.jsonl").open("w", encoding="utf-8") as f:
        for case_id, tr in tracers:
            for span in tr.spans:
                f.write(json.dumps({"case": case_id, **span.to_dict()}, ensure_ascii=False) + "\n")
    (out / "summary.md").write_text(summary_markdown(report, args, meta, cmp, sig), encoding="utf-8")

    print(text)
    if cmp is not None:
        print("\nvs baseline:\n" + cmp.render() + (f"\n{sig[1]}" if sig else ""))
    print(f"\nresults written to {out}/")
    return 1 if (args.fail_on_regression and cmp is not None and not cmp.ok) else 0


def main() -> None:
    sys.exit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()
