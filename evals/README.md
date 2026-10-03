# Real-model eval

16 read-only workspace tasks (lookup, aggregation, search, multi-hop, format, error recovery, calculator, prompt-injection,
path-escape, long-context recall) with answers **derived from the generated files**, scored on a final `ANSWER:` line.
Everything below runs on your machine with your key; nothing here has been run against a real model yet.

## 0. Sanity check (no key, no cost)
```bash
uv run pytest tests/test_real_suite.py -q          # validates tools, fixtures, checkers (oracles pass, saboteurs fail)
uv run python -m evals.run_real --dry-run          # exercises the whole pipeline with a scripted oracle (stamped DRY RUN)
uv run python -m evals.run_real --list
```

## 1. First real run
```bash
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY with --provider openai [--base-url ...]
uv run python -m evals.run_real --model <model-id> --repeats 3 --concurrency 4 \
    --context-window <the model's real window> \
    --price-in <usd per Mtok> --price-out <usd per Mtok> --price-cache-read <..> --price-cache-write <..>
```
Prices are not built in (they change per model): copy them from your provider's pricing page, or cost shows as n/a.
Rough size: 16 cases x 3 repeats = 48 runs, a few model calls each. Try `--only lookup --repeats 1` first.

Output goes to `evals/results/<name>/`:
`summary.md` (read this) - `report.txt` - `report.json` (baseline format) - `runs.jsonl` (per run) - `traces.jsonl` (spans)

## 2. How to get an honest "before -> after"
Change ONE thing, run the same suite, compare. Built-in switches (A/B on a real model):

| Question | Baseline | Variant |
|---|---|---|
| What does prompt caching save? | `--no-cache --name nocache` | `--name cache --baseline evals/results/nocache/report.json` |
| Does compaction preserve what matters? | `--no-compaction` (use a small `--context-window`) | default |
| Is the injection line doing anything? | `--no-injection-defense` | default |
| Did my prompt change help? | `--system-file old.txt` | `--system-file new.txt` |
| Did a harness change regress anything? | last good `report.json` | `--baseline ... --fail-on-regression` (exit 1) |

`summary.md` prints a 95% Wilson interval and tells you when baseline and variant intervals overlap. With ~48 runs the
interval is roughly +/-12 points: **a 5-point difference is noise**. Cost and latency differences from caching are usually
large enough to be real; success-rate differences usually are not unless you add cases or repeats.

## 3. What you may and may not claim
- OK: "On a self-built 16-task suite (3 repeats, N runs), model X + this harness scored S% (95% CI a-b), p95 latency L s, $C per run."
- OK, if the intervals do not overlap: "success rose from A% to B%" - and say which single change caused it.
- Not OK: presenting dry-run numbers, quoting a delta inside the noise, or calling a self-built suite a benchmark.
`summary.md` generates the resume sentence from the actual numbers and refuses to for a dry run.

## 4. Reading failures
`report.txt` clusters failures by likely layer (model / prompt / tool / context / harness / infra). It is triage with
evidence, not proof: for `model` vs `prompt`, change the prompt and re-run. `runs.jsonl` has every answer preview.
