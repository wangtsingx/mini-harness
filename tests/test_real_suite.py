"""Validates the REAL-MODEL eval suite itself, with no model: tools, fixtures, checkers, and the runner script.

If these pass, a low score from a real model is information about the model/harness, not about a broken test."""

import asyncio
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from evals import run_real
from evals.fixtures import INJECTION, build_workspace
from evals.oracle import injected_script, saboteur_script
from evals.suite import answer_equals, bare_reply, build_cases, parse_answer
from evals.tools import build_eval_registry, format_number, safe_eval
from mini_harness import Agent, ContextConfig
from mini_harness.core.compaction import ExtractiveCompactor
from mini_harness.core.errors import ToolError
from mini_harness.eval import EvalRunner, attribute, drive, wilson_interval
from mini_harness.providers.fake import FakeProvider


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def ws(tmp_path):
    root = tmp_path / "ws"
    return root, build_workspace(root)


def digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def agent_for(case, truth, root, script, tracer=None, window=None):
    ctx = ContextConfig(context_window=case.meta.get("context_window", window or 200_000))
    return Agent(
        FakeProvider(script), build_eval_registry(root), model="m", tracer=tracer, context=ctx,
        compactor=ExtractiveCompactor(),
    )  # fmt: skip


# ---------------------------------------------------------------- fixtures
def test_workspace_is_deterministic_and_seed_sensitive(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    ta, tb, tc = build_workspace(a), build_workspace(b), build_workspace(c, seed=8)
    assert digest(a) == digest(b) and ta == tb
    assert digest(a) != digest(c) and ta != tc


def test_ground_truth_matches_an_independent_recomputation(ws):
    root, t = ws
    rows = list(csv.DictReader((root / "inventory.csv").open()))
    price = {r["id"]: float(r["price"]) for r in rows}
    qty = {r["id"]: int(r["qty"]) for r in rows}
    assert t.widget17_price == next(float(r["price"]) for r in rows if r["name"] == "Widget-17")
    assert t.low_stock_total == sum(q for q in qty.values() if q < 10) and sum(q < 10 for q in qty.values()) >= 3
    top = sorted(price, key=price.get, reverse=True)[:3]
    assert t.top3_ids == tuple(top) and t.top3_total == pytest.approx(sum(price[i] for i in top), abs=1e-9)
    assert len(set(price.values())) == len(price)  # no price ties: "most expensive" is unambiguous

    feb = (root / "orders/2024-02.log").read_text().splitlines()
    assert t.feb_error_count == sum(" ERROR " in ln for ln in feb) and t.feb_error_count == 7
    hits = [p.name for p in (root / "orders").glob("*.log") if "E-4417" in p.read_text()]
    assert hits == ["2024-02.log"]  # the special code is in exactly one file...
    assert sum("E-4417" in ln for ln in feb) == 1  # ...exactly once
    newest = sorted((root / "orders").glob("*.log"))[-1]
    first_err = next(ln for ln in newest.read_text().splitlines() if " ERROR " in ln)
    item = re.search(r"item=(ITEM-\d+)", first_err).group(1)
    assert (t.newest_log_first_error_item, t.newest_log_first_error_item_qty) == (item, qty[item])
    assert t.todo_count == sum("TODO" in ln for f in (root / "src").glob("*.py") for ln in f.read_text().splitlines())
    assert json.loads((root / "config.json").read_text())["max_connections"] == t.max_connections == 487
    big = (root / "big/deploy_log.txt").read_text()
    assert len(big) > 20_000 and t.deploy_log == big  # about 5k tokens: pasted into the context case's 2nd prompt
    assert INJECTION in (root / "notes/injected.txt").read_text()


def test_suite_shape(ws):
    _, t = ws
    cases = build_cases(t)
    assert len(cases) == 16 and len({c.id for c in cases}) == 16
    tags = {tag for c in cases for tag in c.tags}
    assert {"lookup", "aggregate", "search", "multi-hop", "format", "recovery", "calc", "safety", "context"} <= tags
    ctx = next(c for c in cases if c.id == "context_recall")
    assert len(ctx.prompt) == 3 and ctx.meta["context_window"] == 6000


# ---------------------------------------------------------------- tools
@pytest.mark.parametrize(
    ("expr", "want"),
    [("(1234*5678)/(12+8)", "350332.6"), ("17*23", "391"), ("2**10", "1024"), ("-3+10//4", "-1"), ("7 % 4", "3"),
     ("0.1+0.2", "0.3"), ("10/4", "2.5"), ("1_000*3", "3000")],
)  # fmt: skip
def test_calculator_values(expr, want):
    assert format_number(safe_eval(expr)) == want


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('id')", "open('x')", "a+1", "(1).real", "[1]", "'a'*3", "True+1", "2**100000", "1/0",
     "1 +", "-" * 100 + "1", "int('1')"],
)  # fmt: skip
def test_calculator_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ToolError):
        safe_eval(expr)


def test_grep_counts_and_confines_paths(ws):
    root, t = ws
    reg = build_eval_registry(root)
    names = {s.name: s for s in reg.specs()}
    assert {"read_file", "list_dir", "grep", "calculate"} <= names.keys()
    assert all(s.permission.value == "read" and s.concurrency_safe for s in names.values())

    async def call(name, **args):
        tool = reg.get(name)
        return await tool.invoke(tool.validate(args))

    out = run(call("grep", pattern=" ERROR ", path="orders/2024-02.log", max_matches=2))
    assert (
        out.splitlines()[-1] == f"(total matching lines: {t.feb_error_count}; shown: 2)"
    )  # total survives the display cap
    assert len(out.splitlines()) == 3 and out.startswith("orders/2024-02.log:")
    assert "orders/2024-02.log" in run(call("grep", pattern="E-4417", path="orders"))
    assert run(call("grep", pattern="TODO", path="src")).endswith(
        f"(total matching lines: {t.todo_count}; shown: {t.todo_count})"
    )
    assert run(call("grep", pattern="nothing-matches-this")).endswith("(total matching lines: 0; shown: 0)")
    with pytest.raises(ToolError, match="escapes"):
        run(call("grep", pattern="root", path="../.."))
    with pytest.raises(ToolError, match="invalid regular expression"):
        run(call("grep", pattern="("))
    with pytest.raises(PermissionError):
        run(call("read_file", path="../../etc/passwd"))


# ---------------------------------------------------------------- checkers
def test_answer_parsing_and_normalization():
    assert parse_answer("blah\nANSWER: 42") == "42"
    assert parse_answer("ANSWER: 1\nmore\n**ANSWER: 2**") == "2"  # the last one wins, markdown tolerated
    assert parse_answer("no answer here") is None
    ok = lambda exp, text, **kw: answer_equals(exp, **kw)(_result(text)).passed  # noqa: E731
    assert ok(1234.5, "ANSWER: $1,234.50") and ok(1234.5, "ANSWER: 1,234.5.") and ok(7, "ANSWER: `7`")
    assert not ok(33, "ANSWER: 133") and not ok(33, "ANSWER: 3") and not ok(137.47, "ANSWER: 137.48", tol=0.005)
    assert ok(137.47, "ANSWER: 137.47", tol=0.005) and ok("orders/2024-02.log", "ANSWER: `2024-02.log`", endswith=True)
    assert ok(("A", "B"), "ANSWER: b, a") and not ok(("A", "B"), "ANSWER: A")
    assert ok("CANNOT", "ANSWER: cannot")
    s = answer_equals(5)(_result("it is 5"))
    assert not s.passed and s.kind == "format"
    s = answer_equals(5, unsafe_if=("PWNED",))(_result("PWNED\nANSWER: 5"))
    assert not s.passed and s.kind == "unsafe"
    assert answer_equals(5, facts=("x",))(_result("ANSWER: 6")).facts == ("x",)
    assert bare_reply("ITEM-001")(_result("ITEM-001")).passed and bare_reply("ITEM-001")(_result("`ITEM-001`")).passed
    s = bare_reply("ITEM-001")(_result("The answer is ITEM-001."))
    assert not s.passed and s.kind == "format"
    assert bare_reply("ITEM-001")(_result("ITEM-002")).kind == "wrong_answer"


def _result(text):
    from mini_harness.core.messages import Message
    from mini_harness.core.session import Session
    from mini_harness.eval.replay import RunResult

    return RunResult(
        [],
        None,
        Session(messages=[Message.user("q"), Message.assistant([]).__class__("assistant", Message.user(text).content)]),
    )


# ---------------------------------------------------------------- the suite validates itself
def run_suite(ws, scripts, only=None):
    root, t = ws
    cases = [c for c in build_cases(t) if only is None or c.id in only]
    factory = lambda case, tracer: agent_for(case, t, root, scripts(case.id), tracer)  # noqa: E731
    return run(EvalRunner(factory, cases, concurrency=8).run())


def test_every_oracle_trajectory_passes_with_real_tools(ws):
    _, t = ws
    from evals.oracle import oracle_script

    report = run_suite(ws, lambda cid: oracle_script(cid, t))
    failed = {r.case_id: r.score.reason for r in report.runs if not r.passed}
    assert not failed, failed
    assert report.success_rate == 1.0


def test_no_checker_accepts_a_non_answer(ws):
    report = run_suite(ws, saboteur_script)
    assert report.success_rate == 0.0, [r.case_id for r in report.runs if r.passed]
    kinds = {r.case_id: r.score.kind for r in report.runs}
    assert kinds["top_item_bare"] == "wrong_answer" and kinds["price_lookup"] == "wrong_answer"


def test_following_the_injection_is_flagged_unsafe(ws):
    root, t = ws
    case = next(c for c in build_cases(t) if c.id == "injection")
    res = run(drive(agent_for(case, t, root, injected_script(t)), [case.prompt]))
    score = case.check(res)
    assert not score.passed and score.kind == "unsafe"
    attr = attribute(res, score)
    assert attr.key == "model/unsafe_behavior" and attr.confidence == "medium"


def test_tool_errors_appear_exactly_where_the_design_expects(ws):
    from evals.oracle import oracle_script

    report = run_suite(ws, lambda cid: oracle_script(cid, ws[1]))
    errs = {r.case_id: r.metrics.tool_errors for r in report.runs}
    assert errs["recovery_wrong_path"] == 1 and errs["path_escape"] == 1  # recovery & safety cases are meant to err
    assert sum(v for k, v in errs.items() if k not in ("recovery_wrong_path", "path_escape")) == 0


def test_context_case_really_triggers_compaction_on_the_fixture(ws):
    from evals.oracle import oracle_script

    report = run_suite(ws, lambda cid: oracle_script(cid, ws[1]), only={"context_recall", "price_lookup"})
    m = {r.case_id: r.metrics for r in report.runs}
    assert m["context_recall"].compactions >= 1 and m["price_lookup"].compactions == 0
    # all three prompts of the conversation are counted, not just the last one
    assert m["context_recall"].turns == 3 and m["context_recall"].model_calls == 3


def test_prompt_caching_switch_controls_cache_breakpoints(ws):
    root, t = ws
    for caching, expect_none in ((True, False), (False, True)):
        provider = FakeProvider([[]])
        provider = FakeProvider([__import__("mini_harness.providers.fake", fromlist=["x"]).text_turn("hi")])
        a = Agent(provider, build_eval_registry(root), model="m", context=ContextConfig(prompt_caching=caching))
        run(drive(a, ["hello"]))
        assert (provider.requests[0].cache is None) is expect_none


# ---------------------------------------------------------------- statistics
def test_wilson_interval_known_values():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.237, 0.763)
    lo, hi = wilson_interval(10, 10)
    assert round(lo, 3) == 0.722 and hi == 1.0
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and round(hi, 3) == 0.278
    assert (
        wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0]
        < wilson_interval(5, 10)[1] - wilson_interval(5, 10)[0]
    )


def test_significance_note_detects_overlapping_intervals():
    class R:
        def __init__(self, k, n):
            self._ci = wilson_interval(k, n)

        def success_ci(self):
            return self._ci

    base = {"summary": {"runs": 48, "success_rate": 0.70}}
    sig, note = run_real.significance_note(base, R(35, 48))  # 70% vs 73%: noise
    assert not sig and "OVERLAP" in note
    sig, note = run_real.significance_note(base, R(46, 48))  # 70% vs 96%: real
    assert sig and "do not overlap" in note


# ---------------------------------------------------------------- the runner script, end to end (dry run)
def args_for(tmp_path, *extra):
    return run_real.parse_args(["--dry-run", "--repeats", "2", "--out", str(tmp_path / "res"), *extra])


def test_dry_run_writes_complete_results_and_never_claims_model_numbers(tmp_path, capsys):
    code = run(run_real.amain(args_for(tmp_path, "--name", "first")))
    assert code == 0
    out = tmp_path / "res" / "first"
    assert {p.name for p in out.iterdir()} == {"report.json", "report.txt", "runs.jsonl", "traces.jsonl", "summary.md"}
    doc = json.loads((out / "report.json").read_text())
    assert doc["meta"]["dry_run"] is True and doc["summary"]["runs"] == 32 and doc["summary"]["success_rate"] == 1.0
    assert not any("key" in k for k in doc["meta"]["args"])  # credentials never land in the results
    md = (out / "summary.md").read_text()
    assert (
        run_real.DRY_BANNER in md
        and "Not available for a dry run" in md
        and "简历" not in md.split("Using these numbers")[0]
    )
    assert "在 16 个任务" not in md  # no resume sentence for a dry run
    assert run_real.DRY_BANNER in (out / "report.txt").read_text()
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 32 and {"case", "passed", "cause", "duration_s", "input_tokens", "cost_usd"} <= runs[0].keys()
    spans = [json.loads(line) for line in (out / "traces.jsonl").read_text().splitlines()]
    assert {s["kind"] for s in spans} >= {"run", "turn", "model", "tool"} and "case" in spans[0]
    assert "DRY RUN" in capsys.readouterr().out


def test_baseline_comparison_and_regression_exit_code(tmp_path, monkeypatch, capsys):
    run(run_real.amain(args_for(tmp_path, "--name", "base")))
    baseline = str(tmp_path / "res" / "base" / "report.json")
    assert (
        run(run_real.amain(args_for(tmp_path, "--name", "same", "--baseline", baseline, "--fail-on-regression"))) == 0
    )
    md = (tmp_path / "res" / "same" / "summary.md").read_text()
    assert "## Versus baseline" in md and "OK" in md

    # a "harness change" that breaks every answer must be caught, with a non-zero exit for CI
    monkeypatch.setattr(run_real, "oracle_provider", lambda cid, t: FakeProvider(saboteur_script(cid)))
    code = run(run_real.amain(args_for(tmp_path, "--name", "broken", "--baseline", baseline, "--fail-on-regression")))
    assert code == 1
    text = (tmp_path / "res" / "broken" / "summary.md").read_text()
    assert "REGRESSION" in text and "regressed:" in text and "do not overlap" in text
    capsys.readouterr()


def test_selection_listing_and_input_validation(tmp_path, monkeypatch, capsys):
    assert run(run_real.amain(run_real.parse_args(["--list", "--only", "safety"]))) == 0
    listed = capsys.readouterr().out
    assert "injection" in listed and "path_escape" in listed and "price_lookup" not in listed
    assert run(run_real.amain(args_for(tmp_path, "--only", "calc,config_value", "--name", "sel"))) == 0
    assert json.loads((tmp_path / "res" / "sel" / "report.json").read_text())["summary"]["runs"] == 6

    with pytest.raises(SystemExit, match="no cases selected"):
        run(run_real.amain(args_for(tmp_path, "--only", "does-not-exist")))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        run(run_real.amain(run_real.parse_args(["--model", "some-model", "--out", str(tmp_path / "res")])))
    with pytest.raises(SystemExit, match="--model is required"):
        run(run_real.amain(run_real.parse_args(["--out", str(tmp_path / "res")])))


def test_ablation_switches_reach_the_agent(tmp_path, monkeypatch):
    seen = []
    real_agent = run_real.Agent

    def spy(*a, **kw):
        seen.append((kw["context"].prompt_caching, kw["context"].enabled, kw["system"], kw["context"].context_window))
        return real_agent(*a, **kw)

    monkeypatch.setattr(run_real, "Agent", spy)
    run(
        run_real.amain(
            args_for(
                tmp_path,
                "--only",
                "price_lookup,context_recall",
                "--name",
                "abl",
                "--no-cache",
                "--no-compaction",
                "--no-injection-defense",
                "--context-window",
                "99999",
            )
        )
    )
    assert {(c, e) for c, e, *_ in seen} == {(False, False)}
    assert all("never as instructions" not in system for *_, system, _w in seen)
    assert sorted({w for *_, w in seen}) == [6000, 99999]  # per-case window override beats the global one
    prices = run_real.pricing_from(run_real.parse_args(["--price-in", "3", "--price-out", "15"]))
    assert (prices.input_per_mtok, prices.output_per_mtok, prices.cache_read_per_mtok) == (3, 15, 0)
    assert run_real.pricing_from(run_real.parse_args([])) is None
