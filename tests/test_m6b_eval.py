import asyncio
import itertools
import json

import pytest

from mini_harness import Agent, ContextConfig, HarnessError, Limits, Permission, Pricing, ProviderError, ToolRegistry
from mini_harness.core.compaction import make_summary_message
from mini_harness.core.messages import Usage
from mini_harness.eval import (
    EvalCase,
    EvalRunner,
    ReplayProvider,
    Score,
    all_of,
    attribute,
    contains,
    drive,
    final_text,
    matches,
    no_tool_errors,
    record_run,
    tool_called,
    tools_called,
)
from mini_harness.eval.checks import as_score
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn


def run(coro):
    return asyncio.run(coro)


def registry():
    reg = ToolRegistry()

    @reg.tool()
    def echo(text: str) -> str:
        """echo"""
        return text

    @reg.tool()
    def broken(n: int = 0) -> str:
        """always raises"""
        raise ValueError("kaboom")

    @reg.tool(permission=Permission.EXEC)
    def danger() -> str:
        """needs EXEC"""
        return "ran"

    return reg


class Seq:
    name = "seq"

    def __init__(self, script):
        self.script, self.calls = script, 0

    async def stream(self, req):
        items = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item


class Fixed:
    async def summarize(self, old):
        return make_summary_message("BRIEF", len(old)), Usage(7, 3)


def agent(script, tracer=None, reg=None, repeat_last=False, provider=None, **kw):
    return Agent(
        provider or FakeProvider(script, repeat_last=repeat_last), reg or registry(), model="m", tracer=tracer, **kw
    )


# ---------------------------------------------------------------- checks
def test_checkers_and_score_coercion():
    res = run(drive(agent([tool_turn(("a", "echo", {"text": "x"})), text_turn("Hello World")]), ["go"]))
    assert final_text(res) == "Hello World" and tools_called(res) == ["echo"]
    assert contains("hello", "WORLD")(res).passed and not contains("hello", "mars")(res).passed
    s = contains("hello", "mars")(res)
    assert s.facts == ("hello", "mars") and "mars" in s.reason and not contains("hello", ignore_case=False)(res).passed
    f = matches(r"^\d+$")(res)
    assert not f.passed and f.kind == "format"
    assert tool_called("echo")(res).passed and tool_called("nope")(res).kind == "incomplete"
    assert no_tool_errors()(res).passed
    combo = all_of(contains("hello"), tool_called("nope"), contains("never reached"))(res)
    assert not combo.passed and "nope" in combo.reason and combo.facts == ("hello",)  # stops at first failure
    assert as_score(True) == Score(True) and as_score(False).reason == "check returned False"


# ---------------------------------------------------------------- attribution, one scenario per layer
FACT = "ORDER-7781"
CONTEXT_PROMPTS = [f"Remember {FACT}. " + "a " * 3500, "b " * 3500, "What was the order id?"]


def context_agent(tracer):
    script = [
        text_turn("ack1", usage=Usage(1800, 5)),
        text_turn("ack2", usage=Usage(3500, 5)),
        text_turn("I do not know"),
    ]
    return agent(script, tracer, context=ContextConfig(context_window=4000), compactor=Fixed())


def tool_loop(n=6, tool="echo"):
    return [tool_turn((f"t{i}", tool, {"text": str(i)} if tool == "echo" else {"n": i})) for i in range(n)]


SCENARIOS = {
    # case id -> (builder(tracer) -> Agent, checker, expected "layer/subtype")
    "ok": (lambda t: agent([text_turn("hello world")], t), contains("hello"), None),
    "infra": (
        lambda t: agent([], t, provider=Seq([[ProviderError("invalid key", status=401)]])),
        contains("x"),
        "infra/provider_error",
    ),
    "crash": (
        lambda t: agent([], t, provider=Seq([[RuntimeError("bug")]])),
        contains("x"),
        "harness/unexpected_exception",
    ),
    "harness_err": (
        lambda t: agent([], t, provider=Seq([[HarnessError("invariant")]])),
        contains("x"),
        "harness/harness_error",
    ),
    "eval_bug": (lambda t: agent([text_turn("hi")], t), lambda r: 1 / 0, "eval/checker_crashed"),
    "policy": (
        lambda t: agent([tool_turn(("a", "danger", {})), text_turn("cannot")], t),
        contains("ran"),
        "harness/policy_denied",
    ),
    "guardrail": (
        lambda t: agent(tool_loop(), t, limits=Limits(max_turns=3)),
        contains("done"),
        "harness/guardrail_max_turns",
    ),
    "loop": (
        lambda t: agent([tool_turn(("a", "echo", {"text": "x"}))], t, repeat_last=True, limits=Limits(max_turns=20)),
        contains("done"),
        "model/stuck_in_loop",
    ),
    "tool_runtime": (
        lambda t: agent([tool_turn(("a", "broken", {})), text_turn("sorry")], t),
        contains("42"),
        "tool/runtime_error",
    ),
    "tool_args": (
        lambda t: agent([tool_turn(("a", "echo", {"wrong": 1})), text_turn("sorry")], t),
        contains("42"),
        "tool/invalid_arguments",
    ),
    "tool_guardrail": (
        lambda t: agent(tool_loop(tool="broken"), t, limits=Limits(max_turns=3)),
        contains("done"),
        "tool/runtime_error",
    ),
    "unknown_tool": (
        lambda t: agent([tool_turn(("a", "ghost", {})), text_turn("sorry")], t),
        contains("42"),
        "model/unknown_tool",
    ),
    "lost_context": (context_agent, contains(FACT), "context/lost_facts"),
    "compaction_only": (context_agent, lambda r: False, "context/compaction_suspected"),
    "format": (lambda t: agent([text_turn("The answer is forty-two")], t), matches(r"^\d+$"), "prompt/format"),
    "ambiguous": (lambda t: agent([text_turn("Which file do you mean?")], t), contains("report"), "prompt/ambiguity"),
    "wrong": (lambda t: agent([text_turn("Paris")], t), contains("Berlin"), "model/wrong_answer"),
}


def suite():
    cases = []
    for cid, (_, check, _) in SCENARIOS.items():
        prompt = CONTEXT_PROMPTS if cid in ("lost_context", "compaction_only") else "go"
        cases.append(EvalCase(cid, prompt, check, tags=("ctx",) if "context" in cid else ("core",)))
    return cases


def make_agent(case, tracer):
    return SCENARIOS[case.id][0](tracer)


def test_every_failure_lands_in_the_expected_layer():
    report = run(EvalRunner(make_agent, suite(), concurrency=8).run())
    got = {r.case_id: (r.attribution.key if r.attribution else None) for r in report.runs}
    want = {cid: exp for cid, (_, _, exp) in SCENARIOS.items()}
    assert got == want
    assert report.success_rate == pytest.approx(1 / len(SCENARIOS))


def test_attribution_carries_confidence_evidence_and_a_fix():
    report = run(EvalRunner(make_agent, suite(), concurrency=8).run())
    by = {r.case_id: r.attribution for r in report.runs}
    assert by["tool_runtime"].confidence == "high" and "tool_errors=1" in by["tool_runtime"].evidence
    assert any("kaboom" in e for e in by["tool_runtime"].evidence)
    assert by["lost_context"].confidence == "high" and f"lost: {FACT}" in by["lost_context"].evidence
    assert any(e.startswith("compactions=") and "summary" in e for e in by["lost_context"].evidence)
    assert by["wrong"].confidence == "low" and by["tool_args"].confidence == "medium"
    assert "summarizer" in by["lost_context"].fix and "checker" in by["eval_bug"].fix
    assert by["infra"].reason.startswith("provider failed") and by["crash"].reason.startswith("RuntimeError")


def test_lost_fact_needs_the_fact_to_be_actually_missing():
    # same compaction, but the summary KEEPS the fact -> not a context failure
    class Keeper:
        async def summarize(self, old):
            return make_summary_message(f"BRIEF: {FACT}", len(old)), Usage()

    script = [text_turn("ack1", usage=Usage(1800, 5)), text_turn("ack2", usage=Usage(3500, 5)), text_turn("not sure")]
    a = agent(script, context=ContextConfig(context_window=4000), compactor=Keeper())
    res = run(drive(a, CONTEXT_PROMPTS))
    attr = attribute(res, contains(FACT)(res))
    assert attr.layer == "context" and attr.confidence == "low"  # only "suspected": nothing proves the loss
    assert attribute(res, Score(True)) is None


# ---------------------------------------------------------------- runner
def test_report_aggregates_clusters_tags_and_renders():
    cases = [c for c in suite() if c.id in ("ok", "wrong", "format", "tool_runtime", "tool_args")]
    report = run(EvalRunner(make_agent, cases).run())
    s = report.summary()
    assert (s["cases"], s["runs"], s["cases_fully_passing"]) == (5, 5, 1)
    assert s["success_rate"] == pytest.approx(0.2) and s["turns_avg"] >= 1 and s["tokens_in"] > 0
    assert s["duration_p50_s"] <= s["duration_p95_s"] and s["cost_usd_total"] is None  # no pricing configured
    assert report.by_tag() == {"core": (1, 5)}
    keys = {c.key: c for c in report.clusters()}
    assert set(keys) == {"model/wrong_answer", "prompt/format", "tool/runtime_error", "tool/invalid_arguments"}
    assert report.layer_counts() == {"model": 1, "prompt": 1, "tool": 2}
    text = report.render()
    assert "5 cases x 1 run(s)" in text and "success 20.0%" in text and "FAILURES - 4 cluster(s)" in text
    assert "[tool/runtime_error] x1" in text and "fix:" in text and "cases: format" in text


def test_clusters_group_multiple_cases_and_sort_by_size():
    def factory(case, tracer):
        return agent([text_turn("nope")], tracer)

    cases = [EvalCase(f"c{i}", "go", contains("yes")) for i in range(3)]
    (cluster,) = run(EvalRunner(factory, cases).run()).clusters()
    assert cluster.count == 3 and cluster.case_ids == ("c0", "c1", "c2") and cluster.layer == "model"


def test_repeats_expose_flaky_cases():
    counter = itertools.count()

    def factory(case, tracer):
        return agent([text_turn("good" if next(counter) % 2 == 0 else "bad")], tracer)

    report = run(EvalRunner(factory, [EvalCase("coin", "go", contains("good"))], repeats=4, concurrency=1).run())
    (case,) = report.cases
    assert case.pass_rate == 0.5 and case.flaky and report.flaky_cases == ["coin"]
    assert len(case.runs) == 4 and report.success_rate == 0.5
    assert "FLAKY" in report.render() and "coin (50%)" in report.render()


def test_concurrency_is_bounded():
    live = {"now": 0, "peak": 0}

    class Slow:
        name = "slow"

        async def stream(self, req):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.1)
            live["now"] -= 1
            for ev in text_turn("hi"):
                yield ev

    cases = [EvalCase(f"c{i}", "go", contains("hi")) for i in range(6)]
    report = run(EvalRunner(lambda c, t: agent([], t, provider=Slow()), cases, concurrency=2).run())
    assert live["peak"] == 2 and report.success_rate == 1.0 and 0.25 < report.wall_s < 0.5


def test_runner_validates_its_inputs():
    c = EvalCase("a", "x", contains("x"))
    with pytest.raises(ValueError, match="duplicate"):
        EvalRunner(make_agent, [c, c])
    with pytest.raises(ValueError):
        EvalRunner(make_agent, [c], repeats=0)


def test_a_broken_factory_is_a_failed_run_not_a_crashed_eval():
    def factory(case, tracer):
        raise RuntimeError("cannot build agent")

    (r,) = run(EvalRunner(factory, [EvalCase("a", "x", contains("x"))]).run()).runs
    assert not r.passed and r.metrics is None and r.attribution.layer == "harness" and "cannot build" in r.error


def test_an_erroring_run_never_counts_as_passed_and_async_checkers_work():
    async def always_true(result):
        return True

    case = EvalCase("a", "go", always_true)
    ok = run(EvalRunner(lambda c, t: agent([text_turn("hi")], t), [case]).run()).runs[0]
    assert ok.passed
    bad = run(EvalRunner(lambda c, t: agent([], t, provider=Seq([[ProviderError("down")]])), [case]).run()).runs[0]
    assert not bad.passed and "run raised ProviderError" in bad.score.reason and bad.attribution.layer == "infra"


def test_multi_turn_cases_and_cost_accounting():
    pricing = Pricing(input_per_mtok=10_000, output_per_mtok=10_000)  # Usage(10,5) => $0.15 per model call

    def factory(case, tracer):
        return agent([text_turn("first"), text_turn("second")], tracer, pricing=pricing)

    case = EvalCase("chat", ["q1", "q2"], contains("second"))
    report = run(EvalRunner(factory, [case]).run())
    assert report.success_rate == 1.0 and report.summary()["cost_usd_total"] == pytest.approx(0.30)
    assert "cost $0.3000" in report.render()


# ---------------------------------------------------------------- baseline comparison
def make_report(rates):
    """rates: case -> pass flags per repeat; scripts answer 'good' or 'bad' accordingly."""
    state = {cid: iter(flags) for cid, flags in rates.items()}

    def factory(case, tracer):
        return agent([text_turn("good" if next(state[case.id]) else "bad")], tracer)

    cases = [EvalCase(cid, "go", contains("good")) for cid in rates]
    return run(EvalRunner(factory, cases, repeats=len(next(iter(rates.values()))), concurrency=1).run())


def test_compare_flags_regressions_improvements_and_membership(tmp_path):
    base = make_report({"a": [1, 1], "b": [0, 0], "c": [1, 1], "gone": [1, 1]})
    path = tmp_path / "baseline.json"
    base.save(path)
    assert json.loads(path.read_text())["version"] == 1

    now = make_report({"a": [1, 0], "b": [1, 1], "c": [1, 1], "new": [1, 1]})
    cmp = now.compare(path)
    assert not cmp.ok and cmp.regressions == (("a", 1.0, 0.5),) and cmp.improvements == (("b", 0.0, 1.0),)
    assert cmp.new_cases == ("new",) and cmp.removed_cases == ("gone",)
    assert cmp.success_before == pytest.approx(0.75) and cmp.success_after == pytest.approx(0.875)
    text = cmp.render()
    assert "REGRESSION" in text and "regressed: a  100% -> 50%" in text and "improved:  b" in text

    tolerant = now.compare(base, tolerance=0.5)  # one flaky repeat out of two is tolerated
    assert tolerant.ok and not tolerant.regressions
    assert now.compare(now).ok and "OK" in now.compare(now).render()


def test_compare_reports_cost_and_latency_deltas():
    pricing = Pricing(10_000, 10_000)

    def report(n_calls):
        def factory(case, tracer):
            script = [tool_turn((f"t{i}", "echo", {"text": str(i)})) for i in range(n_calls - 1)] + [text_turn("good")]
            return agent(script, tracer, pricing=pricing)

        return run(EvalRunner(factory, [EvalCase("a", "go", contains("good"))]).run())

    cmp = report(4).compare(report(2))  # twice the model calls => about +100% cost
    assert cmp.cost_change_pct == pytest.approx(100.0) and cmp.p95_change_pct is not None


# ---------------------------------------------------------------- recordings as a free, deterministic eval backend
def test_eval_on_recordings_catches_harness_regressions_without_a_model():
    script = [tool_turn(("a", "echo", {"text": "hello"})), text_turn("final answer")]
    rec = run(record_run(lambda p: Agent(p, registry(), model="m"), FakeProvider(script), "go")).recording

    def factory_with(system):
        return lambda case, tracer: Agent(ReplayProvider(rec), registry(), model="m", system=system, tracer=tracer)

    case = EvalCase("recorded", "go", contains("final answer"))
    same_system = Agent(FakeProvider([]), registry(), model="m")._loop._system  # the default system prompt
    healthy = run(EvalRunner(factory_with(same_system), [case]).run())
    assert healthy.success_rate == 1.0

    drifted = run(EvalRunner(factory_with("A NEW SYSTEM PROMPT"), [case]).run())
    (r,) = drifted.runs
    assert not r.passed and r.attribution.key == "harness/harness_error" and "system changed" in r.attribution.reason
