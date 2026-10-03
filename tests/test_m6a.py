import asyncio
import contextlib

import pytest

from mini_harness import (
    Agent,
    Compacted,
    ContextConfig,
    JsonlSink,
    Limits,
    Pricing,
    ProviderError,
    RetryPolicy,
    ToolRegistry,
    Tracer,
    TransientToolError,
)
from mini_harness.core.context import ContextManager
from mini_harness.core.messages import Message, Usage
from mini_harness.eval import (
    Recording,
    RecordingProvider,
    ReplayProvider,
    diff_requests,
    fingerprint,
    record_run,
    replay_run,
)
from mini_harness.eval.recording import event_from_dict, event_to_dict, request_to_dict
from mini_harness.observability import (
    NullTracer,
    aggregate,
    default_redact,
    load_trace,
    percentile,
    render_tree,
    summarize,
    summarize_all,
)
from mini_harness.providers.base import (
    CachePlan,
    MessageEnd,
    ModelRequest,
    RetryNotice,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.providers.retry import RetryingProvider
from mini_harness.tools.spec import ToolSpec


def run(coro):
    return asyncio.run(coro)


async def drain(agent, session=None, prompt="hi"):
    return [e async for e in agent.run(prompt, session or agent.new_session())]


def echo_registry(upper=False):
    reg = ToolRegistry()

    @reg.tool()
    def echo(text: str) -> str:
        """echo"""
        return text.upper() if upper else text

    return reg


class Seq:
    """i-th call plays script[i]; exceptions are raised."""

    name = "seq"

    def __init__(self, script):
        self.script, self.calls = script, 0

    async def stream(self, req):
        items = self.script[self.calls]
        self.calls += 1
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item


async def nosleep(_):
    pass


BOOM = ProviderError("overloaded", status=529, retryable=True)
TOOL_SCRIPT = [
    tool_turn(("a", "echo", {"text": "hi"}), usage=Usage(100, 20)),
    text_turn("fin", usage=Usage(150, 5)),
]


def traced_run(script=None, registry=None, **kw):
    tracer = Tracer()
    agent = Agent(FakeProvider(script or TOOL_SCRIPT), registry or echo_registry(), model="m", tracer=tracer, **kw)
    events = run(drain(agent))
    return tracer, events


# ================================================================ tracing
def test_span_tree_structure_and_attributes():
    tracer, _ = traced_run()
    spans = tracer.spans
    kinds = sorted(s.kind for s in spans)
    assert kinds == ["model", "model", "run", "tool", "turn", "turn"]
    run_span = next(s for s in spans if s.kind == "run")
    turns = sorted((s for s in spans if s.kind == "turn"), key=lambda s: s.id)
    assert run_span.parent_id is None and all(t.parent_id == run_span.id for t in turns)
    assert all(s.trace_id == run_span.id for s in spans) and all(s.status == "ok" for s in spans)
    models = sorted((s for s in spans if s.kind == "model"), key=lambda s: s.id)
    assert [m.parent_id for m in models] == [t.id for t in turns]
    assert [m.attrs["stop_reason"] for m in models] == ["tool_use", "end_turn"]
    assert [m.attrs["input_tokens"] for m in models] == [100, 150] and models[0].attrs["provider"] == "fake"
    tool = next(s for s in spans if s.kind == "tool")
    assert tool.parent_id == turns[0].id and tool.name == "echo"
    assert tool.attrs["is_error"] is False and tool.attrs["attempts"] == 1 and "hi" in tool.attrs["args"]
    assert run_span.attrs["reason"] == "completed" and run_span.attrs["turns"] == 2
    assert (run_span.attrs["input_tokens"], run_span.attrs["output_tokens"]) == (250, 25)
    assert all(s.duration_s is not None and s.duration_s >= 0 for s in spans)


def test_tracer_clock_and_status_derivation():
    now = [100.0]
    tracer = Tracer(wall=lambda: 1_700_000_000.0, mono=lambda: now[0])
    s = tracer.start("model", "x")
    now[0] = 100.25
    assert tracer.elapsed(s) == 0.25
    now[0] = 101.0
    tracer.end(s, exc=ValueError("bad"), tokens=3)
    assert s.duration_s == 1.0 and s.status == "error" and s.error == "ValueError: bad" and s.attrs["tokens"] == 3
    c = tracer.start("tool", "y")
    tracer.end(c, exc=asyncio.CancelledError())
    assert c.status == "cancelled" and c.error is None and c.start == 1_700_000_000.0
    assert [x.id for x in tracer.spans] == [s.id, c.id]


def test_model_span_counts_retries_and_resets_ttft():
    provider = RetryingProvider(Seq([[TextDelta("par"), BOOM], text_turn("done")]), sleep=nosleep)
    tracer = Tracer()
    agent = Agent(provider, ToolRegistry(), model="m", tracer=tracer)
    run(drain(agent))
    m = next(s for s in tracer.spans if s.kind == "model")
    assert m.attrs["retries"] == 1 and m.attrs["ttft_s"] is not None and m.status == "ok"


def test_tool_span_records_attempts_and_errors():
    reg, calls = ToolRegistry(), [0]

    @reg.tool(idempotent=True)
    async def flaky() -> str:
        """flaky"""
        calls[0] += 1
        if calls[0] < 3:
            raise TransientToolError("again")
        return "ok"

    @reg.tool()
    def broken() -> str:
        """always fails"""
        raise ValueError("kaboom")

    tracer = Tracer()
    agent = Agent(
        FakeProvider([tool_turn(("a", "flaky", {}), ("b", "broken", {})), text_turn("x")]),
        reg, model="m", tracer=tracer, tool_retry=RetryPolicy(max_attempts=3, base_delay_s=0),
    )  # fmt: skip
    run(drain(agent))
    tools = {s.name: s for s in tracer.spans if s.kind == "tool"}
    assert tools["flaky"].attrs["attempts"] == 3 and tools["flaky"].status == "ok"
    assert tools["broken"].status == "error" and tools["broken"].attrs["is_error"] and "kaboom" in tools["broken"].error


def test_parallel_tool_spans_overlap():
    reg = ToolRegistry()

    @reg.tool()
    async def nap(tag: str) -> str:
        """sleep"""
        await asyncio.sleep(0.15)
        return tag

    tracer = Tracer()
    calls = [(f"t{i}", "nap", {"tag": str(i)}) for i in range(3)]
    agent = Agent(FakeProvider([tool_turn(*calls), text_turn("x")]), reg, model="m", tracer=tracer)
    run(drain(agent))
    tools = [s for s in tracer.spans if s.kind == "tool"]
    assert len(tools) == 3 and max(s.start for s in tools) < min(s.start + s.duration_s for s in tools)


def test_cancellation_ends_every_span():
    reg = ToolRegistry()

    @reg.tool(timeout_s=30)
    async def hang() -> str:
        """hangs"""
        await asyncio.sleep(30)
        return "never"

    tracer = Tracer()
    agent = Agent(FakeProvider([tool_turn(("a", "hang", {}))]), reg, model="m", tracer=tracer)

    async def scenario():
        task = asyncio.create_task(drain(agent))
        await asyncio.sleep(0.15)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    run(scenario())
    by_kind = {s.kind: s for s in tracer.spans}
    assert {k: by_kind[k].status for k in ("tool", "turn", "run")} == {
        "tool": "cancelled",
        "turn": "cancelled",
        "run": "cancelled",
    }
    assert by_kind["run"].attrs["reason"] == "cancelled" and all(s.status != "running" for s in tracer.spans)


def test_provider_failure_marks_model_and_run_as_error():
    tracer = Tracer()
    agent = Agent(Seq([[ProviderError("invalid key", status=401)]]), ToolRegistry(), model="m", tracer=tracer)
    with pytest.raises(ProviderError):
        run(drain(agent))
    by_kind = {s.kind: s for s in tracer.spans}
    assert by_kind["model"].status == "error" and "invalid key" in by_kind["model"].error
    assert by_kind["turn"].status == "error" and by_kind["run"].status == "error"
    assert by_kind["run"].attrs["reason"] == "failed"


class Fixed:
    async def summarize(self, old):
        from mini_harness.core.compaction import make_summary_message

        return make_summary_message("BRIEF", len(old)), Usage(7, 3)


def big_registry():
    reg = ToolRegistry()

    @reg.tool()
    def big(n: int) -> str:
        """3000 chars"""
        return "x" * 3000

    return reg


def big_script():
    return [
        tool_turn((f"t{i}", "big", {"n": i}), usage=Usage(input_tokens=800 * (i + 1), output_tokens=5))
        for i in range(8)
    ] + [text_turn("done")]


def test_compaction_span_only_when_it_happens():
    tracer = Tracer()
    agent = Agent(
        FakeProvider(big_script()), big_registry(), model="m", tracer=tracer, compactor=Fixed(),
        context=ContextConfig(context_window=4000), limits=Limits(max_turns=20),
    )  # fmt: skip
    events = run(drain(agent))
    spans = [s for s in tracer.spans if s.kind == "compaction"]
    assert len(spans) == sum(isinstance(e, Compacted) for e in events) >= 1
    assert spans[0].attrs["tokens_before"] > spans[0].attrs["tokens_after"] and spans[0].parent_id is not None
    quiet, _ = traced_run()  # default window is huge
    assert not [s for s in quiet.spans if s.kind == "compaction"]


def test_redaction_in_previews_and_errors():
    assert default_redact("Authorization: Bearer abcdefghijklmnop") == "Authorization: [REDACTED]"
    assert "sk-abc" not in default_redact("key sk-abcdefghijklmnopqrstuvwx here")
    assert default_redact("api_key=secret123 ok") == "api_key=[REDACTED] ok"
    assert default_redact('{"token": "abc123"}') == '{"token": "[REDACTED]"}'
    assert "AKIA" not in default_redact("AKIAABCDEFGHIJKLMNOP")
    tracer, _ = traced_run([tool_turn(("a", "echo", {"text": "my key sk-abcdefghijklmnopqrstuvwx"})), text_turn("x")])
    tool = next(s for s in tracer.spans if s.kind == "tool")
    assert "[REDACTED]" in tool.attrs["args"] and "sk-abcdefgh" not in tool.attrs["args"]
    t = Tracer()
    s = t.start("tool", "x")
    t.end(s, status="error", error="failed with password=hunter2")
    assert s.error == "failed with password=[REDACTED]"
    assert t.preview("y" * 500).endswith("…[+300]")


def test_jsonl_sink_roundtrip(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(sinks=[JsonlSink(path)])
    agent = Agent(FakeProvider(TOOL_SCRIPT), echo_registry(), model="m", tracer=tracer)
    run(drain(agent))
    assert [s.to_dict() for s in load_trace(path)] == [s.to_dict() for s in tracer.spans]


def test_render_tree_shows_structure():
    tracer, _ = traced_run()
    lines = render_tree(tracer.spans).splitlines()
    assert lines[0].startswith("run") and "reason=completed" in lines[0] and "turns=2" in lines[0]
    assert any("turn 1" in ln and ("├─" in ln or "└─" in ln) for ln in lines)
    tool_line = next(ln for ln in lines if "tool echo" in ln)
    model_line = next(ln for ln in lines if "fake/m" in ln and "stop=tool_use" in ln)
    assert tool_line.startswith("│  ") and model_line.startswith("│  ")  # nested under turn 1
    assert "in=100" in model_line and "attempts" not in tool_line


def test_null_tracer_is_the_zero_cost_default():
    agent = Agent(FakeProvider([text_turn("x")]), ToolRegistry(), model="m")
    run(drain(agent))
    assert isinstance(agent.tracer, NullTracer) and agent.tracer.spans == [] and agent.tracer.preview({"a": 1}) == ""


# ================================================================ metrics
def test_run_metrics_from_spans():
    tracer, _ = traced_run()
    m = summarize(tracer.spans)
    assert (m.turns, m.model_calls, m.tool_calls, m.tool_errors, m.retries, m.compactions) == (2, 2, 1, 0, 0, 0)
    assert (m.input_tokens, m.output_tokens, m.reason, m.status) == (250, 25, "completed", "ok")
    assert m.cache_hit_ratio == 0 and m.tool_error_rate == 0 and m.duration_s >= m.model_time_s >= 0


def test_cache_hit_ratio_and_error_rate():
    reg = ToolRegistry()

    @reg.tool()
    def broken() -> str:
        """fails"""
        raise ValueError("x")

    tracer, _ = traced_run(
        [tool_turn(("a", "broken", {}), usage=Usage(100, 5, 300, 0)), text_turn("x", usage=Usage(100, 5, 300, 0))], reg
    )
    m = summarize(tracer.spans)
    assert m.cache_hit_ratio == pytest.approx(600 / 800) and m.tool_error_rate == 1.0


def test_percentile_and_aggregate_over_runs():
    assert percentile([], 50) is None
    assert (percentile([1, 2, 3, 4, 5], 50), percentile([1, 2, 3, 4, 5], 95), percentile([7], 99)) == (3, 5, 7)
    tracer = Tracer()
    for script in (TOOL_SCRIPT, [tool_turn(("a", "echo", {"text": "x"}))]):
        agent = Agent(
            FakeProvider(script, repeat_last=True),
            echo_registry(),
            model="m",
            tracer=tracer,
            limits=Limits(max_turns=6),
        )
        run(drain(agent))
    runs = summarize_all(tracer.spans)
    assert len(runs) == 2
    agg = aggregate(runs)
    assert agg["runs"] == 2 and agg["completion_rate"] == 0.5 and agg["repeated_calls_rate"] == 0.5
    assert agg["tool_error_rate"] == 0 and agg["duration_p95_s"] >= agg["duration_p50_s"]
    assert aggregate([]) == {"runs": 0}


# ================================================================ recording / replay: codecs
def test_event_and_request_codecs_roundtrip():
    events = [
        TextDelta("你好"),
        ToolCallStart("a", "echo"),
        ToolCallDelta("a", '{"text":'),
        ToolCallEnd("a"),
        MessageEnd("tool_use", Usage(1, 2, 3, 4)),
        RetryNotice(2, 0.5, "boom", True),
    ]
    assert [event_from_dict(event_to_dict(e)) for e in events] == events
    spec = ToolSpec("echo", "d", {"type": "object"}, timeout_s=99, idempotent=True)
    req = ModelRequest("m", "S", [Message.user("q")], [spec], 77, CachePlan(True, (0,)))
    d = request_to_dict(req)
    assert d["tools"] == [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}]  # no timeout etc.
    assert d["cache"] == {"system": True, "message_indices": [0]} and d["messages"][0]["role"] == "user"


def test_diff_requests_pinpoints_changes():
    base = request_to_dict(ModelRequest("m", "S", [Message.user("a"), Message.user("b")], [ToolSpec("t", "d", {})], 10))

    def changed(**kw):
        d = {**base, **kw}
        return d

    assert diff_requests(base, base) == []
    assert "system changed" in diff_requests(base, changed(system="Z"))[0]
    new_tool = {"name": "u", "description": "", "input_schema": {}}
    assert diff_requests(base, changed(tools=base["tools"] + [new_tool])) == ["tool added: u"]
    assert diff_requests(base, changed(tools=[])) == ["tool removed: t"]
    assert diff_requests(base, changed(tools=[{**base["tools"][0], "description": "x"}])) == [
        "tool definition changed: t"
    ]
    msgs = [
        base["messages"][0],
        request_to_dict(ModelRequest("m", "S", [Message.user("CHANGED")], [], 1))["messages"][0],
    ]
    assert "messages[1] differs" in diff_requests(base, changed(messages=msgs))[0]
    assert diff_requests(base, changed(messages=msgs[:1]))[0] == "messages: 2 recorded vs 1 now"
    assert diff_requests(base, changed(system="Z"), ignore={"system"}) == []


# ================================================================ recording / replay: behavior
def factory(**kw):
    registry_upper = kw.pop("upper", False)

    def make(provider):
        return Agent(provider, echo_registry(registry_upper), model="m", **kw)

    return make


SCRIPT = [tool_turn(("a", "echo", {"text": "hello"})), text_turn("final answer")]


def test_record_then_replay_is_identical_and_uses_no_model(tmp_path):
    path = tmp_path / "rec.jsonl"
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it", path=str(path)))
    assert rec.result.error is None and len(rec.recording.calls) == 2
    assert rec.recording.expected[-1][:2] == ["done", "end_turn"]

    loaded = Recording.load(path)  # a different "process": only the file
    assert loaded.meta == rec.recording.meta and loaded.expected == rec.recording.expected
    assert [c.events for c in loaded.calls] == [c.events for c in rec.recording.calls]
    report = run(replay_run(factory(), loaded))
    assert report.ok, report.summary()
    assert (report.calls_used, report.calls_recorded) == (2, 2) and "OK" in report.summary()


def test_replay_detects_prompt_regression():
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it")).recording
    report = run(replay_run(factory(system="A DIFFERENT SYSTEM PROMPT"), rec))
    assert not report.ok and any("system changed" in p for p in report.problems) and report.calls_used == 1


def test_replay_detects_tool_behavior_regression():
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it")).recording
    report = run(replay_run(factory(upper=True), rec))  # echo now shouts
    assert not report.ok
    assert any("HELLO" in p and "tool_result" in p for p in report.problems)  # surfaces in the NEXT prompt
    assert report.calls_used == 2


def test_replay_lenient_mode_collects_all_divergences():
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it")).recording
    report = run(replay_run(factory(system="other"), rec, strict=False))
    assert not report.ok and report.calls_used == 2  # kept going through both calls
    assert sum("system changed" in p for p in report.problems) == 2


def test_replay_reports_missing_and_unused_calls():
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it")).recording
    short = Recording(rec.meta, rec.calls[:1], rec.expected)
    report = run(replay_run(factory(), short))
    assert not report.ok and any("only 1 were recorded" in p for p in report.problems)

    early = run(replay_run(factory(limits=Limits(max_turns=1)), rec))  # harness now stops sooner
    assert not early.ok
    assert any("behavior diverged at step 2" in p and "max_turns" in p for p in early.problems)
    assert any("used 1 of 2 recorded model calls" in p for p in early.problems)


def test_replay_ignore_relaxes_comparison(monkeypatch):
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "do it")).recording
    monkeypatch.setattr(ContextManager, "cache_plan", staticmethod(lambda messages: CachePlan(False, ())))
    strict = run(replay_run(factory(), rec))
    assert not strict.ok and any("cache changed" in p for p in strict.problems)
    assert run(replay_run(factory(), rec, ignore={"cache"})).ok


def test_replay_covers_compaction_and_summarizer_calls():
    class Router:
        name = "router"

        def __init__(self, turns):
            self.turns = list(turns)

        async def stream(self, req):
            events = text_turn("BRIEF") if "You compress an agent" in req.system else self.turns.pop(0)
            for ev in events:
                yield ev

    def make(provider):
        return Agent(
            provider, big_registry(), model="m", context=ContextConfig(context_window=4000), limits=Limits(max_turns=20)
        )

    rec = run(record_run(make, Router(big_script()), "start"))
    events = rec.result.events
    assert any(isinstance(e, Compacted) and e.stage == "summary" for e in events)
    assert any("You compress an agent" in c.request["system"] for c in rec.recording.calls)  # summarizer calls recorded
    report = run(replay_run(make, rec.recording))
    assert report.ok, report.summary()
    assert report.calls_used == len(rec.recording.calls)


def test_replay_multi_prompt_sessions():
    script = [text_turn("a1"), text_turn("a2")]
    rec = run(record_run(factory(), FakeProvider(script), ["q1", "q2"])).recording
    assert rec.meta["prompts"] == ["q1", "q2"] and sum(step[0] == "done" for step in rec.expected) == 2
    assert run(replay_run(factory(), rec)).ok
    wrong = run(replay_run(factory(), rec, prompts=["different", "q2"]))
    assert not wrong.ok and any("messages[0] differs" in p for p in wrong.problems)


def test_failed_runs_are_valid_scenarios():
    provider = Seq([[ProviderError("invalid key", status=401)]])
    rec = run(record_run(factory(), provider, "hi"))
    assert isinstance(rec.result.error, ProviderError) and rec.recording.expected == [["error", "ProviderError"]]
    assert rec.recording.calls[0].events[-1]["t"] == "error"
    assert run(replay_run(factory(), rec.recording)).ok  # the error is reproduced, not just skipped


def test_retries_are_recorded_and_replayed():
    inner = RetryingProvider(Seq([[TextDelta("par"), BOOM], text_turn("complete")]), sleep=nosleep)
    rec = run(record_run(factory(), inner, "hi"))  # recorder is outermost: it sees the RetryNotice
    assert ["retry", 1, True] in rec.recording.expected
    assert any(d["t"] == "retry" for d in rec.recording.calls[0].events)
    assert run(replay_run(factory(), rec.recording)).ok


def test_recording_provider_streams_through_unchanged():
    rp = RecordingProvider(FakeProvider([text_turn("abc")]))
    req = ModelRequest("m", "S", [Message.user("q")], [])

    async def collect():
        return [e async for e in rp.stream(req)]

    assert [e.text for e in run(collect()) if isinstance(e, TextDelta)] == ["a", "bc"]
    assert rp.recording.calls[0].complete and rp.name == "fake"


def test_trace_shape_is_identical_between_recording_and_replay():
    tracers: list[Tracer] = []

    def make(provider):
        t = Tracer()
        tracers.append(t)
        return Agent(provider, echo_registry(), model="m", tracer=t)

    rec = run(record_run(make, FakeProvider(SCRIPT), "do it")).recording
    assert run(replay_run(make, rec)).ok

    def shape(t):
        return [
            (s.kind, s.name, s.parent_id, s.status, s.attrs.get("stop_reason"), s.attrs.get("input_tokens"))
            for s in sorted(t.spans, key=lambda s: s.id)
        ]

    assert shape(tracers[0]) == shape(tracers[1]) and len(tracers[0].spans) == 6


def test_replay_provider_counts_calls():
    rec = run(record_run(factory(), FakeProvider(SCRIPT), "x")).recording
    p = ReplayProvider(rec)
    assert p.calls_used == 0 and fingerprint([]) == []


def test_combine_merges_the_runs_of_one_conversation():
    from mini_harness.observability import combine

    tracer = Tracer()
    pricing = Pricing(10_000, 10_000)  # Usage(10,5) => $0.15 per model call
    agent = Agent(
        FakeProvider([text_turn("a"), text_turn("b"), text_turn("c")]),
        echo_registry(),
        model="m",
        tracer=tracer,
        pricing=pricing,
    )
    session = agent.new_session()
    for prompt in ("q1", "q2", "q3"):
        run(drain(agent, session, prompt))
    runs = summarize_all(tracer.spans)
    assert [r.cost_usd for r in runs] == pytest.approx([0.15, 0.15, 0.15])  # each run reports ITS OWN spend
    m = combine(runs)
    assert (m.turns, m.model_calls, m.input_tokens, m.output_tokens) == (3, 3, 30, 15)
    assert m.cost_usd == pytest.approx(0.45) and m.reason == "completed" and m.status == "ok"
    assert m.duration_s == pytest.approx(sum(r.duration_s for r in runs))
    with pytest.raises(ValueError):
        combine([])
