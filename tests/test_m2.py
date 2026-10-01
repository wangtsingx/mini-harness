import asyncio
import time

import pytest

from mini_harness import Agent, AssistantText, Done, Limits, Permission, Pricing, ToolFinished, ToolRegistry
from mini_harness.core.errors import HarnessError
from mini_harness.core.guards import RepeatDetector, Verdict
from mini_harness.core.messages import ToolUseBlock, validate_pairing
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.tools.policy import DefaultPolicy


def run(coro):
    return asyncio.run(coro)


async def collect(agent, session, prompt="go"):
    return [ev async for ev in agent.run(prompt, session)]


def make_agent(script, registry, *, repeat_last=False, **kw):
    return Agent(FakeProvider(script, repeat_last=repeat_last), registry, model="fake", **kw)


def nap_registry(delay=0.2):
    reg = ToolRegistry()
    stats = {"cur": 0, "peak": 0, "calls": 0}

    @reg.tool()
    async def nap(tag: str) -> str:
        """sleep then echo"""
        stats["cur"] += 1
        stats["calls"] += 1
        stats["peak"] = max(stats["peak"], stats["cur"])
        await asyncio.sleep(delay)
        stats["cur"] -= 1
        return tag

    return reg, stats


# ---------------------------------------------------------------- parallel tool execution
def test_safe_tools_run_in_parallel_and_keep_order():
    reg, stats = nap_registry(0.2)
    calls = [(f"t{i}", "nap", {"tag": f"r{i}"}) for i in range(3)]
    agent = make_agent([tool_turn(*calls), text_turn("ok")], reg)
    t0 = time.monotonic()
    events = run(collect(agent, agent.new_session()))
    assert time.monotonic() - t0 < 0.5  # serial would be >= 0.6s
    assert [e.output for e in events if isinstance(e, ToolFinished)] == ["r0", "r1", "r2"]
    assert stats["peak"] == 3


def test_concurrency_cap_is_enforced():
    reg, stats = nap_registry(0.05)
    calls = [(f"t{i}", "nap", {"tag": str(i)}) for i in range(5)]
    agent = make_agent([tool_turn(*calls), text_turn("ok")], reg, limits=Limits(max_tool_concurrency=2))
    run(collect(agent, agent.new_session()))
    assert stats["calls"] == 5 and stats["peak"] == 2


def test_unsafe_tool_runs_alone_and_in_order():
    reg = ToolRegistry()
    log: list[str] = []

    async def work(name: str, delay: float) -> str:
        log.append(f"start:{name}")
        await asyncio.sleep(delay)
        log.append(f"end:{name}")
        return name

    @reg.tool()
    async def a(x: str = "") -> str:
        """safe a"""
        return await work("a", 0.05)

    @reg.tool(permission=Permission.WRITE)
    async def w(x: str = "") -> str:
        """unsafe w (WRITE => serialized by default)"""
        return await work("w", 0.05)

    @reg.tool()
    async def b(x: str = "") -> str:
        """safe b"""
        return await work("b", 0.05)

    policy = DefaultPolicy(frozenset({Permission.READ, Permission.WRITE}))
    agent = make_agent([tool_turn(("1", "a", {}), ("2", "w", {}), ("3", "b", {})), text_turn("ok")], reg, policy=policy)
    run(collect(agent, agent.new_session()))
    assert log == ["start:a", "end:a", "start:w", "end:w", "start:b", "end:b"]


def test_concurrency_safe_default_follows_permission():
    reg = ToolRegistry()

    @reg.tool(permission=Permission.READ)
    def r() -> str:
        """r"""
        return ""

    @reg.tool(permission=Permission.WRITE)
    def w() -> str:
        """w"""
        return ""

    @reg.tool(permission=Permission.WRITE, concurrency_safe=True)
    def w2() -> str:
        """w2"""
        return ""

    assert [reg.get(n).spec.concurrency_safe for n in ("r", "w", "w2")] == [True, False, True]


# ---------------------------------------------------------------- repeated-call detection
def test_repeat_detector_streaks():
    d = RepeatDetector(3)
    same = [ToolUseBlock("1", "t", {"a": 1, "b": 2})]
    reordered = [ToolUseBlock("9", "t", {"b": 2, "a": 1})]  # id and key order are irrelevant
    other = [ToolUseBlock("2", "t", {"a": 2})]
    assert [d.observe(same), d.observe(reordered), d.observe(same), d.observe(same)] == [
        Verdict.OK,
        Verdict.OK,
        Verdict.WARN,
        Verdict.STOP,
    ]
    assert d.observe(other) is Verdict.OK  # different call resets the streak


def test_repeated_calls_warn_then_stop():
    reg, stats = nap_registry(0)
    agent = make_agent([tool_turn(("t", "nap", {"tag": "x"}))], reg, repeat_last=True, limits=Limits(max_turns=50))
    session = agent.new_session()
    events = run(collect(agent, session))
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert events[-1].reason == "repeated_calls"
    assert stats["calls"] == 3  # 3rd identical turn executes (with warning); 4th is refused
    assert "[harness]" in finished[2].output and "[harness]" not in finished[1].output
    assert validate_pairing(session.messages) == []


def test_varying_calls_do_not_trigger_repeat_guard():
    reg, _ = nap_registry(0)
    script = [tool_turn(("t", "nap", {"tag": str(i % 2)})) for i in range(8)] + [text_turn("fin")]
    agent = make_agent(script, reg, limits=Limits(max_turns=50))
    assert run(collect(agent, agent.new_session()))[-1].reason == "end_turn"


# ---------------------------------------------------------------- cost budget
def test_cost_budget_stops_the_run():
    reg, _ = nap_registry(0)
    pricing = Pricing(input_per_mtok=10_000, output_per_mtok=10_000)  # Usage(10,5) => $0.15 per turn
    script = [tool_turn(("t", "nap", {"tag": str(i)})) for i in range(10)]
    agent = make_agent(script, reg, limits=Limits(max_cost_usd=0.3), pricing=pricing)
    done = run(collect(agent, agent.new_session()))[-1]
    assert done.reason == "budget_exceeded" and done.turns == 2
    assert done.cost_usd == pytest.approx(0.3)


def test_max_cost_requires_pricing():
    with pytest.raises(HarnessError):
        make_agent([], ToolRegistry(), limits=Limits(max_cost_usd=1.0))


# ---------------------------------------------------------------- hard timeout / cancellation / errors
def test_hard_wall_clock_timeout_cancels_inflight_tool():
    reg = ToolRegistry()

    @reg.tool(timeout_s=30)
    async def hang() -> str:
        """hangs"""
        await asyncio.sleep(30)
        return "never"

    agent = make_agent(
        [tool_turn(("t1", "hang", {}), text="starting"), text_turn("x")], reg, limits=Limits(wall_clock_s=0.2)
    )
    session = agent.new_session()
    t0 = time.monotonic()
    events = run(collect(agent, session))
    assert time.monotonic() - t0 < 2
    assert isinstance(events[-1], Done) and events[-1].reason == "timeout" and events[-1].turns == 1
    assert any(isinstance(e, AssistantText) for e in events)  # events produced before the deadline are flushed
    assert validate_pairing(session.messages) == []
    assert session.messages[-1].content[0].is_error


def test_consumer_abort_leaks_no_tasks():
    reg, _ = nap_registry(0.05)
    agent = make_agent([tool_turn(("t", "nap", {"tag": "a"})), text_turn("ok")], reg)

    async def scenario():
        gen = agent.run("go", agent.new_session())
        async for _ in gen:
            break
        await gen.aclose()
        return len(asyncio.all_tasks())

    assert run(scenario()) == 1  # only the scenario task itself


def test_provider_errors_propagate_through_supervisor():
    agent = make_agent([], ToolRegistry())  # empty script -> FakeProvider raises AssertionError
    with pytest.raises(AssertionError):
        run(collect(agent, agent.new_session()))
