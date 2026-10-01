import asyncio
import json

import pytest

from mini_harness import Agent, Compacted, ContextConfig, Done, Limits, ProviderError, ToolRegistry
from mini_harness.core.compaction import (
    ExtractiveCompactor,
    LLMSummarizer,
    find_cut,
    is_summary,
    make_summary_message,
    render_transcript,
    trim_tool_results,
)
from mini_harness.core.context import ContextManager
from mini_harness.core.messages import Message, ToolResultBlock, ToolUseBlock, Usage, validate_pairing
from mini_harness.core.session import Session
from mini_harness.core.tokens import estimate_message, estimate_text
from mini_harness.providers.anthropic import _to_payload as anthropic_payload
from mini_harness.providers.base import CachePlan, ModelRequest
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.providers.openai_compat import _to_payload as openai_payload


def run(coro):
    return asyncio.run(coro)


def pair(i, size, name="big"):
    """assistant tool_use + tool result with `size` chars of output."""
    return [
        Message.assistant([ToolUseBlock(f"t{i}", name, {"n": i})]),
        Message.tool_results([ToolResultBlock(f"t{i}", "x" * size)]),
    ]


def tool_session(pairs=6, size=3000):
    msgs = [Message.user("do the big job")]
    for i in range(pairs):
        msgs += pair(i, size)
    return Session(messages=msgs)


# 4000-token window: trigger at 3200, target 2000, protected tail 1000 tokens
CFG = ContextConfig(context_window=4000)


class Boom:
    async def summarize(self, old):
        raise AssertionError("summarizer must not be called")


class Fixed:
    def __init__(self, text="SUMMARY", usage=None):
        self.text, self.usage, self.seen = text, usage or Usage(7, 3), []

    async def summarize(self, old):
        self.seen.append(list(old))
        return make_summary_message(self.text, len(old)), self.usage


# ---------------------------------------------------------------- estimation
def test_token_estimate_orders_sensibly():
    assert estimate_text("") == 0
    assert estimate_text("a" * 400) == 100
    assert estimate_text("你好世界" * 100) >= 400  # CJK is ~1 token per char
    big = Message.assistant([ToolUseBlock("1", "t", {"data": "y" * 400})])
    assert estimate_message(big) > estimate_message(Message.user("hi"))


def test_estimate_prefers_provider_reported_usage():
    s = Session(messages=[Message.user("a" * 4000), Message.user("b" * 400), Message.user("c" * 40)])
    cm = ContextManager(CFG)
    pure = cm.estimate(s, "", [])
    s.last_prompt_tokens, s.last_prompt_msgs = 5000, 2
    assert cm.estimate(s, "", []) == 5000 + estimate_message(s.messages[2])
    assert cm.estimate(s, "", []) != pure


def test_config_validation():
    with pytest.raises(ValueError):
        ContextConfig(target_ratio=0.9)
    with pytest.raises(ValueError):
        ContextConfig(trim_min_chars=100, trim_preview_chars=100)


# ---------------------------------------------------------------- segmentation
def test_cut_never_orphans_tool_pairs_and_keeps_last_unit():
    s = tool_session(pairs=5, size=400)
    for budget in (0, 50, 120, 500, 5000):
        cut = find_cut(s.messages, budget)
        if cut:
            assert s.messages[cut].role != "tool"
            assert validate_pairing(s.messages[:cut]) == [] and validate_pairing(s.messages[cut:]) == []
        assert cut <= max(i for i, m in enumerate(s.messages) if m.role != "tool")
    assert find_cut(s.messages, 10**9) == 0  # everything fits -> nothing to compact


def test_trim_marks_output_and_is_idempotent():
    s = tool_session(pairs=3, size=3000)
    cut = len(s.messages) - 2
    once = trim_tool_results(s.messages, cut, min_chars=1000, preview_chars=200)
    first = once[2].content[0].content
    assert first.startswith("x" * 200) and "trimmed 2800 chars of 'big' output" in first
    assert once[-1] is s.messages[-1]  # protected tail untouched
    twice = trim_tool_results(once, cut, min_chars=1000, preview_chars=200)
    assert all(a is b for a, b in zip(once, twice, strict=True))


# ---------------------------------------------------------------- ContextManager
def test_below_threshold_does_nothing():
    s = tool_session(pairs=1, size=100)
    assert run(ContextManager(CFG, Boom()).prepare(s, "", [])) is None
    assert s.compactions == 0


def test_stage1_trim_is_enough_and_archives_originals():
    s = tool_session(pairs=6, size=3000)
    original = list(s.messages)
    c = run(ContextManager(CFG, Boom()).prepare(s, "", []))  # Boom: no model call allowed
    assert c.stage == "trim" and c.tokens_after <= 2000 < c.tokens_before and c.usage == Usage()
    assert len(s.messages) == len(original) and validate_pairing(s.messages) == []
    assert "trimmed" in s.messages[2].content[0].content  # old result shrunk
    assert s.messages[-1].content[0].content == "x" * 3000  # recent result intact
    assert c.archived == len(s.archive) > 0 and all(a in original for a in s.archive)
    assert run(ContextManager(CFG, Boom()).prepare(s, "", [])) is None  # stable afterwards (no churn)


def long_text_session(n=8, size=3000):
    msgs = []
    for i in range(n):
        msgs.append(Message.user(f"question {i} " + "q" * size))
        msgs.append(Message.assistant([]) if False else Message("assistant", Message.user("a" * size).content))
    msgs.append(Message.user("the final question"))
    return Session(messages=msgs)


def test_stage2_summarizes_old_segment():
    s = long_text_session()
    comp = Fixed()
    c = run(ContextManager(CFG, comp).prepare(s, "", []))
    assert c.stage == "summary" and c.usage == Usage(7, 3) and c.tokens_after < c.tokens_before
    assert is_summary(s.messages[0]) and sum(is_summary(m) for m in s.messages) == 1
    assert s.messages[-1].text() == "the final question"  # newest request survives
    assert c.archived == len(comp.seen[0]) == len(s.archive) and s.compactions == 1
    assert validate_pairing(s.messages) == []


def test_summary_failure_falls_back_to_extractive():
    class Down:
        async def summarize(self, old):
            raise ProviderError("down", retryable=False)

    s = long_text_session()
    c = run(ContextManager(CFG, Down()).prepare(s, "", []))
    assert c.stage == "extractive" and is_summary(s.messages[0])
    assert "User requests" in s.messages[0].text() and "question" in s.messages[0].text()


def test_rolling_summary_replaces_instead_of_stacking():
    s = long_text_session()
    run(ContextManager(CFG, Fixed("FIRST")).prepare(s, "", []))
    for i in range(8):  # conversation keeps going
        s.messages += [Message.user(f"more {i} " + "m" * 3000), Message("assistant", Message.user("r" * 3000).content)]
    comp = Fixed("SECOND")
    run(ContextManager(CFG, comp).prepare(s, "", []))
    assert sum(is_summary(m) for m in s.messages) == 1 and "SECOND" in s.messages[0].text()
    assert is_summary(comp.seen[0][0]) and "FIRST" in comp.seen[0][0].text()  # old summary was folded in


def test_nothing_to_compact_when_single_huge_unit():
    s = Session(messages=[Message.user("z" * 20000)])
    assert run(ContextManager(CFG, Fixed()).prepare(s, "", [])) is None


def test_disabled_config_never_compacts():
    s = tool_session()
    assert run(ContextManager(ContextConfig(context_window=4000, enabled=False), Boom()).prepare(s, "", [])) is None


# ---------------------------------------------------------------- summarizers
def test_llm_summarizer_request_shape_and_result():
    provider = FakeProvider([text_turn("## Goal\nship it")])
    s = tool_session(pairs=2, size=3000)
    msg, usage = run(LLMSummarizer(provider, "m", max_tokens=321).summarize(s.messages))
    req = provider.requests[0]
    assert req.tools == [] and req.max_tokens == 321 and "never call tools" in req.system
    prompt = req.messages[0].text()
    assert "<transcript>" in prompt and "[tool_call big]" in prompt and "[tool_result big]" in prompt
    assert "+1500 chars" in prompt  # long tool output is clipped in the transcript
    assert is_summary(msg) and "ship it" in msg.text() and usage == Usage(10, 5)


def test_llm_summarizer_rejects_empty_output():
    from mini_harness.core.errors import HarnessError

    with pytest.raises(HarnessError):
        run(LLMSummarizer(FakeProvider([text_turn(" ")]), "m").summarize([Message.user("x")]))


def test_transcript_keeps_head_and_tail_when_oversized():
    msgs = [Message.user("FIRST-GOAL")] + [Message.user(f"filler {i} " + "f" * 1400) for i in range(200)]
    msgs.append(Message.user("LAST-NOTE"))
    t = render_transcript(msgs, max_chars=20_000)
    assert len(t) < 21_000 and "FIRST-GOAL" in t and "LAST-NOTE" in t and "omitted" in t


def test_extractive_compactor_is_deterministic_and_mentions_tools():
    s = tool_session(pairs=2, size=50)
    msg, usage = run(ExtractiveCompactor().summarize(s.messages))
    assert "big x2" in msg.text() and usage == Usage()


# ---------------------------------------------------------------- prompt caching
MSGS = [
    Message.user("q1"),
    Message.assistant([ToolUseBlock("c1", "t", {})]),
    Message.tool_results([ToolResultBlock("c1", "r")]),
]


def test_anthropic_cache_breakpoints():
    req = ModelRequest("m", "SYS", MSGS, [], cache=CachePlan(system=True, message_indices=(0, 2)))
    p = anthropic_payload(req)
    assert p["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    marked = [i for i, m in enumerate(p["messages"]) if "cache_control" in m["content"][-1]]
    assert marked == [0, 2]
    assert anthropic_payload(ModelRequest("m", "SYS", MSGS, []))["system"] == "SYS"  # no plan -> unchanged


def test_anthropic_never_exceeds_four_breakpoints():
    req = ModelRequest("m", "S", MSGS * 3, [], cache=CachePlan(True, tuple(range(9))))
    p = anthropic_payload(req)
    n = sum("cache_control" in m["content"][-1] for m in p["messages"]) + isinstance(p["system"], list)
    assert n == 4


def test_openai_ignores_cache_plan():
    req = ModelRequest("m", "SYS", MSGS, [], cache=CachePlan(True, (0, 2)))
    assert "cache_control" not in json.dumps(openai_payload(req, "max_tokens"))


def test_cache_plan_includes_summary_and_newest():
    plain = ContextManager.cache_plan(MSGS)
    assert plain.system and plain.message_indices == (2,)
    with_summary = ContextManager.cache_plan([make_summary_message("s", 3), *MSGS])
    assert with_summary.message_indices == (0, 3)


def strip_cache(x):
    if isinstance(x, dict):
        return {k: strip_cache(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [strip_cache(v) for v in x]
    return x


def test_prefix_is_byte_stable_between_turns():
    reg = ToolRegistry()

    @reg.tool()
    def echo(text: str) -> str:
        """echo"""
        return text

    provider = FakeProvider(
        [tool_turn(("a", "echo", {"text": "1"})), tool_turn(("b", "echo", {"text": "2"})), text_turn("ok")]
    )
    agent = Agent(provider, reg, model="m", system="SYS")

    async def go():
        return [e async for e in agent.run("hi", agent.new_session())]

    run(go())
    payloads = [strip_cache(anthropic_payload(r)) for r in provider.requests]
    for earlier, later in zip(payloads, payloads[1:], strict=False):
        assert earlier["system"] == later["system"] and earlier["tools"] == later["tools"]
        n = len(earlier["messages"])
        assert json.dumps(earlier["messages"]) == json.dumps(later["messages"][:n])  # append-only history
    assert provider.requests[-1].cache == CachePlan(True, (len(provider.requests[-1].messages) - 1,))


# ---------------------------------------------------------------- end to end through the loop
def test_loop_compacts_midrun_and_continues():
    reg = ToolRegistry()

    @reg.tool()
    def big(n: int) -> str:
        """returns 3000 chars"""
        return "x" * 3000

    script = [
        tool_turn((f"t{i}", "big", {"n": i}), usage=Usage(input_tokens=800 * (i + 1), output_tokens=5))
        for i in range(8)
    ] + [text_turn("all done")]
    provider = FakeProvider(script)
    comp = Fixed("BRIEFING")
    agent = Agent(provider, reg, model="m", context=CFG, compactor=comp, limits=Limits(max_turns=20))
    session = agent.new_session()

    async def go():
        return [e async for e in agent.run("start", session)]

    events = run(go())
    compacted = [e for e in events if isinstance(e, Compacted)]
    assert compacted and events[-1].reason == "end_turn" and isinstance(events[-1], Done)
    assert validate_pairing(session.messages) == [] and session.archive and session.compactions >= 1
    # the request right after compaction is smaller than the one before it
    sizes = [sum(estimate_message(m) for m in r.messages) for r in provider.requests]
    assert min(sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)) < 0
    # first message the model sees after a summary-stage compaction is the briefing
    if any(c.stage == "summary" for c in compacted):
        assert any(is_summary(r.messages[0]) for r in provider.requests)
    # summarizer spend is added to session usage on top of the reported model usage
    reported = sum(800 * (i + 1) for i in range(8)) + 10
    assert session.usage.input_tokens == reported + 7 * len(comp.seen)


def test_loop_records_provider_reported_prompt_size():
    provider = FakeProvider([text_turn("hi", usage=Usage(input_tokens=123, output_tokens=4, cache_read_tokens=1000))])
    agent = Agent(provider, ToolRegistry(), model="m")
    session = agent.new_session()

    async def go():
        return [e async for e in agent.run("hello", session)]

    run(go())
    assert session.last_prompt_tokens == 1123 and session.last_prompt_msgs == 1
