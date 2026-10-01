import asyncio
import contextlib

from mini_harness import Agent, AssistantText, Done, Limits, Permission, ToolFinished, ToolRegistry
from mini_harness.core.messages import (
    Message,
    ToolResultBlock,
    ToolUseBlock,
    close_dangling_tool_uses,
    validate_pairing,
)
from mini_harness.providers.base import StreamAssembler
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.tools.builtin import build_fs_registry


def run(coro):
    return asyncio.run(coro)


async def collect(agent, session, prompt="go"):
    return [ev async for ev in agent.run(prompt, session)]


def make_agent(script, registry, *, repeat_last=False, limits=None, policy=None):
    provider = FakeProvider(script, repeat_last=repeat_last)
    return Agent(provider, registry, model="fake", limits=limits, policy=policy), provider


def test_plain_text_turn(tmp_path):
    agent, _ = make_agent([text_turn("hello world")], build_fs_registry(tmp_path))
    session = agent.new_session()
    events = run(collect(agent, session))
    assert "".join(e.text for e in events if isinstance(e, AssistantText)) == "hello world"
    assert isinstance(events[-1], Done) and events[-1].reason == "end_turn"
    assert validate_pairing(session.messages) == []


def test_tool_roundtrip(tmp_path):
    (tmp_path / "a.txt").write_text("secret-content")
    agent, provider = make_agent(
        [tool_turn(("t1", "read_file", {"path": "a.txt"})), text_turn("done")], build_fs_registry(tmp_path)
    )
    session = agent.new_session()
    events = run(collect(agent, session))
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].output == "secret-content" and not finished[0].is_error
    assert validate_pairing(session.messages) == []
    # second request must carry the tool result back to the model
    assert provider.requests[1].messages[-1].role == "tool"


def test_invalid_args_become_recoverable_error(tmp_path):
    agent, _ = make_agent([tool_turn(("t1", "read_file", {"wrong": 1})), text_turn("ok")], build_fs_registry(tmp_path))
    events = run(collect(agent, agent.new_session()))
    err = next(e for e in events if isinstance(e, ToolFinished))
    assert err.is_error and "Invalid arguments" in err.output
    assert events[-1].reason == "end_turn"  # loop continued after the error


def test_unknown_tool(tmp_path):
    agent, _ = make_agent([tool_turn(("t1", "nope", {})), text_turn("ok")], build_fs_registry(tmp_path))
    err = next(e for e in run(collect(agent, agent.new_session())) if isinstance(e, ToolFinished))
    assert err.is_error and "Unknown tool" in err.output


def test_path_escape_is_blocked(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.txt").write_text("nope")
    agent, _ = make_agent(
        [tool_turn(("t1", "read_file", {"path": "../outside.txt"})), text_turn("ok")], build_fs_registry(ws)
    )
    err = next(e for e in run(collect(agent, agent.new_session())) if isinstance(e, ToolFinished))
    assert err.is_error and "escapes workspace" in err.output


def test_policy_denies_ungranted_permission():
    reg = ToolRegistry()

    @reg.tool(permission=Permission.EXEC)
    def danger() -> str:
        """never runs"""
        return "boom"

    agent, _ = make_agent([tool_turn(("t1", "danger", {})), text_turn("ok")], reg)
    err = next(e for e in run(collect(agent, agent.new_session())) if isinstance(e, ToolFinished))
    assert err.is_error and "Denied by policy" in err.output


def test_max_turns_guardrail(tmp_path):
    agent, _ = make_agent(
        [tool_turn(("t1", "list_dir", {}))], build_fs_registry(tmp_path), repeat_last=True, limits=Limits(max_turns=3)
    )
    session = agent.new_session()
    events = run(collect(agent, session))
    assert events[-1].reason == "max_turns" and events[-1].turns == 3
    assert validate_pairing(session.messages) == []


def test_tool_timeout_is_reported():
    reg = ToolRegistry()

    @reg.tool(timeout_s=0.05)
    async def slow() -> str:
        """sleeps"""
        await asyncio.sleep(1)
        return "late"

    agent, _ = make_agent([tool_turn(("t1", "slow", {})), text_turn("ok")], reg)
    err = next(e for e in run(collect(agent, agent.new_session())) if isinstance(e, ToolFinished))
    assert err.is_error and "timed out" in err.output


def test_cancel_mid_tool_keeps_pairing_invariant():
    reg = ToolRegistry()

    @reg.tool(timeout_s=10)
    async def hang() -> str:
        """hangs"""
        await asyncio.sleep(5)
        return "never"

    agent, _ = make_agent([tool_turn(("t1", "hang", {})), text_turn("ok")], reg)
    session = agent.new_session()

    async def scenario():
        async def consume():
            async for _ in agent.run("go", session):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)  # model call done, tool in flight
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    run(scenario())
    assert validate_pairing(session.messages) == []
    last = session.messages[-1]
    assert last.role == "tool" and last.content[0].is_error


def test_stream_assembler_joins_chunked_json_and_flags_bad_json():
    asm = StreamAssembler()
    for ev in tool_turn(("a", "read_file", {"path": "x.txt"}), text="hi"):
        asm.feed(ev)
    msg = asm.finish().message
    assert msg.text() == "hi" and msg.tool_uses()[0].input == {"path": "x.txt"}

    from mini_harness.core.messages import Usage
    from mini_harness.providers.base import INVALID_JSON_KEY, MessageEnd, ToolCallDelta, ToolCallEnd, ToolCallStart

    bad = StreamAssembler()
    events = [
        ToolCallStart("b", "t"),
        ToolCallDelta("b", '{"path": '),
        ToolCallEnd("b"),
        MessageEnd("tool_use", Usage()),
    ]
    for ev in events:
        bad.feed(ev)
    assert INVALID_JSON_KEY in bad.finish().message.tool_uses()[0].input


def test_close_dangling_is_idempotent():
    msgs = [
        Message.user("q"),
        Message.assistant([ToolUseBlock("1", "x", {}), ToolUseBlock("2", "y", {})]),
        Message.tool_results([ToolResultBlock("1", "ok")]),
    ]
    assert validate_pairing(msgs) != []
    assert close_dangling_tool_uses(msgs) == 1
    assert validate_pairing(msgs) == []
    assert close_dangling_tool_uses(msgs) == 0


def test_schema_generation():
    spec = build_fs_registry(".").get("read_file").spec
    assert spec.input_schema["required"] == ["path"]
    assert spec.input_schema["properties"]["path"]["description"]
    assert spec.input_schema["additionalProperties"] is False
