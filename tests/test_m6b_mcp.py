import asyncio
import json
import sys
from pathlib import Path

import pytest

from mini_harness import Agent, ToolFinished, ToolRegistry
from mini_harness.mcp import (
    McpClient,
    McpError,
    McpServerParams,
    default_permission,
    mcp_tool_name,
    mcp_tools,
    register_mcp_tools,
)
from mini_harness.mcp.client import McpToolInfo, build_env, render_result
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.tools.policy import DefaultPolicy
from mini_harness.tools.spec import Permission

SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))


def params(**env):
    return McpServerParams(sys.executable, (SERVER,), env=env or None)


def run(coro):
    return asyncio.run(coro)


async def run_tool_calls(registry, *calls, policy=None):
    """Drive the real loop: the model calls the given tools once, then answers."""
    agent = Agent(FakeProvider([tool_turn(*calls), text_turn("done")]), registry, model="m", policy=policy)
    events = [e async for e in agent.run("go", agent.new_session())]
    return [e for e in events if isinstance(e, ToolFinished)]


# ---------------------------------------------------------------- unit: naming, rendering, env
def test_tool_name_sanitizing_and_truncation():
    assert mcp_tool_name("my server", "do.thing") == "my_server__do_thing"
    long = mcp_tool_name("srv", "t" * 100)
    assert len(long) == 64 and long == mcp_tool_name("srv", "t" * 100)  # stable
    assert long != mcp_tool_name("srv", "t" * 99 + "u")  # distinct tools stay distinct
    assert mcp_tool_name("a", "b") == "a__b"


def test_render_result_variants():
    assert render_result({"content": [{"type": "text", "text": "hi"}]}).text == "hi"
    r = render_result({"content": [{"type": "image", "data": "AAAA", "mimeType": "image/png"}], "isError": True})
    assert "image omitted: image/png, 4 base64" in r.text and r.is_error
    assert render_result({"content": [], "structuredContent": {"a": 1}}).text == '{"a": 1}'
    assert "resource: file:///x" in render_result({"content": [{"type": "resource_link", "uri": "file:///x"}]}).text
    emb = {"type": "resource", "resource": {"uri": "u", "text": "inline body"}}
    assert render_result({"content": [emb]}).text == "inline body"
    assert "unsupported content type: weird" in render_result({"content": [{"type": "weird"}]}).text


def test_server_env_is_minimal_but_extensible(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    env = build_env({"EXPLICIT_VAR": "yes"})
    assert "SECRET_TOKEN" not in env and env["EXPLICIT_VAR"] == "yes" and "PATH" in env


def test_default_permission_follows_readonly_hint_only():
    assert default_permission(McpToolInfo("t", "", {}, {"readOnlyHint": True})) == Permission.READ
    for ann in ({}, {"readOnlyHint": False}, {"readOnlyHint": "yes"}, {"destructiveHint": False}):
        assert default_permission(McpToolInfo("t", "", {}, ann)) == Permission.EXEC


# ---------------------------------------------------------------- protocol against a real subprocess
def test_handshake_pagination_and_tool_metadata():
    async def scenario():
        async with McpClient("fake", params()) as c:
            assert c.alive and c.server_info["name"] == "fake" and c.protocol_version == "2025-06-18"
            infos = await c.list_tools()  # two pages
            return {i.name: i for i in infos}

    infos = run(scenario())
    assert {"echo", "add", "fail", "slow", "crash", "idem"} <= infos.keys()
    assert infos["echo"].annotations == {"readOnlyHint": True} and infos["add"].input_schema["required"] == ["a", "b"]


def test_registered_tools_have_safe_specs():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            names = await register_mcp_tools(reg, c)
            return names, {n: reg.get(n).spec for n in names}

    names, specs = run(scenario())
    assert "fake__echo" in names and "fake__bad_name_with_spaces" in names
    echo, add, idem = specs["fake__echo"], specs["fake__add"], specs["fake__idem"]
    assert (echo.permission, echo.concurrency_safe) == (Permission.READ, True)
    assert (add.permission, add.concurrency_safe) == (Permission.EXEC, False)  # unannotated => untrusted
    assert idem.idempotent and not echo.idempotent
    assert echo.description == "Echo text back" and echo.input_schema["required"] == ["text"]
    assert all(len(n) <= 64 for n in names)


def test_allow_and_deny_filters():
    async def scenario():
        async with McpClient("fake", params()) as c:
            infos = await c.list_tools()
            only = [t.spec.name for t in mcp_tools(c, infos, allow={"echo", "add"})]
            minus = {t.spec.name for t in mcp_tools(c, infos, deny={"crash", "slow"})}
            return only, minus

    only, minus = run(scenario())
    assert only == ["fake__echo", "fake__add"] and "fake__crash" not in minus and "fake__echo" in minus


def test_end_to_end_through_the_agent_loop_with_default_policy():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            ok = await run_tool_calls(reg, ("1", "fake__echo", {"text": "hello mcp"}))
            denied = await run_tool_calls(reg, ("2", "fake__add", {"a": 2, "b": 3}))  # EXEC not granted
            granted = await run_tool_calls(
                reg,
                ("3", "fake__add", {"a": 2, "b": 3}),
                policy=DefaultPolicy(frozenset({Permission.READ, Permission.EXEC})),
            )
            return ok[0], denied[0], granted[0]

    ok, denied, granted = run(scenario())
    assert (ok.output, ok.is_error) == ("hello mcp", False)
    assert denied.is_error and "Denied by policy" in denied.output
    assert (granted.output, granted.is_error) == ("5", False)


def test_local_schema_validation_gives_model_readable_errors():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            pol = DefaultPolicy(frozenset({Permission.READ, Permission.EXEC}))
            bad_type = await run_tool_calls(reg, ("1", "fake__add", {"a": "x", "b": 1}), policy=pol)
            missing = await run_tool_calls(reg, ("2", "fake__add", {"a": 1}), policy=pol)
            extra = await run_tool_calls(reg, ("3", "fake__add", {"a": 1, "b": 2, "c": 3}), policy=pol)
            return bad_type[0], missing[0], extra[0]

    bad_type, missing, extra = run(scenario())
    assert bad_type.is_error and bad_type.output.startswith("Invalid arguments:") and "a:" in bad_type.output
    assert "'b' is a required property" in missing.output and "'c'" in extra.output


def test_server_reported_errors_are_clean_tool_errors():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            pol = DefaultPolicy(frozenset({Permission.EXEC}))
            return (await run_tool_calls(reg, ("1", "fake__fail", {}), policy=pol))[0]

    r = run(scenario())
    assert r.is_error and r.output == "boom from server"  # no "McpToolError: " noise


def test_non_text_content_and_structured_results():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            out = await run_tool_calls(reg, ("1", "fake__image", {}), ("2", "fake__structured", {}))
            return out

    img, structured = run(scenario())
    assert "image omitted: image/png" in img.output and "caption" in img.output
    assert structured.output == '{"a": 1}'


def test_huge_results_cross_the_pipe_and_are_truncated_by_the_executor():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            return (await run_tool_calls(reg, ("1", "fake__big", {})))[0]

    r = run(scenario())  # 200 KB single JSON line > asyncio's default 64 KiB line limit
    assert not r.is_error and len(r.output) < 21_000 and "truncated" in r.output


def test_unknown_remote_tool_surfaces_the_jsonrpc_error():
    async def scenario():
        async with McpClient("fake", params()) as c:
            with pytest.raises(McpError, match="Unknown tool") as e:
                await c.call_tool("nope", {})
            return e.value.code

    assert run(scenario()) == -32602


def test_timeout_cancels_the_remote_request(tmp_path):
    log = tmp_path / "log.jsonl"

    async def scenario():
        async with McpClient("fake", params(FAKE_MCP_LOG=str(log))) as c:
            infos = {i.name: i for i in await c.list_tools()}
            (slow,) = mcp_tools(c, [infos["slow"]], timeout_s=0.3)
            reg = ToolRegistry()
            reg.register(slow)
            r = (
                await run_tool_calls(reg, ("1", "fake__slow", {}), policy=DefaultPolicy(frozenset({Permission.EXEC})))
            )[0]
            await asyncio.sleep(0.2)  # let the notification reach the server
            return r

    r = run(scenario())
    assert r.is_error and "timed out" in r.output
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    call = next(m["in"] for m in logged if m.get("in", {}).get("method") == "tools/call")
    cancelled = [m["in"] for m in logged if m.get("in", {}).get("method") == "notifications/cancelled"]
    assert cancelled and cancelled[0]["params"]["requestId"] == call["id"]


def test_server_crash_fails_fast_and_never_hangs():
    async def scenario():
        async with McpClient("fake", params()) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            pol = DefaultPolicy(frozenset({Permission.EXEC, Permission.READ}))
            first = (await asyncio.wait_for(run_tool_calls(reg, ("1", "fake__crash", {}), policy=pol), 5))[0]
            second = (await asyncio.wait_for(run_tool_calls(reg, ("2", "fake__echo", {"text": "x"}), policy=pol), 5))[0]
            return first, second, c.alive

    first, second, alive = run(scenario())
    assert first.is_error and "exited (code 3)" in first.output
    assert second.is_error and "exited" in second.output and alive is False


def test_server_requests_are_answered(tmp_path):
    log = tmp_path / "log.jsonl"

    async def scenario():
        async with McpClient("fake", params(FAKE_MCP_LOG=str(log), FAKE_MCP_PING="1")) as c:
            await c.list_tools()
            await asyncio.sleep(0.2)

    run(scenario())
    replies = {
        m["response"]["id"]: m["response"] for m in map(json.loads, log.read_text().splitlines()) if "response" in m
    }
    assert replies["srv-1"] == {"jsonrpc": "2.0", "id": "srv-1", "result": {}}  # ping answered
    assert replies["srv-2"]["error"]["code"] == -32601  # unsupported server request refused, not ignored


def test_server_gets_a_scrubbed_environment(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")

    async def scenario():
        async with McpClient("fake", params(EXPLICIT_VAR="visible")) as c:
            reg = ToolRegistry()
            await register_mcp_tools(reg, c)
            return (
                await run_tool_calls(reg, ("1", "fake__env", {}), policy=DefaultPolicy(frozenset({Permission.EXEC})))
            )[0]

    info = json.loads(run(scenario()).output)
    assert "SECRET_TOKEN" not in info["keys"] and info["explicit"] == "visible"


def test_unsupported_protocol_version_is_rejected_and_cleaned_up():
    async def scenario():
        client = McpClient("fake", params(FAKE_MCP_VERSION="1999-01-01"))
        with pytest.raises(McpError, match="unsupported protocol version"):
            await client.start()
        return client

    client = run(scenario())
    assert not client.alive and client._proc is None


def test_missing_server_binary_is_a_clear_error():
    async def scenario():
        with pytest.raises(McpError, match="cannot start MCP server"):
            await McpClient("ghost", McpServerParams("/definitely/not/a/binary")).start()

    run(scenario())


def test_close_terminates_the_process_and_is_idempotent():
    async def scenario():
        c = McpClient("fake", params())
        await c.start()
        proc = c._proc
        await c.close()
        await c.close()
        return proc.returncode, c.alive

    code, alive = run(scenario())
    assert code is not None and alive is False


def test_calls_after_close_fail_immediately():
    async def scenario():
        c = McpClient("fake", params())
        await c.start()
        await c.close()
        with pytest.raises(McpError, match="closed"):
            await c.list_tools()

    run(scenario())
