import asyncio
import json
import time

import httpx
import pytest

from mini_harness import (
    Agent,
    AssistantText,
    Done,
    Limits,
    ProviderError,
    Retrying,
    RetryPolicy,
    ToolFinished,
    ToolRegistry,
    TransientToolError,
)
from mini_harness.core.messages import Message, ToolResultBlock, ToolUseBlock, Usage, validate_pairing
from mini_harness.providers.anthropic import AnthropicProvider
from mini_harness.providers.base import (
    MessageEnd,
    ModelRequest,
    RetryNotice,
    StreamAssembler,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from mini_harness.providers.openai_compat import OpenAICompatProvider
from mini_harness.providers.openai_compat import _to_payload as openai_payload
from mini_harness.providers.retry import RetryingProvider
from mini_harness.tools.builtin import build_fs_registry
from mini_harness.tools.spec import ToolSpec


def run(coro):
    return asyncio.run(coro)


REQ = ModelRequest("m", "sys", [Message.user("hi")], [])


async def assemble(provider, req=REQ):
    asm = StreamAssembler()
    async for ev in provider.stream(req):
        asm.feed(ev)
    return asm.finish()


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def oa_sse(*chunks):
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def an_sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


# ---------------------------------------------------------------- RetryPolicy
def test_backoff_schedule_and_retry_after():
    p = RetryPolicy(max_attempts=5, base_delay_s=0.5, max_delay_s=20)
    assert [p.delay(n, rng=lambda: 1.0) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]
    assert p.delay(10, rng=lambda: 1.0) == 20  # capped
    assert p.delay(2, rng=lambda: 0.5) == 0.5  # full jitter scales the cap
    assert p.delay(1, retry_after_s=3) == 3  # server hint wins ...
    assert p.delay(1, retry_after_s=999) == 20  # ... but is bounded
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


# ---------------------------------------------------------------- OpenAI adapter
OA_TOOL_STREAM = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "看看"}, "finish_reason": None}]},
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    },
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}}]},
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "list_dir", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    },
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]}}]},
    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 60}},
    },
]


def test_openai_stream_interleaved_tool_calls_and_usage():
    provider = OpenAICompatProvider("k", client=client(lambda r: httpx.Response(200, content=oa_sse(*OA_TOOL_STREAM))))
    res = run(assemble(provider))
    assert res.message.text() == "看看"
    assert [(c.id, c.name, c.input) for c in res.message.tool_uses()] == [
        ("call_a", "read_file", {"path": "a.txt"}),
        ("call_b", "list_dir", {}),
    ]
    assert res.stop_reason == "tool_use"
    # prompt_tokens includes cached tokens -> normalized to Anthropic-style accounting
    assert res.usage == Usage(input_tokens=40, output_tokens=20, cache_read_tokens=60)


def test_openai_missing_ids_get_synthesized():
    chunk = {
        "choices": [
            {
                "delta": {"tool_calls": [{"index": 0, "function": {"name": "t", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    provider = OpenAICompatProvider("k", client=client(lambda r: httpx.Response(200, content=oa_sse(chunk))))
    assert run(assemble(provider)).message.tool_uses()[0].id == "call_0"


def test_openai_payload_shape():
    spec = ToolSpec("read_file", "Read a file", {"type": "object", "properties": {}})
    msgs = [
        Message.user("q"),
        Message.assistant(
            [ToolUseBlock("c1", "read_file", {"path": "a"}), ToolUseBlock("c2", "read_file", {"path": "b"})]
        ),
        Message.tool_results([ToolResultBlock("c1", "ok"), ToolResultBlock("c2", "boom", is_error=True)]),
    ]
    p = openai_payload(ModelRequest("gpt", "SYS", msgs, [spec], max_tokens=77), "max_completion_tokens")
    assert [m["role"] for m in p["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert p["messages"][0]["content"] == "SYS"
    assert p["messages"][2]["content"] is None
    assert json.loads(p["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert p["messages"][3]["tool_call_id"] == "c1" and p["messages"][4]["content"] == "Error: boom"
    assert p["tools"][0]["function"]["name"] == "read_file"
    assert p["max_completion_tokens"] == 77 and "max_tokens" not in p
    assert p["stream"] is True and p["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------- error mapping (both adapters)
@pytest.mark.parametrize("make", [AnthropicProvider, OpenAICompatProvider])
def test_http_errors_are_classified(make):
    def handler(req):
        status = int(req.headers.get("x-status", "0")) or STATUS[0]
        headers = {"retry-after": "2"} if status == 429 else {}
        return httpx.Response(status, headers=headers, json={"error": {"message": "nope"}})

    STATUS = [429]
    provider = make("k", client=client(handler))
    with pytest.raises(ProviderError) as e:
        run(assemble(provider))
    assert e.value.retryable and e.value.status == 429 and e.value.retry_after_s == 2.0

    STATUS[0] = 400
    with pytest.raises(ProviderError) as e:
        run(assemble(provider))
    assert not e.value.retryable and e.value.status == 400

    STATUS[0] = 529
    with pytest.raises(ProviderError) as e:
        run(assemble(provider))
    assert e.value.retryable


def test_truncated_streams_are_retryable_errors():
    oa = OpenAICompatProvider(
        "k", client=client(lambda r: httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'))
    )
    an = AnthropicProvider(
        "k",
        client=client(
            lambda r: httpx.Response(
                200,
                content=an_sse(
                    {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
                ),
            )
        ),
    )
    for provider in (oa, an):
        with pytest.raises(ProviderError) as e:
            run(assemble(provider))
        assert e.value.retryable


def test_transport_failure_is_retryable():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(ProviderError) as e:
        run(assemble(OpenAICompatProvider("k", client=client(handler))))
    assert e.value.retryable


def test_malformed_sse_payload_is_retryable():
    provider = OpenAICompatProvider("k", client=client(lambda r: httpx.Response(200, content=b"data: {oops\n\n")))
    with pytest.raises(ProviderError) as e:
        run(assemble(provider))
    assert e.value.retryable


# ---------------------------------------------------------------- LSP: both vendors -> identical internal message
def test_both_adapters_produce_the_same_internal_message():
    anthropic_body = an_sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "看看"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call_a", "name": "read_file", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": '}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"a.txt"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    )
    openai_body = oa_sse(
        {"choices": [{"delta": {"content": "看看"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": '{"path": '}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9}},
    )
    a = run(assemble(AnthropicProvider("k", client=client(lambda r: httpx.Response(200, content=anthropic_body)))))
    o = run(assemble(OpenAICompatProvider("k", client=client(lambda r: httpx.Response(200, content=openai_body)))))
    assert a.message == o.message and a.stop_reason == o.stop_reason == "tool_use" and a.usage == o.usage


# ---------------------------------------------------------------- RetryingProvider
class Flaky:
    name = "flaky"

    def __init__(self, attempts):
        self.attempts, self.calls = list(attempts), 0

    async def stream(self, req):
        script = self.attempts[min(self.calls, len(self.attempts) - 1)]
        self.calls += 1
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item


OK_TURN = [TextDelta("complete answer"), MessageEnd("end_turn", Usage(1, 1))]
BOOM = ProviderError("overloaded", status=529, retryable=True)


def test_midstream_failure_discards_partial_and_replays():
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    flaky = Flaky([[TextDelta("par"), BOOM], OK_TURN])
    provider = RetryingProvider(flaky, RetryPolicy(max_attempts=3, base_delay_s=1), sleep=fake_sleep, rng=lambda: 1.0)
    agent = Agent(provider, ToolRegistry(), model="m")
    session = agent.new_session()

    async def go():
        return [e async for e in agent.run("hi", session)]

    events = run(go())
    retry = next(e for e in events if isinstance(e, Retrying))
    assert retry.attempt == 1 and retry.discarded_partial and retry.delay_s == 1.0
    assert "".join(e.text for e in events if isinstance(e, AssistantText)) == "parcomplete answer"  # UI saw both...
    assert session.messages[-1].text() == "complete answer"  # ...but history only keeps the good attempt
    assert flaky.calls == 2 and sleeps == [1.0] and events[-1].reason == "end_turn"


def test_failure_before_any_output_reports_no_partial():
    async def nosleep(d):
        pass

    provider = RetryingProvider(Flaky([[BOOM], OK_TURN]), sleep=nosleep)

    async def go():
        return [e async for e in provider.stream(REQ)]

    notices = [e for e in run(go()) if isinstance(e, RetryNotice)]
    assert len(notices) == 1 and notices[0].discarded_partial is False


def test_non_retryable_error_fails_fast():
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    flaky = Flaky([[ProviderError("bad key", status=401, retryable=False)]])
    provider = RetryingProvider(flaky, sleep=fake_sleep)
    with pytest.raises(ProviderError):
        run(assemble(provider))
    assert flaky.calls == 1 and sleeps == []


def test_retries_are_exhausted():
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    flaky = Flaky([[BOOM]])
    provider = RetryingProvider(flaky, RetryPolicy(max_attempts=3), sleep=fake_sleep)
    with pytest.raises(ProviderError):
        run(assemble(provider))
    assert flaky.calls == 3 and len(sleeps) == 2


def test_wall_clock_bounds_backoff_time():
    policy = RetryPolicy(max_attempts=5, base_delay_s=10, max_delay_s=10)
    provider = RetryingProvider(Flaky([[BOOM]]), policy, rng=lambda: 1.0)  # real 10s sleeps
    agent = Agent(provider, ToolRegistry(), model="m", limits=Limits(wall_clock_s=0.3))
    session = agent.new_session()

    async def go():
        return [e async for e in agent.run("hi", session)]

    t0 = time.monotonic()
    events = run(go())
    assert time.monotonic() - t0 < 2
    assert any(isinstance(e, Retrying) for e in events) and events[-1].reason == "timeout"
    assert validate_pairing(session.messages) == []


# ---------------------------------------------------------------- end to end: OpenAI-compatible + retry + tools
def test_end_to_end_openai_with_429_then_tool_loop(tmp_path):
    (tmp_path / "a.txt").write_text("file-content")
    payloads, n = [], [0]

    def handler(req):
        n[0] += 1
        payloads.append(json.loads(req.content))
        if n[0] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        if n[0] == 2:
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
            return httpx.Response(200, content=oa_sse(*chunks))
        return httpx.Response(
            200,
            content=oa_sse(
                {"choices": [{"delta": {"content": "it says file-content"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}},
            ),
        )

    async def nosleep(d):
        pass

    provider = RetryingProvider(OpenAICompatProvider("k", client=client(handler)), sleep=nosleep)
    agent = Agent(provider, build_fs_registry(tmp_path), model="gpt-x")
    session = agent.new_session()

    async def go():
        return [e async for e in agent.run("read a.txt", session)]

    events = run(go())
    assert n[0] == 3
    assert [type(e).__name__ for e in events if isinstance(e, (Retrying, ToolFinished, Done))] == [
        "Retrying",
        "ToolFinished",
        "Done",
    ]
    assert next(e for e in events if isinstance(e, ToolFinished)).output == "file-content"
    assert [m["role"] for m in payloads[2]["messages"]] == ["system", "user", "assistant", "tool"]
    assert payloads[2]["messages"][3]["tool_call_id"] == "call_a"
    assert events[-1].reason == "end_turn"


# ---------------------------------------------------------------- idempotent tool retry
FAST = RetryPolicy(max_attempts=3, base_delay_s=0.0)


def run_tool(registry, name="t", **agent_kw):
    agent = Agent(
        Flaky([[ToolCallStart("x", name), ToolCallEnd("x"), MessageEnd("tool_use", Usage())], OK_TURN]),
        registry,
        model="m",
        tool_retry=FAST,
        **agent_kw,
    )

    async def go():
        return [e async for e in agent.run("go", agent.new_session())]

    return next(e for e in run(go()) if isinstance(e, ToolFinished))


def test_idempotent_tool_retries_transient_errors():
    reg, calls = ToolRegistry(), [0]

    @reg.tool(idempotent=True)
    async def t() -> str:
        """flaky"""
        calls[0] += 1
        if calls[0] < 3:
            raise TransientToolError("try again")
        return "finally"

    result = run_tool(reg)
    assert result.output == "finally" and not result.is_error and calls[0] == 3


def test_non_idempotent_tool_is_never_retried():
    reg, calls = ToolRegistry(), [0]

    @reg.tool()
    async def t() -> str:
        """not idempotent"""
        calls[0] += 1
        raise TransientToolError("temporary")

    result = run_tool(reg)
    assert result.is_error and "TransientToolError" in result.output and calls[0] == 1


def test_deterministic_errors_are_not_retried_even_if_idempotent():
    reg, calls = ToolRegistry(), [0]

    @reg.tool(idempotent=True)
    async def t() -> str:
        """bug"""
        calls[0] += 1
        raise ValueError("bad input")

    assert run_tool(reg).is_error and calls[0] == 1


def test_idempotent_tool_retries_after_timeout():
    reg, calls = ToolRegistry(), [0]

    @reg.tool(idempotent=True, timeout_s=0.05)
    async def t() -> str:
        """slow first time"""
        calls[0] += 1
        if calls[0] == 1:
            await asyncio.sleep(1)
        return "second time lucky"

    assert run_tool(reg).output == "second time lucky" and calls[0] == 2


def test_retry_exhaustion_reports_the_last_error():
    reg, calls = ToolRegistry(), [0]

    @reg.tool(idempotent=True)
    async def t() -> str:
        """always fails"""
        calls[0] += 1
        raise ConnectionError("down")

    result = run_tool(reg)
    assert result.is_error and "ConnectionError" in result.output and calls[0] == 3
