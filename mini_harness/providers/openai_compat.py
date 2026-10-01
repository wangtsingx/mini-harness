"""OpenAI-compatible Chat Completions adapter (OpenAI, Azure-style gateways, vLLM, DeepSeek, ...).

Vendor differences handled here so the core never sees them:
  * system prompt is a leading message, not a top-level field
  * tool calls live on the assistant message as `tool_calls` (arguments = JSON *string*)
  * each tool result is its own `role: tool` message; there is no is_error flag -> prefixed with "Error: "
  * streamed tool calls are indexed deltas that may interleave; id/name only arrive in the first delta
  * `prompt_tokens` INCLUDES cached tokens -> normalized so input_tokens excludes them (like Anthropic)
Assumption: the first delta of a tool call carries both `id` and `function.name` (true for OpenAI-spec servers).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mini_harness.core.errors import ProviderError
from mini_harness.core.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_harness.providers._sse import DONE, sse_json
from mini_harness.providers.base import (
    MessageEnd,
    ModelRequest,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens"}


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens_param: str = "max_tokens",  # newer OpenAI models want "max_completion_tokens"
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._max_tokens_param = max_tokens_param
        self._headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        payload = _to_payload(req, self._max_tokens_param)
        open_calls: dict[int, str] = {}  # tool_call index -> id
        finish: str | None = None
        saw_done = False
        usage = Usage()
        async for data in sse_json(self._client, self._url, payload, self._headers):
            if data is DONE:
                saw_done = True
                continue
            if "error" in data:
                err = data["error"]
                raise ProviderError(f"stream error: {err}", retryable=_error_retryable(err))
            if data.get("usage"):
                usage = _to_usage(data["usage"])
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if text := delta.get("content"):
                    yield TextDelta(text)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if idx not in open_calls:
                        open_calls[idx] = tc.get("id") or f"call_{idx}"
                        yield ToolCallStart(open_calls[idx], fn.get("name") or "")
                    if args := fn.get("arguments"):
                        yield ToolCallDelta(open_calls[idx], args)
                finish = choice.get("finish_reason") or finish
        if finish is None and not saw_done:
            raise ProviderError("stream ended before finish_reason", retryable=True)
        for idx in sorted(open_calls):
            yield ToolCallEnd(open_calls[idx])
        yield MessageEnd(STOP_REASONS.get(finish or "stop", finish or "end_turn"), usage)


def _error_retryable(err: Any) -> bool:
    kind = str(err.get("type", "")) if isinstance(err, dict) else ""
    return kind in {"server_error", "rate_limit_error", "overloaded_error", "api_error"}


def _to_usage(u: dict[str, Any]) -> Usage:
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return Usage(
        input_tokens=max(u.get("prompt_tokens", 0) - cached, 0),
        output_tokens=u.get("completion_tokens", 0),
        cache_read_tokens=cached,
    )


def _to_payload(req: ModelRequest, max_tokens_param: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": req.system}]
    for m in req.messages:
        messages.extend(_to_wire(m))
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        max_tokens_param: req.max_tokens,
    }
    if req.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
            }
            for t in req.tools
        ]
    return payload


def _to_wire(m: Message) -> list[dict[str, Any]]:
    if m.role == "tool":  # one wire message per result
        return [
            {
                "role": "tool",
                "tool_call_id": b.tool_use_id,
                "content": f"Error: {b.content}" if b.is_error else b.content,
            }
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
    text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
    if m.role == "user":
        return [{"role": "user", "content": text}]
    wire: dict[str, Any] = {"role": "assistant", "content": text or None}
    calls = [
        {"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
        for b in m.content
        if isinstance(b, ToolUseBlock)
    ]
    if calls:
        wire["tool_calls"] = calls
    return [wire]
