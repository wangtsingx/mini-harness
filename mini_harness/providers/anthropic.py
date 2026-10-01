"""Anthropic Messages API adapter (streaming, tool use). Retries live in RetryingProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from mini_harness.core.errors import ProviderError
from mini_harness.core.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_harness.providers._sse import sse_json
from mini_harness.providers.base import (
    CachePlan,
    MessageEnd,
    ModelRequest,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
RETRYABLE_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_URL,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url
        self._headers = {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        tool_ids: dict[int, str] = {}  # content block index -> tool id
        in_tok = out_tok = cache_read = cache_write = 0
        stop_reason, saw_stop = "end_turn", False
        async for data in sse_json(self._client, self._url, _to_payload(req), self._headers):
            kind = data.get("type")
            if kind == "message_start":
                u = data["message"].get("usage", {})
                in_tok = u.get("input_tokens", 0)
                cache_read = u.get("cache_read_input_tokens", 0)
                cache_write = u.get("cache_creation_input_tokens", 0)
            elif kind == "content_block_start":
                block = data["content_block"]
                if block["type"] == "tool_use":
                    tool_ids[data["index"]] = block["id"]
                    yield ToolCallStart(block["id"], block["name"])
            elif kind == "content_block_delta":
                delta = data["delta"]
                if delta["type"] == "text_delta":
                    yield TextDelta(delta["text"])
                elif delta["type"] == "input_json_delta":
                    yield ToolCallDelta(tool_ids[data["index"]], delta["partial_json"])
            elif kind == "content_block_stop":
                if data["index"] in tool_ids:
                    yield ToolCallEnd(tool_ids[data["index"]])
            elif kind == "message_delta":
                stop_reason = data["delta"].get("stop_reason") or stop_reason
                out_tok = data.get("usage", {}).get("output_tokens", out_tok)
            elif kind == "message_stop":
                saw_stop = True
            elif kind == "error":
                err = data.get("error", {})
                raise ProviderError(f"stream error: {err}", retryable=err.get("type") in RETRYABLE_ERROR_TYPES)
        if not saw_stop:
            raise ProviderError("stream ended before message_stop", retryable=True)
        yield MessageEnd(stop_reason, Usage(in_tok, out_tok, cache_read, cache_write))


EPHEMERAL = {"type": "ephemeral"}
MAX_BREAKPOINTS = 4  # API limit; one is spent on the static prefix


def _apply_cache(payload: dict[str, Any], plan: CachePlan | None, system: str) -> None:
    """Explicit breakpoints. Order of the cached prefix is tools -> system -> messages."""
    if plan is None:
        return
    if plan.system and system:
        payload["system"] = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
    budget = MAX_BREAKPOINTS - (1 if plan.system and system else 0)
    for i in sorted(plan.message_indices)[-budget:]:  # prefer the newest breakpoints
        if 0 <= i < len(payload["messages"]) and payload["messages"][i]["content"]:
            payload["messages"][i]["content"][-1]["cache_control"] = EPHEMERAL


def _to_payload(req: ModelRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": True,
        "system": req.system,
        "messages": [_to_wire(m) for m in req.messages],
    }
    _apply_cache(payload, req.cache, req.system)
    if req.tools:
        payload["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in req.tools
        ]
    return payload


def _to_wire(m: Message) -> dict[str, Any]:
    """Internal role 'tool' becomes a user message carrying tool_result blocks."""
    blocks: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            if b.text:  # the API rejects empty text blocks
                blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolUseBlock):
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif isinstance(b, ToolResultBlock):
            blocks.append(
                {"type": "tool_result", "tool_use_id": b.tool_use_id, "content": b.content, "is_error": b.is_error}
            )
    return {"role": "assistant" if m.role == "assistant" else "user", "content": blocks}
