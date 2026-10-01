"""Provider abstraction: a narrow protocol + a vendor-neutral stream-event vocabulary.

Adapters translate vendor SSE into these events; StreamAssembler turns events into a Message.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from mini_harness.core.errors import ProviderError
from mini_harness.core.messages import ContentBlock, Message, TextBlock, ToolUseBlock, Usage
from mini_harness.tools.spec import ToolSpec

INVALID_JSON_KEY = "__invalid_json__"  # marks tool args that failed to parse; executor reports it


@dataclass(frozen=True)
class CachePlan:
    """Where a provider with explicit prompt caching should place breakpoints.

    `system=True` marks the end of the static prefix (tools + system). `message_indices` index into
    ModelRequest.messages; the breakpoint goes after the last block of that message.
    """

    system: bool = True
    message_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec]
    max_tokens: int = 4096
    cache: CachePlan | None = None  # hint; adapters without explicit cache control ignore it


# ---- stream events (provider -> loop) ----
@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStart:
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallDelta:
    id: str
    partial_json: str


@dataclass(frozen=True)
class ToolCallEnd:
    id: str


@dataclass(frozen=True)
class MessageEnd:
    stop_reason: str  # end_turn | tool_use | max_tokens | ...
    usage: Usage


@dataclass(frozen=True)
class RetryNotice:
    """Emitted by RetryingProvider before it replays a failed call (ADR-004).

    Everything streamed so far in this call is void: assemblers reset, UIs should discard partial output.
    """

    attempt: int  # the attempt that just failed (1-based)
    delay_s: float
    reason: str
    discarded_partial: bool  # True if events had already been streamed from the failed attempt


StreamEvent = TextDelta | ToolCallStart | ToolCallDelta | ToolCallEnd | MessageEnd | RetryNotice


class Provider(Protocol):
    name: str

    def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]: ...


@dataclass(frozen=True)
class AssembledMessage:
    message: Message
    stop_reason: str
    usage: Usage


class StreamAssembler:
    """Accumulates stream events into one assistant Message.

    Tool arguments arrive as JSON fragments; they are parsed only at ToolCallEnd.
    Tool calls that never ended (truncated stream) are dropped, never executed half-formed.
    """

    def __init__(self) -> None:
        self._blocks: list[ContentBlock] = []
        self._text: list[str] = []
        self._pending: dict[str, tuple[str, list[str]]] = {}
        self._end: MessageEnd | None = None

    def feed(self, ev: StreamEvent) -> None:
        match ev:
            case TextDelta(text):
                self._text.append(text)
            case ToolCallStart(id, name):
                self._flush_text()
                self._pending[id] = (name, [])
            case ToolCallDelta(id, partial):
                if id in self._pending:
                    self._pending[id][1].append(partial)
            case ToolCallEnd(id):
                name, parts = self._pending.pop(id, (None, []))
                if name is not None:
                    self._blocks.append(ToolUseBlock(id, name, _parse_args("".join(parts))))
            case MessageEnd():
                self._end = ev
            case RetryNotice():
                self._reset()

    def finish(self) -> AssembledMessage:
        if self._end is None:
            raise ProviderError("stream ended without MessageEnd", retryable=True)
        self._flush_text()
        return AssembledMessage(Message.assistant(self._blocks), self._end.stop_reason, self._end.usage)

    def _reset(self) -> None:
        self._blocks.clear()
        self._text.clear()
        self._pending.clear()
        self._end = None

    def _flush_text(self) -> None:
        text = "".join(self._text)
        if text:
            self._blocks.append(TextBlock(text))
        self._text.clear()


def _parse_args(raw: str) -> dict:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw}
    return value if isinstance(value, dict) else {INVALID_JSON_KEY: raw}
