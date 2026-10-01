"""Scripted provider for tests, demos and (later) deterministic replay."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from mini_harness.core.messages import Usage
from mini_harness.providers.base import (
    MessageEnd,
    ModelRequest,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)

DEFAULT_USAGE = Usage(10, 5)


def text_turn(text: str, usage: Usage = DEFAULT_USAGE) -> list[StreamEvent]:
    mid = max(1, len(text) // 2)
    return [TextDelta(text[:mid]), TextDelta(text[mid:]), MessageEnd("end_turn", usage)]


def tool_turn(
    *calls: tuple[str, str, dict[str, Any]], text: str = "", usage: Usage = DEFAULT_USAGE
) -> list[StreamEvent]:
    events: list[StreamEvent] = [TextDelta(text)] if text else []
    for call_id, name, args in calls:
        raw = json.dumps(args)
        mid = len(raw) // 2  # split JSON to exercise streamed-argument assembly
        events += [
            ToolCallStart(call_id, name),
            ToolCallDelta(call_id, raw[:mid]),
            ToolCallDelta(call_id, raw[mid:]),
            ToolCallEnd(call_id),
        ]
    events.append(MessageEnd("tool_use", usage))
    return events


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[list[StreamEvent]], *, repeat_last: bool = False) -> None:
        self._script = list(script)
        self._repeat_last = repeat_last
        self.requests: list[ModelRequest] = []

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(req)
        idx = len(self.requests) - 1
        if idx >= len(self._script):
            if not self._repeat_last:
                raise AssertionError("FakeProvider script exhausted")
            idx = len(self._script) - 1
        for ev in self._script[idx]:
            yield ev
