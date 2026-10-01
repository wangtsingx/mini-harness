"""Cheap, dependency-free token estimation.

Used only for *deltas* (new messages since the last measured request) and for sizing compaction
decisions. The ground truth is the provider-reported usage of the last request (see ContextManager.estimate).
"""

from __future__ import annotations

import json
import re

from mini_harness.core.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from mini_harness.tools.spec import ToolSpec

_CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uff00-\uffef]")
MESSAGE_OVERHEAD = 4


def estimate_text(s: str) -> int:
    """~1 token per CJK char, ~1 token per 4 other chars (deliberately slightly conservative)."""
    if not s:
        return 0
    cjk = len(_CJK.findall(s))
    return cjk + (len(s) - cjk + 3) // 4


def estimate_message(m: Message) -> int:
    n = MESSAGE_OVERHEAD
    for b in m.content:
        if isinstance(b, TextBlock):
            n += estimate_text(b.text)
        elif isinstance(b, ToolUseBlock):
            n += estimate_text(b.name) + estimate_text(json.dumps(b.input, ensure_ascii=False)) + 3
        elif isinstance(b, ToolResultBlock):
            n += estimate_text(b.content) + 3
    return n


def estimate_messages(messages: list[Message]) -> int:
    return sum(estimate_message(m) for m in messages)


def estimate_request(system: str, tools: list[ToolSpec], messages: list[Message]) -> int:
    tool_tokens = sum(
        estimate_text(t.name + t.description + json.dumps(t.input_schema, ensure_ascii=False)) for t in tools
    )
    return estimate_text(system) + tool_tokens + estimate_messages(messages)
