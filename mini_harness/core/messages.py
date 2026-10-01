"""Internal, provider-neutral message model (DRY: vendor differences live in adapters only).

Invariant: every ToolUseBlock must be answered by exactly one ToolResultBlock in the
immediately following "tool" message. Both major vendor APIs reject requests otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]  # system prompt is a separate request field


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def user(text: str) -> Message:
        return Message("user", (TextBlock(text),))

    @staticmethod
    def assistant(blocks: list[ContentBlock]) -> Message:
        return Message("assistant", tuple(blocks))

    @staticmethod
    def tool_results(results: list[ToolResultBlock]) -> Message:
        return Message("tool", tuple(results))

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


def validate_pairing(messages: list[Message]) -> list[str]:
    """Return a list of invariant violations (empty list == valid)."""
    problems: list[str] = []
    pending: set[str] = set()
    for i, m in enumerate(messages):
        if m.role == "tool":
            for b in m.content:
                if not isinstance(b, ToolResultBlock):
                    continue
                if b.tool_use_id in pending:
                    pending.discard(b.tool_use_id)
                else:
                    problems.append(f"message[{i}]: tool_result without matching tool_use: {b.tool_use_id}")
        if pending:
            problems.append(f"before message[{i}]: unanswered tool_use: {sorted(pending)}")
            pending.clear()
        if m.role == "assistant":
            pending = {b.id for b in m.tool_uses()}
    if pending:
        problems.append(f"tail: unanswered tool_use: {sorted(pending)}")
    return problems


def close_dangling_tool_uses(
    messages: list[Message], reason: str = "Tool call was interrupted before completion."
) -> int:
    """Repair the tail of the history after cancel/crash so the pairing invariant holds.

    Mutates `messages` in place; returns the number of synthesized error results.
    """
    if not messages:
        return 0
    last = messages[-1]
    if last.role == "assistant":
        calls, answered = last.tool_uses(), set()
    elif last.role == "tool" and len(messages) >= 2 and messages[-2].role == "assistant":
        calls = messages[-2].tool_uses()
        answered = {b.tool_use_id for b in last.content if isinstance(b, ToolResultBlock)}
    else:
        return 0
    missing = [c.id for c in calls if c.id not in answered]
    if not missing:
        return 0
    extra = tuple(ToolResultBlock(i, reason, True) for i in missing)
    if last.role == "tool":
        messages[-1] = Message("tool", last.content + extra, last.meta)
    else:
        messages.append(Message("tool", extra))
    return len(missing)
