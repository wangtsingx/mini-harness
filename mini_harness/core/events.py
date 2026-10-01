"""Events emitted to SDK users (stable public surface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_harness.core.messages import Usage


@dataclass(frozen=True)
class AssistantText:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolFinished:
    id: str
    name: str
    is_error: bool
    output: str


@dataclass(frozen=True)
class Retrying:
    """A model call failed transiently and will be replayed after `delay_s`.

    If `discarded_partial` is True, drop any AssistantText already shown for this model call.
    """

    attempt: int
    delay_s: float
    reason: str
    discarded_partial: bool


@dataclass(frozen=True)
class Compacted:
    """The context was shrunk before a model call (stage: trim | summary | extractive)."""

    stage: str
    tokens_before: int
    tokens_after: int
    archived: int


@dataclass(frozen=True)
class Done:
    # end_turn | max_tokens | max_turns | budget_exceeded | repeated_calls | timeout
    reason: str
    turns: int  # model calls in this run
    usage: Usage  # session-cumulative
    cost_usd: float | None = None  # session-cumulative; None when no Pricing configured


Event = AssistantText | Retrying | Compacted | ToolStarted | ToolFinished | Done
