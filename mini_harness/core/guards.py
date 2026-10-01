"""Loop-health guards."""

from __future__ import annotations

import json
from enum import Enum

from mini_harness.core.messages import ToolUseBlock


class Verdict(Enum):
    OK = "ok"
    WARN = "warn"
    STOP = "stop"


class RepeatDetector:
    """Detects an agent stuck issuing the exact same tool calls turn after turn.

    Streak == threshold -> WARN (execute, but tell the model). Streak > threshold -> STOP.
    Any different call set resets the streak.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._last: tuple[str, ...] | None = None
        self._streak = 0

    def observe(self, calls: list[ToolUseBlock]) -> Verdict:
        sig = tuple(sorted(f"{c.name}:{json.dumps(c.input, sort_keys=True, default=str)}" for c in calls))
        self._streak = self._streak + 1 if sig == self._last else 1
        self._last = sig
        if self._streak > self._threshold:
            return Verdict.STOP
        if self._streak == self._threshold:
            return Verdict.WARN
        return Verdict.OK
