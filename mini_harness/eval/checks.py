"""Scoring primitives for eval cases. A checker looks at a finished run and returns a Score."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from mini_harness.core.events import ToolFinished, ToolStarted
from mini_harness.eval.replay import RunResult


@dataclass(frozen=True)
class Score:
    passed: bool
    reason: str = ""
    # why it failed, in the checker's own terms: wrong_answer | format | incomplete | unsafe | checker_error | other
    kind: str = "wrong_answer"
    # strings the right answer depends on. Lets attribution tell "the model got it wrong" from
    # "the information was compacted away": facts that exist only in archived history point at the context layer.
    facts: tuple[str, ...] = ()


Checker = Callable[[RunResult], "Score | bool"]


def final_text(result: RunResult) -> str:
    """The last assistant message with text."""
    for m in reversed(result.session.messages):
        if m.role == "assistant" and m.text():
            return m.text()
    return ""


def tools_called(result: RunResult) -> list[str]:
    return [e.name for e in result.events if isinstance(e, ToolStarted)]


def tool_errors(result: RunResult) -> list[ToolFinished]:
    return [e for e in result.events if isinstance(e, ToolFinished) and e.is_error]


def contains(*phrases: str, ignore_case: bool = True) -> Checker:
    """Passes when every phrase appears in the final answer."""

    def check(result: RunResult) -> Score:
        text = final_text(result)
        hay = text.lower() if ignore_case else text
        missing = [p for p in phrases if (p.lower() if ignore_case else p) not in hay]
        return Score(not missing, f"missing in answer: {missing}" if missing else "", "wrong_answer", tuple(phrases))

    return check


def matches(pattern: str, *, kind: str = "format") -> Checker:
    """Passes when the regex matches the final answer. Failures default to the 'format' kind."""
    rx = re.compile(pattern, re.DOTALL)

    def check(result: RunResult) -> Score:
        ok = rx.search(final_text(result)) is not None
        return Score(ok, "" if ok else f"answer does not match /{pattern}/", kind)

    return check


def tool_called(name: str) -> Checker:
    def check(result: RunResult) -> Score:
        ok = name in tools_called(result)
        return Score(ok, "" if ok else f"tool {name!r} was never called", "incomplete")

    return check


def no_tool_errors() -> Checker:
    def check(result: RunResult) -> Score:
        errs = tool_errors(result)
        return Score(not errs, f"{len(errs)} tool call(s) failed" if errs else "", "other")

    return check


def all_of(*checks: Checker) -> Checker:
    """First failing check wins (its reason and kind are reported)."""

    def check(result: RunResult) -> Score:
        facts: tuple[str, ...] = ()
        for c in checks:
            s = as_score(c(result))
            facts += s.facts
            if not s.passed:
                return Score(False, s.reason, s.kind, facts)
        return Score(True, "", facts=facts)

    return check


def as_score(value: Score | bool) -> Score:
    return value if isinstance(value, Score) else Score(bool(value), "" if value else "check returned False")
