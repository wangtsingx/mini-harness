"""Deterministic replay: re-run the harness against recorded model responses, with zero model calls.

What it proves: given the same model behavior, today's harness still (a) builds the same prompts, (b) executes tools
the same way, (c) compacts, retries and applies guardrails the same way, (d) emits the same events.
Tools run for real during replay, so use deterministic/pure tools (or a sandbox) in recorded scenarios.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mini_harness.core.errors import HarnessError, ProviderError
from mini_harness.core.events import AssistantText, Compacted, Done, Event, Retrying, ToolFinished, ToolStarted
from mini_harness.core.session import Session
from mini_harness.eval.recording import (
    Recording,
    RecordingProvider,
    diff_requests,
    event_from_dict,
    request_to_dict,
)
from mini_harness.providers.base import ModelRequest, Provider, StreamEvent
from mini_harness.sdk import Agent

AgentFactory = Callable[[Provider], Agent]


class ReplayDivergence(HarnessError):
    """The harness asked the model something different from what was recorded."""

    def __init__(self, call_index: int, problems: list[str]) -> None:
        super().__init__(f"call #{call_index + 1}: " + "; ".join(problems))
        self.call_index = call_index
        self.problems = problems


class ReplayProvider:
    name = "replay"

    def __init__(self, recording: Recording, *, ignore: Iterable[str] = (), strict: bool = True) -> None:
        self._rec = recording
        self._ignore = frozenset(ignore)
        self._strict = strict
        self._next = 0
        self.divergences: list[ReplayDivergence] = []

    @property
    def calls_used(self) -> int:
        return self._next

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        i = self._next
        self._next += 1
        if i >= len(self._rec.calls):
            raise self._diverge(i, [f"harness made call #{i + 1} but only {len(self._rec.calls)} were recorded"], True)
        call = self._rec.calls[i]
        problems = diff_requests(call.request, request_to_dict(req), self._ignore)
        if problems:
            self._diverge(i, problems, self._strict)
        for d in call.events:
            if d["t"] == "error":
                raise ProviderError(
                    d["message"], status=d["status"], retryable=d["retryable"], retry_after_s=d["retry_after_s"]
                )
            yield event_from_dict(d)
        if not call.complete:
            raise ProviderError("recorded call ended abnormally", retryable=False)

    def _diverge(self, i: int, problems: list[str], raise_it: bool) -> ReplayDivergence:
        err = ReplayDivergence(i, problems)
        self.divergences.append(err)
        if raise_it:
            raise err
        return err


# ---------------------------------------------------------------- behavioral fingerprint
def fingerprint(events: Sequence[Event], error: BaseException | None = None) -> list[list[Any]]:
    """Timing-free, JSON-able summary of a run's observable behavior (streamed text is joined)."""
    out: list[list[Any]] = []
    text: list[str] = []

    def flush() -> None:
        if text:
            out.append(["text", "".join(text)])
            text.clear()

    for ev in events:
        if isinstance(ev, AssistantText):
            text.append(ev.text)
            continue
        flush()
        if isinstance(ev, Retrying):
            out.append(["retry", ev.attempt, ev.discarded_partial])  # delay is random jitter: excluded
        elif isinstance(ev, Compacted):
            out.append(["compacted", ev.stage, ev.tokens_before, ev.tokens_after, ev.archived])
        elif isinstance(ev, ToolStarted):
            out.append(["tool_start", ev.name, json.dumps(ev.input, sort_keys=True, ensure_ascii=False)])
        elif isinstance(ev, ToolFinished):
            out.append(["tool_end", ev.name, ev.is_error, ev.output])
        elif isinstance(ev, Done):
            u = ev.usage
            out.append(
                [
                    "done",
                    ev.reason,
                    ev.turns,
                    [u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens],
                ]
            )
    flush()
    if error is not None and not isinstance(error, ReplayDivergence):
        out.append(["error", type(error).__name__])
    return out


# ---------------------------------------------------------------- driving a scenario
@dataclass
class RunResult:
    events: list[Event]
    error: BaseException | None
    session: Session


async def drive(agent: Agent, prompts: Sequence[str | None]) -> RunResult:
    """Run prompts in order on one session. Exceptions are captured (a failing run is a valid scenario)."""
    session = agent.new_session()
    events: list[Event] = []
    error: BaseException | None = None
    try:
        for p in prompts:
            async for ev in agent.run(p, session):
                events.append(ev)
    except Exception as e:  # noqa: BLE001
        error = e
    return RunResult(events, error, session)


@dataclass(frozen=True)
class RecordedRun:
    recording: Recording
    result: RunResult


async def record_run(
    make_agent: AgentFactory,
    provider: Provider,
    prompts: Sequence[str | None] | str,
    *,
    path: str | None = None,
    meta: dict[str, Any] | None = None,
) -> RecordedRun:
    """Run a scenario against a real (or fake) provider and capture it, including the expected behavior."""
    plist = [prompts] if isinstance(prompts, str) else list(prompts)
    recording = Recording(meta={**(meta or {}), "prompts": plist})
    rp = RecordingProvider(provider, recording, path)
    result = await drive(make_agent(rp), plist)
    recording.expected = fingerprint(result.events, result.error)
    if path:
        from mini_harness.eval.recording import expected_line

        with open(path, "a", encoding="utf-8") as f:
            f.write(expected_line(recording.expected) + "\n")
    return RecordedRun(recording, result)


@dataclass(frozen=True)
class ReplayReport:
    ok: bool
    problems: tuple[str, ...]
    calls_recorded: int
    calls_used: int
    expected: list[list[Any]] | None
    actual: list[list[Any]]

    def summary(self) -> str:
        if self.ok:
            return f"replay OK: {self.calls_used}/{self.calls_recorded} recorded calls, behavior identical"
        return "replay FAILED:\n" + "\n".join(f"  - {p}" for p in self.problems)


async def replay_run(
    make_agent: AgentFactory,
    recording: Recording,
    *,
    prompts: Sequence[str | None] | None = None,
    ignore: Iterable[str] = (),
    strict: bool = True,
) -> ReplayReport:
    """Replay a recording through the CURRENT harness (built by `make_agent`) and report any behavioral drift."""
    provider = ReplayProvider(recording, ignore=ignore, strict=strict)
    plist = list(prompts) if prompts is not None else list(recording.meta.get("prompts", []))
    if not plist:
        raise ValueError("no prompts: record with record_run() or pass prompts=")
    result = await drive(make_agent(provider), plist)
    actual = fingerprint(result.events, result.error)

    problems = [f"prompt divergence in {d}" for d in provider.divergences]
    if isinstance(result.error, ReplayDivergence) and result.error not in provider.divergences:
        problems.append(str(result.error))
    if not provider.divergences and recording.expected is not None:
        exp = recording.expected
        for i in range(max(len(exp), len(actual))):
            e = exp[i] if i < len(exp) else "<nothing>"
            a = actual[i] if i < len(actual) else "<nothing>"
            if e != a:
                problems.append(f"behavior diverged at step {i}: expected {e!r}, got {a!r}")
                break
    if provider.calls_used < len(recording.calls) and not provider.divergences:
        problems.append(f"harness used {provider.calls_used} of {len(recording.calls)} recorded model calls")
    return ReplayReport(
        not problems, tuple(problems), len(recording.calls), provider.calls_used, recording.expected, actual
    )


__all__ = [
    "ReplayDivergence", "ReplayProvider", "ReplayReport", "RecordedRun", "RunResult",
    "drive", "fingerprint", "record_run", "replay_run",
]  # fmt: skip
