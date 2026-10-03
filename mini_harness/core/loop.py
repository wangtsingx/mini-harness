"""AgentLoop: the control flow only (SRP). Model I/O, tools, context, persistence and tracing are injected (DIP).

Wall-clock limits and cancellation plumbing live in `supervisor.py`, not here.

Structure: run() owns guardrails, compaction and the run-level lifecycle; _turn() is exactly one model call plus
its tool round. Checkpoints are written at turn boundaries only, so a stored state is always a valid conversation.
Spans: run -> {compaction, turn -> {model, tool*}}.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace

from mini_harness.core.context import ContextConfig, ContextManager
from mini_harness.core.errors import HarnessError
from mini_harness.core.events import AssistantText, Compacted, Done, Event, Retrying, ToolFinished, ToolStarted
from mini_harness.core.guards import RepeatDetector, Verdict
from mini_harness.core.limits import Limits
from mini_harness.core.messages import Message, ToolResultBlock, close_dangling_tool_uses, validate_pairing
from mini_harness.core.pricing import Pricing
from mini_harness.core.session import Session
from mini_harness.observability.tracer import NullTracer, Span, Tracer
from mini_harness.providers.base import (
    ModelRequest,
    Provider,
    RetryNotice,
    StreamAssembler,
    TextDelta,
    ToolCallStart,
)
from mini_harness.tools.executor import ToolExecutor
from mini_harness.tools.registry import ToolRegistry

REPEAT_WARNING = (
    "\n\n[harness] You have issued the same tool call(s) several times in a row with identical arguments. "
    "Change your approach or answer with what you already have; another identical repeat will stop this run."
)
REPEAT_STOPPED = "Not executed: the harness stopped this run because identical tool calls kept repeating."
INTERRUPTED_STATUSES = {"running", "cancelled", "failed"}

Checkpointer = Callable[[Session, str], Awaitable[object]]


class AgentLoop:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        *,
        model: str,
        system: str,
        limits: Limits | None = None,
        pricing: Pricing | None = None,
        context: ContextManager | None = None,
        checkpoint: Checkpointer | None = None,
        tracer: Tracer | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._limits = limits or Limits()
        if self._limits.max_cost_usd is not None and pricing is None:
            raise HarnessError("Limits.max_cost_usd requires a Pricing table")
        self._provider = provider
        self._registry = registry
        self._executor = executor
        self._model = model
        self._system = system
        self._pricing = pricing
        self._context = context or ContextManager(ContextConfig(enabled=False))
        self._checkpoint = checkpoint
        self._tracer = tracer or NullTracer()
        self._max_tokens = max_tokens

    def done(self, reason: str, turns: int, session: Session) -> Done:
        cost = self._pricing.cost(session.usage) if self._pricing else None
        return Done(reason, turns, session.usage, cost)

    async def _save(self, session: Session, label: str) -> None:
        if self._checkpoint is not None:
            await self._checkpoint(session, label)

    async def _finish(self, session: Session, reason: str, turns: int) -> Done:
        session.status = "completed" if reason == "end_turn" else reason
        await self._save(session, f"done: {reason}")
        return self.done(reason, turns, session)

    async def run(self, session: Session, user_input: str | None) -> AsyncIterator[Event]:
        """One run. `user_input=None` continues from the current state (e.g. after recovery or a rewind to a
        user checkpoint); it requires the history to end with a user or tool message."""
        lim = self._limits
        if user_input is not None:
            session.messages.append(Message.user(user_input))
        elif not session.messages or session.messages[-1].role == "assistant":
            raise HarnessError("nothing to continue: history must end with a user or tool message")
        session.status = "running"
        run_span = self._tracer.start(
            "run", "run", session_id=session.id, branch=session.branch, resumed=user_input is None
        )
        detector = RepeatDetector(lim.repeat_call_threshold)
        usage_before, turns = session.usage, 0
        try:
            if user_input is not None:
                await self._save(session, f"user: {user_input.strip().replace(chr(10), ' ')[:60]}")
            while True:
                # ---- guardrails, checked before every model call
                if turns >= lim.max_turns:
                    yield await self._finish(session, "max_turns", turns)
                    return
                if session.usage.total >= lim.max_total_tokens:
                    yield await self._finish(session, "budget_exceeded", turns)
                    return
                if lim.max_cost_usd is not None and self._pricing.cost(session.usage) >= lim.max_cost_usd:
                    yield await self._finish(session, "budget_exceeded", turns)
                    return

                # ---- keep the prompt inside the window (trim -> summarize); history stays append-only otherwise
                cspan = self._tracer.start("compaction", "compaction", run_span)
                try:
                    compaction = await self._context.prepare(session, self._system, self._registry.specs())
                except BaseException as e:
                    self._tracer.end(cspan, exc=e)
                    raise
                if compaction is not None:  # no span for the (common) no-op case
                    self._tracer.end(
                        cspan, stage=compaction.stage, tokens_before=compaction.tokens_before,
                        tokens_after=compaction.tokens_after, archived=compaction.archived,
                    )  # fmt: skip
                    session.usage = session.usage + compaction.usage
                    yield Compacted(
                        compaction.stage, compaction.tokens_before, compaction.tokens_after, compaction.archived
                    )
                turns += 1
                session.turns += 1

                async with contextlib.aclosing(self._turn(session, turns, run_span, detector)) as turn_events:
                    async for ev in turn_events:
                        yield ev
                        if isinstance(ev, Done):
                            return
        except asyncio.CancelledError:
            session.status = "cancelled"
            raise
        except Exception:
            session.status = "failed"
            raise
        finally:
            # Cancel / timeout / consumer abort / crash: keep history valid, and persist what we have.
            exc = sys.exc_info()[1]
            close_dangling_tool_uses(session.messages)
            if session.status in INTERRUPTED_STATUSES:
                if session.status == "running":  # consumer stopped iterating
                    session.status = "cancelled"
                with contextlib.suppress(Exception):  # best effort; the last turn-boundary checkpoint still stands
                    await self._save(session, "interrupted")
            used = session.usage
            self._tracer.end(
                run_span, exc=exc, reason=session.status, turns=turns,
                input_tokens=used.input_tokens - usage_before.input_tokens,
                output_tokens=used.output_tokens - usage_before.output_tokens,
                cache_read_tokens=used.cache_read_tokens - usage_before.cache_read_tokens,
                cache_write_tokens=used.cache_write_tokens - usage_before.cache_write_tokens,
                cost_usd=(self._pricing.cost(used) - self._pricing.cost(usage_before)) if self._pricing else None,
            )  # fmt: skip

    async def _turn(
        self, session: Session, turns: int, run_span: Span, detector: RepeatDetector
    ) -> AsyncIterator[Event]:
        """One model call and its tool round. Yields a final Done if the run ends in this turn."""
        tracer = self._tracer
        turn_span = tracer.start("turn", f"turn {turns}", run_span, index=turns)
        completed = False  # set once the turn's work is done; closing the generator after that is not a cancel
        try:
            request = self._build_request(session)
            result = None
            retries, ttft = 0, None
            model_span = tracer.start(
                "model", self._model, turn_span, model=self._model, provider=getattr(self._provider, "name", "?"),
                messages=len(request.messages),
            )  # fmt: skip
            try:
                assembler = StreamAssembler()
                async for ev in self._provider.stream(request):
                    if ttft is None and isinstance(ev, TextDelta | ToolCallStart):
                        ttft = tracer.elapsed(model_span)
                    if isinstance(ev, TextDelta):
                        yield AssistantText(ev.text)
                    elif isinstance(ev, RetryNotice):
                        retries += 1
                        ttft = None  # the failed attempt's first token no longer counts
                        yield Retrying(ev.attempt, ev.delay_s, ev.reason, ev.discarded_partial)
                    assembler.feed(ev)
                result = assembler.finish()
            finally:
                u = result.usage if result else None
                tracer.end(
                    model_span, exc=sys.exc_info()[1], retries=retries, ttft_s=ttft,
                    stop_reason=result.stop_reason if result else None,
                    input_tokens=u.input_tokens if u else 0, output_tokens=u.output_tokens if u else 0,
                    cache_read_tokens=u.cache_read_tokens if u else 0,
                    cache_write_tokens=u.cache_write_tokens if u else 0,
                )  # fmt: skip

            session.usage = session.usage + result.usage
            prompt_tokens = result.usage.input_tokens + result.usage.cache_read_tokens + result.usage.cache_write_tokens
            if prompt_tokens:  # ground truth for the next sizing decision
                session.last_prompt_tokens, session.last_prompt_msgs = prompt_tokens, len(request.messages)
            if result.message.content:
                session.messages.append(result.message)

            # ---- no tool calls: the run is over
            calls = result.message.tool_uses()
            if not calls:
                reason = "max_tokens" if result.stop_reason == "max_tokens" else "end_turn"
                done = await self._finish(session, reason, turns)
                completed = True
                yield done
                return

            # ---- loop-health check
            verdict = detector.observe(calls)
            if verdict is Verdict.STOP:
                stopped = [ToolResultBlock(c.id, REPEAT_STOPPED, True) for c in calls]
                session.messages.append(Message.tool_results(stopped))  # keep pairing valid
                done = await self._finish(session, "repeated_calls", turns)
                completed = True
                yield done
                return

            # ---- execute tools (parallel where safe), feed results back
            for c in calls:
                yield ToolStarted(c.id, c.name, c.input)
            results = await self._executor.run_all(calls, turn_span)
            if verdict is Verdict.WARN:
                results[-1] = replace(results[-1], content=results[-1].content + REPEAT_WARNING)
            names = {c.id: c.name for c in calls}
            for r in results:
                yield ToolFinished(r.tool_use_id, names[r.tool_use_id], r.is_error, r.content)
            session.messages.append(Message.tool_results(results))
            await self._save(session, f"turn {session.turns}")
        finally:
            exc = sys.exc_info()[1]
            tracer.end(turn_span, exc=None if completed and isinstance(exc, GeneratorExit) else exc)

    def _build_request(self, session: Session) -> ModelRequest:
        problems = validate_pairing(session.messages)
        if problems:
            raise HarnessError(f"message history invariant violated: {problems}")
        return ModelRequest(
            model=self._model,
            system=self._system,
            messages=list(session.messages),
            tools=self._registry.specs(),
            max_tokens=self._max_tokens,
            cache=self._context.cache_plan(session.messages) if self._context.caching_enabled else None,
        )
