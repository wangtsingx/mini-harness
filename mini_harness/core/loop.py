"""AgentLoop: the control flow only (SRP). Model I/O, tools and storage are injected (DIP).

Wall-clock limits and cancellation plumbing live in `supervisor.py`, not here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

from mini_harness.core.context import ContextConfig, ContextManager
from mini_harness.core.errors import HarnessError
from mini_harness.core.events import AssistantText, Compacted, Done, Event, Retrying, ToolFinished, ToolStarted
from mini_harness.core.guards import RepeatDetector, Verdict
from mini_harness.core.limits import Limits
from mini_harness.core.messages import Message, ToolResultBlock, close_dangling_tool_uses, validate_pairing
from mini_harness.core.pricing import Pricing
from mini_harness.core.session import Session
from mini_harness.providers.base import ModelRequest, Provider, RetryNotice, StreamAssembler, TextDelta
from mini_harness.tools.executor import ToolExecutor
from mini_harness.tools.registry import ToolRegistry

REPEAT_WARNING = (
    "\n\n[harness] You have issued the same tool call(s) several times in a row with identical arguments. "
    "Change your approach or answer with what you already have; another identical repeat will stop this run."
)
REPEAT_STOPPED = "Not executed: the harness stopped this run because identical tool calls kept repeating."


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
        self._max_tokens = max_tokens

    def done(self, reason: str, turns: int, session: Session) -> Done:
        cost = self._pricing.cost(session.usage) if self._pricing else None
        return Done(reason, turns, session.usage, cost)

    async def run(self, session: Session, user_input: str) -> AsyncIterator[Event]:
        lim = self._limits
        session.messages.append(Message.user(user_input))
        detector = RepeatDetector(lim.repeat_call_threshold)
        turns = 0
        try:
            while True:
                # ---- guardrails, checked before every model call
                if turns >= lim.max_turns:
                    yield self.done("max_turns", turns, session)
                    return
                if session.usage.total >= lim.max_total_tokens:
                    yield self.done("budget_exceeded", turns, session)
                    return
                if lim.max_cost_usd is not None and self._pricing.cost(session.usage) >= lim.max_cost_usd:
                    yield self.done("budget_exceeded", turns, session)
                    return
                # ---- keep the prompt inside the window (trim -> summarize); history stays append-only otherwise
                compaction = await self._context.prepare(session, self._system, self._registry.specs())
                if compaction is not None:
                    session.usage = session.usage + compaction.usage
                    yield Compacted(
                        compaction.stage, compaction.tokens_before, compaction.tokens_after, compaction.archived
                    )
                turns += 1
                session.turns += 1

                # ---- one model call (streamed)
                request = self._build_request(session)
                assembler = StreamAssembler()
                async for ev in self._provider.stream(request):
                    if isinstance(ev, TextDelta):
                        yield AssistantText(ev.text)
                    elif isinstance(ev, RetryNotice):
                        yield Retrying(ev.attempt, ev.delay_s, ev.reason, ev.discarded_partial)
                    assembler.feed(ev)
                result = assembler.finish()
                session.usage = session.usage + result.usage
                prompt_tokens = (
                    result.usage.input_tokens + result.usage.cache_read_tokens + result.usage.cache_write_tokens
                )
                if prompt_tokens:  # ground truth for the next sizing decision
                    session.last_prompt_tokens, session.last_prompt_msgs = prompt_tokens, len(request.messages)
                if result.message.content:
                    session.messages.append(result.message)

                # ---- no tool calls: the run is over
                calls = result.message.tool_uses()
                if not calls:
                    reason = "max_tokens" if result.stop_reason == "max_tokens" else "end_turn"
                    yield self.done(reason, turns, session)
                    return

                # ---- loop-health check
                verdict = detector.observe(calls)
                if verdict is Verdict.STOP:
                    stopped = [ToolResultBlock(c.id, REPEAT_STOPPED, True) for c in calls]
                    session.messages.append(Message.tool_results(stopped))  # keep pairing valid
                    yield self.done("repeated_calls", turns, session)
                    return

                # ---- execute tools (parallel where safe), feed results back
                for c in calls:
                    yield ToolStarted(c.id, c.name, c.input)
                results = await self._executor.run_all(calls)
                if verdict is Verdict.WARN:
                    results[-1] = replace(results[-1], content=results[-1].content + REPEAT_WARNING)
                names = {c.id: c.name for c in calls}
                for r in results:
                    yield ToolFinished(r.tool_use_id, names[r.tool_use_id], r.is_error, r.content)
                session.messages.append(Message.tool_results(results))
        finally:
            # Cancel / timeout / consumer abort / crash mid-tool: keep history valid for the next request.
            close_dangling_tool_uses(session.messages)

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
            cache=self._context.cache_plan(session.messages),
        )
