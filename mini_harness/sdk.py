"""Public facade. Composition root: wires concrete parts into the loop."""

from __future__ import annotations

from collections.abc import AsyncIterator

from mini_harness.core.compaction import Compactor, LLMSummarizer
from mini_harness.core.context import ContextConfig, ContextManager
from mini_harness.core.errors import HarnessError
from mini_harness.core.events import Event
from mini_harness.core.limits import Limits
from mini_harness.core.loop import AgentLoop
from mini_harness.core.pricing import Pricing
from mini_harness.core.retry import RetryPolicy
from mini_harness.core.session import Session
from mini_harness.core.supervisor import supervise
from mini_harness.observability.tracer import NullTracer, Tracer
from mini_harness.providers.base import Provider
from mini_harness.session.manager import SessionManager
from mini_harness.session.models import SessionStore
from mini_harness.tools.executor import DEFAULT_TOOL_RETRY, ToolExecutor
from mini_harness.tools.policy import DefaultPolicy, Policy
from mini_harness.tools.registry import ToolRegistry

DEFAULT_SYSTEM = "You are a helpful agent. Use the provided tools when they help; be concise."


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        *,
        model: str,
        system: str = DEFAULT_SYSTEM,
        policy: Policy | None = None,
        limits: Limits | None = None,
        pricing: Pricing | None = None,
        tool_retry: RetryPolicy = DEFAULT_TOOL_RETRY,
        context: ContextConfig | None = None,
        compactor: Compactor | None = None,
        store: SessionStore | None = None,
        tracer: Tracer | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._limits = limits or Limits()
        self._tracer = tracer or NullTracer()
        executor = ToolExecutor(
            registry,
            policy or DefaultPolicy(),
            max_concurrency=self._limits.max_tool_concurrency,
            retry=tool_retry,
            tracer=self._tracer,
        )
        self._sessions = SessionManager(store) if store is not None else None
        ctx_config = context or ContextConfig()
        context_manager = ContextManager(ctx_config, compactor or LLMSummarizer(provider, model))
        self._loop = AgentLoop(
            provider,
            registry,
            executor,
            model=model,
            system=system,
            limits=self._limits,
            pricing=pricing,
            context=context_manager,
            checkpoint=self._sessions.checkpoint if self._sessions else None,
            tracer=self._tracer,
            max_tokens=max_tokens,
        )

    @property
    def tracer(self) -> Tracer:
        """Spans recorded so far (empty unless a Tracer was passed)."""
        return self._tracer

    @property
    def sessions(self) -> SessionManager:
        """Persistence workflows (open / rewind / checkpoints). Requires `store=`."""
        if self._sessions is None:
            raise HarnessError("this Agent has no store; pass store=SQLiteStore(...)")
        return self._sessions

    def new_session(self, session_id: str | None = None) -> Session:
        return Session(id=session_id) if session_id else Session()

    async def open_session(self, session_id: str) -> Session:
        """Reload a persisted session at its latest checkpoint (crash-safe)."""
        return await self.sessions.open(session_id)

    async def rewind(self, session_id: str, checkpoint_id: int) -> Session:
        """Branch back to an earlier checkpoint; history is preserved. Does not undo external side effects."""
        return await self.sessions.rewind(session_id, checkpoint_id)

    def run(self, user_input: str | None, session: Session) -> AsyncIterator[Event]:
        """Stream events for one run. Pass None to continue from the current state (after open/rewind).

        Cancel by cancelling the consuming task or calling aclose()."""
        turns_before = session.turns
        return supervise(
            self._loop.run(session, user_input),
            timeout_s=self._limits.wall_clock_s,
            on_timeout=lambda: self._loop.done("timeout", session.turns - turns_before, session),
        )
