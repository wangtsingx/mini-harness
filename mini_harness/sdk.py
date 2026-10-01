"""Public facade. Composition root: wires concrete parts into the loop."""

from __future__ import annotations

from collections.abc import AsyncIterator

from mini_harness.core.compaction import Compactor, LLMSummarizer
from mini_harness.core.context import ContextConfig, ContextManager
from mini_harness.core.events import Event
from mini_harness.core.limits import Limits
from mini_harness.core.loop import AgentLoop
from mini_harness.core.pricing import Pricing
from mini_harness.core.retry import RetryPolicy
from mini_harness.core.session import Session
from mini_harness.core.supervisor import supervise
from mini_harness.providers.base import Provider
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
        max_tokens: int = 4096,
    ) -> None:
        self._limits = limits or Limits()
        executor = ToolExecutor(
            registry,
            policy or DefaultPolicy(),
            max_concurrency=self._limits.max_tool_concurrency,
            retry=tool_retry,
        )
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
            max_tokens=max_tokens,
        )

    def new_session(self) -> Session:
        return Session()

    def run(self, user_input: str, session: Session) -> AsyncIterator[Event]:
        """Stream events for one user turn. Cancel by cancelling the consuming task or calling aclose()."""
        turns_before = session.turns
        return supervise(
            self._loop.run(session, user_input),
            timeout_s=self._limits.wall_clock_s,
            on_timeout=lambda: self._loop.done("timeout", session.turns - turns_before, session),
        )
