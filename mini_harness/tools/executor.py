"""Tool execution pipeline: lookup -> validate -> policy -> run (timeout) -> truncate.

Design rules
- ADR-003: tool failures NEVER raise; they become is_error results (keeps tool_use/tool_result pairing).
- M2 scheduling: consecutive concurrency-safe calls run in parallel (TaskGroup); a non-safe call runs
  alone, in order. A global Semaphore caps in-flight tool executions across all runs of this executor.
- Results are always returned in the same order as the calls.
"""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from mini_harness.core.errors import TransientToolError
from mini_harness.core.messages import ToolResultBlock, ToolUseBlock
from mini_harness.core.retry import RetryPolicy
from mini_harness.providers.base import INVALID_JSON_KEY
from mini_harness.tools.policy import Policy
from mini_harness.tools.registry import Tool, ToolRegistry

MAX_RESULT_CHARS = 20_000
TRANSIENT = (TimeoutError, ConnectionError, TransientToolError)
DEFAULT_TOOL_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.2, max_delay_s=2.0)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: Policy,
        *,
        max_concurrency: int = 8,
        retry: RetryPolicy = DEFAULT_TOOL_RETRY,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._retry = retry
        self._registry = registry
        self._policy = policy
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run_all(self, calls: list[ToolUseBlock]) -> list[ToolResultBlock]:
        results: list[ToolResultBlock] = []
        for batch in self._batches(calls):
            if len(batch) == 1:
                results.append(await self._run_limited(batch[0]))
                continue
            async with asyncio.TaskGroup() as tg:  # children never raise; parent cancel cancels all
                tasks = [tg.create_task(self._run_limited(c)) for c in batch]
            results.extend(t.result() for t in tasks)
        return results

    def _batches(self, calls: list[ToolUseBlock]) -> list[list[ToolUseBlock]]:
        """Group consecutive safe calls; every unsafe call is its own batch (preserves ordering)."""
        batches: list[list[ToolUseBlock]] = []
        prev_safe = False
        for c in calls:
            tool = self._registry.get(c.name)
            safe = tool is None or tool.spec.concurrency_safe  # unknown tool -> instant error, harmless
            if safe and prev_safe:
                batches[-1].append(c)
            else:
                batches.append([c])
            prev_safe = safe
        return batches

    async def _run_limited(self, call: ToolUseBlock) -> ToolResultBlock:
        async with self._sem:
            try:
                return await self.run_one(call)
            except Exception as e:  # noqa: BLE001 - e.g. a broken Policy must not kill sibling tasks
                return _err(call, f"Internal error: {type(e).__name__}: {e}")

    async def run_one(self, call: ToolUseBlock) -> ToolResultBlock:
        tool = self._registry.get(call.name)
        if tool is None:
            return _err(call, f"Unknown tool '{call.name}'. Available: {self._registry.names()}")
        if INVALID_JSON_KEY in call.input:
            return _err(call, f"Arguments were not valid JSON: {call.input[INVALID_JSON_KEY][:200]!r}")
        try:
            args = tool.validate(call.input)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, x['loc'])) or '<root>'}: {x['msg']}" for x in e.errors())
            return _err(call, f"Invalid arguments: {problems}")

        decision = await self._policy.check(tool.spec, args)
        if not decision.allowed:
            return _err(call, f"Denied by policy: {decision.reason}")

        try:
            output = await self._invoke(tool, args)
        except TimeoutError:
            return _err(call, f"Tool timed out after {tool.spec.timeout_s}s")
        except Exception as e:  # noqa: BLE001 - tool bugs must not crash the loop
            return _err(call, f"{type(e).__name__}: {e}")
        # CancelledError is BaseException: it propagates, and the loop repairs the history.
        return ToolResultBlock(call.id, _truncate(str(output)))

    async def _invoke(self, tool: Tool, args: dict) -> object:
        """Run with timeout. Only idempotent tools are re-run, and only after a *transient* failure
        (timeout, connection error, TransientToolError); deterministic errors are never retried."""
        attempts = self._retry.max_attempts if tool.spec.idempotent else 1
        for attempt in range(1, attempts + 1):
            try:
                async with asyncio.timeout(tool.spec.timeout_s):
                    return await tool.invoke(args)
            except TRANSIENT:
                if attempt >= attempts:
                    raise
                await asyncio.sleep(self._retry.delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover


def _err(call: ToolUseBlock, message: str) -> ToolResultBlock:
    return ToolResultBlock(call.id, message, is_error=True)


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_RESULT_CHARS} chars]"
