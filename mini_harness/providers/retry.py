"""RetryingProvider: a decorator adding backoff retries to ANY Provider (OCP: adapters stay retry-free).

ADR-004: a stream that dies midway is not resumed. The partial output is discarded and the whole call
is replayed; a RetryNotice tells downstream consumers to drop what they saw from the failed attempt.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable

from mini_harness.core.errors import ProviderError
from mini_harness.core.retry import RetryPolicy
from mini_harness.providers.base import ModelRequest, Provider, RetryNotice, StreamEvent


class RetryingProvider:
    def __init__(
        self,
        inner: Provider,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._inner = inner
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self.name = inner.name

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async for ev in self._inner.stream(req):
                    emitted = True
                    yield ev
                return
            except ProviderError as e:
                if not e.retryable or attempt >= self._policy.max_attempts:
                    raise
                delay = self._policy.delay(attempt, e.retry_after_s, self._rng)
                yield RetryNotice(attempt, delay, str(e), discarded_partial=emitted)
                await self._sleep(delay)
