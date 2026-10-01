from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter. Shared by model calls and idempotent tools."""

    max_attempts: int = 4  # total attempts, including the first
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def delay(
        self,
        attempt: int,
        retry_after_s: float | None = None,
        rng: Callable[[], float] = random.random,
    ) -> float:
        """Delay before retrying after failed attempt number `attempt` (1-based).

        A server-provided Retry-After wins, bounded by max_delay_s so a run can never hang on it.
        """
        if retry_after_s is not None:
            return min(max(retry_after_s, 0.0), self.max_delay_s)
        cap = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return rng() * cap
