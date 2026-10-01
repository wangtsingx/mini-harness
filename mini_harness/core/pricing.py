from __future__ import annotations

from dataclasses import dataclass

from mini_harness.core.messages import Usage


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Supplied by the caller: prices change and differ per model.

    Assumes cache read/write tokens are reported separately from input_tokens (as Anthropic does).
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0

    def cost(self, u: Usage) -> float:
        return (
            u.input_tokens * self.input_per_mtok
            + u.output_tokens * self.output_per_mtok
            + u.cache_read_tokens * self.cache_read_per_mtok
            + u.cache_write_tokens * self.cache_write_per_mtok
        ) / 1_000_000
