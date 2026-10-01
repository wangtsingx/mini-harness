class HarnessError(Exception):
    """Unrecoverable harness/config error (bugs, broken invariants)."""


class ProviderError(HarnessError):
    """Model-provider failure. `retryable` tells RetryingProvider whether a retry can help."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class TransientToolError(Exception):
    """Raise from a tool to signal a temporary failure. Idempotent tools are retried on it."""
