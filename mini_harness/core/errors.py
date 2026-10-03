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


class SessionNotFound(HarnessError):
    pass


class CheckpointNotFound(HarnessError):
    pass


class StaleSessionError(HarnessError):
    """The branch head moved since this Session was loaded (concurrent writer or reused session id)."""


class ToolError(Exception):
    """Raise from a tool for an EXPECTED failure. The message goes to the model verbatim (no exception-type prefix)."""


class ArgumentError(ValueError):
    """Tool arguments failed validation. `problems` are short, model-readable messages."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
