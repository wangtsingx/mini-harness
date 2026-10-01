from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    NETWORK = "network"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: Permission = Permission.READ
    timeout_s: float = 30.0
    concurrency_safe: bool = True  # may run in parallel with other safe calls of the same turn
    idempotent: bool = False  # safe to auto-retry after failure (consumed in M3)
