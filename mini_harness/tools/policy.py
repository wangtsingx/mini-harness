from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mini_harness.tools.spec import Permission, ToolSpec


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


class Policy(Protocol):
    async def check(self, spec: ToolSpec, args: dict[str, Any]) -> Decision: ...


class DefaultPolicy:
    """Least privilege: only explicitly granted permission classes are allowed (default: READ)."""

    def __init__(self, allowed: frozenset[Permission] = frozenset({Permission.READ})) -> None:
        self._allowed = allowed

    async def check(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        if spec.permission in self._allowed:
            return Decision(True)
        return Decision(False, f"permission '{spec.permission.value}' is not granted")
