"""Built-in read-only filesystem tools, confined to a workspace directory."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mini_harness.tools.registry import ToolRegistry
from mini_harness.tools.spec import Permission


def build_fs_registry(workspace: Path | str) -> ToolRegistry:
    root = Path(workspace).resolve()
    registry = ToolRegistry()

    def safe(rel: str) -> Path:
        p = (root / rel).resolve()  # resolve() first: defeats '..' and symlink escapes
        if not p.is_relative_to(root):
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    @registry.tool(permission=Permission.READ, timeout_s=10)
    def read_file(path: Annotated[str, Field(description="File path relative to the workspace")]) -> str:
        """Read a UTF-8 text file from the workspace."""
        return safe(path).read_text(encoding="utf-8", errors="replace")

    @registry.tool(permission=Permission.READ, timeout_s=10)
    def list_dir(path: Annotated[str, Field(description="Directory relative to the workspace")] = ".") -> str:
        """List entries of a workspace directory (directories end with '/')."""
        entries = sorted(safe(path).iterdir(), key=lambda p: (p.is_file(), p.name))
        return "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries) or "(empty)"

    return registry
