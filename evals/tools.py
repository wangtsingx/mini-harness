"""Read-only tools for the real-model eval: the workspace file tools + grep + a safe calculator."""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mini_harness.core.errors import ToolError
from mini_harness.tools.builtin import build_fs_registry
from mini_harness.tools.registry import ToolRegistry
from mini_harness.tools.spec import Permission

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}  # fmt: skip
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_FILE_BYTES = 2_000_000


def safe_eval(expression: str) -> int | float:
    """Arithmetic only: numbers, + - * / // % ** and parentheses. No names, calls or attributes."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}") from e

    def ev(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 50:
            raise ToolError("expression is nested too deeply")
        if isinstance(node, ast.Expression):
            return ev(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand, depth + 1))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left, depth + 1), ev(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or (abs(left) > 1e6 and abs(right) > 10)):
                raise ToolError("exponent too large")
            try:
                return _BIN[type(node.op)](left, right)
            except ZeroDivisionError as e:
                raise ToolError("division by zero") from e
        raise ToolError("unsupported expression: only numbers and + - * / // % ** ( ) are allowed")

    return ev(tree)


def format_number(x: int | float) -> str:
    return str(x) if isinstance(x, int) else f"{x:.12g}"


def build_eval_registry(workspace: Path | str) -> ToolRegistry:
    root = Path(workspace).resolve()
    registry = build_fs_registry(root)

    def safe(rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ToolError("path escapes the workspace")
        return p

    @registry.tool(permission=Permission.READ, timeout_s=15)
    def grep(
        pattern: Annotated[str, Field(description="Python regular expression (case-sensitive)")],
        path: Annotated[str, Field(description="File or directory relative to the workspace")] = ".",
        max_matches: Annotated[int, Field(description="How many matching lines to print", ge=1, le=200)] = 50,
    ) -> str:
        """Search file contents for lines matching a regex. Prints `file:line:text` and ALWAYS ends with the total
        number of matching lines, so it can be used for counting."""
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}") from e
        target = safe(path)
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        shown: list[str] = []
        total = 0
        for f in files:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            rel = f.relative_to(root)
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(shown) < max_matches:
                        shown.append(f"{rel}:{n}:{line}")
        shown.append(f"(total matching lines: {total}; shown: {min(total, max_matches)})")
        return "\n".join(shown)

    @registry.tool(permission=Permission.READ, timeout_s=5)
    def calculate(expression: Annotated[str, Field(description="Arithmetic, e.g. (1234*5678)/20")]) -> str:
        """Evaluate an arithmetic expression exactly. Use it instead of doing arithmetic in your head."""
        return format_number(safe_eval(expression))

    return registry
