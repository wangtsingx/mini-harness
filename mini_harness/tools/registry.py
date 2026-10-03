"""Tool registration: a decorated function -> ToolSpec (JSON Schema via pydantic) + validator."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from mini_harness.tools.spec import Permission, ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., Any]
    args_model: type[BaseModel] | None = None  # subclasses (e.g. McpTool) may validate differently

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Raises pydantic.ValidationError (or ArgumentError) on bad input; extra keys are rejected."""
        assert self.args_model is not None
        model = self.args_model(**raw)
        return {name: getattr(model, name) for name in type(model).model_fields}

    async def invoke(self, args: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**args)
        return await asyncio.to_thread(self.fn, **args)  # never block the event loop


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def tool(
        self,
        *,
        permission: Permission = Permission.READ,
        timeout_s: float = 30.0,
        name: str | None = None,
        concurrency_safe: bool | None = None,
        idempotent: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a function as a tool.

        `concurrency_safe` defaults to True only for READ tools: anything that can mutate state
        or the outside world is serialized unless the author explicitly opts in.
        """
        safe = permission == Permission.READ if concurrency_safe is None else concurrency_safe

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(_build_tool(fn, name or fn.__name__, permission, timeout_s, safe, idempotent))
            return fn

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]


def _build_tool(
    fn: Callable[..., Any], name: str, permission: Permission, timeout_s: float, safe: bool, idempotent: bool
) -> Tool:
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (hints.get(pname, Any), default)
    model = create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    spec = ToolSpec(name, inspect.getdoc(fn) or name, schema, permission, timeout_s, safe, idempotent)
    return Tool(spec, fn, model)
