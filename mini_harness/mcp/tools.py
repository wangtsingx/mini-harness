"""Expose MCP server tools as ordinary harness Tools (the loop cannot tell the difference: LSP).

Security stance
- Permission defaults to EXEC unless the server says readOnlyHint. Annotations come from the server, which may be
  malicious or buggy, so they only RELAX nothing by themselves: DefaultPolicy still denies EXEC. Pass `permission_for`
  to pin permissions per tool, and `allow=` to expose only the tools you reviewed.
- Tool descriptions and results are untrusted text that reaches the model (prompt-injection surface).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Collection
from typing import Any

import jsonschema

from mini_harness.core.errors import ArgumentError, ToolError
from mini_harness.mcp.client import McpClient, McpToolInfo
from mini_harness.tools.registry import Tool, ToolRegistry
from mini_harness.tools.spec import Permission, ToolSpec

MAX_NAME = 64  # limit shared by the major model APIs: ^[a-zA-Z0-9_-]{1,64}$
MAX_DESCRIPTION = 2000


def mcp_tool_name(server: str, tool: str) -> str:
    """`server__tool`, sanitized to the API-safe charset; long names get a stable hash suffix."""
    raw = f"{server}__{tool}"
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if len(name) <= MAX_NAME:
        return name
    return f"{name[: MAX_NAME - 9]}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def default_permission(info: McpToolInfo) -> Permission:
    return Permission.READ if info.annotations.get("readOnlyHint") is True else Permission.EXEC


class McpTool(Tool):
    def __init__(self, spec: ToolSpec, client: McpClient, remote_name: str, schema: dict[str, Any]) -> None:
        super().__init__(spec=spec, fn=lambda **_: None, args_model=None)
        self._client = client
        self._remote_name = remote_name
        self._validator: jsonschema.protocols.Validator | None = None
        try:  # an invalid server schema must not break the tool: the server validates too
            cls = jsonschema.validators.validator_for(schema)
            cls.check_schema(schema)
            self._validator = cls(schema)
        except Exception:  # noqa: BLE001
            self._validator = None

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._validator is not None:
            errors = sorted(self._validator.iter_errors(raw), key=lambda e: [str(p) for p in e.absolute_path])
            if errors:
                raise ArgumentError(
                    [f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors[:5]]
                )
        return dict(raw)

    async def invoke(self, args: dict[str, Any]) -> str:
        from mini_harness.mcp.client import McpError

        try:
            result = await self._client.call_tool(self._remote_name, args)
        except McpError as e:
            raise ToolError(f"MCP error: {e}") from e
        if result.is_error:
            raise ToolError(result.text or "the MCP tool reported an error")
        return result.text


def mcp_tools(
    client: McpClient,
    infos: list[McpToolInfo],
    *,
    prefix: str | None = None,
    permission_for: Callable[[McpToolInfo], Permission] = default_permission,
    timeout_s: float = 60.0,
    allow: Collection[str] | None = None,
    deny: Collection[str] = (),
) -> list[McpTool]:
    tools: list[McpTool] = []
    for info in infos:
        if (allow is not None and info.name not in allow) or info.name in deny:
            continue
        spec = ToolSpec(
            name=mcp_tool_name(prefix or client.name, info.name),
            description=(info.description or info.name)[:MAX_DESCRIPTION],
            input_schema=info.input_schema,
            permission=permission_for(info),
            timeout_s=timeout_s,
            concurrency_safe=permission_for(info) == Permission.READ,
            idempotent=info.annotations.get("idempotentHint") is True,
        )
        tools.append(McpTool(spec, client, info.name, info.input_schema))
    return tools


async def register_mcp_tools(registry: ToolRegistry, client: McpClient, **options: Any) -> list[str]:
    """List the server's tools and register them. Returns the registered (prefixed) names."""
    tools = mcp_tools(client, await client.list_tools(), **options)
    for t in tools:
        registry.register(t)
    return [t.spec.name for t in tools]
