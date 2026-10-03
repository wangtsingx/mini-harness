"""Minimal MCP (Model Context Protocol) client over stdio: JSON-RPC 2.0, newline-delimited.

Scope: initialize handshake, tools/list (paginated), tools/call, cancellation, ping. Not implemented: HTTP transport,
resources/prompts, sampling, list_changed refresh. Everything a server sends is UNTRUSTED input.

Safety defaults
- The server process gets a minimal environment (PATH, HOME, ...) plus only the variables you pass explicitly:
  your API keys are not inherited by accident.
- A call that is cancelled or times out tells the server (notifications/cancelled) and never leaves a dangling future.
- If the server dies, every pending and future call fails fast with McpError (no hangs).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mini_harness.core.errors import HarnessError

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SAFE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SHELL",
    "TERM",
    "SYSTEMROOT",
)
MAX_LINE_BYTES = 16 * 1024 * 1024  # one JSON-RPC message per line; tool results can be large


class McpError(HarnessError):
    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class McpServerParams:
    command: str
    args: Sequence[str] = ()
    env: Mapping[str, str] | None = None  # added on top of the minimal safe environment
    cwd: str | None = None


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)  # hints from the server: untrusted


@dataclass(frozen=True)
class McpCallResult:
    text: str  # content rendered for the model
    is_error: bool


def build_env(extra: Mapping[str, str] | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    env.update(extra or {})
    return env


def render_result(result: dict[str, Any]) -> McpCallResult:
    """Flatten MCP content blocks into text. Binary payloads are described, never inlined."""
    parts: list[str] = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} omitted: {block.get('mimeType', '?')}, {len(block.get('data', ''))} base64 chars]")
        elif kind == "resource_link":
            parts.append(f"[resource: {block.get('uri', '?')}]")
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(res["text"] if "text" in res else f"[binary resource: {res.get('uri', '?')}]")
        else:
            parts.append(f"[unsupported content type: {kind}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return McpCallResult("\n".join(parts), bool(result.get("isError")))


class McpClient:
    def __init__(
        self,
        name: str,
        params: McpServerParams,
        *,
        request_timeout_s: float = 30.0,
        client_name: str = "mini-harness",
        client_version: str = "0.7.0",
    ) -> None:
        self.name = name
        self._params = params
        self._request_timeout_s = request_timeout_s
        self._client_info = {"name": client_name, "version": client_version}
        self._proc: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=50)
        self._dead: McpError | None = None
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None

    # ------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> McpClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._dead is None

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    async def start(self) -> None:
        p = self._params
        try:
            self._proc = await asyncio.create_subprocess_exec(
                p.command, *p.args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=build_env(p.env), cwd=p.cwd, limit=MAX_LINE_BYTES,
            )  # fmt: skip
        except OSError as e:
            raise McpError(f"cannot start MCP server {self.name!r}: {e}") from e
        self._reader = asyncio.create_task(self._read_loop(), name=f"mcp-read-{self.name}")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name=f"mcp-stderr-{self.name}")
        try:
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": SUPPORTED_PROTOCOL_VERSIONS[0],
                    "capabilities": {},
                    "clientInfo": self._client_info,
                },
                timeout=self._request_timeout_s,
            )
            version = result.get("protocolVersion")
            if version not in SUPPORTED_PROTOCOL_VERSIONS:
                raise McpError(f"server {self.name!r} speaks unsupported protocol version {version!r}")
            self.protocol_version = version
            self.server_info = result.get("serverInfo") or {}
            self.capabilities = result.get("capabilities") or {}
            await self._notify("notifications/initialized")
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        self._dead = self._dead or McpError(f"MCP server {self.name!r} was closed")
        self._fail_pending(self._dead)
        with contextlib.suppress(Exception):
            proc.stdin.close()  # EOF is the polite way to ask a stdio server to exit
        try:
            await asyncio.wait_for(proc.wait(), 2.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 2.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        for task in (self._reader, self._stderr_task):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (self._reader, self._stderr_task) if t), return_exceptions=True)

    # ------------------------------------------------------------ API
    async def list_tools(self) -> list[McpToolInfo]:
        tools: list[McpToolInfo] = []
        cursor: str | None = None
        for _ in range(100):  # pagination guard against a server that never ends
            result = await self._request(
                "tools/list", {"cursor": cursor} if cursor else {}, timeout=self._request_timeout_s
            )
            for t in result.get("tools", []):
                tools.append(
                    McpToolInfo(
                        t["name"], t.get("description") or t.get("title") or "",
                        t.get("inputSchema") or {"type": "object", "properties": {}}, t.get("annotations") or {},
                    )
                )  # fmt: skip
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        raise McpError(f"server {self.name!r}: tools/list pagination did not terminate")

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> McpCallResult:
        """No internal timeout: the caller (ToolExecutor) owns it. Cancellation is forwarded to the server."""
        result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        return render_result(result)

    # ------------------------------------------------------------ JSON-RPC plumbing
    async def _send(self, msg: dict[str, Any]) -> None:
        if self._dead is not None or self._proc is None:
            raise self._dead or McpError(f"MCP server {self.name!r} is not running")
        data = (json.dumps(msg, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        try:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
        except (ConnectionError, BrokenPipeError) as e:
            raise McpError(f"MCP server {self.name!r} closed its input: {e}") from e

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def _request(self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        if self._dead is not None:
            raise self._dead
        rid = next(self._ids)
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            await self._send(msg)
            async with asyncio.timeout(timeout):
                return await fut
        except (asyncio.CancelledError, TimeoutError):
            self._pending.pop(rid, None)
            if self._dead is None:  # tell the server to stop working on it (best effort)
                with contextlib.suppress(Exception):
                    await self._notify("notifications/cancelled", {"requestId": rid, "reason": "client gave up"})
            raise
        finally:
            self._pending.pop(rid, None)

    def _fail_pending(self, exc: McpError) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def _read_loop(self) -> None:
        assert self._proc is not None
        stdout = self._proc.stdout
        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr.append(f"[non-JSON on stdout] {line[:200]!r}")  # tolerate stray output
                    continue
                if isinstance(msg, dict):
                    await self._dispatch(msg)
        except (ValueError, asyncio.LimitOverrunError) as e:  # a single line above MAX_LINE_BYTES
            self._stderr.append(f"[protocol error] {e}")
        finally:
            code = self._proc.returncode if self._proc else None
            if self._dead is None:
                tail = f" stderr: {self.stderr_tail()[-300:]}" if self._stderr else ""
                self._dead = McpError(
                    f"MCP server {self.name!r} exited{'' if code is None else f' (code {code})'}.{tail}"
                )
            self._fail_pending(self._dead)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):  # response to one of our requests
            fut = self._pending.pop(msg["id"], None)
            if fut is None or fut.done():
                return
            if "error" in msg:
                err = msg["error"] or {}
                fut.set_exception(
                    McpError(f"{err.get('message', 'error')} (code {err.get('code')})", code=err.get("code"))
                )
            else:
                fut.set_result(msg["result"])
        elif "id" in msg and "method" in msg:  # request FROM the server: answer ping, refuse the rest
            if msg["method"] == "ping":
                reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
            else:
                reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not found"}}
            with contextlib.suppress(Exception):
                await self._send(reply)
        # notifications (e.g. tools/list_changed, logging) are ignored

    async def _drain_stderr(self) -> None:
        assert self._proc is not None
        with contextlib.suppress(Exception):
            async for raw in self._proc.stderr:
                self._stderr.append(raw.decode("utf-8", "replace").rstrip())
