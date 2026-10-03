"""A tiny stdlib-only MCP server (stdio, newline-delimited JSON-RPC) used by the test-suite.

Env: FAKE_MCP_LOG=<file> logs every incoming message; FAKE_MCP_VERSION=<v> overrides the protocol version;
FAKE_MCP_PING=1 makes the server send requests of its own after `initialized`.
"""

import json
import os
import sys
import threading
import time

LOG = os.environ.get("FAKE_MCP_LOG")
VERSION = os.environ.get("FAKE_MCP_VERSION", "2025-06-18")
OBJ = {"type": "object", "properties": {}}


def log(obj):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(obj) + "\n")


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def tool(name, desc, schema=OBJ, **annotations):
    t = {"name": name, "description": desc, "inputSchema": schema}
    if annotations:
        t["annotations"] = annotations
    return t


PAGE1 = [
    tool(
        "echo",
        "Echo text back",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        readOnlyHint=True,
    ),
    tool(
        "add",
        "Add two numbers",
        {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    ),
]
PAGE2 = [
    tool("fail", "Always reports an error"),
    tool("slow", "Sleeps for a while"),
    tool("big", "Returns a huge text", readOnlyHint=True),
    tool("image", "Returns an image block", readOnlyHint=True),
    tool("structured", "Returns structuredContent only", readOnlyHint=True),
    tool("crash", "Kills the server"),
    tool("env", "Reports environment keys"),
    tool("idem", "Idempotent writer", idempotentHint=True, readOnlyHint=False),
    tool("bad.name with spaces", "Name needs sanitizing", readOnlyHint=True),
]


def text(s, error=False):
    return {"content": [{"type": "text", "text": s}], "isError": error}


def call(name, args, mid):
    if name == "echo":
        return text(args["text"])
    if name == "add":
        return text(str(args["a"] + args["b"]))
    if name == "fail":
        return text("boom from server", error=True)
    if name == "slow":
        time.sleep(3)
        return text("slow done")
    if name == "big":
        return text("x" * 200_000)
    if name == "image":
        return {
            "content": [{"type": "image", "data": "AAAA", "mimeType": "image/png"}, {"type": "text", "text": "caption"}]
        }
    if name == "structured":
        return {"content": [], "structuredContent": {"a": 1}}
    if name == "crash":
        os._exit(3)
    if name == "env":
        return text(json.dumps({"keys": sorted(os.environ), "explicit": os.environ.get("EXPLICIT_VAR")}))
    if name in ("idem", "bad.name with spaces"):
        return text("ok")
    return None


def handle(msg):
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    log({"in": msg})
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                },
            }
        )
    elif method == "notifications/initialized":
        if os.environ.get("FAKE_MCP_PING"):
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
            send({"jsonrpc": "2.0", "id": "srv-2", "method": "roots/list"})
    elif method == "tools/list":
        if params.get("cursor") == "page2":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": PAGE2}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": PAGE1, "nextCursor": "page2"}})
    elif method == "tools/call":
        result = call(params["name"], params.get("arguments") or {}, mid)
        if result is None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {params['name']}"}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": result})


for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "tools/call" and msg["params"]["name"] == "slow":
        threading.Thread(target=handle, args=(msg,), daemon=True).start()  # keep reading stdin meanwhile
    elif "method" not in msg:
        log({"response": msg})  # the client's answer to one of OUR requests
    else:
        handle(msg)
