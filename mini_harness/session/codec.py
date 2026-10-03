"""Message <-> JSON. Digests use canonical JSON (sorted keys); stored bodies keep insertion order so a
reloaded conversation re-serializes byte-identically (keeps provider prompt caches valid after resume)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mini_harness.core.messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock


def _block_to_dict(b: ContentBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": "tool_result", "tool_use_id": b.tool_use_id, "content": b.content, "is_error": b.is_error}


def _block_from_dict(d: dict[str, Any]) -> ContentBlock:
    kind = d["type"]
    if kind == "text":
        return TextBlock(d["text"])
    if kind == "tool_use":
        return ToolUseBlock(d["id"], d["name"], d["input"])
    if kind == "tool_result":
        return ToolResultBlock(d["tool_use_id"], d["content"], d.get("is_error", False))
    raise ValueError(f"unknown block type: {kind!r}")


def message_to_dict(m: Message) -> dict[str, Any]:
    return {"role": m.role, "content": [_block_to_dict(b) for b in m.content], "meta": m.meta}


def message_from_dict(d: dict[str, Any]) -> Message:
    return Message(d["role"], tuple(_block_from_dict(b) for b in d["content"]), dict(d.get("meta") or {}))


def encode(m: Message) -> tuple[str, str]:
    """(digest, body) for a message."""
    d = message_to_dict(m)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), json.dumps(d, ensure_ascii=False)


def decode(body: str) -> Message:
    return message_from_dict(json.loads(body))
