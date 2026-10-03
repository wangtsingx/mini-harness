"""Record every model call (request + streamed events) so a run can be replayed without a model.

Place RecordingProvider as the OUTERMOST provider (around RetryingProvider): the recording then holds exactly what
the loop saw, including RetryNotice events, and the summarizer's calls (same provider) are captured in order.
WARNING: recordings contain full prompts and tool outputs. Treat them like the data they came from.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_harness.core.errors import ProviderError
from mini_harness.core.messages import Usage
from mini_harness.providers.base import (
    CachePlan,
    MessageEnd,
    ModelRequest,
    Provider,
    RetryNotice,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from mini_harness.session.codec import message_to_dict

FORMAT_VERSION = 1


# ---------------------------------------------------------------- (de)serialization
def event_to_dict(ev: StreamEvent) -> dict[str, Any]:
    match ev:
        case TextDelta(text):
            return {"t": "text", "text": text}
        case ToolCallStart(id, name):
            return {"t": "call_start", "id": id, "name": name}
        case ToolCallDelta(id, partial):
            return {"t": "call_delta", "id": id, "partial_json": partial}
        case ToolCallEnd(id):
            return {"t": "call_end", "id": id}
        case MessageEnd(stop_reason, usage):
            u = [usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens]
            return {"t": "end", "stop_reason": stop_reason, "usage": u}
        case RetryNotice(attempt, delay_s, reason, discarded):
            return {
                "t": "retry",
                "attempt": attempt,
                "delay_s": delay_s,
                "reason": reason,
                "discarded_partial": discarded,
            }
    raise TypeError(f"unknown stream event: {ev!r}")


def event_from_dict(d: dict[str, Any]) -> StreamEvent:
    match d["t"]:
        case "text":
            return TextDelta(d["text"])
        case "call_start":
            return ToolCallStart(d["id"], d["name"])
        case "call_delta":
            return ToolCallDelta(d["id"], d["partial_json"])
        case "call_end":
            return ToolCallEnd(d["id"])
        case "end":
            return MessageEnd(d["stop_reason"], Usage(*d["usage"]))
        case "retry":
            return RetryNotice(d["attempt"], d["delay_s"], d["reason"], d["discarded_partial"])
    raise ValueError(f"unknown event type: {d['t']!r}")


def error_to_dict(e: ProviderError) -> dict[str, Any]:
    return {
        "t": "error",
        "message": str(e),
        "status": e.status,
        "retryable": e.retryable,
        "retry_after_s": e.retry_after_s,
    }


def _cache_to_dict(plan: CachePlan | None) -> dict[str, Any] | None:
    return None if plan is None else {"system": plan.system, "message_indices": list(plan.message_indices)}


def request_to_dict(req: ModelRequest) -> dict[str, Any]:
    """What the model is actually shown. Tool timeouts/permissions are not part of the prompt, so not recorded."""
    return {
        "model": req.model,
        "system": req.system,
        "max_tokens": req.max_tokens,
        "messages": [message_to_dict(m) for m in req.messages],
        "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in req.tools],
        "cache": _cache_to_dict(req.cache),
    }  # fmt: skip


def request_digest(d: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------- diffing
def _clip(s: str, n: int = 100) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _msg_preview(m: dict[str, Any]) -> str:
    parts = []
    for b in m["content"]:
        if b["type"] == "text":
            parts.append(b["text"])
        elif b["type"] == "tool_use":
            parts.append(f"tool_use {b['name']}({json.dumps(b['input'], ensure_ascii=False)})")
        else:
            parts.append(f"tool_result{' ERROR' if b.get('is_error') else ''} {b['content']}")
    return _clip(f"{m['role']}: " + " | ".join(parts))


def diff_requests(recorded: dict[str, Any], actual: dict[str, Any], ignore: Iterable[str] = ()) -> list[str]:
    """Human-readable differences between a recorded request and the one the harness builds now."""
    skip = set(ignore)
    problems: list[str] = []
    for key in ("model", "system", "max_tokens", "cache"):
        if key not in skip and recorded[key] != actual[key]:
            problems.append(f"{key} changed: {_clip(str(recorded[key]))!r} -> {_clip(str(actual[key]))!r}")
    if "tools" not in skip and recorded["tools"] != actual["tools"]:
        old = {t["name"]: t for t in recorded["tools"]}
        new = {t["name"]: t for t in actual["tools"]}
        for name in sorted(old.keys() - new.keys()):
            problems.append(f"tool removed: {name}")
        for name in sorted(new.keys() - old.keys()):
            problems.append(f"tool added: {name}")
        for name in sorted(old.keys() & new.keys()):
            if old[name] != new[name]:
                problems.append(f"tool definition changed: {name}")
        if not problems and [t["name"] for t in recorded["tools"]] != [t["name"] for t in actual["tools"]]:
            problems.append("tool order changed (breaks prompt-cache prefix stability)")
    if "messages" not in skip and recorded["messages"] != actual["messages"]:
        a, b = recorded["messages"], actual["messages"]
        if len(a) != len(b):
            problems.append(f"messages: {len(a)} recorded vs {len(b)} now")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                problems.append(f"messages[{i}] differs: recorded {_msg_preview(x)!r} vs now {_msg_preview(y)!r}")
                break
    return problems


# ---------------------------------------------------------------- recording model
@dataclass
class RecordedCall:
    request: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = True


@dataclass
class Recording:
    meta: dict[str, Any] = field(default_factory=dict)
    calls: list[RecordedCall] = field(default_factory=list)
    expected: list[list[Any]] | None = None  # behavioral fingerprint of the recorded run (see replay.fingerprint)

    def to_lines(self) -> list[str]:
        lines = [json.dumps({"kind": "header", "version": FORMAT_VERSION, "meta": self.meta}, ensure_ascii=False)]
        lines += [call_line(c) for c in self.calls]
        if self.expected is not None:
            lines.append(expected_line(self.expected))
        return lines

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.to_lines()) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> Recording:
        rec = Recording()
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            d = json.loads(raw)
            if d["kind"] == "header":
                if d["version"] != FORMAT_VERSION:
                    raise ValueError(f"unsupported recording version {d['version']}")
                rec.meta = d["meta"]
            elif d["kind"] == "call":
                rec.calls.append(RecordedCall(d["request"], d["events"], d["complete"]))
            elif d["kind"] == "expected":
                rec.expected = d["fingerprint"]
        return rec


def call_line(c: RecordedCall) -> str:
    return json.dumps(
        {"kind": "call", "request": c.request, "events": c.events, "complete": c.complete}, ensure_ascii=False
    )


def expected_line(fp: list[list[Any]]) -> str:
    return json.dumps({"kind": "expected", "fingerprint": fp}, ensure_ascii=False)


class RecordingProvider:
    """Decorator: forwards to `inner`, capturing each call. With `path`, appends a JSONL line per finished call
    (a crash mid-run keeps everything recorded so far; the file is truncated when the provider is created)."""

    def __init__(self, inner: Provider, recording: Recording | None = None, path: str | Path | None = None) -> None:
        self._inner = inner
        self.recording = recording or Recording()
        self.name = inner.name
        self._path = Path(path) if path else None
        if self._path:
            self._path.write_text(
                json.dumps({"kind": "header", "version": FORMAT_VERSION, "meta": self.recording.meta}) + "\n",
                encoding="utf-8",
            )

    def _append(self, line: str) -> None:
        if self._path:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        call = RecordedCall(request_to_dict(req), complete=False)
        self.recording.calls.append(call)
        try:
            async for ev in self._inner.stream(req):
                call.events.append(event_to_dict(ev))
                yield ev
            call.complete = True
        except ProviderError as e:
            call.events.append(error_to_dict(e))
            call.complete = True  # a failed call is a complete, replayable outcome
            raise
        finally:
            self._append(call_line(call))
