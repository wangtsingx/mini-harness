"""Tracing: Run -> Turn -> {Model, Tool, Compaction} spans.

Design notes
- Zero cost by default: Agent uses NullTracer unless you pass a Tracer.
- Spans are recorded when they END (a span still open at crash time is simply absent).
- Everything that can carry user data (tool args, error text) goes through `preview()`: redacted + clipped.
- Sinks are plain callables `(Span) -> None`; JsonlSink is provided. Export to OpenTelemetry etc. = one small sink.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PLAIN = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
_KEYED = re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)\"?\s*[=:]\s*\"?)[^\s\"',}]+")


def default_redact(text: str) -> str:
    """Best-effort masking of common credentials. Not a security boundary: don't trace secrets you can avoid."""
    for pat in _PLAIN:
        text = pat.sub("[REDACTED]", text)
    return _KEYED.sub(r"\1[REDACTED]", text)


@dataclass
class Span:
    id: int
    trace_id: int  # id of the root (run) span
    parent_id: int | None
    kind: str  # run | turn | model | tool | compaction
    name: str
    start: float  # epoch seconds
    t0: float = field(default=0.0, repr=False)  # monotonic origin (not exported)
    duration_s: float | None = None
    status: str = "running"  # ok | error | cancelled
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "trace_id": self.trace_id, "parent_id": self.parent_id, "kind": self.kind,
            "name": self.name, "start": self.start, "duration_s": self.duration_s, "status": self.status,
            "error": self.error, "attrs": self.attrs,
        }  # fmt: skip

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Span:
        return Span(
            d["id"], d["trace_id"], d["parent_id"], d["kind"], d["name"], d["start"],
            duration_s=d["duration_s"], status=d["status"], error=d["error"], attrs=dict(d["attrs"]),
        )  # fmt: skip


class Tracer:
    def __init__(
        self,
        sinks: Sequence[Callable[[Span], None]] = (),
        *,
        keep: bool = True,
        redact: Callable[[str], str] = default_redact,
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.spans: list[Span] = []
        self._sinks = list(sinks)
        self._keep = keep
        self._redact = redact
        self._wall = wall
        self._mono = mono
        self._next_id = 0

    def start(self, kind: str, name: str, parent: Span | None = None, **attrs: Any) -> Span:
        self._next_id += 1
        return Span(
            self._next_id, parent.trace_id if parent else self._next_id, parent.id if parent else None,
            kind, name, self._wall(), self._mono(), attrs=dict(attrs),
        )  # fmt: skip

    def end(
        self,
        span: Span,
        *,
        exc: BaseException | None = None,
        status: str | None = None,
        error: str | None = None,
        **attrs: Any,
    ) -> None:
        """Close a span. Pass `exc=sys.exc_info()[1]` from a finally block to derive status automatically."""
        span.attrs.update(attrs)
        span.duration_s = self._mono() - span.t0
        if exc is not None:
            if isinstance(exc, asyncio.CancelledError | GeneratorExit):
                status = status or "cancelled"
            else:
                status = status or "error"
                error = error or f"{type(exc).__name__}: {exc}"
        span.status = status or "ok"
        span.error = self._redact(error) if error else None
        if self._keep:
            self.spans.append(span)
        for sink in self._sinks:
            sink(span)

    def elapsed(self, span: Span) -> float:
        return self._mono() - span.t0

    def preview(self, obj: Any, limit: int = 200) -> str:
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
        text = self._redact(text)
        return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit}]"


class NullTracer(Tracer):
    """Default: records nothing, serializes nothing."""

    def __init__(self) -> None:
        super().__init__(keep=False)

    def end(self, span: Span, **_: Any) -> None:  # type: ignore[override]
        return None

    def elapsed(self, span: Span) -> float:
        return 0.0

    def preview(self, obj: Any, limit: int = 200) -> str:
        return ""


class JsonlSink:
    """Appends one JSON line per finished span (blocking, tiny writes)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __call__(self, span: Span) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")


def load_trace(path: str | Path) -> list[Span]:
    with Path(path).open(encoding="utf-8") as f:
        return [Span.from_dict(json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------- rendering
def _fmt_duration(d: float | None) -> str:
    if d is None:
        return "?"
    return f"{d * 1000:.0f}ms" if d < 1 else f"{d:.2f}s"


def _describe(s: Span) -> str:
    a = s.attrs
    flag = "" if s.status == "ok" else f" [{s.status}]"
    head = s.name if s.kind == "turn" else (f"{s.kind} {s.name}" if s.kind == "tool" else s.kind)
    parts: list[str] = []
    if s.kind == "run":
        parts = [
            f"reason={a.get('reason')}",
            f"turns={a.get('turns')}",
            f"tokens={a.get('input_tokens', 0) + a.get('output_tokens', 0)}",
        ]
    elif s.kind == "model":
        parts = [
            f"{a.get('provider')}/{a.get('model')}",
            f"in={a.get('input_tokens')}",
            f"out={a.get('output_tokens')}",
        ]
        if a.get("cache_read_tokens"):
            parts.append(f"cache_read={a['cache_read_tokens']}")
        if a.get("ttft_s") is not None:
            parts.append(f"ttft={_fmt_duration(a['ttft_s'])}")
        parts.append(f"stop={a.get('stop_reason')}")
        if a.get("retries"):
            parts.append(f"retries={a['retries']}")
    elif s.kind == "tool":
        if a.get("attempts", 1) > 1:
            parts.append(f"attempts={a['attempts']}")
        parts.append(f"args={a.get('args')}")
    elif s.kind == "compaction":
        parts = [str(a.get("stage")), f"{a.get('tokens_before')}->{a.get('tokens_after')} tokens"]
    tail = f"  {' '.join(parts)}" if parts else ""
    err = f"  !! {s.error}" if s.error else ""
    return f"{head}{flag} {_fmt_duration(s.duration_s)}{tail}{err}"


def render_tree(spans: Sequence[Span]) -> str:
    """Human-readable tree of finished spans, one root per run."""
    by_id = {s.id: s for s in spans}
    children: dict[int | None, list[Span]] = {}
    for s in spans:
        children.setdefault(s.parent_id if s.parent_id in by_id else None, []).append(s)
    for group in children.values():
        group.sort(key=lambda s: (s.start, s.id))
    lines: list[str] = []

    def walk(node: Span, prefix: str, last: bool, root: bool) -> None:
        lines.append(_describe(node) if root else f"{prefix}{'└─ ' if last else '├─ '}{_describe(node)}")
        kids = children.get(node.id, [])
        child_prefix = "" if root else prefix + ("   " if last else "│  ")
        for i, k in enumerate(kids):
            walk(k, child_prefix, i == len(kids) - 1, False)

    for r in children.get(None, []):
        walk(r, "", True, True)
    return "\n".join(lines)
