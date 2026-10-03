"""ContextManager: decides WHEN to compact and commits the result; also plans prompt-cache breakpoints.

Prompt-caching contract (why compaction is conservative):
  request prefix = [tools][system][messages...]. A provider can reuse its cache only while that prefix is
  byte-identical, so between compactions history is strictly append-only; tools/system never change per turn;
  trimming/summarizing happens in ONE step when the threshold is crossed, then the new prefix is stable again.
"""

from __future__ import annotations

from dataclasses import dataclass

from mini_harness.core.compaction import (
    Compactor,
    ExtractiveCompactor,
    find_cut,
    is_summary,
    trim_tool_results,
)
from mini_harness.core.errors import HarnessError
from mini_harness.core.messages import Message, Usage
from mini_harness.core.session import Session
from mini_harness.core.tokens import estimate_message, estimate_messages, estimate_request
from mini_harness.providers.base import CachePlan
from mini_harness.tools.spec import ToolSpec


@dataclass(frozen=True)
class ContextConfig:
    context_window: int = 200_000  # tokens of the model you use; set it explicitly
    compact_threshold: float = 0.8  # start compacting at this fraction of the window
    target_ratio: float = 0.5  # aim to end up below this fraction
    recent_ratio: float = 0.25  # newest messages up to this fraction of the window are never touched
    trim_min_chars: int = 1_000  # tool results longer than this (in the old part) are trimmed
    trim_preview_chars: int = 200
    enabled: bool = True
    min_gain_ratio: float = 0.05  # don't summarize an old segment smaller than this fraction of the window
    prompt_caching: bool = True  # False = never send cache breakpoints (for A/B cost measurements)

    def __post_init__(self) -> None:
        if not (0 < self.recent_ratio < self.target_ratio < self.compact_threshold < 1):
            raise ValueError("require 0 < recent_ratio < target_ratio < compact_threshold < 1")
        if self.trim_preview_chars >= self.trim_min_chars:
            raise ValueError("trim_preview_chars must be < trim_min_chars")


@dataclass(frozen=True)
class Compaction:
    stage: str  # trim | summary | extractive | skipped (summary was not smaller: history unchanged)
    tokens_before: int
    tokens_after: int
    archived: int  # original messages moved to Session.archive
    usage: Usage  # tokens spent on summarization (zero for trim/extractive)


class ContextManager:
    def __init__(
        self,
        config: ContextConfig | None = None,
        compactor: Compactor | None = None,
        fallback: Compactor | None = None,
    ) -> None:
        self._cfg = config or ContextConfig()
        self._compactor = compactor
        self._fallback = fallback or ExtractiveCompactor()

    @property
    def caching_enabled(self) -> bool:
        return self._cfg.prompt_caching

    # ------------------------------------------------------------ size
    def estimate(self, session: Session, system: str, tools: list[ToolSpec]) -> int:
        """Last provider-reported prompt size + estimate of what was appended since (else pure estimate)."""
        if 0 < session.last_prompt_msgs <= len(session.messages):
            return session.last_prompt_tokens + estimate_messages(session.messages[session.last_prompt_msgs :])
        return estimate_request(system, tools, session.messages)

    # ------------------------------------------------------------ compaction
    async def prepare(self, session: Session, system: str, tools: list[ToolSpec]) -> Compaction | None:
        cfg = self._cfg
        if not cfg.enabled:
            return None
        before = self.estimate(session, system, tools)
        if before < cfg.compact_threshold * cfg.context_window:
            return None
        msgs = session.messages
        cut = find_cut(msgs, int(cfg.recent_ratio * cfg.context_window))
        if cut <= 0:
            return None  # nothing old enough to compact
        target = cfg.target_ratio * cfg.context_window
        old = msgs[:cut]
        old_tokens = estimate_messages(old)

        # stage 1: trim bulky old tool results
        trimmed = trim_tool_results(msgs, cut, min_chars=cfg.trim_min_chars, preview_chars=cfg.trim_preview_chars)
        after = before - (old_tokens - estimate_messages(trimmed[:cut]))
        if after <= target:
            changed = [o for o, t in zip(old, trimmed[:cut], strict=True) if o is not t]
            return self._commit(session, trimmed, changed, "trim", before, after, Usage())

        # stage 2: fold the old segment into one summary message - but only if there is something worth reclaiming.
        # (A huge protected tail, e.g. one giant tool result, cannot be compacted; summarizing the few small
        # messages before it would add tokens and re-trigger on every turn.)
        old_trimmed = estimate_messages(trimmed[:cut])
        if old_trimmed < cfg.min_gain_ratio * cfg.context_window:
            return None
        summary, usage, stage = await self._summarize(trimmed[:cut])
        summary_tokens = estimate_message(summary)
        if summary_tokens >= old_trimmed:  # the summary did not shrink anything: keep history, report the spend
            return Compaction("skipped", before, before, 0, usage)
        new = [summary, *msgs[cut:]]
        after = before - (old_tokens - summary_tokens)
        return self._commit(session, new, old, stage, before, after, usage)

    async def _summarize(self, old: list[Message]) -> tuple[Message, Usage, str]:
        if self._compactor is not None:
            try:
                summary, usage = await self._compactor.summarize(old)
                return summary, usage, "summary"
            except HarnessError:  # provider down / empty summary: degrade, don't die (CancelledError passes)
                pass
        summary, usage = await self._fallback.summarize(old)
        return summary, usage, "extractive"

    @staticmethod
    def _commit(
        session: Session, new: list[Message], archived: list[Message], stage: str, before: int, after: int, usage: Usage
    ) -> Compaction:
        session.messages = new  # replace the list: callers holding the old one see an unchanged snapshot
        session.archive.extend(archived)  # originals are never lost (M5 persists them)
        session.compactions += 1
        session.last_prompt_tokens, session.last_prompt_msgs = max(after, 0), len(new)
        return Compaction(stage, before, after, len(archived), usage)

    # ------------------------------------------------------------ caching
    @staticmethod
    def cache_plan(messages: list[Message]) -> CachePlan:
        """Breakpoints: static prefix (tools+system), the stable summary (if any), and the newest message."""
        idx: set[int] = set()
        if messages and is_summary(messages[0]):
            idx.add(0)
        if messages:
            idx.add(len(messages) - 1)
        return CachePlan(system=True, message_indices=tuple(sorted(idx)))
