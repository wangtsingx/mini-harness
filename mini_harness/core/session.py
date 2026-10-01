from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from mini_harness.core.messages import Message, Usage


@dataclass
class Session:
    """In-memory conversation state (M5 introduces persistence behind a SessionStore)."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0  # model calls made in this session (cumulative)
    archive: list[Message] = field(default_factory=list)  # originals removed by compaction (never sent)
    compactions: int = 0
    # Provider-reported size of the last request and how many messages it contained (ground truth for sizing)
    last_prompt_tokens: int = 0
    last_prompt_msgs: int = 0
