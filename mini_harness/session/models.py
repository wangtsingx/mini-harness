"""Persistence contracts: value objects + the SessionStore protocol (ISP: narrow, storage-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mini_harness.core.messages import Message, Usage


@dataclass(frozen=True)
class Snapshot:
    """Full, immutable state of a session at one point (a checkpoint)."""

    session_id: str
    branch: str
    parent_id: int | None  # checkpoint this state descends from (optimistic-concurrency token)
    messages: tuple[Message, ...]
    archive: tuple[Message, ...]
    usage: Usage
    turns: int
    compactions: int
    last_prompt_tokens: int
    last_prompt_msgs: int
    status: str
    checkpoint_id: int | None = None  # set when loaded


@dataclass(frozen=True)
class CheckpointMeta:
    id: int
    session_id: str
    branch: str
    parent_id: int | None
    turn: int
    created_at: float
    label: str
    status: str
    n_messages: int


@dataclass(frozen=True)
class SessionInfo:
    id: str
    created_at: float
    updated_at: float
    status: str
    current_branch: str
    n_checkpoints: int


@dataclass(frozen=True)
class BranchInfo:
    name: str
    head: int | None
    parent_branch: str | None
    fork_point: int | None


class SessionStore(Protocol):
    async def save(self, snap: Snapshot, label: str = "") -> CheckpointMeta:
        """Append a checkpoint to snap.branch. Raises StaleSessionError unless snap.parent_id is the branch head."""

    async def load(self, session_id: str, checkpoint_id: int | None = None, *, branch: str | None = None) -> Snapshot:
        """Latest checkpoint of `branch` (default: the session's current branch), or a specific checkpoint."""

    async def list_checkpoints(self, session_id: str, branch: str | None = None) -> list[CheckpointMeta]:
        """Lineage of the branch head (oldest first), including history from before any fork."""

    async def fork(self, session_id: str, checkpoint_id: int, name: str | None = None) -> str:
        """Create a branch whose head is `checkpoint_id` and make it current. Never alters existing history."""

    async def branches(self, session_id: str) -> list[BranchInfo]: ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    async def delete_session(self, session_id: str) -> None: ...
