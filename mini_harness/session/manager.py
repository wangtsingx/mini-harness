"""SessionManager: Session <-> Snapshot translation and the recovery/rewind workflows."""

from __future__ import annotations

from mini_harness.core.messages import close_dangling_tool_uses
from mini_harness.core.session import Session
from mini_harness.session.models import CheckpointMeta, SessionStore, Snapshot

RECOVERY_REASON = "Not executed: the process was interrupted before this tool call finished."


def to_snapshot(session: Session) -> Snapshot:
    return Snapshot(
        session_id=session.id,
        branch=session.branch,
        parent_id=session.head,
        messages=tuple(session.messages),
        archive=tuple(session.archive),
        usage=session.usage,
        turns=session.turns,
        compactions=session.compactions,
        last_prompt_tokens=session.last_prompt_tokens,
        last_prompt_msgs=session.last_prompt_msgs,
        status=session.status,
    )


def from_snapshot(snap: Snapshot) -> Session:
    return Session(
        id=snap.session_id,
        messages=list(snap.messages),
        usage=snap.usage,
        turns=snap.turns,
        archive=list(snap.archive),
        compactions=snap.compactions,
        last_prompt_tokens=snap.last_prompt_tokens,
        last_prompt_msgs=snap.last_prompt_msgs,
        branch=snap.branch,
        head=snap.checkpoint_id,
        status=snap.status,
    )


class SessionManager:
    def __init__(self, store: SessionStore) -> None:
        self._store = store

    @property
    def store(self) -> SessionStore:
        return self._store

    async def checkpoint(self, session: Session, label: str = "") -> CheckpointMeta:
        """Persist the current state as a new checkpoint on the session's branch."""
        meta = await self._store.save(to_snapshot(session), label)
        session.head = meta.id
        return meta

    async def open(self, session_id: str) -> Session:
        """Latest state of the session's current branch, crash-safe.

        A process killed mid-tool leaves a checkpoint from before the tool ran, so history is normally already
        consistent; any dangling tool_use (e.g. from an older writer) is closed with an explicit error result.
        """
        session = from_snapshot(await self._store.load(session_id))
        close_dangling_tool_uses(session.messages, RECOVERY_REASON)
        return session

    async def rewind(self, session_id: str, checkpoint_id: int) -> Session:
        """Go back to `checkpoint_id` on a NEW branch; the old branch and all its checkpoints stay intact.

        Only the conversation state is rewound. Side effects already made in the outside world (files written,
        API calls, emails) are NOT undone.
        """
        snap = await self._store.load(session_id, checkpoint_id)  # validates ownership first
        name = await self._store.fork(session_id, checkpoint_id)
        session = from_snapshot(snap)
        session.branch, session.head = name, checkpoint_id
        close_dangling_tool_uses(session.messages, RECOVERY_REASON)
        return session

    async def checkpoints(self, session_id: str, branch: str | None = None) -> list[CheckpointMeta]:
        return await self._store.list_checkpoints(session_id, branch)
