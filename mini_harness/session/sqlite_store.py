"""SQLite SessionStore (stdlib only).

Model (ADR-008): immutable, git-like snapshots over content-addressed messages.
  messages     content-addressed rows (a message stored once, however many checkpoints reference it)
  checkpoints  append-only; each = (parent, ordered message ids, archive ids, counters). Never updated.
  branches     movable pointers (head) into the checkpoint tree; rewinding = new pointer, history untouched
Single connection + lock; blocking work runs in a thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from mini_harness.core.errors import CheckpointNotFound, HarnessError, SessionNotFound, StaleSessionError
from mini_harness.core.messages import Message, Usage
from mini_harness.session import codec
from mini_harness.session.models import BranchInfo, CheckpointMeta, SessionInfo, Snapshot

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT, digest TEXT NOT NULL UNIQUE, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL,
    status TEXT NOT NULL, current_branch TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    branch TEXT NOT NULL, parent_id INTEGER REFERENCES checkpoints(id),
    turn INTEGER NOT NULL, created_at REAL NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL,
    n_messages INTEGER NOT NULL, state TEXT NOT NULL, message_ids TEXT NOT NULL, archive_ids TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_checkpoints_session ON checkpoints(session_id, id);
CREATE TABLE IF NOT EXISTS branches(
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL, head INTEGER, parent_branch TEXT, fork_point INTEGER,
    PRIMARY KEY(session_id, name));
"""
META_COLS = "id, session_id, branch, parent_id, turn, created_at, label, status, n_messages"


def _meta(r: sqlite3.Row) -> CheckpointMeta:
    return CheckpointMeta(
        r["id"], r["session_id"], r["branch"], r["parent_id"], r["turn"], r["created_at"],
        r["label"], r["status"], r["n_messages"],
    )  # fmt: skip


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._ids: dict[str, int] = {}  # digest -> message id (only rows known to be committed)
        self._init(str(path) != ":memory:")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ async facade
    async def save(self, snap: Snapshot, label: str = "") -> CheckpointMeta:
        return await asyncio.to_thread(self._save, snap, label)

    async def load(self, session_id: str, checkpoint_id: int | None = None, *, branch: str | None = None) -> Snapshot:
        return await asyncio.to_thread(self._load, session_id, checkpoint_id, branch)

    async def list_checkpoints(self, session_id: str, branch: str | None = None) -> list[CheckpointMeta]:
        return await asyncio.to_thread(self._list_checkpoints, session_id, branch)

    async def fork(self, session_id: str, checkpoint_id: int, name: str | None = None) -> str:
        return await asyncio.to_thread(self._fork, session_id, checkpoint_id, name)

    async def branches(self, session_id: str) -> list[BranchInfo]:
        return await asyncio.to_thread(self._branches, session_id)

    async def list_sessions(self) -> list[SessionInfo]:
        return await asyncio.to_thread(self._list_sessions)

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    # ------------------------------------------------------------------ implementation
    def _init(self, is_file: bool) -> None:
        with self._lock:
            c = self._conn
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise HarnessError(f"database schema v{version} is newer than this library (v{SCHEMA_VERSION})")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=5000")
            if is_file:
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA synchronous=NORMAL")
            c.executescript(SCHEMA)
            c.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            c.commit()

    def _intern(self, m: Message, pending: dict[str, int]) -> int:
        digest, body = codec.encode(m)
        known = self._ids.get(digest) or pending.get(digest)
        if known:
            return known
        c = self._conn
        c.execute("INSERT OR IGNORE INTO messages(digest, body) VALUES(?, ?)", (digest, body))
        pending[digest] = c.execute("SELECT id FROM messages WHERE digest=?", (digest,)).fetchone()[0]
        return pending[digest]

    def _save(self, snap: Snapshot, label: str) -> CheckpointMeta:
        now = time.time()
        pending: dict[str, int] = {}
        with self._lock, self._conn as c:  # one atomic transaction
            c.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at, status, current_branch) VALUES(?,?,?,?,?)",
                (snap.session_id, now, now, snap.status, snap.branch),
            )
            c.execute("INSERT OR IGNORE INTO branches(session_id, name) VALUES(?, ?)", (snap.session_id, snap.branch))
            head = c.execute(
                "SELECT head FROM branches WHERE session_id=? AND name=?", (snap.session_id, snap.branch)
            ).fetchone()["head"]
            if head != snap.parent_id:
                raise StaleSessionError(
                    f"session {snap.session_id!r} branch {snap.branch!r}: head is {head}, "
                    f"but this state descends from {snap.parent_id} (concurrent writer, or id already in use)"
                )
            message_ids = [self._intern(m, pending) for m in snap.messages]
            archive_ids = [self._intern(m, pending) for m in snap.archive]
            state = json.dumps(
                {
                    "usage": [
                        snap.usage.input_tokens,
                        snap.usage.output_tokens,
                        snap.usage.cache_read_tokens,
                        snap.usage.cache_write_tokens,
                    ],
                    "compactions": snap.compactions,
                    "last_prompt_tokens": snap.last_prompt_tokens,
                    "last_prompt_msgs": snap.last_prompt_msgs,
                }  # fmt: skip
            )
            cur = c.execute(
                "INSERT INTO checkpoints(session_id, branch, parent_id, turn, created_at, label, status, n_messages,"
                " state, message_ids, archive_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snap.session_id, snap.branch, snap.parent_id, snap.turns, now, label, snap.status,
                 len(snap.messages), state, json.dumps(message_ids), json.dumps(archive_ids)),
            )  # fmt: skip
            cp_id = cur.lastrowid
            c.execute("UPDATE branches SET head=? WHERE session_id=? AND name=?", (cp_id, snap.session_id, snap.branch))
            c.execute(
                "UPDATE sessions SET updated_at=?, status=?, current_branch=? WHERE id=?",
                (now, snap.status, snap.branch, snap.session_id),
            )
        self._ids.update(pending)  # only after commit: a rolled-back insert must never be cached
        return CheckpointMeta(
            cp_id, snap.session_id, snap.branch, snap.parent_id, snap.turns, now, label, snap.status, len(snap.messages)
        )

    def _fetch_messages(self, ids: list[int]) -> tuple[Message, ...]:
        found: dict[int, Message] = {}
        unique = sorted(set(ids))
        for i in range(0, len(unique), 500):  # stay under SQLite's bound-variable limit
            chunk = unique[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(f"SELECT id, body FROM messages WHERE id IN ({marks})", chunk):
                found[r["id"]] = codec.decode(r["body"])
        missing = set(ids) - found.keys()
        if missing:
            raise HarnessError(f"corrupt store: missing messages {sorted(missing)[:5]}")
        return tuple(found[i] for i in ids)

    def _head(self, session_id: str, branch: str | None) -> tuple[str, int | None]:
        s = self._conn.execute("SELECT current_branch FROM sessions WHERE id=?", (session_id,)).fetchone()
        if s is None:
            raise SessionNotFound(session_id)
        name = branch or s["current_branch"]
        b = self._conn.execute("SELECT head FROM branches WHERE session_id=? AND name=?", (session_id, name)).fetchone()
        if b is None:
            raise CheckpointNotFound(f"unknown branch {name!r} in session {session_id!r}")
        return name, b["head"]

    def _load(self, session_id: str, checkpoint_id: int | None, branch: str | None) -> Snapshot:
        with self._lock:
            if checkpoint_id is None:
                name, checkpoint_id = self._head(session_id, branch)
                if checkpoint_id is None:
                    raise CheckpointNotFound(f"branch {name!r} has no checkpoints")
            elif self._conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                raise SessionNotFound(session_id)
            r = self._conn.execute(
                "SELECT * FROM checkpoints WHERE id=? AND session_id=?", (checkpoint_id, session_id)
            ).fetchone()
            if r is None:
                raise CheckpointNotFound(f"checkpoint {checkpoint_id} does not belong to session {session_id!r}")
            state: dict[str, Any] = json.loads(r["state"])
            return Snapshot(
                session_id=session_id,
                branch=r["branch"],
                parent_id=r["parent_id"],
                messages=self._fetch_messages(json.loads(r["message_ids"])),
                archive=self._fetch_messages(json.loads(r["archive_ids"])),
                usage=Usage(*state["usage"]),
                turns=r["turn"],
                compactions=state["compactions"],
                last_prompt_tokens=state["last_prompt_tokens"],
                last_prompt_msgs=state["last_prompt_msgs"],
                status=r["status"],
                checkpoint_id=r["id"],
            )

    def _list_checkpoints(self, session_id: str, branch: str | None) -> list[CheckpointMeta]:
        with self._lock:
            _, head = self._head(session_id, branch)
            if head is None:
                return []
            rows = self._conn.execute(
                f"""WITH RECURSIVE chain AS (
                        SELECT {META_COLS} FROM checkpoints WHERE id=?
                        UNION ALL
                        SELECT {", ".join("c." + col.strip() for col in META_COLS.split(","))}
                        FROM checkpoints c JOIN chain ON c.id = chain.parent_id)
                    SELECT * FROM chain ORDER BY id""",
                (head,),
            ).fetchall()
            return [_meta(r) for r in rows]

    def _fork(self, session_id: str, checkpoint_id: int, name: str | None) -> str:
        with self._lock, self._conn as c:
            r = c.execute(
                "SELECT branch FROM checkpoints WHERE id=? AND session_id=?", (checkpoint_id, session_id)
            ).fetchone()
            if r is None:
                if c.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                    raise SessionNotFound(session_id)
                raise CheckpointNotFound(f"checkpoint {checkpoint_id} does not belong to session {session_id!r}")
            existing = {x["name"] for x in c.execute("SELECT name FROM branches WHERE session_id=?", (session_id,))}
            if name is None:
                n = len(existing) + 1
                while f"branch-{n}" in existing:
                    n += 1
                name = f"branch-{n}"
            elif name in existing:
                raise HarnessError(f"branch {name!r} already exists")
            c.execute(
                "INSERT INTO branches(session_id, name, head, parent_branch, fork_point) VALUES(?,?,?,?,?)",
                (session_id, name, checkpoint_id, r["branch"], checkpoint_id),
            )
            c.execute("UPDATE sessions SET current_branch=?, updated_at=? WHERE id=?", (name, time.time(), session_id))
            return name

    def _branches(self, session_id: str) -> list[BranchInfo]:
        with self._lock:
            self._head(session_id, None)  # raises SessionNotFound
            rows = self._conn.execute(
                "SELECT name, head, parent_branch, fork_point FROM branches WHERE session_id=? ORDER BY rowid",
                (session_id,),
            )
            return [BranchInfo(r["name"], r["head"], r["parent_branch"], r["fork_point"]) for r in rows]

    def _list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.created_at, s.updated_at, s.status, s.current_branch,"
                " (SELECT COUNT(*) FROM checkpoints c WHERE c.session_id = s.id) AS n"
                " FROM sessions s ORDER BY s.updated_at DESC, s.rowid DESC"
            )
            return [
                SessionInfo(r["id"], r["created_at"], r["updated_at"], r["status"], r["current_branch"], r["n"])
                for r in rows
            ]

    def _delete_session(self, session_id: str) -> None:
        """Removes the session's checkpoints and branches. Shared message rows are left (content-addressed)."""
        with self._lock, self._conn as c:
            c.execute("UPDATE checkpoints SET parent_id=NULL WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))  # cascades to checkpoints, branches
