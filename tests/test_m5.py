import asyncio
import contextlib
import sqlite3
import subprocess
import sys
import time

import pytest

from mini_harness import (
    Agent,
    CheckpointNotFound,
    ContextConfig,
    HarnessError,
    Limits,
    ProviderError,
    SessionNotFound,
    StaleSessionError,
    ToolRegistry,
)
from mini_harness.core.compaction import make_summary_message
from mini_harness.core.messages import Message, ToolResultBlock, ToolUseBlock, Usage, validate_pairing
from mini_harness.core.session import Session
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.session import codec
from mini_harness.session.manager import RECOVERY_REASON
from mini_harness.session.models import Snapshot
from mini_harness.session.sqlite_store import SQLiteStore


def run(coro):
    return asyncio.run(coro)


def snap(session_id="s", branch="main", parent=None, texts=("a",), status="running", turns=0):
    return Snapshot(
        session_id, branch, parent, tuple(Message.user(t) for t in texts), (), Usage(1, 2, 3, 4),
        turns, 0, 0, 0, status,
    )  # fmt: skip


def echo_registry():
    reg = ToolRegistry()

    @reg.tool()
    def echo(text: str) -> str:
        """echo"""
        return text

    return reg


async def drain(agent, session, prompt="hi"):
    return [e async for e in agent.run(prompt, session)]


async def labels(agent, sid, branch=None):
    return [c.label for c in await agent.sessions.checkpoints(sid, branch)]


# ---------------------------------------------------------------- codec
def test_codec_roundtrip_and_canonical_digest():
    m = Message(
        "assistant",
        (
            Message.user("你好 🌏").content[0],
            ToolUseBlock("c1", "t", {"z": 1, "a": [1, 2]}),
        ),
        {"kind": "x", "n": 3},
    )
    d1, body = codec.encode(m)
    back = codec.decode(body)
    assert back == m and list(back.content[1].input) == ["z", "a"]  # key order survives (cache-stable resume)
    same_but_reordered = Message("assistant", m.content[:1] + (ToolUseBlock("c1", "t", {"a": [1, 2], "z": 1}),), m.meta)
    assert codec.encode(same_but_reordered)[0] == d1  # digest is canonical
    tr = Message.tool_results([ToolResultBlock("c1", "boom", True)])
    assert codec.decode(codec.encode(tr)[1]) == tr


# ---------------------------------------------------------------- store: snapshots, lineage, branches
def test_save_load_roundtrip_and_message_dedup():
    store = SQLiteStore()
    meta1 = run(store.save(snap(texts=("a",), turns=1), "first"))
    meta2 = run(store.save(snap(parent=meta1.id, texts=("a", "b"), turns=2), "second"))
    loaded = run(store.load("s"))
    assert loaded.checkpoint_id == meta2.id and loaded.parent_id == meta1.id
    assert [m.text() for m in loaded.messages] == ["a", "b"] and loaded.usage == Usage(1, 2, 3, 4)
    assert loaded.turns == 2 and loaded.status == "running"
    assert store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2  # "a" stored once
    old = run(store.load("s", meta1.id))
    assert [m.text() for m in old.messages] == ["a"]  # past checkpoints stay readable


def test_optimistic_concurrency_rejects_stale_writers_and_rolls_back():
    store = SQLiteStore()
    cp1 = run(store.save(snap()))
    run(store.save(snap(parent=cp1.id, texts=("a", "b"))))
    with pytest.raises(StaleSessionError):
        run(store.save(snap(parent=cp1.id, texts=("a", "c"))))  # forked from an old head without fork()
    with pytest.raises(StaleSessionError):
        run(store.save(snap(parent=None)))  # id already in use
    assert len(run(store.list_checkpoints("s"))) == 2


def test_lineage_fork_and_branch_isolation():
    store = SQLiteStore()
    cp1 = run(store.save(snap(texts=("a",)), "one"))
    cp2 = run(store.save(snap(parent=cp1.id, texts=("a", "b")), "two"))
    cp3 = run(store.save(snap(parent=cp2.id, texts=("a", "b", "c")), "three"))

    name = run(store.fork("s", cp2.id))
    assert name == "branch-2" and run(store.fork("s", cp1.id)) == "branch-3"
    assert run(store.load("s", branch="branch-2")).checkpoint_id == cp2.id  # new branch starts at the fork point

    cp4 = run(store.save(snap(branch="branch-2", parent=cp2.id, texts=("a", "b", "X")), "alt"))
    assert [c.label for c in run(store.list_checkpoints("s", "branch-2"))] == ["one", "two", "alt"]
    assert [c.label for c in run(store.list_checkpoints("s", "main"))] == ["one", "two", "three"]  # untouched
    assert run(store.load("s", branch="main")).checkpoint_id == cp3.id
    assert run(store.load("s", cp4.id)).branch == "branch-2"
    info = {b.name: b for b in run(store.branches("s"))}
    assert info["branch-2"].fork_point == cp2.id and info["branch-2"].parent_branch == "main"
    assert info["main"].head == cp3.id


def test_cross_session_access_is_rejected():
    store = SQLiteStore()
    a = run(store.save(snap(session_id="A")))
    run(store.save(snap(session_id="B")))
    with pytest.raises(CheckpointNotFound):
        run(store.load("B", a.id))
    with pytest.raises(CheckpointNotFound):
        run(store.fork("B", a.id))
    with pytest.raises(SessionNotFound):
        run(store.load("nope"))
    with pytest.raises(CheckpointNotFound):
        run(store.load("A", branch="ghost"))


def test_list_and_delete_sessions():
    store = SQLiteStore()
    cp = run(store.save(snap(session_id="A", status="completed")))
    run(store.save(snap(parent=cp.id, session_id="A", texts=("a", "b"), status="completed")))
    run(store.save(snap(session_id="B")))
    infos = {s.id: s for s in run(store.list_sessions())}
    assert infos["A"].n_checkpoints == 2 and infos["A"].status == "completed" and infos["B"].n_checkpoints == 1
    run(store.delete_session("A"))
    assert [s.id for s in run(store.list_sessions())] == ["B"]
    with pytest.raises(SessionNotFound):
        run(store.load("A"))
    assert store._conn.execute("SELECT COUNT(*) FROM checkpoints WHERE session_id='A'").fetchone()[0] == 0


def test_file_store_persists_across_instances_and_checks_schema(tmp_path):
    db = tmp_path / "h.db"
    s1 = SQLiteStore(db)
    run(s1.save(snap(texts=("persisted",))))
    s1.close()
    s2 = SQLiteStore(db)
    assert run(s2.load("s")).messages[0].text() == "persisted"
    s2.close()
    raw = sqlite3.connect(db)
    raw.execute("PRAGMA user_version=99")
    raw.commit()
    raw.close()
    with pytest.raises(HarnessError, match="newer"):
        SQLiteStore(db)


# ---------------------------------------------------------------- loop integration
def test_checkpoints_are_written_at_turn_boundaries():
    store = SQLiteStore()
    agent = Agent(
        FakeProvider([tool_turn(("a", "echo", {"text": "1"})), text_turn("fin")]),
        echo_registry(), model="m", store=store,
    )  # fmt: skip
    session = agent.new_session("s1")
    run(drain(agent, session))
    assert run(labels(agent, "s1")) == ["user: hi", "turn 1", "done: end_turn"]
    cps = run(agent.sessions.checkpoints("s1"))
    assert [c.n_messages for c in cps] == [1, 3, 4] and cps[-1].status == "completed"
    assert session.status == "completed" and session.head == cps[-1].id
    for c in cps:  # every stored state is a valid conversation
        assert validate_pairing(list(run(store.load("s1", c.id)).messages)) == []


def test_stop_reasons_become_session_status():
    store = SQLiteStore()
    agent = Agent(
        FakeProvider([tool_turn(("a", "echo", {"text": "x"}))], repeat_last=True),
        echo_registry(), model="m", store=store, limits=Limits(max_turns=1),
    )  # fmt: skip
    session = agent.new_session("s1")
    run(drain(agent, session))
    assert session.status == "max_turns" and run(store.list_sessions())[0].status == "max_turns"


def test_cancel_mid_tool_saves_and_session_can_be_resumed(tmp_path):
    db = tmp_path / "h.db"
    reg = ToolRegistry()

    @reg.tool(timeout_s=30)
    async def hang() -> str:
        """hangs"""
        await asyncio.sleep(30)
        return "never"

    agent = Agent(FakeProvider([tool_turn(("a", "hang", {}))]), reg, model="m", store=SQLiteStore(db))
    session = agent.new_session("s1")

    async def scenario():
        async def consume():
            async for _ in agent.run("go", session):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.15)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    run(scenario())
    assert session.status == "cancelled" and validate_pairing(session.messages) == []

    # a fresh process-like agent resumes it and finishes the job
    agent2 = Agent(FakeProvider([text_turn("recovered")]), reg, model="m", store=SQLiteStore(db))

    async def resume():
        s = await agent2.open_session("s1")
        assert [c.label for c in await agent2.sessions.checkpoints("s1")] == ["user: go", "interrupted"]
        assert s.status == "cancelled" and s.messages[-1].content[0].is_error
        events = [e async for e in agent2.run(None, s)]
        return s, events

    s, events = run(resume())
    assert events[-1].reason == "end_turn" and s.messages[-1].text() == "recovered"
    assert s.status == "completed" and validate_pairing(s.messages) == []


def test_hard_timeout_is_persisted_as_interrupted():
    reg = ToolRegistry()

    @reg.tool(timeout_s=30)
    async def hang() -> str:
        """hangs"""
        await asyncio.sleep(30)
        return "never"

    store = SQLiteStore()
    agent = Agent(
        FakeProvider([tool_turn(("a", "hang", {}))]), reg, model="m", store=store, limits=Limits(wall_clock_s=0.2)
    )
    events = run(drain(agent, agent.new_session("s1")))
    assert events[-1].reason == "timeout"
    cps = run(agent.sessions.checkpoints("s1"))
    assert cps[-1].label == "interrupted" and cps[-1].status == "cancelled"


class Seq:
    """Provider whose i-th call plays script[i]; exceptions in a script are raised."""

    name = "seq"

    def __init__(self, script):
        self.script, self.calls = script, 0

    async def stream(self, req):
        items = self.script[self.calls]
        self.calls += 1
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item


def test_provider_failure_is_persisted_and_run_can_continue_later():
    store = SQLiteStore()
    bad = Seq([tool_turn(("a", "echo", {"text": "1"})), [ProviderError("invalid key", status=401)]])
    agent = Agent(bad, echo_registry(), model="m", store=store)
    session = agent.new_session("s1")
    with pytest.raises(ProviderError):
        run(drain(agent, session))
    assert session.status == "failed"
    cps = run(agent.sessions.checkpoints("s1"))
    assert [c.label for c in cps] == ["user: hi", "turn 1", "interrupted"] and cps[-1].status == "failed"

    agent2 = Agent(FakeProvider([text_turn("back online")]), echo_registry(), model="m", store=store)

    async def resume():
        s = await agent2.open_session("s1")
        await _consume(agent2.run(None, s))
        return s

    s = run(resume())
    assert s.status == "completed" and s.messages[-1].text() == "back online"


async def _consume(gen):
    return [e async for e in gen]


def test_open_repairs_a_dangling_tool_use():
    store = SQLiteStore()
    dangling = (Message.user("q"), Message.assistant([ToolUseBlock("c1", "echo", {"text": "x"})]))
    run(store.save(Snapshot("s1", "main", None, dangling, (), Usage(), 1, 0, 0, 0, "running")))
    agent = Agent(FakeProvider([text_turn("ok")]), echo_registry(), model="m", store=store)
    s = run(agent.open_session("s1"))
    assert validate_pairing(s.messages) == [] and s.messages[-1].content[0].content == RECOVERY_REASON


def test_continue_requires_a_resumable_history():
    agent = Agent(FakeProvider([text_turn("x")]), ToolRegistry(), model="m")
    with pytest.raises(HarnessError, match="nothing to continue"):
        run(_consume(agent.run(None, Session())))
    done = Session(messages=[Message.user("q"), Message("assistant", Message.user("a").content)])
    with pytest.raises(HarnessError, match="nothing to continue"):
        run(_consume(agent.run(None, done)))


def test_agent_without_store_explains_itself():
    agent = Agent(FakeProvider([]), ToolRegistry(), model="m")
    with pytest.raises(HarnessError, match="store"):
        _ = agent.sessions


# ---------------------------------------------------------------- compaction state survives
class Fixed:
    def __init__(self):
        self.seen = []

    async def summarize(self, old):
        self.seen.append(len(old))
        return make_summary_message("BRIEFING", len(old)), Usage(7, 3)


def test_compaction_state_and_archive_roundtrip():
    reg = ToolRegistry()

    @reg.tool()
    def big(n: int) -> str:
        """3000 chars"""
        return "x" * 3000

    script = [
        tool_turn((f"t{i}", "big", {"n": i}), usage=Usage(input_tokens=800 * (i + 1), output_tokens=5))
        for i in range(8)
    ] + [text_turn("done")]
    store = SQLiteStore()
    agent = Agent(
        FakeProvider(script), reg, model="m", store=store, compactor=Fixed(),
        context=ContextConfig(context_window=4000), limits=Limits(max_turns=20),
    )  # fmt: skip
    session = agent.new_session("s1")
    run(drain(agent, session))
    assert session.compactions >= 1 and session.archive
    reopened = run(agent.open_session("s1"))
    assert reopened.messages == session.messages and reopened.archive == session.archive
    assert (reopened.usage, reopened.turns, reopened.compactions) == (session.usage, session.turns, session.compactions)
    assert (reopened.last_prompt_tokens, reopened.last_prompt_msgs) == (
        session.last_prompt_tokens,
        session.last_prompt_msgs,
    )


# ---------------------------------------------------------------- rewind / branching end to end
def test_rewind_branches_without_touching_history():
    store = SQLiteStore()
    agent = Agent(
        FakeProvider([text_turn("answer one"), text_turn("answer two"), text_turn("alternative")]),
        ToolRegistry(), model="m", store=store,
    )  # fmt: skip

    async def scenario():
        s = agent.new_session("s1")
        await _consume(agent.run("q1", s))
        await _consume(agent.run("q2", s))
        main_before = await agent.sessions.checkpoints("s1")
        assert [c.label for c in main_before] == ["user: q1", "done: end_turn", "user: q2", "done: end_turn"]

        back = await agent.rewind("s1", main_before[1].id)  # right after the first answer
        assert [m.text() for m in back.messages] == ["q1", "answer one"] and back.branch == "branch-2"
        await _consume(agent.run("q2 but different", back))

        assert [c.label for c in await agent.sessions.checkpoints("s1", "branch-2")] == [
            "user: q1", "done: end_turn", "user: q2 but different", "done: end_turn",
        ]  # fmt: skip
        assert [c.id for c in await agent.sessions.checkpoints("s1", "main")] == [c.id for c in main_before]
        latest = await agent.open_session("s1")  # current branch is the new one
        assert latest.branch == "branch-2" and latest.messages[-1].text() == "alternative"
        original = await store.load("s1", branch="main")
        assert original.messages[-1].text() == "answer two"

    run(scenario())


def test_rewind_to_a_user_checkpoint_enables_retry():
    store = SQLiteStore()
    agent = Agent(
        FakeProvider([text_turn("bad answer"), text_turn("better answer")]), ToolRegistry(), model="m", store=store
    )

    async def scenario():
        s = agent.new_session("s1")
        await _consume(agent.run("question", s))
        user_cp = (await agent.sessions.checkpoints("s1"))[0]
        retry = await agent.rewind("s1", user_cp.id)
        assert [m.text() for m in retry.messages] == ["question"]
        await _consume(agent.run(None, retry))  # regenerate the answer
        return retry

    retry = run(scenario())
    assert [m.text() for m in retry.messages] == ["question", "better answer"]


def test_second_writer_on_same_head_is_rejected():
    store = SQLiteStore()
    agent = Agent(FakeProvider([text_turn("a"), text_turn("b")]), ToolRegistry(), model="m", store=store)

    async def scenario():
        s0 = agent.new_session("s1")
        await _consume(agent.run("first", s0))
        a, b = await agent.open_session("s1"), await agent.open_session("s1")
        await _consume(agent.run("from a", a))
        with pytest.raises(StaleSessionError):
            await _consume(agent.run("from b", b))  # b still descends from the old head

    run(scenario())
    # a fresh Session reusing an existing id is rejected too
    agent2 = Agent(FakeProvider([text_turn("x")]), ToolRegistry(), model="m", store=store)
    with pytest.raises(StaleSessionError):
        run(_consume(agent2.run("hi", agent2.new_session("s1"))))


# ---------------------------------------------------------------- real crash: kill -9
CRASH_SCRIPT = """
import asyncio, sys
from mini_harness import Agent, ToolRegistry
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.session.sqlite_store import SQLiteStore

reg = ToolRegistry()

@reg.tool()
def quick() -> str:
    "fast tool"
    return "quick-result"

@reg.tool(timeout_s=120)
async def hang() -> str:
    "never returns"
    await asyncio.sleep(120)
    return "never"

provider = FakeProvider([tool_turn(("a", "quick", {})), tool_turn(("b", "hang", {})), text_turn("x")])
agent = Agent(provider, reg, model="m", store=SQLiteStore(sys.argv[1]))

async def main():
    async for _ in agent.run("start the job", agent.new_session("crashy")):
        pass

asyncio.run(main())
"""


def test_process_killed_mid_tool_is_recoverable(tmp_path):
    db = tmp_path / "crash.db"
    proc = subprocess.Popen([sys.executable, "-c", CRASH_SCRIPT, str(db)], stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"worker exited early: {proc.stderr.read().decode()[-500:]}")
            if db.exists():
                with contextlib.suppress(sqlite3.Error):
                    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    got = [r[0] for r in ro.execute("SELECT label FROM checkpoints")]
                    ro.close()
                    if "turn 1" in got:
                        break
            time.sleep(0.05)
        else:
            pytest.fail("worker never reached 'turn 1'")
        time.sleep(0.3)  # let the second turn start its hanging tool
        proc.kill()  # SIGKILL: no finally blocks, no cleanup
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    agent = Agent(FakeProvider([text_turn("finished after crash")]), echo_registry(), model="m", store=SQLiteStore(db))

    async def recover():
        s = await agent.open_session("crashy")
        assert [c.label for c in await agent.sessions.checkpoints("crashy")] == ["user: start the job", "turn 1"]
        assert s.status == "running"  # nobody got to record a final status - that's what a crash looks like
        assert validate_pairing(s.messages) == []
        assert s.messages[-1].role == "tool" and s.messages[-1].content[0].content == "quick-result"
        await _consume(agent.run(None, s))
        return s

    s = run(recover())
    assert s.status == "completed" and s.messages[-1].text() == "finished after crash"
