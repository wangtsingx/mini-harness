"""Demo CLI.

Offline:    python examples/cli.py --fake
Anthropic:  ANTHROPIC_API_KEY=... python examples/cli.py
Persistence: add --db harness.db   (then --list-sessions, or --session ID to create/resume;
            in the REPL: /cp lists checkpoints, /rewind N branches back to checkpoint N)
Observability: --trace trace.jsonl   prints a span tree + metrics after the run
Record/replay: --record rec.jsonl --prompt "..."   captures every model call;
               --replay rec.jsonl                  re-runs the CURRENT harness against it (no model, exit 1 on drift)
MCP tools:     --mcp "fs=npx -y @modelcontextprotocol/server-filesystem /tmp"   (repeatable; name=command args)
               MCP tools are EXEC-level unless the server marks them readOnly, so add --grant exec to allow them.
OpenAI or any compatible server:
            OPENAI_API_KEY=... python examples/cli.py --provider openai --model gpt-4o [--base-url http://localhost:8000/v1]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shlex
from pathlib import Path

from mini_harness import (
    Agent,
    AssistantText,
    Compacted,
    Done,
    JsonlSink,
    Limits,
    Retrying,
    ToolFinished,
    ToolStarted,
    Tracer,
)
from mini_harness.eval import Recording, record_run, replay_run
from mini_harness.mcp import McpClient, McpServerParams, mcp_tools
from mini_harness.observability import render_tree, summarize_all
from mini_harness.providers.base import Provider
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.providers.retry import RetryingProvider
from mini_harness.session.sqlite_store import SQLiteStore
from mini_harness.tools.builtin import build_fs_registry
from mini_harness.tools.policy import DefaultPolicy
from mini_harness.tools.spec import Permission


def make_provider(args: argparse.Namespace) -> Provider:
    if args.fake:
        return FakeProvider(
            [
                tool_turn(("t1", "list_dir", {"path": "."}), text="我先看看工作区里有什么。"),
                text_turn("以上就是工作区的内容。"),
            ]
        )
    if args.provider == "openai":
        from mini_harness.providers.openai_compat import DEFAULT_BASE_URL, OpenAICompatProvider

        inner: Provider = OpenAICompatProvider(os.environ["OPENAI_API_KEY"], base_url=args.base_url or DEFAULT_BASE_URL)
    else:
        from mini_harness.providers.anthropic import AnthropicProvider

        inner = AnthropicProvider(os.environ["ANTHROPIC_API_KEY"])
    return RetryingProvider(inner)  # transient failures: exponential backoff + jitter


async def run_prompt(agent: Agent, session, prompt: str | None) -> None:
    async for ev in agent.run(prompt, session):
        match ev:
            case AssistantText(text):
                print(text, end="", flush=True)
            case Retrying(attempt, delay_s, reason, discarded):
                note = " (partial output above is void)" if discarded else ""
                print(f"\n  [retry #{attempt} in {delay_s:.1f}s] {reason[:100]}{note}")
            case Compacted(stage, before, after, archived):
                print(f"\n  [context compacted: {stage}, ~{before} -> ~{after} tokens, {archived} messages archived]")
            case ToolStarted(_, name, args):
                print(f"\n  [tool] {name}({args})")
            case ToolFinished(_, _, is_error, output):
                print(f"  [tool:{'error' if is_error else 'ok'}] {output[:120]!r}")
            case Done(reason, turns, usage, cost):
                extra = f" cost=${cost:.4f}" if cost is not None else ""
                print(f"\n  [done] reason={reason} turns={turns} tokens={usage.total}{extra}")


def print_trace(tracer: Tracer) -> None:
    print("\n" + render_tree(tracer.spans))
    for m in summarize_all(tracer.spans):
        print(
            f"  metrics: {m.duration_s:.2f}s turns={m.turns} model_calls={m.model_calls} tool_calls={m.tool_calls} "
            f"tool_errors={m.tool_errors} retries={m.retries} compactions={m.compactions} "
            f"tokens in/out={m.input_tokens}/{m.output_tokens} cache_hit={m.cache_hit_ratio:.0%}"
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="scripted offline model")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: api.openai.com/v1)")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--db", help="SQLite file: checkpoint every turn; enables --session / --list-sessions")
    ap.add_argument("--session", help="session id to create or resume (needs --db)")
    ap.add_argument("--list-sessions", action="store_true")
    ap.add_argument("--trace", help="write spans to this JSONL file and print a tree + metrics at the end")
    ap.add_argument("--record", help="record all model calls of a one-shot --prompt run into this JSONL file")
    ap.add_argument("--replay", help="replay a recording through the current harness (no model calls)")
    ap.add_argument("--mcp", action="append", default=[], metavar="NAME=CMD", help="attach an MCP stdio server")
    ap.add_argument(
        "--grant",
        action="append",
        default=[],
        choices=[p.value for p in Permission],
        help="permission classes to allow besides read (repeatable)",
    )
    ap.add_argument("--prompt", help="one-shot prompt (default: interactive REPL)")
    ap.add_argument("--model", default=os.environ.get("HARNESS_MODEL"))
    args = ap.parse_args()
    model = args.model or ("claude-sonnet-5-5" if args.provider == "anthropic" else None)
    if model is None and not args.fake and not args.replay:
        ap.error("--model (or HARNESS_MODEL) is required for --provider openai")
    if (args.session or args.list_sessions) and not args.db:
        ap.error("--session / --list-sessions need --db")
    if (args.record or args.replay) and (args.db or args.session):
        ap.error("--record/--replay run on a fresh in-memory session; don't combine with --db/--session")

    policy = DefaultPolicy(frozenset({Permission.READ, *(Permission(g) for g in args.grant)}))
    async with contextlib.AsyncExitStack() as stack:
        extra_tools = []
        for spec in args.mcp:
            name, _, command = spec.partition("=")
            argv = shlex.split(command)
            if not name or not argv:
                ap.error(f"--mcp expects NAME=COMMAND [ARGS...], got {spec!r}")
            client = await stack.enter_async_context(McpClient(name, McpServerParams(argv[0], argv[1:])))
            tools = mcp_tools(client, await client.list_tools())
            extra_tools += tools
            print(f"  [mcp {name}: " + ", ".join(f"{t.spec.name} ({t.spec.permission.value})" for t in tools) + "]")
        await run_cli(args, ap, model, extra_tools, policy)


async def run_cli(args, ap, model, extra_tools, policy) -> None:
    store = SQLiteStore(args.db) if args.db else None
    if args.list_sessions:
        for s in await store.list_sessions():
            print(f"{s.id}  {s.status:<16} branch={s.current_branch} checkpoints={s.n_checkpoints}")
        return

    tracer = None
    if args.trace:
        Path(args.trace).write_text("")
        tracer = Tracer(sinks=[JsonlSink(args.trace)])

    def make_agent(p: Provider) -> Agent:
        registry = build_fs_registry(Path(args.workspace))
        for t in extra_tools:
            registry.register(t)
        return Agent(
            p, registry, model=model or "fake", limits=Limits(max_turns=15), store=store, tracer=tracer, policy=policy
        )

    if args.replay:
        report = await replay_run(make_agent, Recording.load(args.replay))
        print(report.summary())
        if tracer:
            print_trace(tracer)
        raise SystemExit(0 if report.ok else 1)

    provider = make_provider(args)
    try:
        if args.record:
            prompt = args.prompt or ("看看工作区里有什么" if args.fake else None)
            if prompt is None:
                ap.error("--record needs --prompt")
            rr = await record_run(make_agent, provider, prompt, path=args.record)
            final = next((m.text() for m in reversed(rr.result.session.messages) if m.role == "assistant"), "")
            print(f"  [recorded {len(rr.recording.calls)} model calls -> {args.record}]\n{final}")
            if rr.result.error:
                print(f"  [run ended with {type(rr.result.error).__name__}: {rr.result.error}]")
            if tracer:
                print_trace(tracer)
            return

        agent = make_agent(provider)
        known = {s.id for s in await store.list_sessions()} if store else set()
        if args.session and args.session in known:
            session = await agent.open_session(args.session)
            print(f"  [resumed {session.id} on {session.branch}: {len(session.messages)} messages, {session.status}]")
            if session.messages[-1].role != "assistant" and not args.prompt:
                print("  [the previous run was interrupted; continuing it]")
                await run_prompt(agent, session, None)
        else:
            session = agent.new_session(args.session)

        if args.prompt or (args.fake and not args.session):
            await run_prompt(agent, session, args.prompt or "看看工作区里有什么")
        elif not args.fake:
            while line := (await asyncio.to_thread(input, "you> ")).strip():
                if store and line == "/cp":
                    for cp in await agent.sessions.checkpoints(session.id):
                        print(f"  #{cp.id:<4} {cp.branch:<10} {cp.label:<40} ({cp.n_messages} msgs, {cp.status})")
                elif store and line.startswith("/rewind "):
                    session = await agent.rewind(session.id, int(line.split()[1]))
                    print(f"  [rewound onto new branch {session.branch}; {len(session.messages)} messages]")
                    if session.messages[-1].role == "user":
                        await run_prompt(agent, session, None)  # retry the answer
                else:
                    await run_prompt(agent, session, line)
        if tracer:
            print_trace(tracer)
    finally:
        closer = getattr(getattr(provider, "_inner", provider), "aclose", None)
        if closer:
            await closer()
        if store:
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
