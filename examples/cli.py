"""Demo CLI.

Offline:    python examples/cli.py --fake
Anthropic:  ANTHROPIC_API_KEY=... python examples/cli.py
OpenAI or any compatible server:
            OPENAI_API_KEY=... python examples/cli.py --provider openai --model gpt-4o [--base-url http://localhost:8000/v1]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from mini_harness import Agent, AssistantText, Compacted, Done, Limits, Retrying, ToolFinished, ToolStarted
from mini_harness.providers.base import Provider
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn
from mini_harness.providers.retry import RetryingProvider
from mini_harness.tools.builtin import build_fs_registry


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


async def run_prompt(agent: Agent, session, prompt: str) -> None:
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


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="scripted offline model")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: api.openai.com/v1)")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--prompt", help="one-shot prompt (default: interactive REPL)")
    ap.add_argument("--model", default=os.environ.get("HARNESS_MODEL"))
    args = ap.parse_args()
    model = args.model or ("claude-sonnet-5-5" if args.provider == "anthropic" else None)
    if model is None and not args.fake:
        ap.error("--model (or HARNESS_MODEL) is required for --provider openai")

    provider = make_provider(args)
    agent = Agent(provider, build_fs_registry(Path(args.workspace)), model=model or "fake", limits=Limits(max_turns=15))
    session = agent.new_session()
    try:
        if args.prompt or args.fake:
            await run_prompt(agent, session, args.prompt or "看看工作区里有什么")
        else:
            while (line := await asyncio.to_thread(input, "you> ")).strip():
                await run_prompt(agent, session, line)
    finally:
        closer = getattr(getattr(provider, "_inner", provider), "aclose", None)
        if closer:
            await closer()


if __name__ == "__main__":
    asyncio.run(main())
