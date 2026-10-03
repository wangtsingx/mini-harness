"""Offline eval demo (scripted models, no API key):  uv run python examples/eval_demo.py

Shows the full loop: run a suite -> clustered failures with a likely layer and fix -> save a baseline ->
re-run after a 'change' -> compare against the baseline.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from mini_harness import Agent, Limits, Pricing, ToolRegistry
from mini_harness.eval import EvalCase, EvalRunner, contains, matches
from mini_harness.providers.fake import FakeProvider, text_turn, tool_turn

registry = ToolRegistry()


@registry.tool()
def lookup(key: str) -> str:
    """Look a value up in the knowledge base."""
    if key == "capital_of_france":
        return "Paris"
    raise KeyError(key)


def scripted(script, **kw):
    return lambda tracer: Agent(
        FakeProvider(script), registry, model="demo", tracer=tracer, pricing=Pricing(3.0, 15.0), **kw
    )


def suite(fixed: bool):
    """`fixed` plays the role of 'after my change': one tool-argument bug and one wrong answer get fixed."""
    arg = "capital_of_france" if fixed else "france_capital"
    scenarios = {
        "capital": (
            scripted(
                [
                    tool_turn(("a", "lookup", {"key": arg})),
                    text_turn("The capital is Paris." if fixed else "I could not find it."),
                ]
            ),
            contains("Paris"),
        ),
        "format": (scripted([text_turn("It is forty-two.")]), matches(r"^\d+$")),
        "math": (scripted([text_turn("2+2 is 4." if fixed else "2+2 is 5.")]), contains("4")),
        "loop": (
            scripted([tool_turn(("a", "lookup", {"key": "x"}))] * 8, limits=Limits(max_turns=3)),
            contains("done"),
        ),
        "greeting": (scripted([text_turn("Hello there!")]), contains("hello")),
    }
    return scenarios


async def evaluate(fixed: bool):
    scenarios = suite(fixed)
    cases = [EvalCase(cid, "go", check, tags=("demo",)) for cid, (_, check) in scenarios.items()]
    return await EvalRunner(lambda case, tracer: scenarios[case.id][0](tracer), cases, concurrency=4).run()


async def main() -> None:
    before = await evaluate(fixed=False)
    print(before.render())
    baseline = Path(tempfile.mkdtemp()) / "baseline.json"
    before.save(baseline)

    print("\n" + "=" * 70 + "\nafter the change\n" + "=" * 70)
    after = await evaluate(fixed=True)
    print(after.render())
    print("\nvs baseline:\n" + after.compare(baseline).render())


if __name__ == "__main__":
    asyncio.run(main())
