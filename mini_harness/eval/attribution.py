"""Failure attribution: which LAYER of the system most plausibly caused a failed eval run, with evidence.

This is triage, not proof. Rules run in precedence order and the first match wins; every verdict carries a
confidence and the evidence it is based on. The point is to route a failure to the right kind of fix:

  infra    provider outage, auth, rate limits          -> fix the environment, not the agent
  eval     the checker itself crashed                  -> fix the test
  harness  bug, guardrail too tight, policy denial     -> fix the runtime / its configuration
  context  needed information was compacted away       -> tune compaction / summarizer
  tool     tool errors, unclear schema/description     -> fix the tool
  prompt   format/ambiguity problems                   -> fix instructions
  model    ordinary reasoning error (the residual)     -> stronger model, examples, decomposition
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mini_harness.core.errors import HarnessError, ProviderError
from mini_harness.core.events import Compacted, Done, ToolStarted
from mini_harness.core.messages import Message
from mini_harness.eval.checks import Score, final_text, tool_errors
from mini_harness.eval.replay import RunResult

LAYERS = ("infra", "eval", "harness", "context", "tool", "prompt", "model")

FIXES = {
    "infra": "Provider/environment problem: check credentials, quotas, retry policy and provider status; re-run.",
    "eval": "The checker raised an exception: fix the test, not the agent.",
    "harness": "Runtime issue: fix the bug, or adjust limits/policy (max_turns, token budget, permissions).",
    "context": "Information was lost to compaction: raise the window/threshold, widen the protected tail, "
    "or make the summarizer keep exact identifiers.",
    "tool": "Fix the tool: clearer name/description/schema, better error messages, or fix its runtime failure.",
    "prompt": "Clarify the instructions: state the output format explicitly, add an example, remove ambiguity.",
    "model": "Reasoning error with a healthy run: try a stronger model, add examples, decompose the task; "
    "check variance with repeats before concluding.",
}


@dataclass(frozen=True)
class Attribution:
    layer: str
    subtype: str
    confidence: str  # high | medium | low
    reason: str
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fix(self) -> str:
        return FIXES[self.layer]

    @property
    def key(self) -> str:
        return f"{self.layer}/{self.subtype}"


def _text(messages: list[Message]) -> str:
    out: list[str] = []
    for m in messages:
        for b in m.content:
            out.append(getattr(b, "text", None) or getattr(b, "content", None) or str(getattr(b, "input", "")))
    return "\n".join(str(x) for x in out)


def _tool_error_kind(output: str) -> str:
    if output.startswith("Invalid arguments") or output.startswith("Arguments were not valid JSON"):
        return "invalid_arguments"
    if output.startswith("Unknown tool"):
        return "unknown_tool"
    if output.startswith("Denied by policy"):
        return "policy_denied"
    if output.startswith("Tool timed out"):
        return "timeout"
    return "runtime_error"


def lost_facts(result: RunResult, facts: tuple[str, ...]) -> list[str]:
    """Facts present in compacted-away originals but absent from the context the model finally had."""
    if not facts or not result.session.archive:
        return []
    archived = _text(result.session.archive).lower()
    working = _text(result.session.messages).lower()
    return [f for f in facts if f.lower() in archived and f.lower() not in working]


def attribute(result: RunResult, score: Score) -> Attribution | None:
    """None for a passing run."""
    if score.passed:
        return None
    err = result.error
    if isinstance(err, ProviderError):
        return Attribution("infra", "provider_error", "high", f"provider failed: {err}", (f"status={err.status}",))
    if err is not None:
        sub = "harness_error" if isinstance(err, HarnessError) else "unexpected_exception"
        return Attribution("harness", sub, "high", f"{type(err).__name__}: {err}")
    if score.kind == "checker_error":
        return Attribution("eval", "checker_crashed", "high", score.reason)

    done = next((e for e in reversed(result.events) if isinstance(e, Done)), None)
    stop = done.reason if done else "none"
    errors = tool_errors(result)
    kinds = [_tool_error_kind(e.output) for e in errors]
    ev = [
        f"stop={stop}",
        f"tool_calls={sum(isinstance(e, ToolStarted) for e in result.events)}",
        f"tool_errors={len(errors)}",
    ]
    compactions = [e for e in result.events if isinstance(e, Compacted)]
    if compactions:
        ev.append("compactions=" + ",".join(c.stage for c in compactions))
    sample = [f"{e.name}: {e.output[:100]}" for e in errors[:2]]

    if "policy_denied" in kinds:
        return Attribution(
            "harness", "policy_denied", "high", "a tool call was denied by the permission policy", (*ev, *sample)
        )

    lost = lost_facts(result, score.facts)
    if lost:
        return Attribution(
            "context", "lost_facts", "high",
            f"facts {lost} existed only in compacted-away history", (*ev, *[f"lost: {x}" for x in lost]),
        )  # fmt: skip

    if stop in ("max_turns", "budget_exceeded", "max_tokens", "repeated_calls"):
        if errors:
            worst = max(set(kinds), key=kinds.count)
            return Attribution(
                "tool", worst, "medium", f"run hit '{stop}' while tools kept failing ({worst})", (*ev, *sample)
            )
        if stop == "repeated_calls":
            return Attribution("model", "stuck_in_loop", "medium", "the agent repeated identical tool calls", tuple(ev))
        return Attribution("harness", f"guardrail_{stop}", "medium", f"run stopped by guardrail '{stop}'", tuple(ev))

    if errors:
        worst = max(set(kinds), key=kinds.count)
        if worst == "unknown_tool":
            return Attribution(
                "model", "unknown_tool", "medium", "the model called a tool that does not exist", (*ev, *sample)
            )
        conf = "high" if worst in ("runtime_error", "timeout") else "medium"
        return Attribution("tool", worst, conf, f"{len(errors)} tool call(s) failed ({worst})", (*ev, *sample))

    if compactions:
        return Attribution(
            "context", "compaction_suspected", "low", "context was compacted during a failed run", tuple(ev)
        )

    text = final_text(result).strip()
    if score.kind == "unsafe":
        return Attribution(
            "model", "unsafe_behavior", "medium", score.reason or "unsafe or fabricated output", tuple(ev)
        )
    if score.kind == "format":
        return Attribution("prompt", "format", "medium", score.reason or "answer had the wrong format", tuple(ev))
    if not any(isinstance(e, ToolStarted) for e in result.events) and text.endswith(("?", "？")):
        return Attribution(
            "prompt",
            "ambiguity",
            "medium",
            "the agent asked a question instead of acting",
            (*ev, f"final: {text[:80]}"),
        )

    return Attribution("model", "wrong_answer", "low", score.reason or "healthy run, wrong result", tuple(ev))
