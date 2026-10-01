"""Shared HTTP+SSE plumbing for vendor adapters (DRY): status mapping, transport-error mapping."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mini_harness.core.errors import ProviderError

DONE: Any = object()  # sentinel yielded for the "[DONE]" line


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None  # HTTP-date form is ignored; the backoff schedule applies


def _retryable_status(status: int) -> bool:
    return status in (408, 425, 429) or status >= 500


async def sse_json(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], headers: dict[str, str]
) -> AsyncIterator[Any]:
    """POST `payload`, yield each SSE `data:` line as parsed JSON (or DONE).

    Every failure is raised as ProviderError with the right `retryable` flag, so adapters and the
    retry layer never need to know about httpx.
    """
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise ProviderError(
                    f"HTTP {resp.status_code}: {body}",
                    status=resp.status_code,
                    retryable=_retryable_status(resp.status_code),
                    retry_after_s=_retry_after(resp),
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    yield DONE
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ProviderError(f"malformed SSE payload: {raw[:100]!r}", retryable=True) from e
                yield data
    except httpx.TransportError as e:  # timeouts, resets, incomplete chunked reads...
        raise ProviderError(f"transport error: {type(e).__name__}: {e}", retryable=True) from e
